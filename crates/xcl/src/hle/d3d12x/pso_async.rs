//! Non-blocking pipeline creation. Translating a console shader takes milliseconds; compiling
//! the HLSL with DXC takes 0.2 s to seconds, and a title creates thousands of pipelines while
//! it loads. When a pipeline's DXIL is not in the disk cache yet, the title gets a stand-in
//! pipeline (shaders that draw nothing) at once and the real one is built on a worker; binding
//! (`SetPipelineState`, console list replay, the two-pass lookups) uses the real one as soon
//! as it is there. Until then a graphics pipeline draws nothing and a compute pipeline's
//! dispatches are skipped: a bind never builds a pipeline on the title's thread, and waits for
//! one only up to `XCL_PSO_WAIT_MS` (`resolve_wait`), so no dispatch is put off to the submit.
//! `XCL_ASYNC_PSO=0` builds every pipeline in the call, as before.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex, OnceLock};

struct Pending {
    /// the build, until a worker takes it
    job: Mutex<Option<Job>>,
    /// the same build, for when the pipeline was released (`shader_code`)
    again: Mutex<Option<Job>>,
    /// its code counts, and a compute one may be released when not used for a while
    releasable: bool,
    /// it was built in this run and released since: the next bind builds it itself
    gone: std::sync::atomic::AtomicBool,
    /// bytes of its compiled code (0: not used yet) and when it was last used
    code: AtomicUsize,
    last: std::sync::atomic::AtomicU64,
    /// the build went to the workers (a lazy one goes on its first `resolve`)
    sent: std::sync::atomic::AtomicBool,
    /// the real pipeline state (raw pointer), 0 while building or when building failed
    real: AtomicUsize,
    /// the driver pipeline binds get till then (`shared`); 0: the stand-in is one itself
    driver: AtomicUsize,
    done: Mutex<bool>,
    cv: Condvar,
    compute: bool,
    /// of its shaders (`used`); 0 = none
    key: u64,
    /// it was used (`used::note` is told once)
    used: std::sync::atomic::AtomicBool,
    /// the stand-in was released before the real pipeline was built
    orphan: std::sync::atomic::AtomicBool,
    /// a bind went on without it (past `wait_budget`, or with none): binds do not wait for it
    /// until it is built
    late: std::sync::atomic::AtomicBool,
    /// its dispatches may be left out while it is built (`image_only`): binds do not wait for it
    spared: std::sync::atomic::AtomicBool,
    demand: Mutex<Demand>,
}

/// Whether one of the title's threads waits for the pipeline, and the preload worker building it.
#[derive(Default)]
struct Demand {
    wanted: bool,
    /// a call put off to the submit needs it (`need_by_submit`)
    by_submit: bool,
    /// thread id of the preload worker (lowest priority) building it, 0 = none
    preload_thread: u32,
}

thread_local! {
    /// what this worker is building
    static BUILDING: std::cell::RefCell<Option<Arc<Pending>>> = const { std::cell::RefCell::new(None) };
}

/// This thread is a worker and one of the title's threads waits for what it is building: the
/// build must not stand back for the title's sake (`real::big_compile_turn`).
pub(super) fn build_is_wanted() -> bool {
    BUILDING.with(|b| b.borrow().as_ref().is_some_and(|p| p.demand.lock().unwrap().wanted))
}

/// Whether one of the title's threads waits for what this thread is doing, to be asked later and
/// from another thread: a worker's build as `build_is_wanted`; a thread that is not XCL's is the
/// title's own.
pub fn wanted_here() -> crate::gpu::recomp::dxc::Wanted {
    if let Some(p) = BUILDING.with(|b| b.borrow().clone()) {
        return Box::new(move || p.demand.lock().unwrap().wanted);
    }
    let own = crate::state::boot().is_some() && !std::thread::current().name().is_some_and(|n| n.starts_with("xcl-"));
    Box::new(move || own)
}

/// A worker's build of `p` (`preload`: by one of the lowest-priority workers).
fn build(p: &Arc<Pending>, job: Job, preload: bool) {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, GetCurrentThreadId, SetThreadPriority, THREAD_PRIORITY_LOWEST, THREAD_PRIORITY_NORMAL};
    BUILDING.with(|b| *b.borrow_mut() = Some(p.clone()));
    if preload {
        let mut d = p.demand.lock().unwrap();
        d.preload_thread = unsafe { GetCurrentThreadId() };
        if d.wanted {
            unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL) };
        }
    }
    run(p, job);
    if preload {
        // (under the lock: `want` does not raise this thread after it)
        let mut d = p.demand.lock().unwrap();
        d.preload_thread = 0;
        if d.wanted {
            unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST) };
        }
    }
    BUILDING.with(|b| *b.borrow_mut() = None);
}

impl Pending {
    /// One of the title's threads is about to wait for it: the worker building it goes on now,
    /// at a normal priority.
    fn want(&self) {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Threading::{OpenThread, SetThreadPriority, THREAD_PRIORITY_NORMAL, THREAD_SET_LIMITED_INFORMATION};
        let mut d = self.demand.lock().unwrap();
        if !std::mem::replace(&mut d.wanted, true) && d.preload_thread != 0 {
            unsafe {
                let h = OpenThread(THREAD_SET_LIMITED_INFORMATION, 0, d.preload_thread);
                if !h.is_null() {
                    SetThreadPriority(h, THREAD_PRIORITY_NORMAL);
                    CloseHandle(h);
                }
            }
        }
    }

    fn note_use(&self) {
        self.last.store(super::shader_code::now_ms(), Ordering::Release);
        if self.key != 0 && !self.used.swap(true, Ordering::Relaxed) {
            used::note(self.key);
        }
    }
}

/// stand-in handed to the title → its real pipeline
static PENDING: Mutex<Option<HashMap<usize, Arc<Pending>>>> = Mutex::new(None);

fn run(p: &Pending, mut job: Job) {
    let real = job();
    if p.releasable {
        *p.again.lock().unwrap() = Some(job);
    }
    complete(p, real);
}

/// The build is done (`real`: its pipeline, 0 = failed).
fn complete(p: &Pending, real: usize) {
    if real == 0 {
        FAILED.fetch_add(1, Ordering::Relaxed);
    } else {
        BUILT.fetch_add(1, Ordering::Relaxed);
    }
    let orphan = {
        let mut done = p.done.lock().unwrap();
        let orphan = p.orphan.load(Ordering::Acquire);
        if !orphan {
            p.real.store(real, Ordering::Release);
        }
        *done = true;
        p.cv.notify_all();
        orphan
    };
    // outside `done`: the Release override takes PENDING (`counts` takes PENDING, then `done`)
    if orphan {
        release(real);
    }
}
static BUILT: AtomicUsize = AtomicUsize::new(0);
static FAILED: AtomicUsize = AtomicUsize::new(0);
/// binds waiting right now for a pipeline still being built (`title_held`)
static HELD: AtomicUsize = AtomicUsize::new(0);

/// The PC calls of the title's threads (`pc_call`): the ones going on, and how long the ones
/// that ended took since the last frame was shown.
struct PcCalls {
    /// (id, when it began)
    now: Vec<(u64, std::time::Instant)>,
    /// the frames shown when `spent` began to count (`None` before the first observation)
    frame: Option<u64>,
    /// when the latest frame advance was first observed; calls that started earlier are clamped here
    frame_since: Option<std::time::Instant>,
    /// the time of the PC calls that ended since then
    spent: std::time::Duration,
}

impl PcCalls {
    const fn new() -> Self {
        Self { now: Vec::new(), frame: None, frame_since: None, spent: std::time::Duration::ZERO }
    }

    /// `frame` frames have been shown: the calls before the last of them are not this wait's.
    fn at(&mut self, frame: u64, now: std::time::Instant) {
        match self.frame {
            None => self.frame = Some(frame),
            Some(prev) if frame > prev => {
                self.frame = Some(frame);
                self.frame_since = Some(now);
                self.spent = std::time::Duration::ZERO;
            }
            Some(_) => {}
        }
    }

    fn clamp(&self, since: std::time::Instant) -> std::time::Instant {
        self.frame_since.map_or(since, |f| since.max(f))
    }

    fn begin(&mut self, id: u64, frame: u64, now: std::time::Instant) {
        self.at(frame, now);
        self.now.push((id, now));
    }

    fn end(&mut self, id: u64, frame: u64, now: std::time::Instant) {
        self.at(frame, now);
        if let Some(i) = self.now.iter().position(|c| c.0 == id) {
            let (_, since) = self.now.swap_remove(i);
            self.spent += now.saturating_duration_since(self.clamp(since));
        }
    }

    /// Whether a call going on holds the title: the PC calls since the last frame, it included,
    /// have taken `PC_CALL_HOLDS_AFTER`.
    fn holds(&mut self, frame: u64, now: std::time::Instant) -> bool {
        self.at(frame, now);
        let spent = self.spent;
        self.now.iter().any(|(_, since)| spent + now.saturating_duration_since(self.clamp(*since)) >= PC_CALL_HOLDS_AFTER)
    }
}

static PC_CALLS: Mutex<PcCalls> = Mutex::new(PcCalls::new());
/// From this much time in PC calls since the last frame on, a PC call holds the title: the
/// console's calls return at once.
const PC_CALL_HOLDS_AFTER: std::time::Duration = std::time::Duration::from_millis(250);

/// Whether one of the title's threads is held where a console title never waits: a bind is
/// waiting for a pipeline still being built, or a PC call is going on after such calls have
/// kept the title's threads for long (`pc_call`). `kernelx::Sleep` does not count that time.
pub fn title_held() -> bool {
    HELD.load(Ordering::Acquire) != 0 || held_by_pc_call()
}

fn held_by_pc_call() -> bool {
    PC_CALLS.lock().unwrap().holds(super::present::presents(), std::time::Instant::now())
}

/// A call of the title into D3D12 that XCL or the PC's driver can keep for long where the
/// console's takes no time: a submission (XCL's uploads and tile mappings before the lists, the
/// driver's compiling in them), a texture's placement, a pipeline built in the call. While it
/// lasts the title is held (`title_held`), once the PC calls since the last frame shown, this
/// one included, have taken `PC_CALL_HOLDS_AFTER`: one long call, or many short ones in a row
/// (a retail title placed 16 384 textures in 29 s without a frame, 1 - 50 ms each). A title
/// that shows its frames is not held by its submissions.
pub struct PcCall(u64);

pub fn pc_call() -> PcCall {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    PC_CALLS.lock().unwrap().begin(id, super::present::presents(), std::time::Instant::now());
    PcCall(id)
}

impl Drop for PcCall {
    fn drop(&mut self) {
        PC_CALLS.lock().unwrap().end(self.0, super::present::presents(), std::time::Instant::now());
    }
}

/// Whether this thread is building a pipeline for `build` (a worker, not one of the title's).
pub(super) fn building_here() -> bool {
    BUILDING.with(|b| b.borrow().is_some())
}

/// A bind's wait for a pipeline still being built (`title_held`).
struct Held;

impl Held {
    fn new() -> Held {
        HELD.fetch_add(1, Ordering::AcqRel);
        Held
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        HELD.fetch_sub(1, Ordering::AcqRel);
    }
}

pub fn enabled() -> bool {
    std::env::var("XCL_ASYNC_PSO").map_or(true, |v| v != "0")
}

/// Builds the real pipeline: its raw pointer, 0 = failed.
pub type Job = Box<dyn FnMut() -> usize + Send>;

/// pipelines a submit will wait for: the workers take these before their queue (`need_by_submit`)
static NEEDED: Mutex<std::collections::VecDeque<Arc<Pending>>> = Mutex::new(std::collections::VecDeque::new());

fn next_needed() -> Option<Arc<Pending>> {
    NEEDED.lock().unwrap().pop_front()
}

/// Drop an unstarted orphaned job outside all locks and wake any concurrent waiter.
fn drop_unstarted_job(p: &Pending) {
    let job = p.job.lock().unwrap().take();
    if let Some(job) = job {
        {
            let mut done = p.done.lock().unwrap();
            *done = true;
            p.cv.notify_all();
        }
        drop(job);
    }
}

type Work = Box<dyn FnOnce() + Send>;

fn queue() -> &'static Mutex<mpsc::Sender<Work>> {
    static Q: OnceLock<Mutex<mpsc::Sender<Work>>> = OnceLock::new();
    Q.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<Work>();
        let rx = Arc::new(Mutex::new(rx));
        // translation recurses deeply on big programs; DXC itself runs on its own pool
        let n = crate::gpu::recomp::dxc::compile_threads();
        for k in 0..n {
            let rx = rx.clone();
            let _ = std::thread::Builder::new().name(format!("xcl-pso-{k}")).stack_size(64 << 20).spawn(move || loop {
                let next = rx.lock().unwrap().recv();
                let Ok(work) = next else { return };
                while let Some(p) = next_needed() {
                    build_queued(&p);
                }
                work();
            });
        }
        Mutex::new(tx)
    })
}

/// Run a background PSO build task on the `xcl-pso-*` worker pool.
pub(super) fn run_on_worker(work: impl FnOnce() + Send + 'static) {
    let _ = queue().lock().unwrap().send(Box::new(work));
}

fn enqueue(p: Arc<Pending>) {
    let _ = queue().lock().unwrap().send(Box::new(move || build_queued(&p)));
}

fn build_queued(p: &Arc<Pending>) {
    // (released by the title meanwhile)
    if p.orphan.load(Ordering::Acquire) {
        drop_unstarted_job(p);
        return;
    }
    // (None: a preload worker took it)
    let job = p.job.lock().unwrap().take();
    if let Some(job) = job {
        build(p, job, false);
    }
}

/// The caller also drains its batch. Multiple workers can therefore build
/// different chains without ever waiting for a free worker to start a piece.
pub(super) fn build_pieces(jobs: Vec<Job>) -> Vec<usize> {
    let count = jobs.len();
    let jobs = Arc::new(Mutex::new(jobs.into_iter().enumerate().collect::<Vec<_>>()));
    let (send, recv) = mpsc::channel();
    let drain = move || loop {
        let job = jobs.lock().unwrap().pop();
        let Some((i, mut job)) = job else { return };
        let _ = send.send((i, job()));
    };
    // the helpers build for this worker's pipeline: the title waiting for it waits for them too
    let parent = BUILDING.with(|b| b.borrow().clone());
    for _ in 1..count.min(crate::gpu::recomp::dxc::compile_threads()) {
        let (drain, parent) = (drain.clone(), parent.clone());
        let _ = queue().lock().unwrap().send(Box::new(move || {
            let before = BUILDING.with(|b| b.replace(parent));
            drain();
            BUILDING.with(|b| *b.borrow_mut() = before);
        }));
    }
    drain();
    let mut results = vec![0; count];
    for (i, result) in recv.iter().take(count) { results[i] = result; }
    results
}

/// Release a pipeline state XCL owns (raw pointer, 0 = none).
pub fn release(p: usize) {
    if p != 0 {
        unsafe { drop(<windows::Win32::Graphics::Direct3D12::ID3D12PipelineState as windows::core::Interface>::from_raw(p as *mut std::ffi::c_void)) };
    }
}

/// `standin` went to the title; `job` builds the real pipeline on a worker.
pub fn submit(standin: usize, compute: bool, job: Job) {
    submit_as(standin, compute, job, true, 0, super::shader_code::limit() != 0);
}

/// Lazy pipeline creation (`XCL_LAZY_PSO=0`: off): a title's pipelines with console shaders get
/// their stand-in when made and are built when first used – the first bind of one sends it to
/// the workers. A retail game makes ~51 000 pipelines in its first 30 s and uses few; each one
/// built stays in the driver (~400 KB), and the process ran out of commit.
pub fn lazy() -> bool {
    static L: OnceLock<bool> = OnceLock::new();
    *L.get_or_init(|| enabled() && std::env::var("XCL_LAZY_PSO").map_or(true, |v| v != "0"))
}

/// `submit`, sent to the workers only on its first `resolve` (`lazy`) – or now, at a low
/// priority, when an earlier run used a pipeline of these shaders (`key`, 0 = none; `used`).
pub fn submit_lazy(standin: usize, compute: bool, job: Job, key: u64) {
    submit_as(standin, compute, job, false, key, super::shader_code::limit() != 0);
}

/// Register a permanently gated stand-in pipeline (e.g. RT compute held back under
/// `XCL_RT_GATE=hold`) with `done = true`, `job = None`, `sent = true`, and `real = 0`.
/// Never enqueues a worker job, never translates or compiles, maps `standin -> driver` cleanly
/// for `SetPipelineState`/`resolve`/`resolve_wait`, reports `is_pending(standin) == false`,
/// and releases cleanly via `released()`.
pub fn submit_gated(standin: usize, driver: usize) {
    let p = Arc::new(Pending {
        again: Mutex::new(None),
        releasable: false,
        gone: Default::default(),
        code: AtomicUsize::new(0),
        last: Default::default(),
        job: Mutex::new(None),
        sent: std::sync::atomic::AtomicBool::new(true),
        real: AtomicUsize::new(0),
        driver: AtomicUsize::new(driver),
        done: Mutex::new(true),
        cv: Condvar::new(),
        compute: true,
        key: 0,
        used: Default::default(),
        orphan: Default::default(),
        late: std::sync::atomic::AtomicBool::new(true),
        spared: std::sync::atomic::AtomicBool::new(false),
        demand: Default::default(),
    });
    PENDING.lock().unwrap().get_or_insert_with(HashMap::new).insert(standin, p);
}

fn submit_as(standin: usize, compute: bool, job: Job, send: bool, key: u64, releasable: bool) {
    let p = Arc::new(Pending {
        again: Mutex::new(None),
        releasable,
        gone: Default::default(),
        code: AtomicUsize::new(0),
        last: Default::default(),
        job: Mutex::new(Some(job)),
        sent: std::sync::atomic::AtomicBool::new(send),
        real: AtomicUsize::new(0),
        driver: AtomicUsize::new(0),
        done: Mutex::new(false),
        cv: Condvar::new(),
        compute,
        key,
        used: Default::default(),
        orphan: Default::default(),
        late: Default::default(),
        spared: Default::default(),
        demand: Default::default(),
    });
    PENDING.lock().unwrap().get_or_insert_with(HashMap::new).insert(standin, p.clone());
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        crate::xlog_info!("D3D12.X: pipelines whose shaders are not compiled yet get a stand-in and are built in the background");
    });
    if send {
        enqueue(p);
    } else if used::known(key) {
        if let Some(q) = preload() {
            q.list.lock().unwrap().push_back(p);
            q.cv.notify_one();
        }
    }
}

/// The keys of the pipelines earlier runs used (`used-pipelines.txt` beside the DXIL cache):
/// what `xcl shaders --precompile --used-only` builds.
pub fn earlier_used_keys() -> std::collections::HashSet<u64> {
    used::parse(&std::fs::read_to_string(crate::gpu::recomp::dxc::cache_dir().join("used-pipelines.txt")).unwrap_or_default())
}

/// The pipelines a title used, by a hash of their shaders, kept between runs
/// (`used-pipelines.txt` beside the DXIL cache, a key a line).
///
/// A lazy pipeline is built at its first use: until then a graphics one draws nothing, and a
/// compute one's dispatches are skipped (after a wait of up to `XCL_PSO_WAIT_MS`). A retail
/// title's first frames of its world use hundreds of pipelines for the first time at once. The
/// pipelines of an earlier run are built while the title loads instead (`preload`): measured on
/// a build that skips a dispatch instead of waiting for its pipeline, ~2 170 dispatches were
/// skipped in those frames without this and 94 – 150 with it.
mod used {
    use std::sync::{Arc, Condvar, Mutex, OnceLock, RwLock};
    use std::{collections::HashSet, fmt::Write as _, io::Write, path::PathBuf, thread::JoinHandle, time::{Duration, Instant}};

    /// Coalesce first-use bursts into a single disk write off the bind lock.
    const COALESCE_WINDOW: Duration = Duration::from_millis(25);
    /// Flush immediately once a burst reaches this many unique keys.
    const BATCH_IMMEDIATE: usize = 64;

    enum Sink {
        LazyFile(PathBuf, Option<std::fs::File>),
        #[cfg(test)]
        Writer(Box<dyn Write + Send>),
    }

    impl Sink {
        fn write_batch(&mut self, bytes: &[u8]) {
            match self {
                Self::LazyFile(path, file) => {
                    if file.is_none() {
                        if let Some(dir) = path.parent() { let _ = std::fs::create_dir_all(dir); }
                        *file = std::fs::OpenOptions::new().create(true).append(true).open(&*path).ok();
                    }
                    if file.as_mut().is_some_and(|f| f.write_all(bytes).is_err() || f.flush().is_err()) { *file = None; }
                }
                #[cfg(test)]
                Self::Writer(w) => if w.write_all(bytes).is_ok() { let _ = w.flush(); },
            }
        }
    }

    #[derive(Default)]
    struct JournalState {
        seen: HashSet<u64>,
        pending: Vec<u64>,
        enqueued_seq: u64,
        completed_seq: u64,
        flush_now: bool,
        closed: bool,
        writer_stopped: bool,
    }

    struct JournalShared {
        state: Mutex<JournalState>,
        wake_cv: Condvar,
        flushed_cv: Condvar,
    }

    struct StopGuard<'a>(&'a JournalShared);
    impl Drop for StopGuard<'_> {
        fn drop(&mut self) {
            self.0.state.lock().unwrap_or_else(|e| e.into_inner()).writer_stopped = true;
            self.0.flushed_cv.notify_all();
        }
    }

    pub(super) struct Journal {
        shared: Arc<JournalShared>,
        writer: Option<JoinHandle<()>>,
    }

    impl Journal {
        fn new_shared(state: JournalState) -> Arc<JournalShared> {
            Arc::new(JournalShared { state: Mutex::new(state), wake_cv: Condvar::new(), flushed_cv: Condvar::new() })
        }
        fn in_memory(seen: HashSet<u64>) -> Self {
            Self { shared: Self::new_shared(JournalState { seen, writer_stopped: true, ..Default::default() }), writer: None }
        }
        fn with_sink(seen: HashSet<u64>, sink: Sink, coalesce: Duration) -> Self {
            let shared = Self::new_shared(JournalState { seen, ..Default::default() });
            let bg = Arc::clone(&shared);
            let writer = std::thread::Builder::new().name("xcl-pso-used".into()).spawn(move || writer_loop(&bg, sink, coalesce)).ok();
            if writer.is_none() {
                shared.state.lock().unwrap().writer_stopped = true;
                shared.flushed_cv.notify_all();
            }
            Self { shared, writer }
        }
        #[cfg(test)]
        pub(super) fn with_writer(w: Box<dyn Write + Send>, c: Duration) -> Self { Self::with_sink(HashSet::new(), Sink::Writer(w), c) }
        #[cfg(test)]
        pub(super) fn with_file(p: PathBuf, c: Duration) -> Self { Self::with_sink(HashSet::new(), Sink::LazyFile(p, None), c) }

        pub(super) fn note(&self, key: u64) {
            if key == 0 { return; }
            let mut g = self.shared.state.lock().unwrap();
            if !g.seen.insert(key) || g.writer_stopped { return; }
            g.pending.push(key);
            g.enqueued_seq += 1;
            if matches!(g.pending.len(), 1 | BATCH_IMMEDIATE) { self.shared.wake_cv.notify_one(); }
        }

        #[cfg(test)]
        pub(super) fn noted(&self, key: u64) -> bool { self.shared.state.lock().unwrap().seen.contains(&key) }

        #[allow(dead_code)]
        pub(super) fn flush(&self) {
            let mut g = self.shared.state.lock().unwrap();
            let target = g.enqueued_seq;
            if g.completed_seq >= target || g.writer_stopped { return; }
            g.flush_now = true;
            self.shared.wake_cv.notify_one();
            while g.completed_seq < target && !g.writer_stopped { g = self.shared.flushed_cv.wait(g).unwrap(); }
        }
    }

    impl Drop for Journal {
        fn drop(&mut self) {
            self.shared.state.lock().unwrap_or_else(|e| e.into_inner()).closed = true;
            self.shared.wake_cv.notify_one();
            if let Some(h) = self.writer.take() { let _ = h.join(); }
        }
    }

    fn writer_loop(shared: &JournalShared, mut sink: Sink, coalesce: Duration) {
        let _stop = StopGuard(shared);
        let mut buf = String::new();
        loop {
            let (batch, batch_seq) = {
                let mut g = shared.state.lock().unwrap();
                while g.pending.is_empty() && !g.closed {
                    if std::mem::take(&mut g.flush_now) { shared.flushed_cv.notify_all(); }
                    g = shared.wake_cv.wait(g).unwrap();
                }
                if g.pending.is_empty() && g.closed { break; }
                if !coalesce.is_zero() {
                    let deadline = Instant::now() + coalesce;
                    while g.pending.len() < BATCH_IMMEDIATE && !g.flush_now && !g.closed {
                        let Some(rem) = deadline.checked_duration_since(Instant::now()).filter(|d| !d.is_zero()) else { break };
                        let (guard, res) = shared.wake_cv.wait_timeout(g, rem).unwrap();
                        g = guard;
                        if res.timed_out() { break; }
                    }
                }
                g.flush_now = false;
                (std::mem::take(&mut g.pending), g.enqueued_seq)
            };
            buf.clear();
            buf.reserve(batch.len() * 17);
            for key in batch { let _ = writeln!(buf, "{key:016x}"); }
            sink.write_batch(buf.as_bytes());
            shared.state.lock().unwrap().completed_seq = batch_seq;
            shared.flushed_cv.notify_all();
        }
    }

    struct Used {
        earlier: RwLock<HashSet<u64>>,
        now: Journal,
    }

    /// The keys in a `used-pipelines.txt` (lines that are not a hexadecimal key are passed over).
    pub fn parse(text: &str) -> HashSet<u64> {
        text.lines().filter_map(|l| u64::from_str_radix(l.trim(), 16).ok()).filter(|&k| k != 0).collect()
    }

    fn get() -> &'static Used {
        static U: OnceLock<Used> = OnceLock::new();
        U.get_or_init(|| {
            // (a test neither reads nor writes the user's file)
            if cfg!(test) {
                return Used { earlier: RwLock::new(HashSet::new()), now: Journal::in_memory(HashSet::new()) };
            }
            let path = crate::gpu::recomp::dxc::cache_dir().join("used-pipelines.txt");
            let earlier = parse(&std::fs::read_to_string(&path).unwrap_or_default());
            if !earlier.is_empty() {
                crate::xlog_info!("D3D12.X: the pipelines of {} shader sets an earlier run used are built ahead of their first use", earlier.len());
            }
            let now = Journal::with_sink(earlier.clone(), Sink::LazyFile(path, None), COALESCE_WINDOW);
            Used { earlier: RwLock::new(earlier), now }
        })
    }

    /// An earlier run used a pipeline of these shaders.
    pub fn known(key: u64) -> bool {
        key != 0 && get().earlier.read().unwrap().contains(&key)
    }

    /// A pipeline of these shaders is used now: kept for the next run.
    pub fn note(key: u64) {
        get().now.note(key);
    }

    /// (tests) This run used a pipeline of these shaders.
    #[cfg(test)]
    pub fn noted(key: u64) -> bool {
        get().now.noted(key)
    }

    #[cfg(test)]
    pub fn earlier_run_used(key: u64) {
        get().earlier.write().unwrap().insert(key);
    }
}

/// The workers that build the pipelines of an earlier run ahead of their first use
/// (`XCL_PSO_PRELOAD=0`: off). Apart from the ones a bind sends to (`queue`) and at the lowest
/// priority: a pipeline that is bound is not held up by the ones made ahead (whoever takes
/// `job` first builds it), and the title's own threads are not either.
struct Preload {
    list: Mutex<std::collections::VecDeque<Arc<Pending>>>,
    cv: Condvar,
}

fn preload() -> Option<&'static Preload> {
    static Q: OnceLock<Option<Preload>> = OnceLock::new();
    static STARTED: std::sync::Once = std::sync::Once::new();
    let q = Q
        .get_or_init(|| {
            if std::env::var("XCL_PSO_PRELOAD").is_ok_and(|v| v.trim() == "0") {
                return None;
            }
            Some(Preload { list: Mutex::new(Default::default()), cv: Condvar::new() })
        })
        .as_ref()?;
    STARTED.call_once(|| {
        let n = (crate::gpu::recomp::dxc::compile_threads() / 4).clamp(2, 4);
        for k in 0..n {
            let _ = std::thread::Builder::new().name(format!("xcl-pso-pre-{k}")).stack_size(64 << 20).spawn(move || {
                unsafe {
                    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_LOWEST};
                    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
                }
                static DONE: AtomicUsize = AtomicUsize::new(0);
                loop {
                    let p = {
                        let mut g = q.list.lock().unwrap();
                        loop {
                            if let Some(p) = g.pop_front() {
                                break p;
                            }
                            g = q.cv.wait(g).unwrap();
                        }
                    };
                    // (released by the title meanwhile)
                    if p.orphan.load(Ordering::Acquire) {
                        drop_unstarted_job(&p);
                        continue;
                    }
                    // (None: a worker a bind sent it to took it)
                    let job = p.job.lock().unwrap().take();
                    if let Some(job) = job {
                        build(&p, job, true);
                        let n = DONE.fetch_add(1, Ordering::Relaxed) + 1;
                        if n.is_power_of_two() {
                            crate::xlog_info!("D3D12.X: {n} pipelines built ahead of their first use");
                        }
                    }
                }
            });
        }
    });
    Some(q)
}

#[cfg(test)]
mod used_tests {
    use super::*;

    /// A PC call that has kept one of the title's threads for long holds the title, a short one
    /// does not, and the hold ends with the call.
    #[test]
    fn a_long_pc_call_holds_the_title() {
        let mut calls = PcCalls::new();
        let began = std::time::Instant::now();
        calls.begin(1, 0, began);
        calls.begin(2, 0, began + 2 * PC_CALL_HOLDS_AFTER);
        assert!(!calls.holds(0, began), "a call that has just begun holds the title");
        let later = began + 2 * PC_CALL_HOLDS_AFTER;
        assert!(calls.holds(0, later), "a PC call has lasted {:?}, but the title is not held", 2 * PC_CALL_HOLDS_AFTER);
        calls.end(1, 0, later);
        calls.end(2, 0, later);
        assert!(!calls.holds(0, later), "the calls are over, the title is still held");
    }

    /// Short PC calls add up between two frames: none of them is long, but after many in a row
    /// the one going on holds the title. The title's own time between them is not held, and a
    /// frame shown ends it.
    #[test]
    fn short_pc_calls_between_two_frames_hold_the_title() {
        let mut calls = PcCalls::new();
        let step = PC_CALL_HOLDS_AFTER / 10;
        let mut now = std::time::Instant::now();
        for id in 0..20 {
            calls.begin(id, 7, now);
            assert!(id >= 9 || !calls.holds(7, now + step), "{id} short calls hold the title already");
            now += step;
            calls.end(id, 7, now);
        }
        assert!(!calls.holds(7, now), "no PC call is going on, the title is held");
        calls.begin(20, 7, now);
        assert!(calls.holds(7, now), "twenty short PC calls since the last frame and one going on, but the title is not held");
        assert!(!calls.holds(8, now), "a frame was shown, the calls before it still hold the title");
        calls.end(20, 8, now);
    }

    /// While a bind waits (within its budget) for a pipeline still being built, the title is held
    /// (`title_held`): `kernelx::Sleep` does not count that time.
    #[test]
    fn a_bind_waiting_for_a_build_holds_the_title() {
        let standin = 0x7E57_B002usize;
        let (started, building) = mpsc::channel();
        let (go, finish) = mpsc::channel::<()>();
        submit_lazy(standin, true, Box::new(move || {
            let _ = started.send(());
            let _ = finish.recv();
            0x5252
        }), 0);
        let bind = std::thread::spawn(move || resolve_wait_for(standin, Some(std::time::Duration::from_secs(30))));
        building.recv_timeout(std::time::Duration::from_secs(10)).expect("the pipeline is being built");
        // (built by a worker, with the bind about to wait for it)
        let t = std::time::Instant::now();
        while !title_held() && t.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(title_held(), "a bind waits for a build, but the title is not held");
        go.send(()).unwrap();
        assert_eq!(bind.join().unwrap(), 0x5252);
        let _ = released(standin);
    }

    /// A stand-in that is no driver object (`shared`) never reaches the driver: binds get the
    /// driver pipeline shared with other stand-ins until the real one is built, and after a build
    /// that failed.
    #[test]
    fn a_bind_of_a_shared_stand_in_gets_its_driver_pipeline() {
        let (fails, builds, driver) = (0x7E57_B011usize, 0x7E57_B012usize, 0x7E57_B013usize);
        submit_lazy(fails, false, Box::new(|| 0), 0);
        submit_lazy(builds, false, Box::new(|| 0x1234), 0);
        for s in [fails, builds] {
            shared(s, driver);
            assert_eq!(entry(s).unwrap().stand(s), driver);
        }
        assert_eq!(resolve_wait_for(fails, None), driver, "not built yet");
        assert_eq!(wait_built(fails), driver, "its build failed");
        assert_eq!(resolve(fails), driver);
        assert_eq!(resolve_wait_for(builds, None), driver, "not built yet");
        assert_eq!(wait_built(builds), 0x1234);
        assert_eq!(resolve_for_bind(builds), 0x1234);
        // (0x1234 is no pipeline: taken, not released)
        assert_eq!((released(fails), released(builds)), (0, 0x1234));
    }

    /// A bind does not wait for a compute pipeline that writes only images: it gets the stand-in
    /// at once, with no budget, the build goes to the workers
    /// unwanted, and once built the bind gets the real one. Without `image_only` a bind with a
    /// budget waits for it.
    #[test]
    fn a_bind_does_not_wait_for_a_pipeline_that_writes_only_images() {
        let (spared, waited) = (0x7E57_B0A1usize, 0x7E57_B0A2usize);
        let (go, finish) = mpsc::channel::<()>();
        let (tx, rx) = mpsc::channel();
        submit_lazy(spared, true, Box::new(move || {
            let _ = tx.send(wanted_here()());
            let _ = finish.recv();
            0x5A5A
        }), 0);
        image_only(spared);
        // (on a thread: a bind that waits would wait for `go`, below)
        let (bound_tx, bound) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = bound_tx.send((resolve_wait_for(spared, None), resolve_wait_for(spared, None)));
        });
        let binds = bound.recv_timeout(std::time::Duration::from_secs(5));
        if binds.is_err() {
            let _ = go.send(());
        }
        assert_eq!(binds.expect("the binds waited for the build"), (spared, spared), "still being built: the stand-in");
        assert!(is_pending(spared));
        assert!(!rx.recv_timeout(std::time::Duration::from_secs(10)).expect("a worker builds it"), "nobody waits: the build is not wanted");
        go.send(()).unwrap();
        let start = std::time::Instant::now();
        while is_pending(spared) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(resolve_wait_for(spared, None), 0x5A5A, "built by a worker");
        // the same build without the mark: a bind with a budget waits for it
        submit_lazy(waited, true, Box::new(|| {
            std::thread::sleep(std::time::Duration::from_millis(100));
            0x5B5B
        }), 0);
        assert_eq!(resolve_wait_for(waited, Some(std::time::Duration::from_secs(30))), 0x5B5B);
        let _ = released(spared);
        let _ = released(waited);
    }

    #[test]
    fn a_pipeline_a_submit_needs_is_built_before_the_queue() {
        let order = Arc::new(Mutex::new(Vec::new()));
        let backlog = 8 * crate::gpu::recomp::dxc::compile_threads();
        let standins: Vec<usize> = (0..=backlog).map(|i| 0x7E57_E000usize + i).collect();
        for &standin in &standins {
            let order = order.clone();
            submit(standin, true, Box::new(move || {
                std::thread::sleep(std::time::Duration::from_millis(50));
                order.lock().unwrap().push(standin);
                0x6161
            }));
        }
        let last = *standins.last().unwrap();
        need_by_submit(last);
        let start = std::time::Instant::now();
        while standins.iter().any(|&s| is_pending(s)) && start.elapsed() < std::time::Duration::from_secs(30) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let order = order.lock().unwrap();
        assert_eq!(order.len(), standins.len());
        let at = order.iter().position(|&s| s == last).unwrap();
        assert!(at < order.len() / 2, "made last and needed by a submit, it was built {at} of {}", order.len());
        for &standin in &standins {
            let _ = released(standin);
        }
    }

    /// What a bind got before its pipeline was built stays "not built" for it after the build
    /// ends (`bound_unbuilt`), where `is_pending` already says built: `ExecuteIndirect` skipped
    /// on that and sent the stand-in to the PC call. A failed build's stand-in and a pipeline that
    /// is no stand-in are not unbuilt.
    #[test]
    fn a_stand_in_bound_before_its_build_ended_stays_unbuilt_for_that_bind() {
        let (standin, failed) = (0x7E57_B0D1usize, 0x7E57_B0D2usize);
        let (go, finish) = mpsc::channel::<()>();
        submit_lazy(standin, true, Box::new(move || {
            let _ = finish.recv();
            0x6161
        }), 0);
        // (a bind that does not wait: the stand-in while it is built)
        image_only(standin);
        let bound = resolve_wait_for(standin, None);
        assert_eq!(bound, standin);
        assert!(is_pending(standin) && bound_unbuilt(standin, bound));
        go.send(()).unwrap();
        let start = std::time::Instant::now();
        while is_pending(standin) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(resolve(standin), 0x6161);
        assert!(!is_pending(standin), "built");
        assert!(bound_unbuilt(standin, bound), "the bind got the stand-in, not the pipeline built after it");
        assert!(!bound_unbuilt(standin, resolve_wait_for(standin, None)), "a bind after the build gets the real pipeline");
        submit(failed, true, Box::new(|| 0));
        let start = std::time::Instant::now();
        while is_pending(failed) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(resolve(failed), failed);
        assert!(!bound_unbuilt(failed, failed), "a failed build's stand-in is not waiting for a build");
        assert!(!bound_unbuilt(0x7E57_B0D3, 0x7E57_B0D3), "no stand-in");
        let _ = released(standin);
        let _ = released(failed);
    }

    /// A bind waits for a pipeline still being built no longer than its budget and goes on with
    /// the stand-in; later binds do not wait for it again; once built it is the real one.
    #[test]
    fn a_bind_waits_for_a_slow_pipeline_no_longer_than_its_budget() {
        let standin = 0x7E57_B001usize;
        submit(standin, true, Box::new(|| {
            std::thread::sleep(std::time::Duration::from_millis(600));
            0x5151
        }));
        let budget = Some(std::time::Duration::from_millis(30));
        let t = std::time::Instant::now();
        assert_eq!(resolve_wait_for(standin, budget), standin, "not built within the budget: the stand-in");
        assert!(t.elapsed() < std::time::Duration::from_millis(400), "waited {:?}", t.elapsed());
        let t = std::time::Instant::now();
        assert_eq!(resolve_wait_for(standin, budget), standin);
        assert!(t.elapsed() < std::time::Duration::from_millis(20), "a late pipeline is not waited for again ({:?})", t.elapsed());
        let start = std::time::Instant::now();
        while is_pending(standin) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(resolve_wait_for(standin, budget), 0x5151, "built by a worker");
        let _ = released(standin);
    }

    /// A bind with no budget neither builds a cold pipeline on its own thread nor waits for it,
    /// compute or graphics: it gets the stand-in at once, a worker builds the pipeline wanted, and
    /// once built binds get the real one.
    #[test]
    fn a_bind_with_no_budget_neither_builds_nor_waits_for_the_pipeline() {
        for (standin, compute, real) in [(0x7E57_B0C1usize, true, 0x5C5Cusize), (0x7E57_B0C2, false, 0x5D5D)] {
            let (go, finish) = mpsc::channel::<()>();
            let (tx, rx) = mpsc::channel();
            submit_lazy(standin, compute, Box::new(move || {
                let _ = tx.send((std::thread::current().name().map(str::to_owned), wanted_here()()));
                let _ = finish.recv();
                real
            }), 0);
            // (on a thread: a bind that builds or waits would wait for `go`, below)
            let (bound_tx, bound) = mpsc::channel();
            std::thread::spawn(move || {
                let _ = bound_tx.send((resolve_wait_for(standin, None), resolve_wait_for(standin, None)));
            });
            let binds = bound.recv_timeout(std::time::Duration::from_secs(5));
            if binds.is_err() {
                let _ = go.send(());
            }
            assert_eq!(binds.expect("the binds built or waited for the pipeline"), (standin, standin), "still being built: the stand-in");
            let (thread, wanted) = rx.recv_timeout(std::time::Duration::from_secs(10)).expect("a worker builds it");
            assert!(thread.as_deref().is_some_and(|n| n.starts_with("xcl-pso")), "built on {thread:?}, not on a worker");
            assert!(wanted, "the title binds it: its build is wanted");
            assert!(is_pending(standin));
            go.send(()).unwrap();
            assert_eq!(wait_built(standin), real);
            assert_eq!(resolve_wait_for(standin, None), real, "built: the real one");
            assert!(!is_pending(standin));
            let _ = released(standin);
        }
    }

    /// A compile a worker's build starts is wanted once a bind waits for the build – asked from
    /// another thread, as the DXC workers do; a thread that builds nothing (no title runs here)
    /// wants nothing.
    #[test]
    fn a_waiting_bind_makes_the_builds_compiles_wanted() {
        let standin = 0x7E57_B005usize;
        let (tx, rx) = mpsc::channel();
        submit(standin, true, Box::new(move || {
            tx.send(wanted_here()).unwrap();
            let t = std::time::Instant::now();
            while !build_is_wanted() && t.elapsed() < std::time::Duration::from_secs(10) {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            0x5454
        }));
        let wanted = rx.recv().unwrap();
        assert!(!wanted(), "nobody waits yet");
        assert_eq!(resolve_wait_for(standin, Some(std::time::Duration::from_secs(30))), 0x5454);
        assert!(wanted());
        assert!(!wanted_here()());
        let _ = released(standin);
    }

    /// A worker's build hears that a bind waits for it (and only its own build does).
    #[test]
    fn a_waiting_bind_makes_the_build_wanted() {
        let standin = 0x7E57_B004usize;
        let (started_tx, started) = mpsc::channel();
        submit(standin, true, Box::new(move || {
            let unwanted_first = !build_is_wanted();
            started_tx.send(()).unwrap();
            let t = std::time::Instant::now();
            while !build_is_wanted() && t.elapsed() < std::time::Duration::from_secs(10) {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            if unwanted_first && build_is_wanted() { 0x5353 } else { 0 }
        }));
        started.recv().unwrap();
        assert!(!build_is_wanted(), "this thread builds nothing");
        assert_eq!(resolve_wait_for(standin, Some(std::time::Duration::from_secs(30))), 0x5353, "the bind's wait reached the build");
        let _ = released(standin);
    }

    #[test]
    fn the_keys_of_a_file_are_read_and_other_lines_passed_over() {
        let keys = used::parse("00000000000000ff\n\nnot a key\n  1a2b \n0\n");
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&0xff) && keys.contains(&0x1a2b));
    }

    #[test]
    fn used_note_does_not_block_on_slow_writer() {
        use std::{io::Write, time::Duration};
        struct Blocking(Option<mpsc::Sender<()>>, mpsc::Receiver<()>, Arc<Mutex<Vec<u8>>>);
        impl Write for Blocking {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                if let Some(tx) = self.0.take() { let _ = tx.send(()); let _ = self.1.recv(); }
                self.2.lock().unwrap().extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
        }
        let ((entered_tx, entered_rx), (release_tx, release_rx)) = (mpsc::channel(), mpsc::channel());
        let out = Arc::new(Mutex::new(Vec::new()));
        let j = used::Journal::with_writer(Box::new(Blocking(Some(entered_tx), release_rx, out.clone())), Duration::ZERO);
        j.note(0x1111);
        entered_rx.recv_timeout(Duration::from_secs(5)).expect("writer entered write_all");
        j.note(0x2222);
        j.note(0x3333);
        assert!(j.noted(0x1111) && j.noted(0x2222) && j.noted(0x3333), "note/noted must not block on writer I/O");
        release_tx.send(()).unwrap();
        j.flush();
        assert_eq!(used::parse(std::str::from_utf8(&out.lock().unwrap()).unwrap()), [0x1111, 0x2222, 0x3333].into());
    }

    #[test]
    fn used_burst_coalesces_writes_and_deduplicates_keys() {
        use std::{io::Write, time::Duration};
        struct Counting(Arc<Mutex<(usize, Vec<u8>)>>);
        impl Write for Counting {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                let mut g = self.0.lock().unwrap();
                g.0 += 1;
                g.1.extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
        }
        let rec = Arc::new(Mutex::new((0usize, Vec::new())));
        let j = used::Journal::with_writer(Box::new(Counting(rec.clone())), Duration::from_millis(500));
        for k in (0u64..=40).chain(0u64..=40) { j.note(k); }
        j.flush();
        let (writes, bytes) = { let g = rec.lock().unwrap(); (g.0, g.1.clone()) };
        let text = String::from_utf8(bytes).unwrap();
        assert!(writes >= 1 && writes < 40, "expected coalesced writes (< 40), got {writes}");
        assert_eq!(text.lines().count(), 40);
        assert_eq!(used::parse(&text), (1u64..=40).collect());
    }

    #[test]
    fn used_flush_drop_lazy_file_and_io_error_are_graceful() {
        use std::{io::Write, time::Duration};
        let dir = std::env::temp_dir().join(format!("xcl-used-journal-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("nested").join("used-pipelines.txt");
        let j = used::Journal::with_file(path.clone(), Duration::from_secs(5));
        assert!(!dir.exists() && !path.exists(), "dir/file must open lazily on first batch write");
        j.note(0xAA01);
        j.note(0xAA02);
        j.flush();
        assert_eq!(used::parse(&std::fs::read_to_string(&path).unwrap()), [0xAA01, 0xAA02].into());
        j.note(0xAA03);
        drop(j);
        assert_eq!(used::parse(&std::fs::read_to_string(&path).unwrap()), [0xAA01, 0xAA02, 0xAA03].into());
        let _ = std::fs::remove_dir_all(&dir);

        struct Failing(Arc<AtomicUsize>);
        impl Write for Failing {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                self.0.fetch_add(1, Ordering::Relaxed);
                Err(std::io::Error::other("disk full"))
            }
            fn flush(&mut self) -> std::io::Result<()> { Err(std::io::Error::other("disk full")) }
        }
        let attempts = Arc::new(AtomicUsize::new(0));
        let j = used::Journal::with_writer(Box::new(Failing(attempts.clone())), Duration::ZERO);
        j.note(0xBAD1);
        j.note(0xBAD2);
        j.flush();
        j.note(0xBAD3);
        j.flush();
        assert!(attempts.load(Ordering::Acquire) >= 1 && j.noted(0xBAD1) && j.noted(0xBAD2) && j.noted(0xBAD3));
    }

    /// A lazy pipeline of shaders an earlier run used is built without a bind; another one is
    /// built only at its first use, and that use is kept for the next run.
    #[test]
    fn a_pipeline_an_earlier_run_used_is_built_ahead_of_its_first_use() {
        let (known, unknown) = (0x7E57_7001usize, 0x7E57_7002usize);
        let (known_key, unknown_key) = (0x5EED_0001u64, 0x5EED_0002u64);
        used::earlier_run_used(known_key);
        submit_lazy(known, false, Box::new(|| 0x4321), known_key);
        submit_lazy(unknown, false, Box::new(|| 0x4322), unknown_key);
        let start = std::time::Instant::now();
        while is_pending(known) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!is_pending(known), "built by the workers that build ahead, with no bind");
        assert_eq!(resolve(known), 0x4321);
        assert!(is_pending(unknown), "nobody used it and no earlier run did");
        assert!(!used::noted(unknown_key));
        assert_eq!(resolve_wait_for(unknown, None), unknown, "a bind does not wait for it");
        assert_eq!(wait_built(unknown), 0x4322);
        assert!(used::noted(unknown_key) && used::noted(known_key), "a first use is kept for the next run");
        released(known);
        released(unknown);
    }

    /// A compute pipeline not used for a while is released and built again by its next bind; one
    /// in use, or a thread's, is not freed.
    #[test]
    fn an_unused_compute_pipeline_is_released_and_built_again() {
        use windows::core::Interface;
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Direct3D12::*;
        unsafe {
            let Ok(dev) = crate::gpu::agility::create_device(None, D3D_FEATURE_LEVEL_11_0) else {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            };
            let mut sig = None;
            D3D12SerializeRootSignature(&D3D12_ROOT_SIGNATURE_DESC::default(), D3D_ROOT_SIGNATURE_VERSION_1, &mut sig, None).unwrap();
            let sig = sig.unwrap();
            let rs: ID3D12RootSignature = dev.CreateRootSignature(0, std::slice::from_raw_parts(sig.GetBufferPointer() as *const u8, sig.GetBufferSize())).unwrap();
            let code = super::super::real::compile("[numthreads(1, 1, 1)] void main() {}", "cs_5_0");
            let builds = Arc::new(AtomicUsize::new(0));
            let (n, dev, rs) = (builds.clone(), dev.into_raw() as usize, rs.into_raw() as usize);
            let standin = 0x7E57_C0DEusize;
            let job: Job = Box::new(move || {
                n.fetch_add(1, Ordering::AcqRel);
                let (dev, rs) = (dev as *mut std::ffi::c_void, rs as *mut std::ffi::c_void);
                let dev = ID3D12Device::from_raw_borrowed(&dev).unwrap();
                let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute(rs),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: code.as_ptr() as _, BytecodeLength: code.len() },
                    ..Default::default()
                };
                dev.CreateComputePipelineState::<ID3D12PipelineState>(&desc).map_or(0, |p| p.into_raw() as usize)
            });
            submit_as(standin, true, job, false, 0, true);
            shared(standin, 0x7E57_C0DF);
            let built = wait_built(standin);
            assert_eq!(resolve_wait_for(standin, None), built);
            assert_ne!(built, 0x7E57_C0DF);
            release_unused(0);
            assert_eq!(resolve_wait_for(standin, None), built, "just used");
            let p = entry(standin).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(super::super::shader_code::IDLE_MS + 50));
            release_unused(0);
            assert!(is_pending(standin) && p.real.load(Ordering::Acquire) == 0, "released");
            // (the thread's reference: a list it was about to record keeps it)
            assert!(super::super::shader_code::Ref::new(built).is_some());
            assert_eq!(builds.load(Ordering::Acquire), 1);
            assert_ne!(resolve_wait_for(standin, None), 0x7E57_C0DF);
            assert_eq!(builds.load(Ordering::Acquire), 2);
            // (with a time budget too: not behind what the workers have to build first)
            std::thread::sleep(std::time::Duration::from_millis(super::super::shader_code::IDLE_MS + 50));
            release_unused(0);
            assert!(is_pending(standin));
            assert_ne!(resolve_wait_for(standin, Some(std::time::Duration::from_nanos(1))), 0x7E57_C0DF);
            assert_eq!(builds.load(Ordering::Acquire), 3);
            release(released(standin));
        }
    }

    #[test]
    fn an_unused_graphics_pipeline_is_released_when_over_limit() {
        use windows::core::Interface;
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Direct3D12::*;
        unsafe {
            let Ok(dev) = crate::gpu::agility::create_device(None, D3D_FEATURE_LEVEL_11_0) else {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            };
            let mut sig = None;
            D3D12SerializeRootSignature(&D3D12_ROOT_SIGNATURE_DESC::default(), D3D_ROOT_SIGNATURE_VERSION_1, &mut sig, None).unwrap();
            let sig = sig.unwrap();
            let rs: ID3D12RootSignature = dev.CreateRootSignature(0, std::slice::from_raw_parts(sig.GetBufferPointer() as *const u8, sig.GetBufferSize())).unwrap();
            let code = super::super::real::compile("[numthreads(1, 1, 1)] void main() {}", "cs_5_0");
            let builds = Arc::new(AtomicUsize::new(0));
            let (n, dev, rs) = (builds.clone(), dev.into_raw() as usize, rs.into_raw() as usize);
            let standin = 0x7E57_C0E0usize;
            let job: Job = Box::new(move || {
                n.fetch_add(1, Ordering::AcqRel);
                let (dev, rs) = (dev as *mut std::ffi::c_void, rs as *mut std::ffi::c_void);
                let dev = ID3D12Device::from_raw_borrowed(&dev).unwrap();
                let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute(rs),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: code.as_ptr() as _, BytecodeLength: code.len() },
                    ..Default::default()
                };
                dev.CreateComputePipelineState::<ID3D12PipelineState>(&desc).map_or(0, |p| p.into_raw() as usize)
            });
            submit_as(standin, false, job, false, 0, true);
            shared(standin, 0x7E57_C0E1);
            let built = wait_built(standin);
            assert_eq!(resolve_wait_for(standin, None), built);
            assert_ne!(built, 0x7E57_C0E1);
            release_unused_with(0, super::super::shader_code::IDLE_MS);
            assert_eq!(resolve_wait_for(standin, None), built, "recently used graphics pipeline must stay");
            let p = entry(standin).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(super::super::shader_code::IDLE_MS + 50));
            release_unused_with(0, super::super::shader_code::IDLE_MS);
            assert!(is_pending(standin) && p.real.load(Ordering::Acquire) == 0, "idle graphics pipeline must be released when over limit");
            assert_ne!(resolve_wait_for(standin, None), 0x7E57_C0E1);
            assert_eq!(builds.load(Ordering::Acquire), 2);
            release(released(standin));
        }
    }

    static WORKER_SATURATION_LOCK: Mutex<()> = Mutex::new(());

    /// A queued normal worker job whose stand-in was released before a worker starts it is
    /// skipped as orphaned instead of entering the compiler.
    #[test]
    fn a_released_queued_pipeline_is_not_built_by_normal_workers() {
        let _sat = WORKER_SATURATION_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use std::sync::atomic::AtomicBool;
        use std::time::{Duration, Instant};

        struct DropFlag(Arc<AtomicBool>);
        impl Drop for DropFlag {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }

        let n = crate::gpu::recomp::dxc::compile_threads();
        let (started_tx, started_rx) = mpsc::channel();
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let mut blockers = Vec::with_capacity(n);
        for k in 0..n {
            let standin = 0x7E57_9720usize + k;
            let started_tx = started_tx.clone();
            let gate = gate.clone();
            submit(standin, false, Box::new(move || {
                let _ = started_tx.send(());
                let mut open = gate.0.lock().unwrap();
                while !*open {
                    open = gate.1.wait(open).unwrap();
                }
                0x9720 + k
            }));
            blockers.push(standin);
        }
        for _ in 0..n {
            started_rx.recv_timeout(Duration::from_secs(10)).expect("normal workers saturated");
        }

        let target = 0x7E57_97FFusize;
        let ran = Arc::new(AtomicBool::new(false));
        let dropped = Arc::new(AtomicBool::new(false));
        let ran_job = ran.clone();
        let drop_flag = DropFlag(dropped.clone());
        submit(target, false, Box::new(move || {
            let _keep = &drop_flag;
            ran_job.store(true, Ordering::Release);
            0x9721
        }));
        assert_eq!(released(target), 0);

        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();

        let deadline = Instant::now() + Duration::from_secs(10);
        while !dropped.load(Ordering::Acquire) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(dropped.load(Ordering::Acquire), "queued orphan job was never drained");
        assert!(!ran.load(Ordering::Acquire), "released job reached the compiler");

        for standin in blockers {
            let start = Instant::now();
            while is_pending(standin) && start.elapsed() < Duration::from_secs(10) {
                std::thread::sleep(Duration::from_millis(2));
            }
            let _ = released(standin);
        }
    }

    #[test]
    fn pc_call_crossing_frame_boundary_does_not_charge_prior_frame_time() {
        let mut calls = PcCalls::new();
        let t0 = std::time::Instant::now();
        let ms = std::time::Duration::from_millis;
        calls.begin(1, 7, t0);
        // Frame 8 shown after 290 ms; call 1 ends 10 ms later -> only 10 ms counts on frame 8.
        let t_frame8 = t0 + PC_CALL_HOLDS_AFTER + ms(40);
        assert!(!calls.holds(8, t_frame8), "call from frame 7 must not immediately hold frame 8");
        calls.end(1, 8, t_frame8 + ms(10));
        let t_next = t_frame8 + ms(11);
        calls.begin(2, 8, t_next);
        assert!(!calls.holds(8, t_next), "pre-frame duration leaked into frame 8's budget");
        // A call genuinely stuck >= PC_CALL_HOLDS_AFTER after frame 8 still holds the title.
        assert!(!calls.holds(8, t_next + PC_CALL_HOLDS_AFTER - ms(20)));
        assert!(calls.holds(8, t_next + PC_CALL_HOLDS_AFTER));
        calls.end(2, 8, t_next + PC_CALL_HOLDS_AFTER);

        // Also when frame 9 is first noticed inside end() itself, pre-frame time is clamped to 0.
        let t_f9 = t_next + PC_CALL_HOLDS_AFTER + ms(10);
        calls.begin(3, 8, t_f9);
        let t_f9_end = t_f9 + PC_CALL_HOLDS_AFTER + ms(50);
        calls.end(3, 9, t_f9_end);
        calls.begin(4, 9, t_f9_end + ms(1));
        assert!(!calls.holds(9, t_f9_end + ms(1)), "end() on new frame must not charge prior-frame duration");
        calls.end(4, 9, t_f9_end + ms(1) + PC_CALL_HOLDS_AFTER);

        // A stale lower frame number must never rewind self.frame or wipe frame 9's spent budget.
        let t_stale = t_f9_end + PC_CALL_HOLDS_AFTER + ms(10);
        calls.begin(5, 8, t_stale);
        assert!(calls.holds(9, t_stale), "stale lower frame rewound frame 9 or cleared its spent budget");
        calls.end(5, 9, t_stale);
    }

    #[test]
    fn released_queued_pipeline_drops_job_immediately_before_workers_drain() {
        let _sat = WORKER_SATURATION_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use std::sync::atomic::AtomicBool;
        use std::time::{Duration, Instant};

        struct DropFlag(Arc<AtomicBool>);
        impl Drop for DropFlag {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }
        struct OpenGate(Arc<(Mutex<bool>, Condvar)>);
        impl Drop for OpenGate {
            fn drop(&mut self) {
                *self.0 .0.lock().unwrap() = true;
                self.0 .1.notify_all();
            }
        }

        // Saturate normal workers and verify released() drops the queued job immediately.
        let n = crate::gpu::recomp::dxc::compile_threads();
        let (started_tx, started_rx) = mpsc::channel();
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let open_normal = OpenGate(gate.clone());
        let mut blockers = Vec::with_capacity(n);
        for k in 0..n {
            let standin = 0x7E57_A100usize + k;
            let (tx, g) = (started_tx.clone(), gate.clone());
            submit(standin, false, Box::new(move || {
                let _ = tx.send(());
                let mut open = g.0.lock().unwrap();
                while !*open {
                    open = g.1.wait(open).unwrap();
                }
                0xA100 + k
            }));
            blockers.push(standin);
        }
        for _ in 0..n {
            started_rx.recv_timeout(Duration::from_secs(10)).expect("normal workers saturated");
        }
        let target_n = 0x7E57_A1FFusize;
        let (ran_n, dropped_n) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
        let (ran_job, flag_n) = (ran_n.clone(), DropFlag(dropped_n.clone()));
        submit(target_n, false, Box::new(move || {
            let _keep = &flag_n;
            ran_job.store(true, Ordering::Release);
            0xA1FF
        }));
        assert_eq!(released(target_n), 0);
        assert!(dropped_n.load(Ordering::Acquire), "normal queued job was not dropped immediately on released()");
        drop(open_normal);
        assert!(!ran_n.load(Ordering::Acquire));
        for standin in blockers {
            let start = Instant::now();
            while is_pending(standin) && start.elapsed() < Duration::from_secs(10) {
                std::thread::sleep(Duration::from_millis(2));
            }
            let _ = released(standin);
        }

        // Saturate preload workers and verify released() drops the preloaded job immediately.
        let m = (crate::gpu::recomp::dxc::compile_threads() / 4).clamp(2, 4);
        let (p_tx, p_rx) = mpsc::channel();
        let p_gate = Arc::new((Mutex::new(false), Condvar::new()));
        let open_pre = OpenGate(p_gate.clone());
        let mut p_blockers = Vec::with_capacity(m);
        for k in 0..m {
            let standin = 0x7E57_A200usize + k;
            let key = 0x5EED_A200u64 + k as u64;
            used::earlier_run_used(key);
            let (tx, g) = (p_tx.clone(), p_gate.clone());
            submit_lazy(standin, false, Box::new(move || {
                let _ = tx.send(());
                let mut open = g.0.lock().unwrap();
                while !*open {
                    open = g.1.wait(open).unwrap();
                }
                0xA200 + k
            }), key);
            p_blockers.push(standin);
        }
        for _ in 0..m {
            p_rx.recv_timeout(Duration::from_secs(10)).expect("preload workers saturated");
        }
        let target_p = 0x7E57_A2FFusize;
        let key_p = 0x5EED_A2FFu64;
        used::earlier_run_used(key_p);
        let (ran_p, dropped_p) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
        let (ran_job, flag_p) = (ran_p.clone(), DropFlag(dropped_p.clone()));
        submit_lazy(target_p, false, Box::new(move || {
            let _keep = &flag_p;
            ran_job.store(true, Ordering::Release);
            0xA2FF
        }), key_p);
        assert_eq!(released(target_p), 0);
        assert!(dropped_p.load(Ordering::Acquire), "preload queued job was not dropped immediately on released()");
        drop(open_pre);
        assert!(!ran_p.load(Ordering::Acquire));
        for standin in p_blockers {
            let start = Instant::now();
            while is_pending(standin) && start.elapsed() < Duration::from_secs(10) {
                std::thread::sleep(Duration::from_millis(2));
            }
            let _ = released(standin);
        }
    }
}

fn entry(pso: usize) -> Option<Arc<Pending>> {
    PENDING.lock().unwrap().as_ref()?.get(&pso).cloned()
}

impl Pending {
    /// What a bind of the stand-in `pso` gives the driver while the pipeline is not built.
    fn stand(&self, pso: usize) -> usize {
        match self.driver.load(Ordering::Acquire) {
            0 => pso,
            d => d,
        }
    }
}

impl Pending {
    /// Build it here if it was released: that takes the driver milliseconds (its cache has the
    /// code), and a turn in the workers' queue – behind first builds of seconds – minutes.
    fn build_again(&self) {
        if self.gone.swap(false, Ordering::AcqRel) {
            let job = self.job.lock().unwrap().take();
            if let Some(job) = job {
                run(self, job);
            }
        }
    }

    /// The pipeline to bind for `pso`: this thread's until it is bound (`shader_code`).
    fn take(&self, pso: usize) -> usize {
        let mut first = false;
        let real = {
            let _done = self.done.lock().unwrap();
            let real = self.real.load(Ordering::Acquire);
            if real == 0 {
                return self.stand(pso);
            }
            if self.releasable {
                unsafe { super::shader_code::Ref::new(real) }.map(super::shader_code::hold);
            }
            if self.releasable && self.code.load(Ordering::Acquire) == 0 {
                self.code.store(unsafe { super::shader_code::count(real) }.max(1), Ordering::Release);
                first = true;
            }
            real
        };
        if first {
            let limit = super::shader_code::limit();
            if limit != 0 {
                release_unused(limit);
            }
        } else if self.releasable {
            let limit = super::shader_code::limit();
            if limit != 0 && super::shader_code::used() > limit {
                static LAST_SWEEP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                let now = super::shader_code::now_ms();
                let prev = LAST_SWEEP.load(Ordering::Relaxed);
                if now >= prev + 250 && LAST_SWEEP.compare_exchange(prev, now, Ordering::AcqRel, Ordering::Relaxed).is_ok() {
                    release_unused(limit);
                }
            }
        }
        real
    }
}

/// Release the pipelines used longest ago when there is more compiled code in use than
/// `shader_code::limit`: idle compute pipelines first (`IDLE_MS`), then idle graphics pipelines
/// (`GFX_IDLE_MS`) only if compute eviction alone could not bring usage under `limit`. Their next
/// bind builds them again (from the driver's cache).
fn release_unused(limit: usize) {
    release_unused_with(limit, 10_000);
}

fn release_unused_with(limit: usize, gfx_idle_ms: u64) {
    use super::shader_code::{choose, in_recordings, let_go_done, now_ms, used, IDLE_MS};
    if used() <= limit {
        return;
    }
    let_go_done();
    let recorded = in_recordings();
    let evict = |compute: bool, idle_ms: u64| -> usize {
        let now = now_ms();
        let all: Vec<Arc<Pending>> = PENDING.lock().unwrap().as_ref().map_or(Vec::new(), |m| {
            m.values()
                .filter(|p| {
                    p.releasable
                        && p.compute == compute
                        && p.driver.load(Ordering::Acquire) != 0
                        && p.code.load(Ordering::Acquire) != 0
                        && p.last.load(Ordering::Acquire) + idle_ms <= now
                        && !recorded.contains(&p.real.load(Ordering::Acquire))
                })
                .cloned()
                .collect()
        });
        let sizes: Vec<(usize, u64)> = all.iter().map(|p| (p.code.load(Ordering::Acquire), p.last.load(Ordering::Acquire))).collect();
        let mut n = 0;
        for i in choose(&sizes, used(), limit, now) {
            let p = &all[i];
            let real = {
                // (a build or a bind of it is going on: not unused)
                let Ok(mut done) = p.done.try_lock() else { continue };
                if !*done || p.last.load(Ordering::Acquire) + idle_ms > now_ms() || p.orphan.load(Ordering::Acquire) {
                    continue;
                }
                let real = p.real.swap(0, Ordering::AcqRel);
                if real == 0 {
                    continue;
                }
                p.code.store(0, Ordering::Release);
                *p.job.lock().unwrap() = p.again.lock().unwrap().take();
                p.sent.store(false, Ordering::Release);
                p.late.store(false, Ordering::Release);
                p.gone.store(true, Ordering::Release);
                p.demand.lock().unwrap().by_submit = false;
                *done = false;
                real
            };
            release(real);
            n += 1;
        }
        n
    };
    let n_cs = evict(true, IDLE_MS);
    let n_gfx = if used() > limit { evict(false, gfx_idle_ms) } else { 0 };
    let n = n_cs + n_gfx;
    if n != 0 {
        static RELEASED: AtomicUsize = AtomicUsize::new(0);
        let all = RELEASED.fetch_add(n, Ordering::Relaxed) + n;
        crate::xlog_info!(
            "D3D12.X: {n} pipelines not used lately released ({n_cs} compute, {n_gfx} graphics; {} MB of shader code left in use; {all} released so far)",
            used() >> 20
        );
    }
}

/// Stand-in `standin` is no driver object (`standin::shell`): binds get `driver` in its place.
/// Told before the title has the stand-in.
pub fn shared(standin: usize, driver: usize) {
    if let Some(p) = entry(standin) {
        p.driver.store(driver, Ordering::Release);
    }
}

/// The pipeline to use for `pso` now: the real one once built, else the stand-in's driver
/// pipeline (`pso` itself when it is one).
pub fn resolve(pso: usize) -> usize {
    let Some(p) = entry(pso) else { return pso };
    p.note_use();
    match p.real.load(Ordering::Acquire) {
        0 => {
            // a lazy one is wanted now: to the workers
            let stand = p.stand(pso);
            if !p.sent.swap(true, Ordering::AcqRel) {
                enqueue(p);
            }
            stand
        }
        _ => p.take(pso),
    }
}

/// `resolve` for a bind that needs the pipeline itself (a dispatch, a draw): waits up to
/// `wait_budget` for a pipeline still being built, else gets the stand-in at once.
pub fn resolve_wait(pso: usize) -> usize {
    resolve_wait_for(pso, wait_budget())
}

/// `resolve_wait`, waiting at most `budget` (None: not at all).
fn resolve_wait_for(pso: usize, budget: Option<std::time::Duration>) -> usize {
    let Some(p) = entry(pso) else { return pso };
    p.note_use();
    if p.spared.load(Ordering::Acquire) {
        return resolve_spared(&p, pso);
    }
    let Some(budget) = budget else { return resolve_unwaited(&p, pso) };
    // (the title's thread is held from here until the pipeline is built or the budget is over)
    let _held = (!*p.done.lock().unwrap()).then(Held::new);
    resolve_within(&p, pso, budget)
}

/// How long a bind waits for a pipeline still being built (`XCL_PSO_WAIT_MS`; unset or 0: not at
/// all, `resolve_unwaited`). A retail game builds compute programs on its own threads: the driver
/// takes up to 30 s for the biggest translated ones, and the title's watchdog ends it when its
/// main thread has been held 30 s.
pub(super) fn wait_budget() -> Option<std::time::Duration> {
    static B: OnceLock<Option<std::time::Duration>> = OnceLock::new();
    *B.get_or_init(|| std::env::var("XCL_PSO_WAIT_MS").ok().and_then(|v| v.trim().parse::<u64>().ok()).filter(|&ms| ms > 0).map(std::time::Duration::from_millis))
}

/// Whether a bind or a submit waits for a pipeline still being built until it is built. Not since
/// a bind with no budget gets the stand-in at once (`resolve_unwaited`): `list_x::defer` then puts
/// no call off to the submit.
pub fn waits_until_built() -> bool {
    false
}

/// `resolve_wait` for at most `budget`: the build goes to the workers (not this thread), and a
/// pipeline not built by then is `late` – it and later binds of it get the stand-in (their
/// dispatches are skipped) until a worker has built it.
fn resolve_within(p: &Arc<Pending>, pso: usize, budget: std::time::Duration) -> usize {
    p.build_again();
    if *p.done.lock().unwrap() {
        return p.take(pso);
    }
    if !p.sent.swap(true, Ordering::AcqRel) {
        enqueue(p.clone());
    }
    if p.late.load(Ordering::Acquire) {
        return p.stand(pso);
    }
    p.want();
    let (done, _) = p.cv.wait_timeout_while(p.done.lock().unwrap(), budget, |d| !*d).unwrap();
    let built = *done;
    drop(done);
    if built {
        return p.take(pso);
    }
    p.late.store(true, Ordering::Release);
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    if n < 16 || n.is_power_of_two() {
        crate::xlog_warn!("D3D12.X: a pipeline still being built after {} ms: its binds go on without it until it is built ({} so far)", budget.as_millis(), n + 1);
    }
    p.stand(pso)
}

/// `resolve_wait` with no budget: the real pipeline once built, else the stand-in at once (its
/// dispatches and draws are left out), with the build sent to the workers.
///
/// The title's thread neither builds the pipeline nor waits for it. A bind that built a cold
/// pipeline itself and waited for it kept a retail title's thread in the driver's compile of a
/// 2.4 MB compute program for 466 s, with no frame shown after it (it never finished). Unlike
/// `resolve_spared` the build is `wanted`: the title binds it now, so it goes before the builds
/// made ahead and does not wait for free commit (up to `real::big_compile_turn`'s cap).
fn resolve_unwaited(p: &Arc<Pending>, pso: usize) -> usize {
    p.build_again();
    match p.real.load(Ordering::Acquire) {
        0 => {
            // (before it is sent: the worker's compiles see it wanted from their start)
            p.want();
            if !p.sent.swap(true, Ordering::AcqRel) {
                enqueue(p.clone());
            }
            if !*p.done.lock().unwrap() && !p.late.swap(true, Ordering::AcqRel) {
                static N: AtomicUsize = AtomicUsize::new(0);
                let n = N.fetch_add(1, Ordering::Relaxed);
                if n < 16 || n.is_power_of_two() {
                    crate::xlog_info!("D3D12.X: a pipeline bound while it is still being built: its binds go on without it until it is ({} so far)", n + 1);
                }
            }
            p.stand(pso)
        }
        _ => p.take(pso),
    }
}

/// `XCL_PSO_IMAGE_SKIP=1`: a bind does not wait for a big compute program that writes nothing
/// but images (`gpu::effects::big_and_image_only`); its dispatches are left out until a worker
/// has built it. Off by default: what such a program draws is missing for that long.
///
/// A retail title's world start binds one ray-tracing resolve program per material (73 – 118
/// KiB of console code, about 1 MB of DXIL each): an RDNA2 driver takes 22 – 172 s for each,
/// one bind at a time, and the title stands still for all of them. `XCL_PSO_WAIT_MS` goes on without every late
/// program, also the culling passes whose counts an `ExecuteIndirect` reads next – a GPU hang.
pub fn image_skip() -> bool {
    static S: OnceLock<bool> = OnceLock::new();
    *S.get_or_init(|| std::env::var("XCL_PSO_IMAGE_SKIP").is_ok_and(|v| v.trim() == "1"))
}

/// The compute pipeline behind stand-in `standin` writes nothing but images: binds go on
/// without it while it is built (`resolve_spared`). Told before the title has the stand-in.
pub fn image_only(standin: usize) {
    if let Some(p) = entry(standin) {
        p.spared.store(true, Ordering::Release);
    }
}

/// `resolve_wait` for a pipeline binds do not wait for (`image_only`): the real one once built,
/// else the stand-in, with the build sent to the workers. Nobody waits, so the build is not
/// `wanted`: a big one takes its turn among the driver's compiles and waits for free commit.
fn resolve_spared(p: &Arc<Pending>, pso: usize) -> usize {
    p.build_again();
    match p.real.load(Ordering::Acquire) {
        0 => {
            if !p.sent.swap(true, Ordering::AcqRel) {
                enqueue(p.clone());
            }
            if !*p.done.lock().unwrap() && !p.late.swap(true, Ordering::AcqRel) {
                static N: AtomicUsize = AtomicUsize::new(0);
                let n = N.fetch_add(1, Ordering::Relaxed);
                if n < 16 || n.is_power_of_two() {
                    crate::xlog_info!("D3D12.X: a compute program that writes only images is still being built: its dispatches are left out until it is ({} so far)", n + 1);
                }
            }
            p.stand(pso)
        }
        _ => p.take(pso),
    }
}

/// What stand-in `pso` is (being) built from, for a diagnostic line (`dred::mark_pipeline`: a
/// pipeline bound with no label of its own). None: not a stand-in.
pub fn describe(pso: usize) -> Option<String> {
    let p = entry(pso)?;
    let state = if p.real.load(Ordering::Acquire) != 0 { "" } else { ", not built yet" };
    Some(match (p.compute, p.key) {
        (true, 0) => format!("a compute pipeline {pso:#x}{state}"),
        (true, key) => format!("cs {key:016x}{state}"),
        (false, key) => format!("a graphics pipeline {pso:#x} (shaders {key:016x}){state}"),
    })
}

/// For binding: compute pipelines are waited for up to `XCL_PSO_WAIT_MS` (`resolve_wait`),
/// graphics ones are not.
pub fn resolve_for_bind(pso: usize) -> usize {
    match entry(pso) {
        Some(p) if p.compute => resolve_wait(pso),
        Some(_) => resolve(pso),
        None => pso,
    }
}

/// The title released stand-in `pso` for good: forget it (its address may come back for
/// another object) and return its real pipeline to release (0: none, or still being built –
/// the worker's result is then released when it arrives).
pub fn released(pso: usize) -> usize {
    let Some(p) = PENDING.lock().unwrap().as_mut().and_then(|m| m.remove(&pso)) else { return 0 };
    {
        let done = p.done.lock().unwrap();
        if *done {
            let real = p.real.swap(0, Ordering::AcqRel);
            drop(done);
            super::shader_code::let_go(real);
            return real;
        }
        // the worker stores its result into a Pending nobody will read: mark it orphaned
        p.orphan.store(true, Ordering::Release);
    }
    drop_unstarted_job(&p);
    0
}

/// The pipeline of stand-in `pso` once its build is over: the real one, or the stand-in's
/// driver pipeline when the build failed; `pso` itself at once when it is no stand-in (also with
/// `XCL_ASYNC_PSO=0`: pipelines are built in the call, none is pending). Waits as long as the
/// build takes – for tools and tests, never for a bind (`resolve_wait`).
pub fn wait_built(pso: usize) -> usize {
    let Some(p) = entry(pso) else { return pso };
    if !p.sent.swap(true, Ordering::AcqRel) {
        enqueue(p.clone());
    }
    let mut done = p.done.lock().unwrap();
    while !*done {
        done = p.cv.wait(done).unwrap();
    }
    drop(done);
    match p.real.load(Ordering::Acquire) {
        0 => p.stand(pso),
        r => r,
    }
}

/// `pso` is a stand-in whose real pipeline is still being built.
pub fn is_pending(pso: usize) -> bool {
    entry(pso).is_some_and(|p| !*p.done.lock().unwrap())
}

/// One of the title's threads is about to wait for `pso`.
pub fn want(pso: usize) {
    if let Some(p) = entry(pso) {
        p.want();
    }
}

/// A call with `pso` was put off to the submit: its build goes before the workers' queue.
pub fn need_by_submit(pso: usize) {
    let Some(p) = entry(pso) else { return };
    if !std::mem::replace(&mut p.demand.lock().unwrap().by_submit, true) {
        NEEDED.lock().unwrap().push_back(p);
        let _ = queue().lock().unwrap().send(Box::new(|| ()));
    }
}

/// `bound` – what `resolve_wait(standin)` gave a bind – is the stand-in of a pipeline that was not
/// built yet then: still being built, or built since. Not after a failed build (that one is the
/// stand-in for good). Unlike `is_pending(standin)` after the bind, a build that ends between
/// the two does not make the stand-in look like the title's pipeline.
pub fn bound_unbuilt(standin: usize, bound: usize) -> bool {
    entry(standin).is_some_and(|p| {
        let done = p.done.lock().unwrap();
        bound == p.stand(standin) && !(*done && p.real.load(Ordering::Acquire) == 0)
    })
}

/// (built in the background, failed, still building)
pub fn counts() -> (usize, usize, usize) {
    let pending = PENDING.lock().unwrap().as_ref().map_or(0, |m| m.values().filter(|p| !*p.done.lock().unwrap()).count());
    (BUILT.load(Ordering::Relaxed), FAILED.load(Ordering::Relaxed), pending)
}

#[cfg(test)]
mod gated_tests {
    use super::*;

    #[test]
    fn submit_gated_maps_driver_without_pending_or_enqueue() {
        let standin = 0x7E57_9A7Eusize;
        let driver = 0x7E57_9A7Fusize;
        submit_gated(standin, driver);
        assert!(!is_pending(standin));
        assert_eq!(resolve(standin), driver);
        assert_eq!(resolve_wait(standin), driver);
        assert!(!bound_unbuilt(standin, driver));
        assert_eq!(released(standin), 0);
    }
}
