//! Compute pipelines not used for a while are released and made again when they are bound.
//!
//! NVIDIA's driver keeps the machine code of the pipelines its lists use in one heap of at most
//! 512 MB. A submission that needs more removes the device: `DXGI_ERROR_DEVICE_REMOVED`, nothing
//! in DRED and no event in the system log, as there was no GPU fault. The wave64 programs of a
//! console title are several times their size on 32-lane hardware, and a retail title's world is
//! past that in its first seconds. Code goes into the heap when a pipeline is first used and out
//! of it with the pipeline.
//!
//! A pipeline that is released can still be in a recorded list. So `pso_async::resolve` and its
//! kin give the thread a reference to it, a bind hands that to the allocator the list records on
//! (`bound`), and the allocator's `Reset` – the title's word that the GPU is past its lists –
//! gives it back (`allocator_reset`).
//!
//! `XCL_SHADER_CODE_MB`: the compiled code to keep in use (`GetCachedBlob` sizes, which are about
//! 1/9 less than what the code takes in the heap; 0: no limit). Default: 320 on NVIDIA, no limit
//! elsewhere.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::ID3D12PipelineState;

/// a thread's references to pipelines it resolved and has not bound yet, at most
const HELD_MAX: usize = 4;
/// a pipeline used this recently is not released
pub const IDLE_MS: u64 = 1000;
const NVIDIA: u32 = 0x10DE;

/// A reference to a pipeline.
#[derive(PartialEq)]
pub struct Ref(ID3D12PipelineState);

impl Ref {
    /// # Safety
    /// `pso` is a live pipeline.
    pub unsafe fn new(pso: usize) -> Option<Self> {
        ID3D12PipelineState::from_raw_borrowed(&(pso as *mut c_void)).cloned().map(Self)
    }
}

thread_local! {
    static HELD: RefCell<VecDeque<Ref>> = const { RefCell::new(VecDeque::new()) };
}

/// allocator → the pipelines its recordings use
static BOUND: Mutex<Option<HashMap<usize, HashMap<usize, Ref>>>> = Mutex::new(None);
/// pipeline that was used → bytes of its compiled code, until the driver frees it
static SIZES: Mutex<Option<HashMap<usize, usize>>> = Mutex::new(None);
/// their sum
static USED: AtomicUsize = AtomicUsize::new(0);

fn nvidia() -> Option<bool> {
    use windows::Win32::Graphics::Direct3D12::ID3D12Device;
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIAdapter, IDXGIFactory4};
    let raw = super::real::device_raw();
    unsafe {
        let dev = ID3D12Device::from_raw_borrowed(&raw)?;
        let factory: IDXGIFactory4 = CreateDXGIFactory1().ok()?;
        let adapter: IDXGIAdapter = factory.EnumAdapterByLuid(dev.GetAdapterLuid()).ok()?;
        Some(adapter.GetDesc().ok()?.VendorId == NVIDIA)
    }
}

/// How many bytes of compiled code to keep in use (0: no limit).
pub fn limit() -> usize {
    static L: OnceLock<usize> = OnceLock::new();
    *L.get_or_init(|| {
        let mb = std::env::var("XCL_SHADER_CODE_MB").ok().and_then(|v| v.trim().parse().ok());
        // (the tests give `pso_async` pipelines that are none)
        mb.unwrap_or_else(|| if !cfg!(test) && nvidia() == Some(true) { 320usize } else { 0 }) << 20
    })
}

pub fn now_ms() -> u64 {
    static T: OnceLock<Instant> = OnceLock::new();
    T.get_or_init(Instant::now).elapsed().as_millis() as u64
}

/// This thread is about to use pipeline `r`.
pub fn hold(r: Ref) {
    let old = HELD.with(|h| {
        let mut h = h.borrow_mut();
        if h.contains(&r) {
            return Some(r);
        }
        h.push_back(r);
        if h.len() > HELD_MAX {
            h.pop_front()
        } else {
            None
        }
    });
    // (outside the borrow: the last reference's `Release` comes back into XCL)
    drop(old);
}

/// The title is done with the pipeline `pso`: this thread's reference to it goes.
pub fn let_go(pso: usize) {
    let old = HELD.with(|h| {
        let mut h = h.borrow_mut();
        let i = h.iter().position(|r| r.0.as_raw() as usize == pso)?;
        h.remove(i)
    });
    drop(old);
}

/// `list` records with the pipelines this thread resolved: they live until its allocator's `Reset`.
pub fn bound(list: usize) {
    let held = HELD.with(|h| std::mem::take(&mut *h.borrow_mut()));
    if held.is_empty() {
        return;
    }
    let allocator = match super::alloc_pool::enabled() {
        true => super::alloc_pool::recording_backing(list).unwrap_or(0),
        false => super::list_x::allocator_of(list),
    };
    keep(allocator, held);
}

/// `bound` for a list recording on the PC allocator `allocator`.
pub fn bound_on(allocator: usize) {
    let held = HELD.with(|h| std::mem::take(&mut *h.borrow_mut()));
    if !held.is_empty() {
        keep(allocator, held);
    }
}

fn keep(allocator: usize, held: VecDeque<Ref>) {
    let mut left = Vec::new();
    {
        let mut g = BOUND.lock().unwrap();
        let m = g.get_or_insert_with(HashMap::new).entry(allocator).or_default();
        for r in held {
            left.extend(m.insert(r.0.as_raw() as usize, r));
        }
    }
    drop(left);
}

/// `allocator` was reset or released: the GPU is past its recordings.
pub fn allocator_reset(allocator: usize) {
    let gone = BOUND.lock().unwrap().as_mut().and_then(|m| m.remove(&allocator));
    drop(gone);
}

/// Pipeline `pso` is used: its code counts until it is freed (`freed`).
///
/// # Safety
/// `pso` is a live pipeline.
pub unsafe fn count(pso: usize) -> usize {
    if let Some(&n) = SIZES.lock().unwrap().as_ref().and_then(|m| m.get(&pso)) {
        return n;
    }
    // (outside the lock: the driver's call)
    let n = super::real::cached_blob_size(pso as *mut c_void)
        + super::real::dispatch_chain::tail_code_bytes(pso)
        + super::real::two_pass_code_bytes(pso);
    if SIZES.lock().unwrap().get_or_insert_with(HashMap::new).insert(pso, n).is_none() {
        USED.fetch_add(n, Ordering::AcqRel);
    }
    n
}

/// The last reference to pipeline `pso` goes.
pub fn freed(pso: usize) {
    if let Some(n) = SIZES.lock().unwrap().as_mut().and_then(|m| m.remove(&pso)) {
        USED.fetch_sub(n, Ordering::AcqRel);
    }
}

/// The pipelines recordings use: released, they would take their room until those are done.
pub fn in_recordings() -> HashSet<usize> {
    let mut out = BOUND.lock().unwrap().as_ref().map_or(HashSet::new(), |m| m.values().flat_map(|p| p.keys().copied()).collect());
    super::real::dispatch_chain::recorded_pipelines(&mut out);
    out
}

/// Give back what the allocators hold whose lists are all reset and done on the GPU.
pub fn let_go_done() {
    let all: Vec<usize> = BOUND.lock().unwrap().as_ref().map_or(Vec::new(), |m| m.keys().copied().collect());
    for a in super::alloc_pool::done(&all) {
        allocator_reset(a);
    }
}

pub fn used() -> usize {
    USED.load(Ordering::Acquire)
}

/// Which of `pipelines` (bytes of code, when last used) to release for `used` bytes to come down
/// to 7/8 of `limit`: the ones not used for the longest time, and none used in the last `IDLE_MS`.
pub fn choose(pipelines: &[(usize, u64)], used: usize, limit: usize, now: u64) -> HashSet<usize> {
    let mut order: Vec<usize> = (0..pipelines.len()).filter(|&i| pipelines[i].1 + IDLE_MS <= now).collect();
    order.sort_by_key(|&i| pipelines[i].1);
    let (mut left, mut out) = (used, HashSet::new());
    for i in order {
        if left <= limit / 8 * 7 {
            break;
        }
        left = left.saturating_sub(pipelines[i].0);
        out.insert(i);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_longest_unused_go_first_and_only_as_many_as_it_takes() {
        // 100 each, 400 used, the limit 320 (7/8: 280): two have to go
        let p = [(100, 5_000), (100, 1_000), (100, 3_000), (100, 9_500)];
        assert_eq!(choose(&p, 400, 320, 10_000), HashSet::from([1, 2]));
        assert_eq!(choose(&p, 280, 320, 10_000), HashSet::new());
        // never one just used, also when that leaves too much
        assert_eq!(choose(&p, 400, 8, 10_000), HashSet::from([0, 1, 2]));
    }

    /// A pipeline a list was recorded with lives until the list's allocator is reset.
    #[test]
    fn a_bound_pipeline_lives_until_its_allocator_is_reset() {
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
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&rs),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: code.as_ptr() as _, BytecodeLength: code.len() },
                ..Default::default()
            };
            let pso: ID3D12PipelineState = dev.CreateComputePipelineState(&desc).unwrap();
            let raw = pso.as_raw() as usize;
            let refs = || {
                type F = unsafe extern "system" fn(usize) -> u32;
                let slot = |k: usize| std::mem::transmute::<usize, F>(*(*(raw as *const *const usize)).add(k));
                slot(1)(raw);
                slot(2)(raw)
            };
            let take = || HELD.with(|h| std::mem::take(&mut *h.borrow_mut()));
            assert_eq!(refs(), 1);
            let allocator = 0x7E57_A110usize;
            hold(Ref::new(raw).unwrap());
            hold(Ref::new(raw).unwrap());
            assert_eq!(refs(), 2, "held once by this thread");
            keep(allocator, take());
            hold(Ref::new(raw).unwrap());
            keep(allocator, take());
            assert_eq!(refs(), 2, "held once by the allocator");
            assert!(in_recordings().contains(&raw));
            allocator_reset(allocator);
            assert_eq!(refs(), 1);
            assert!(!in_recordings().contains(&raw));
        }
    }
}
