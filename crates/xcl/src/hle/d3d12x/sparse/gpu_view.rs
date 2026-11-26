//! Buffers in a GPU view of the title's memory, readable by shaders (`XCL_GPU_VIEW_BUFFERS=1`).
//!
//! A title reserves 64 KiB pages twice, once for the CPU and once with no CPU access (`kernelx::
//! mirrors`), and gives the GPU addresses of the second. Those lie outside the memory windows, so
//! a shader reads zeros at them: a retail title's culling reads its object lists, and its mesh
//! shaders their geometry, from there.
//!
//! The tiles a buffer view covers are mapped in free address space of a sparse window (video
//! memory only: nothing is committed for the CPU) and hold what the title's pages hold: their
//! bytes go up when the view is made and again when the title writes them. The view's address is
//! replaced by the window's. Only what is read is mirrored: a write through it is lost, so UAVs
//! keep their address.

use std::collections::{HashMap, HashSet};

use super::*;

/// The window space a chunk of the title's view takes. Views inside one chunk share it; a view
/// across chunks gets a space of its own (its bytes have to be contiguous).
const EXTENT: usize = 1 << 20;
/// Views longer than this keep their address.
const MOST: usize = 64 << 20;
/// Minimum submit age before budget-pressure LRU eviction can reclaim a still-backed tile,
/// ensuring tiles referenced in the current frame or in-flight submissions are never evicted.
const EVICT_GRACE_SUBMITS: u64 = 16;
/// Rotating title tiles inspected per submit for unmapped/decommitted reclamation.
const SWEEP_PER_SUBMIT: usize = 16;
/// Extra cold tiles reclaimed alongside a budget eviction to amortize tile-mapping updates.
const EVICT_BATCH_EXTRA: usize = 32;

pub fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_GPU_VIEW_BUFFERS").is_ok_and(|v| v.trim() == "1"))
}

/// `XCL_GPU_VIEW_MB` (512): the video memory the mirrors may take.
fn budget() -> usize {
    static B: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *B.get_or_init(|| std::env::var("XCL_GPU_VIEW_MB").ok().and_then(|v| v.trim().parse().ok()).filter(|&m| m > 0).unwrap_or(512usize) << 20)
}

#[derive(Default)]
pub(super) struct Mirrors {
    /// (first byte of the title's chunks, chunks) → the window space of them
    extents: HashMap<(usize, usize), usize>,
    /// window tile → title tile (an address in the GPU view)
    title_of: FxHashMap<usize, usize>,
    /// where the bytes of a title tile are read from (its own address, or the CPU view's)
    read: FxHashMap<usize, usize>,
    /// dense list of keys in `read` for O(SWEEP_PER_SUBMIT) rotating checks without cloning all keys
    read_keys: Vec<usize>,
    /// title tile → index in `read_keys`
    read_idx: FxHashMap<usize, usize>,
    /// read tile → window tiles whose bytes are there
    by_read: FxHashMap<usize, Vec<usize>>,
    /// window tiles none of whose bytes have gone up yet (or whose pages changed place)
    fresh: Vec<usize>,
    /// window tile → submit epoch when last referenced (`base_for`, `show`, or dirty upload)
    last_used: FxHashMap<usize, u64>,
    /// title tiles whose backing read page has been observed committed at least once
    backed: HashSet<usize>,
    /// title tiles detected as dead/unmapped during `refresh_sources()` awaiting decommit in `sync()`
    pending_dead: Vec<usize>,
    /// monotonically incremented once per `sync()` submission
    epoch: u64,
    /// the next title tile the slow check looks at
    turn: usize,
    /// GPU-view mappings seen (`kernelx::gpu_view_map_gen`)
    gen: u64,
}

static MIRRORS: Mutex<Option<Mirrors>> = Mutex::new(None);

impl Mirrors {
    fn track_read_key(&mut self, title: usize) {
        if !self.read_idx.contains_key(&title) {
            let idx = self.read_keys.len();
            self.read_keys.push(title);
            self.read_idx.insert(title, idx);
        }
    }

    fn untrack_read_key(&mut self, title: usize) {
        if let Some(idx) = self.read_idx.remove(&title) {
            self.read_keys.swap_remove(idx);
            if let Some(&moved) = self.read_keys.get(idx) {
                self.read_idx.insert(moved, idx);
            }
        }
    }

    /// Title tile `title` is shown at window tile `win`, its bytes read from `read`.
    pub(super) fn show(&mut self, win: usize, title: usize, read: usize) {
        self.title_of.insert(win, title);
        self.track_read_key(title);
        self.read.insert(title, read);
        self.by_read.entry(read).or_default().push(win);
        self.fresh.push(win);
        self.last_used.insert(win, self.epoch);
    }

    fn touch(&mut self, win: usize) {
        if self.title_of.contains_key(&win) {
            self.last_used.insert(win, self.epoch);
        }
    }

    fn set_read(&mut self, title: usize, read: usize) {
        self.track_read_key(title);
        let old = self.read.insert(title, read);
        if old == Some(read) {
            return;
        }
        let wins: Vec<usize> = self.title_of.iter().filter(|&(_, &t)| t == title).map(|(&w, _)| w).collect();
        if let Some(o) = old {
            if let Some(v) = self.by_read.get_mut(&o) {
                v.retain(|w| !wins.contains(w));
                if v.is_empty() {
                    self.by_read.remove(&o);
                }
            }
        }
        self.by_read.entry(read).or_default().extend(&wins);
        self.fresh.extend(wins);
    }

    /// Remove `wins` from all mirror tables and prune orphaned `by_read`, `read`, `backed`, and
    /// `fresh` entries. Returns the number of mapped window tiles removed.
    fn evict_windows(&mut self, wins: &[usize]) -> usize {
        if wins.is_empty() {
            return 0;
        }
        let mut evicted = 0usize;
        let mut affected_titles: HashSet<usize> = HashSet::new();
        let mut touched_reads: HashSet<usize> = HashSet::new();
        for &w in wins {
            let Some(title) = self.title_of.remove(&w) else { continue };
            self.last_used.remove(&w);
            affected_titles.insert(title);
            if let Some(&r) = self.read.get(&title) {
                touched_reads.insert(r);
            }
            evicted += 1;
        }
        if evicted == 0 {
            return 0;
        }
        for r in touched_reads {
            if let Some(v) = self.by_read.get_mut(&r) {
                v.retain(|w| self.title_of.contains_key(w));
                if v.is_empty() {
                    self.by_read.remove(&r);
                }
            }
        }
        self.fresh.retain(|w| self.title_of.contains_key(w));
        for &t in self.title_of.values() {
            affected_titles.remove(&t);
        }
        for title in affected_titles {
            self.read.remove(&title);
            self.untrack_read_key(title);
            self.backed.remove(&title);
        }
        evicted
    }

    /// Evict every window tile showing any title in `titles` whose tiles were not touched in the
    /// current submit epoch, in a single pass over `title_of` and one `evict_windows` call.
    fn evict_titles_if_idle(&mut self, titles: &[usize]) -> Vec<usize> {
        if titles.is_empty() {
            return Vec::new();
        }
        let wanted: HashSet<usize> = titles.iter().copied().collect();
        let mut by_title: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
        for (&w, &t) in &self.title_of {
            if wanted.contains(&t) {
                by_title.entry(t).or_default().push(w);
            }
        }
        let mut wins = Vec::new();
        for (_, ws) in by_title {
            if !ws.iter().any(|w| self.last_used.get(w).copied().unwrap_or(0) >= self.epoch) {
                wins.extend(ws);
            }
        }
        if !wins.is_empty() {
            self.evict_windows(&wins);
        }
        wins
    }

    /// Evict every window tile showing `title` if none were touched in the current submit epoch.
    #[cfg(test)]
    fn evict_title_if_idle(&mut self, title: usize) -> Vec<usize> {
        self.evict_titles_if_idle(&[title])
    }

    /// Select up to `want` least-recently-used window tiles at least `min_age` submits old and
    /// outside `protect`, ordered oldest-first.
    fn lru_eviction_candidates(&self, want: usize, min_age: u64, protect: &HashSet<usize>) -> Vec<usize> {
        if want == 0 {
            return Vec::new();
        }
        let mut cold: Vec<(u64, usize)> = self
            .title_of
            .keys()
            .filter(|w| !protect.contains(w))
            .filter_map(|&w| {
                let last = self.last_used.get(&w).copied().unwrap_or(0);
                (self.epoch.saturating_sub(last) >= min_age).then_some((last, w))
            })
            .collect();
        cold.sort_unstable();
        cold.truncate(want);
        cold.into_iter().map(|(_, w)| w).collect()
    }

    /// The window tiles and pages a set of written pages (`dirty`, title or CPU addresses) is
    /// shown at: (window page, the read address of its bytes).
    pub(super) fn shown_at(&self, dirty: &[usize]) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for &d in dirty {
            let tile = d & !(TILE - 1);
            for &w in self.by_read.get(&tile).map(Vec::as_slice).unwrap_or(&[]) {
                out.push((w + (d - tile), d));
            }
        }
        out
    }

    pub(super) fn tiles(&self) -> usize {
        self.title_of.len()
    }
}

/// Decommit evicted mirror window tiles in coalesced contiguous runs so `sparse::State` parks,
/// unmaps, and returns their 64 KiB VRAM tiles (and empty 64 MiB pools) safely.
unsafe fn decommit_window_tiles(s: &mut State, wins: &mut [usize]) {
    if wins.is_empty() {
        return;
    }
    wins.sort_unstable();
    let mut i = 0;
    while i < wins.len() {
        let start = wins[i];
        let mut end = start + TILE;
        i += 1;
        while i < wins.len() && (wins[i] == end || wins[i] == end - TILE) {
            if wins[i] == end {
                end += TILE;
            }
            i += 1;
        }
        s.decommit_range(start, end - start);
    }
}

/// Whether `title` (a 64 KiB tile in a GPU view) has lost its backing: either the GPU-view
/// reservation was released, the page was explicitly unmapped via `CopyPageMappingsBatchX`, or
/// a previously committed backing tile has been completely decommitted. Returns `(dead, read)`.
unsafe fn check_title_tile(title: usize, was_backed: bool) -> (bool, usize) {
    if crate::hle::kernelx::mirrors::source(title).is_none() {
        return (true, 0);
    }
    let read = tile_source(title);
    if read == 0 {
        return (true, 0);
    }
    if was_backed
        && query_tile_holes(read) == 0xFFFF
        && !crate::hle::kernelx::lazy::is_lazy(read)
        && !crate::hle::kernelx::physical_page_at(read)
    {
        return (true, read);
    }
    (false, read)
}

/// Copies `n` bytes from the title's memory at `at`; false when the page went away (nothing read).
unsafe fn read_title(at: usize, out: *mut u8, n: usize) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    let mut got = 0usize;
    ReadProcessMemory(windows_sys::Win32::System::Threading::GetCurrentProcess(), at as *const c_void, out as *mut c_void, n, &mut got) != 0 && got == n
}

impl State {
    /// Brings the mirrored tiles in step with the title's memory: whole tiles that are new, and
    /// the pages the title wrote (`dirty`). Before the title's lists; `m` is held by the caller.
    pub(super) unsafe fn mirror_up(&mut self, m: &mut Mirrors, dirty: &[usize]) -> usize {
        let mut pages: Vec<(usize, usize)> = m.shown_at(dirty);
        let dirty_count = pages.len();
        for &(win, _) in &pages[..dirty_count] {
            m.touch(win & !(TILE - 1));
        }
        for win in std::mem::take(&mut m.fresh) {
            let Some(&title) = m.title_of.get(&win) else { continue };
            let read = m.read[&title];
            if read == 0 {
                continue;
            }
            pages.extend((0..PAGES_PER_TILE).map(|i| (win + i * PAGE, read + i * PAGE)));
        }
        pages.sort_unstable();
        pages.dedup();
        let mut runs: Vec<(usize, usize)> = Vec::new();
        let mut cached_pages: FxHashMap<usize, [u8; PAGE]> = FxHashMap::default();
        let mut bytes = [0u8; PAGE];
        for (win, read) in pages {
            let tile = win & !(TILE - 1);
            let Some(entry) = self.mapped.get_mut(&tile) else { continue };
            if !read_title(read, bytes.as_mut_ptr(), PAGE) {
                continue;
            }
            if let Some(&title) = m.title_of.get(&tile) {
                m.backed.insert(title);
            }
            let i = (win - tile) / PAGE;
            entry.holes &= !(1u16 << i);
            let before = runs.len();
            let last_len = runs.last().map_or(0, |r| r.1);
            scan_page(&bytes, win, &mut entry.sent[i], &mut runs);
            if runs.len() != before || runs.last().is_some_and(|r| r.1 != last_len) {
                cached_pages.insert(win, bytes);
            }
        }
        let n = runs.iter().map(|r| r.1).sum();
        // (the title tile a run lies in never changes within it: a run is at most a page here, but
        // `push_run` joins neighbours, and two neighbouring pages may come from two places)
        let split: Vec<(usize, usize)> = runs.iter().flat_map(|&(a, l)| {
            let mut out = Vec::new();
            let (mut a, end) = (a, a + l);
            while a < end {
                let e = ((a & !(PAGE - 1)) + PAGE).min(end);
                out.push((a, e - a));
                a = e;
            }
            out
        }).collect();
        self.upload_runs(&split, &|dst, p, len| {
            let page = dst & !(PAGE - 1);
            if let Some(buf) = cached_pages.get(&page) {
                let off = dst - page;
                std::ptr::copy_nonoverlapping(buf.as_ptr().add(off), p, len);
                true
            } else {
                false
            }
        });
        n
    }
}

/// Where GPU-view tile `t` reads its bytes from (0 = unmapped via `CopyPageMappingsBatchX`).
/// Must be called without holding `sparse::STATE` or `MIRRORS` to preserve lock ordering with `kernelx`.
fn tile_source(t: usize) -> usize {
    match crate::hle::kernelx::mirrors::mapped_source(t) {
        Some(Some(a)) => a & !(TILE - 1),
        Some(None) => 0,
        None => super::super::title_textures::mirror_source(t) & !(TILE - 1),
    }
}

/// The address a view of `len` bytes at `va` gets when it lies in a GPU view of the title's and
/// in no memory window: where the same bytes can be read by a shader. None: it keeps `va`.
pub fn base_for(va: usize, len: usize) -> Option<usize> {
    if !on() || len == 0 || len > MOST {
        return None;
    }
    let u = crate::gpu::uma::get()?;
    if u.contains(va) || !u.has_sparse() {
        return None;
    }
    let last = va.checked_add(len)?.checked_sub(1)?;
    crate::hle::kernelx::mirrors::source(va)?;
    crate::hle::kernelx::mirrors::source(last)?;
    let (t0, t1) = (va & !(TILE - 1), (last & !(TILE - 1)) + TILE);
    let first = t0 & !(EXTENT - 1);
    let n = (t1 - first).div_ceil(EXTENT);
    let (known, missing_tiles) = {
        let mut mg = MIRRORS.lock().unwrap();
        let m = mg.get_or_insert_with(Mirrors::default);
        if let Some(&w) = m.extents.get(&(first, n)) {
            let missing: Vec<usize> = (t0..t1).step_by(TILE).filter(|t| !m.title_of.contains_key(&(w + (t - first)))).collect();
            if missing.is_empty() {
                return Some(w + (va - first));
            }
            (Some(w), missing)
        } else {
            (None, (t0..t1).step_by(TILE).collect())
        }
    };
    let win = match known {
        Some(w) => w,
        None => {
            let (w, _) = u.sparse_alloc(n * EXTENT, EXTENT)?;
            *MIRRORS.lock().unwrap().get_or_insert_with(Mirrors::default).extents.entry((first, n)).or_insert(w)
        }
    };
    // Resolve backing sources and hole masks BEFORE taking `sparse::STATE` so we never call into
    // `kernelx` (which locks `AWAY` -> `sparse::STATE`) while holding `sparse::STATE`.
    let resolved: FxHashMap<usize, (usize, u16)> = missing_tiles
        .iter()
        .map(|&t| {
            let r = tile_source(t);
            let holes = if r == 0 { u16::MAX } else { unsafe { query_tile_holes(r) } };
            (t, (r, holes))
        })
        .collect();
    // (held: STATE, then MIRRORS, as in `sync`)
    let mut g = lock_state();
    let s = g.as_mut()?;
    let mut mg = MIRRORS.lock().unwrap();
    let m = mg.get_or_insert_with(Mirrors::default);
    let new: Vec<usize> = (t0..t1).step_by(TILE).filter(|t| !m.title_of.contains_key(&(win + (t - first)))).collect();
    for t in (t0..t1).step_by(TILE) {
        m.touch(win + (t - first));
    }
    if !new.is_empty() {
        let limit = budget() / TILE;
        if m.tiles() + new.len() > limit {
            let need = (m.tiles() + new.len()).saturating_sub(limit);
            let protect: HashSet<usize> = (t0..t1).step_by(TILE).map(|t| win + (t - first)).collect();
            let want = need.saturating_add(EVICT_BATCH_EXTRA);
            let mut victims = m.lru_eviction_candidates(want, EVICT_GRACE_SUBMITS, &protect);
            if victims.len() >= need {
                let n = m.evict_windows(&victims);
                // SAFETY: evicted mirror tiles live in sparse window VA with no CPU backing.
                unsafe { decommit_window_tiles(s, &mut victims) };
                static EVICTED: AtomicU64 = AtomicU64::new(0);
                let prev = EVICTED.fetch_add(n as u64, Ordering::Relaxed);
                let total = prev + n as u64;
                if prev == 0 || (total >> 10) > (prev >> 10) {
                    crate::xlog_info!("D3D12.X: GPU view mirrors evicted {n} cold tiles ({} MB total reclaimed, {} tiles live)", (total as usize * TILE) >> 20, m.tiles());
                }
            }
        }
        if m.tiles() + new.len() > limit {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| crate::xlog_warn!("D3D12.X: the buffers of the GPU view take more than XCL_GPU_VIEW_MB: the rest keeps its address"));
            return None;
        }
        // whole runs of new tiles at once (a mapping call and a signal for each run)
        let mut i = 0;
        while i < new.len() {
            let mut j = i + 1;
            while j < new.len() && new[j] == new[j - 1] + TILE {
                j += 1;
            }
            let (a, b) = (new[i], win + (new[i] - first));
            // SAFETY: the tiles are address space of a window, with no CPU memory behind them
            unsafe { s.ensure_mapped(b, (j - i) * TILE, |w| resolved.get(&(a + (w - b))).map_or(u16::MAX, |&(_, h)| h)) };
            for &t in &new[i..j] {
                let w = win + (t - first);
                if s.mapped.contains_key(&w) {
                    let (read, holes) = resolved.get(&t).copied().unwrap_or((0, u16::MAX));
                    m.show(w, t, read);
                    if read != 0 && holes != 0xFFFF {
                        m.backed.insert(t);
                    }
                }
            }
            i = j;
        }
        static N: AtomicU64 = AtomicU64::new(0);
        let k = N.fetch_add(1, Ordering::Relaxed) + 1;
        if k.is_power_of_two() {
            crate::xlog_info!("D3D12.X: buffers of the GPU view: {} tiles mirrored ({} MB), latest view {va:#x} ({len} B)", m.tiles(), (m.tiles() * TILE) >> 20);
        }
    }
    if (t0..t1).step_by(TILE).any(|t| !m.title_of.contains_key(&(win + (t - first)))) {
        return None;
    }
    Some(win + (va - first))
}

/// The pages of the CPU (and pool) side whose writes have to be seen: for the write watch.
pub fn sources() -> Vec<(usize, usize)> {
    MIRRORS.lock().unwrap().as_ref().map_or(Vec::new(), |m| m.by_read.keys().copied().filter(|&r| r != 0).map(|r| (r, r + TILE)).collect())
}

/// Refresh tile backing addresses and detect unmapped/decommitted tiles outside `sparse::STATE`
/// (called before `lock_state()` in `before_execute` so `kernelx::AWAY` is never taken under `STATE`).
pub(super) fn refresh_sources() {
    let changed = crate::hle::kernelx::mirrors::take_changed_buffers();
    let gen = crate::hle::kernelx::gpu_view_map_gen();
    let look: Vec<(usize, bool)> = {
        let mut g = MIRRORS.lock().unwrap();
        let Some(m) = g.as_mut().filter(|m| m.tiles() != 0) else { return };
        let full = std::mem::replace(&mut m.gen, gen) != gen;
        let selected: Vec<usize> = if full {
            m.read_keys.clone()
        } else {
            let n = m.read_keys.len();
            let sweep = SWEEP_PER_SUBMIT.min(n);
            let mut out = Vec::with_capacity(sweep + changed.len());
            if !changed.is_empty() {
                let mut seen: HashSet<usize> = HashSet::new();
                for &(a, b) in &changed {
                    let lo = a & !(TILE - 1);
                    let hi = b.saturating_add(TILE - 1) & !(TILE - 1);
                    if lo >= hi {
                        continue;
                    }
                    let tiles = (hi - lo) / TILE;
                    if tiles <= m.read.len() {
                        for t in (lo..hi).step_by(TILE) {
                            if m.read.contains_key(&t) && seen.insert(t) {
                                out.push(t);
                            }
                        }
                    } else {
                        for &t in &m.read_keys {
                            if t >= lo && t < hi && seen.insert(t) {
                                out.push(t);
                            }
                        }
                    }
                }
                for i in 0..sweep {
                    let t = m.read_keys[(m.turn + i) % n];
                    if seen.insert(t) {
                        out.push(t);
                    }
                }
            } else {
                for i in 0..sweep {
                    out.push(m.read_keys[(m.turn + i) % n]);
                }
            }
            out
        };
        m.turn = m.turn.wrapping_add(SWEEP_PER_SUBMIT);
        selected.into_iter().map(|t| (t, m.backed.contains(&t))).collect()
    };
    if look.is_empty() {
        return;
    }
    let updates: Vec<(usize, bool, usize, bool)> = look
        .into_iter()
        .map(|(t, was_backed)| {
            let (dead, r) = unsafe { check_title_tile(t, was_backed) };
            let now_backed = !dead && !was_backed && r != 0 && unsafe { query_tile_holes(r) } != 0xFFFF;
            (t, dead, r, now_backed)
        })
        .collect();
    let mut g = MIRRORS.lock().unwrap();
    if let Some(m) = g.as_mut() {
        for (t, dead, r, now_backed) in updates {
            if !m.read.contains_key(&t) {
                continue;
            }
            if dead {
                m.pending_dead.push(t);
            } else {
                if now_backed {
                    m.backed.insert(t);
                }
                m.set_read(t, r);
            }
        }
    }
}

/// `State::mirror_up` for a submit (`before_execute`; `s` is the locked state).
pub(super) unsafe fn sync(s: &mut State, dirty: &[usize]) {
    let mut g = MIRRORS.lock().unwrap();
    let Some(m) = g.as_mut().filter(|m| m.tiles() != 0) else { return };
    m.epoch = m.epoch.wrapping_add(1);
    let pending_dead = std::mem::take(&mut m.pending_dead);
    let mut dead_wins = m.evict_titles_if_idle(&pending_dead);
    if !dead_wins.is_empty() {
        let evicted = dead_wins.len();
        decommit_window_tiles(s, &mut dead_wins);
        static UNMAPPED_EVICTED: AtomicU64 = AtomicU64::new(0);
        let prev = UNMAPPED_EVICTED.fetch_add(evicted as u64, Ordering::Relaxed);
        let total = prev + evicted as u64;
        if prev == 0 || (total >> 10) > (prev >> 10) {
            crate::xlog_info!("D3D12.X: GPU view mirrors reclaimed {evicted} unmapped tiles ({} MB total, {} tiles live)", (total as usize * TILE) >> 20, m.tiles());
        }
    }
    let n = s.mirror_up(m, dirty);
    static SENT: AtomicU64 = AtomicU64::new(0);
    static CALLS: AtomicU64 = AtomicU64::new(0);
    let total = SENT.fetch_add(n as u64, Ordering::Relaxed) + n as u64;
    let k = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
    if k.is_power_of_two() && k >= 64 {
        crate::xlog_info!("D3D12.X: buffers of the GPU view: {} tiles, {} MB sent in {k} submits", m.tiles(), total >> 20);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_written_page_is_shown_at_every_window_tile_it_is_read_for() {
        let mut m = Mirrors::default();
        m.show(0x10_0000, 0xAA_0000, 0xAA_0000);
        m.show(0x20_0000, 0xAA_0000, 0xAA_0000);
        m.show(0x30_0000, 0xBB_0000, 0xCC_0000);
        assert_eq!(m.tiles(), 3);
        let mut a = m.shown_at(&[0xAA_3000]);
        a.sort();
        assert_eq!(a, vec![(0x10_3000, 0xAA_3000), (0x20_3000, 0xAA_3000)]);
        assert_eq!(m.shown_at(&[0xCC_1000]), vec![(0x30_1000, 0xCC_1000)], "by the read address, not the title's");
        assert!(m.shown_at(&[0xBB_1000]).is_empty());
        m.fresh.clear();
        // the page behind a title tile moved: its window tiles are sent whole again
        m.set_read(0xBB_0000, 0xDD_0000);
        assert_eq!(m.fresh, vec![0x30_0000]);
        assert!(m.shown_at(&[0xCC_1000]).is_empty());
        assert_eq!(m.shown_at(&[0xDD_2000]), vec![(0x30_2000, 0xDD_2000)]);
        m.set_read(0xBB_0000, 0xDD_0000);
        assert_eq!(m.fresh, vec![0x30_0000], "the same place again is nothing new");
    }

    #[test]
    fn refresh_sources_tracks_copy_page_mappings_without_locking_sparse_state() {
        use crate::hle::kernelx::mirrors::{apply_batches, changed_gen, reserved, released, PAGE};
        let (cpu, gpu, size) = (0x7190_0000_0000usize, 0x7196_0000_0000usize, 4 * PAGE);
        let pool = 0x719A_0000_0000usize;
        reserved(cpu, size, 0x2040_2000, PAGE_READWRITE);
        reserved(gpu, size, 0x2040_2000, 0x801);
        let tile = gpu + PAGE;
        assert_eq!(tile_source(tile), cpu + PAGE);
        #[repr(C)]
        struct Batch { dest: u64, ranges_n: u32, _pad: u32, ranges: u64 }
        let g0 = changed_gen();
        let r_map = [[0u32, 2, 1]]; // map 1 page to pool + 2*PAGE
        let b_map = [Batch { dest: tile as u64, ranges_n: 1, _pad: 0, ranges: r_map.as_ptr() as u64 }];
        unsafe { assert!(apply_batches(1, b_map.as_ptr() as usize, pool, 8).is_some()) };
        assert!(changed_gen() > g0);
        assert_eq!(tile_source(tile), pool + 2 * PAGE);
        // Unmap the page -> tile_source returns 0 (never read as address 0)
        let r_unmap = [[2u32, 0, 1]];
        let b_unmap = [Batch { dest: tile as u64, ranges_n: 1, _pad: 0, ranges: r_unmap.as_ptr() as u64 }];
        unsafe { assert!(apply_batches(1, b_unmap.as_ptr() as usize, pool, 8).is_some()) };
        assert_eq!(tile_source(tile), 0);
        released(cpu);
        released(gpu);
    }

    #[test]
    fn evict_windows_cleans_all_indices_and_orphaned_sources() {
        let mut m = Mirrors::default();
        m.show(0x10_0000, 0xAA_0000, 0xCC_0000);
        m.show(0x20_0000, 0xAA_0000, 0xCC_0000);
        m.show(0x30_0000, 0xBB_0000, 0xDD_0000);
        m.backed.insert(0xAA_0000);
        m.backed.insert(0xBB_0000);
        // Evicting one of two window aliases keeps the shared title/read entry alive.
        assert_eq!(m.evict_windows(&[0x10_0000]), 1);
        assert_eq!(m.tiles(), 2);
        assert!(m.read.contains_key(&0xAA_0000));
        assert_eq!(m.shown_at(&[0xCC_1000]), vec![(0x20_1000, 0xCC_1000)]);
        assert!(!m.fresh.contains(&0x10_0000));
        // Evicting the last alias removes title, backed, and empty by_read source entries.
        assert_eq!(m.evict_windows(&[0x20_0000]), 1);
        assert_eq!(m.tiles(), 1);
        assert!(!m.read.contains_key(&0xAA_0000));
        assert!(!m.backed.contains(&0xAA_0000));
        assert!(!m.by_read.contains_key(&0xCC_0000));
        assert_eq!(m.shown_at(&[0xDD_2000]), vec![(0x30_2000, 0xDD_2000)]);
    }

    #[test]
    fn lru_eviction_respects_grace_window_and_protected_tiles() {
        let mut m = Mirrors::default();
        m.epoch = 1;
        m.show(0x10_0000, 0xA0_0000, 0xA0_0000);
        m.epoch = 5;
        m.show(0x20_0000, 0xB0_0000, 0xB0_0000);
        m.epoch = 20;
        m.show(0x30_0000, 0xC0_0000, 0xC0_0000);
        m.epoch = 25;
        // At epoch 25 with grace 16: 0x10 (age 24) and 0x20 (age 20) are cold; 0x30 (age 5) is hot.
        let protect: HashSet<usize> = [0x10_0000].into_iter().collect();
        assert_eq!(m.lru_eviction_candidates(4, EVICT_GRACE_SUBMITS, &protect), vec![0x20_0000]);
        assert_eq!(m.lru_eviction_candidates(4, EVICT_GRACE_SUBMITS, &HashSet::new()), vec![0x10_0000, 0x20_0000]);
        // Touching 0x10 makes it hot in the current epoch, so evict_title_if_idle skips it too.
        m.touch(0x10_0000);
        assert!(m.evict_title_if_idle(0xA0_0000).is_empty());
        assert_eq!(m.evict_title_if_idle(0xB0_0000), vec![0x20_0000]);
    }

    /// The bytes of the title's pages reach the window tiles mapped for them, and again when the
    /// title writes a page; a page that went away is left as it is.
    #[test]
    fn gpu_the_titles_bytes_reach_the_window_tiles_shown_for_them() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE};
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let dev = crate::gpu::agility::create_device(None, D3D_FEATURE_LEVEL_11_0).unwrap_or_else(|_| {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                crate::gpu::agility::create_device(&warp, D3D_FEATURE_LEVEL_11_0).expect("WARP D3D12 device")
            });
            let window = VirtualAlloc(std::ptr::null(), TILE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) as usize;
            let source = VirtualAlloc(std::ptr::null(), TILE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) as usize;
            assert!(window % TILE == 0 && source % TILE == 0);
            for i in 0..TILE / 4 {
                *((source + 4 * i) as *mut u32) = i as u32 * 7 + 1;
            }
            let mut reserved: Option<ID3D12Resource> = None;
            dev.CreateReservedResource(&buffer_desc(TILE as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut reserved).expect("reserved buffer");
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("queue");
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            let event = windows::Win32::System::Threading::CreateEventW(None, false, false, None).expect("event");
            let rb_alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("alloc");
            let rb_list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &rb_alloc, None).expect("list");
            rb_list.Close().expect("close");
            let mut slots = Vec::new();
            for _ in 0..UPLOAD_SLOTS {
                let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("alloc");
                let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).expect("list");
                list.Close().expect("close");
                slots.push(UploadSlot { alloc, list, fence_val: 0 });
            }
            let mut s = State {
                dev,
                queue,
                fence,
                value: 0,
                event,
                windows: vec![(window, reserved.expect("reserved buffer"))],
                pools: Vec::new(),
                free: Vec::new(),
                retired: Vec::new(),
                mapped: FxHashMap::default(),
                pinned: FxHashMap::default(),
                parked: FxHashMap::default(),
                state_gen: 0,
                quarantined: VecDeque::new(),
                queue_fences: FxHashMap::default(),
                to_zero: Vec::new(),
                staging: None,
                ring_write: 0,
                ring_submitted: 0,
                ring_free: 0,
                inflight: VecDeque::new(),
                zero_buf: None,
                slots,
                next_slot: 0,
                readback: None,
                rb_alloc,
                rb_list,
                rb_last_fence: 0,
                waited: FxHashMap::default(),
                runs: None,
                scratch_runs: Vec::new(),
            };
            let mut m = Mirrors::default();
            s.ensure_mapped(window, TILE, |_| query_tile_holes(source));
            m.show(window, 0x7FFF_0000, source);
            let at = |i: usize| *((window + 4 * i) as *const u32);
            let shown = |s: &mut State, m: &mut Mirrors, dirty: &[usize]| {
                s.mirror_up(m, dirty);
                s.readback_ranges(&[(window, TILE)]);
            };
            shown(&mut s, &mut m, &[]);
            assert!((0..TILE / 4).all(|i| at(i) == i as u32 * 7 + 1), "the whole tile at first");
            // a page the title wrote (and one it did not)
            *((source + 0x3010) as *mut u32) = 0xDEAD_BEEF;
            shown(&mut s, &mut m, &[source + 0x3000]);
            assert_eq!((at(0x3010 / 4), at(0x3014 / 4)), (0xDEAD_BEEF, (0x3014 / 4) as u32 * 7 + 1));
            // not reported as written: not sent
            *((source + 0x4010) as *mut u32) = 0x1234_5678;
            shown(&mut s, &mut m, &[]);
            assert_eq!(at(0x4010 / 4), (0x4010 / 4) as u32 * 7 + 1);
            // the page is gone: what the GPU has stays, and nothing faults
            assert_ne!(VirtualFree(source as *mut c_void, 0, MEM_RELEASE), 0);
            m.fresh.push(window);
            shown(&mut s, &mut m, &[source + 0x5000]);
            assert_eq!(at(0x3010 / 4), 0xDEAD_BEEF);
            s.wait_for(s.value);
            drop(s);
            VirtualFree(window as *mut c_void, 0, MEM_RELEASE);
        }
    }
}
