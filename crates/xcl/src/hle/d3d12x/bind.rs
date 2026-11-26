//! Root arguments of console code, bound when it dispatches with a translated pipeline.
//!
//! A translated console shader reads its root arguments the console's way (user SGPRs s4…,
//! the rest from a table at s[0:1]: `console_list::user_data`) and memory through XCL's
//! shader ABI (u0 io, u1 res, u2… unified memory). So on a command list of console code the
//! compute root signature and root arguments are kept here – packed by the root signature's
//! layout – instead of going to the PC list, and each Dispatch with a translated pipeline
//! binds the ABI root signature, writes the user SGPRs (`io`) and the argument table to
//! unified memory and dispatches. That memory stays until the list's allocator is reset.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use super::console_list::{self, Layout, Param};

/// Last XCL ABI root signature + root UAV addresses bound on one stage of a PC command list.
#[derive(Clone, Copy, Default)]
struct AbiBound {
    rs: usize,
    views: [u64; 6],
}

impl AbiBound {
    /// Diff `(rs, views)` against `self` and update `self`, returning `(set_rs, dirty_uav_mask)`.
    fn diff_and_update(&mut self, rs: usize, views: [u64; 6]) -> (bool, u8) {
        if self.rs != rs {
            self.rs = rs;
            self.views = views;
            return (true, 0x3F);
        }
        let mut mask = 0u8;
        for k in 0..6 {
            if self.views[k] != views[k] {
                self.views[k] = views[k];
                mask |= 1u8 << k;
            }
        }
        (false, mask)
    }
}

unsafe fn emit_abi_bindings(list: &ID3D12GraphicsCommandList, gfx: bool, rs: &ID3D12RootSignature, set_rs: bool, uav_mask: u8, views: &[u64; 6]) {
    if gfx {
        if set_rs {
            list.SetGraphicsRootSignature(rs);
        }
        let mut m = uav_mask;
        while m != 0 {
            let k = m.trailing_zeros();
            list.SetGraphicsRootUnorderedAccessView(k, views[k as usize]);
            m &= m - 1;
        }
    } else {
        if set_rs {
            list.SetComputeRootSignature(rs);
        }
        let mut m = uav_mask;
        while m != 0 {
            let k = m.trailing_zeros();
            list.SetComputeRootUnorderedAccessView(k, views[k as usize]);
            m &= m - 1;
        }
    }
}

fn uma_va_in(u: &crate::gpu::uma::Uma, windows: &[u64], p: usize) -> Option<u64> {
    let k = u.window_of(p)?;
    let base = *windows.get(k)?;
    (base != 0).then(|| base + (p - u.window(k).0) as u64)
}

/// Paired persistently-mapped `UPLOAD` slab and L1 VRAM `DEFAULT` (`ALLOW_UNORDERED_ACCESS`) slab
/// for a recording's `io` (`u0`) user-SGPR blocks. CPU writes `io` bytes into `upload` during
/// command-list recording and `prelude_list_for` copies `0..used` into `vram` once before the
/// segment executes, avoiding PCIe/system-RAM stalls on every shader's first `u0` load.
struct IoBlock {
    vram: ID3D12Resource,
    upload: ID3D12Resource,
    mapped: usize,
    vram_va: u64,
    cap: u64,
    used: u64,
    dev_key: usize,
}

impl Drop for IoBlock {
    fn drop(&mut self) {
        if self.mapped != 0 {
            unsafe { self.upload.Unmap(0, None) };
            self.mapped = 0;
        }
    }
}

/// Cached resolved ABI metadata for a translated PSO on one command list, skipping repeated
/// `pso_async` / `real` map lookups across consecutive draws/dispatches with the same PSO.
#[derive(Clone, Copy, Debug)]
struct CachedPsoAbi {
    query_pso: usize,
    real_pso: usize,
    rs_raw: usize,
    user_sgprs: Option<u32>,
    is_gfx: bool,
}

type PreludeEntry = (ID3D12CommandAllocator, ID3D12GraphicsCommandList, usize, i32);

#[derive(Default)]
struct PreludeScratchSlot {
    buf: Option<(ID3D12Resource, u64, usize)>,
    used: u64,
    buf_idx: usize,
}

#[derive(Clone, Copy, Debug)]
struct PreludeScaleJob {
    buf_idx: usize,
    dst_off: u32,
    count: u32,
    stride: u32,
    mesh_offset: u32,
    split: u32,
    cap: u32,
    src_win: u32,
    src_off: u32,
    has_cnt: u32,
    cnt_win: u32,
    cnt_off: u32,
}

#[derive(Default)]
struct State {
    layout: Option<Arc<Layout>>,
    args: Vec<u8>,
    /// the graphics root signature's layout and arguments
    layout_gfx: Option<Arc<Layout>>,
    args_gfx: Vec<u8>,
    /// the title's graphics root signature (a PC object), bound before a draw with an
    /// untranslated pipeline – created with it – since SetGraphicsRootSignature binds nothing
    rs_gfx: usize,
    /// the title's compute root signature (a PC object), preserved for internal-pass restoration
    rs_cs: usize,
    pso: usize,
    /// 32-bit addresses of the view heap and the sampler heap set (0: none)
    heaps: [u32; 2],
    allocator: usize,
    /// unified memory of this recording (io, argument tables)
    uma: Vec<usize>,
    /// where the next draw's io and argument table go (blocks in `uma`)
    arena: crate::gpu::uma::Arena,
    res_buf: Option<(ID3D12Resource, u64, usize)>,
    res_retired: Vec<(ID3D12Resource, u64, usize)>,
    /// rewritten indirect arguments (`scratch_args`): the block, its bytes in use, and the
    /// bytes of blocks this recording took
    scratch: Option<(ID3D12Resource, u64, usize)>,
    scratch_used: u64,
    scratch_taken: u64,
    /// hoisted two-pass indirect argument scaling recorded into `xcl prelude`
    prelude_blocked: bool,
    prelude_scratch: PreludeScratchSlot,
    prelude_bufs: Vec<ID3D12Resource>,
    prelude_scales: Vec<PreludeScaleJob>,
    /// the next dispatch's thread limits (`IO_THREAD_LIMIT`; a `DispatchX` with thread
    /// dimensions), taken by its `bind_abi`
    limits: [u32; 3],
    /// cached UMA window GPU VAs for this recording
    uma_windows: Option<Vec<u64>>,
    /// one reusable 16-byte UMA allocation for empty inline root-argument tables
    dummy_table_p: usize,
    /// one reusable 256-byte zeroed `res` (`u1`) GPU VA per recording
    dummy_res_va: u64,
    /// generation counters bumped on root-argument / heap mutations for `io_cache_*`
    args_gen_gfx: u64,
    args_gen_cs: u64,
    io_cache_gfx: Option<(u64, u64)>,
    io_cache_cs: Option<(u64, Option<u32>, u64)>,
    io_cache_tp_cs: Option<(u64, u64)>,
    io_cache_tp_ms: Option<(u64, u64)>,
    /// last bound PC ABI root signature and root UAVs (u0..u5) per stage
    pc_abi_gfx: AbiBound,
    pc_abi_cs: AbiBound,
    /// true once `list_x::ensure_view_heap` confirmed required shared heaps are bound on this list
    heaps_ensured: bool,
    /// cached resolved ABI metadata for the active translated PSO on this list
    pso_abi_cache: Option<CachedPsoAbi>,
    /// L1 VRAM `io` (`u0`) slabs populated during this recording and copied in `prelude_list_for`
    io_blocks: Vec<IoBlock>,
    /// lazily cached device eligible for VRAM `io` on this segment
    vram_io_dev: Option<Option<ID3D12Device>>,
    /// closed prelude list owned by this recording (kept alive until backing allocator reset)
    prelude_owned: Option<PreludeEntry>,
    prelude_built: Option<ID3D12GraphicsCommandList>,
}

/// A recording's blocks of rewritten indirect arguments, and most bytes of them (past it, its
/// two-pass ExecuteIndirects are skipped)
const SCRATCH_BLOCK: u64 = 1 << 20;
const SCRATCH_CAP: u64 = 16 << 20;
/// Most bytes of buffers kept for reuse (`FREE_RES`)
const FREE_RES_CAP: u64 = 256 << 20;
/// Size of one pooled VRAM `io` (`u0`) slab (64 KiB = 256 distinct 256-byte `io` slots).
const IO_BLOCK_BYTES: u64 = 64 << 10;
/// Most bytes of VRAM `io` slabs kept for reuse (`FREE_IO`).
const FREE_IO_CAP: u64 = 64 << 20;
/// Most recycled prelude (allocator, list) pairs kept for reuse (`FREE_PRELUDE`).
const FREE_PRELUDE_CAP: usize = 256;

static ALLOC_RES: Mutex<Option<HashMap<usize, Vec<(ID3D12Resource, u64, usize)>>>> = Mutex::new(None);
static FREE_RES: Mutex<Vec<(ID3D12Resource, u64, usize)>> = Mutex::new(Vec::new());
static ALLOC_IO: Mutex<Option<HashMap<usize, Vec<IoBlock>>>> = Mutex::new(None);
static FREE_IO: Mutex<Vec<IoBlock>> = Mutex::new(Vec::new());
static ALLOC_PRELUDE: Mutex<Option<HashMap<usize, Vec<PreludeEntry>>>> = Mutex::new(None);
static FREE_PRELUDE: Mutex<Vec<PreludeEntry>> = Mutex::new(Vec::new());

fn hold_res_until_reset(allocator: usize, mut bufs: Vec<(ID3D12Resource, u64, usize)>) {
    if bufs.is_empty() || allocator == 0 {
        return;
    }
    ALLOC_RES.lock().unwrap().get_or_insert_with(HashMap::new).entry(allocator).or_default().append(&mut bufs);
}

fn hold_io_until_reset(allocator: usize, mut blocks: Vec<IoBlock>) {
    if blocks.is_empty() || allocator == 0 {
        return;
    }
    ALLOC_IO.lock().unwrap().get_or_insert_with(HashMap::new).entry(allocator).or_default().append(&mut blocks);
}

fn hold_prelude_until_reset(allocator: usize, entry: Option<PreludeEntry>) {
    let Some(entry) = entry else { return };
    if allocator == 0 {
        return;
    }
    ALLOC_PRELUDE.lock().unwrap().get_or_insert_with(HashMap::new).entry(allocator).or_default().push(entry);
}

pub(super) fn release_allocator_res(allocator: usize) {
    if let Some(bufs) = ALLOC_RES.lock().unwrap().as_mut().and_then(|m| m.remove(&allocator)) {
        let mut free = FREE_RES.lock().unwrap();
        for b in bufs {
            let held: u64 = free.iter().map(|f| f.1).sum();
            if pool_keeps(free.len(), held, b.1) {
                free.push(b);
            }
        }
    }
    if let Some(blocks) = ALLOC_IO.lock().unwrap().as_mut().and_then(|m| m.remove(&allocator)) {
        let mut free = FREE_IO.lock().unwrap();
        for mut b in blocks {
            b.used = 0;
            let held: u64 = free.iter().map(|f| f.cap).sum();
            if io_pool_keeps(held, b.cap) {
                free.push(b);
            }
        }
    }
    if let Some(entries) = ALLOC_PRELUDE.lock().unwrap().as_mut().and_then(|m| m.remove(&allocator)) {
        let mut free = FREE_PRELUDE.lock().unwrap();
        for e in entries {
            if free.len() < FREE_PRELUDE_CAP {
                free.push(e);
            }
        }
    }
}

/// Whether `FREE_RES`, holding `count` buffers of `held` bytes, keeps one of `bytes` more.
fn pool_keeps(count: usize, held: u64, bytes: u64) -> bool {
    count < 16 && held + bytes <= FREE_RES_CAP
}

/// Whether `FREE_IO`, holding `held` bytes of VRAM `io` slabs, keeps one of `bytes` more.
fn io_pool_keeps(held: u64, bytes: u64) -> bool {
    held.saturating_add(bytes) <= FREE_IO_CAP
}

/// `XCL_VRAM_IO=0`: keep `io` (`u0`) user-SGPR buffers in UMA system RAM instead of L1 VRAM.
fn vram_io_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_VRAM_IO").map_or(true, |v| v.trim() != "0"))
}

/// Return the device if `this` can place `io` (`u0`) slots in L1 VRAM (wrapped DIRECT/COMPUTE
/// command list whose prelude list runs before it on submit).
unsafe fn vram_io_device(this: *mut c_void) -> Option<ID3D12Device> {
    if !vram_io_on()
        || super::title_list::wrapper_of(this).is_none()
        || !matches!(
            super::list_x::tracked_list_type(this),
            Some(D3D12_COMMAND_LIST_TYPE_DIRECT | D3D12_COMMAND_LIST_TYPE_COMPUTE)
        )
    {
        return None;
    }
    super::real::list_device(this)
}

unsafe fn create_io_block(dev: &ID3D12Device, cap: u64) -> Option<IoBlock> {
    let vram_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: cap,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut vram: Option<ID3D12Resource> = None;
    dev.CreateCommittedResource(
        &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, CreationNodeMask: 1, VisibleNodeMask: 1, ..Default::default() },
        D3D12_HEAP_FLAG_NONE,
        &vram_desc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        None,
        &mut vram,
    )
    .ok()?;
    let vram = vram?;
    let _ = vram.SetName(windows::core::w!("xcl io vram"));
    let vram_va = vram.GetGPUVirtualAddress();
    if vram_va == 0 {
        return None;
    }
    let up_desc = D3D12_RESOURCE_DESC { Flags: D3D12_RESOURCE_FLAG_NONE, ..vram_desc };
    let mut upload: Option<ID3D12Resource> = None;
    dev.CreateCommittedResource(
        &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, CreationNodeMask: 1, VisibleNodeMask: 1, ..Default::default() },
        D3D12_HEAP_FLAG_NONE,
        &up_desc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        None,
        &mut upload,
    )
    .ok()?;
    let upload = upload?;
    let _ = upload.SetName(windows::core::w!("xcl io upload"));
    let mut mapped = std::ptr::null_mut();
    let empty_read = D3D12_RANGE { Begin: 0, End: 0 };
    upload.Map(0, Some(&empty_read), Some(&mut mapped)).ok()?;
    if mapped.is_null() {
        return None;
    }
    Some(IoBlock { vram, upload, mapped: mapped as usize, vram_va, cap, used: 0, dev_key: dev.as_raw() as usize })
}

/// Write `bytes` into a 256-byte-aligned slot of `io_blocks` and return its L1 VRAM GPU VA.
unsafe fn alloc_vram_io_in(io_blocks: &mut Vec<IoBlock>, allocator: usize, prelude_built: bool, dev: &ID3D12Device, bytes: &[u8]) -> Option<u64> {
    if allocator == 0 || prelude_built {
        return None;
    }
    let take = (bytes.len().max(16) as u64).next_multiple_of(256);
    if let Some(b) = io_blocks.last_mut().filter(|b| b.used + take <= b.cap) {
        let off = b.used;
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), (b.mapped as *mut u8).add(off as usize), bytes.len());
        b.used += take;
        return Some(b.vram_va + off);
    }
    let dev_key = dev.as_raw() as usize;
    let mut block = {
        let mut free = FREE_IO.lock().unwrap();
        free.iter().position(|b| b.dev_key == dev_key && b.cap >= take).map(|i| free.swap_remove(i))
    };
    if block.is_none() {
        block = create_io_block(dev, take.max(IO_BLOCK_BYTES));
    }
    let mut b = block?;
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), b.mapped as *mut u8, bytes.len());
    b.used = take;
    let va = b.vram_va;
    io_blocks.push(b);
    Some(va)
}

/// Build (once per recording) and return the closed PC prelude command list that uploads all
/// L1 VRAM `io` (`u0`) slabs and runs any hoisted batched two-pass indirect argument scaling
/// for `seg` before `seg` executes on the queue.
pub(super) unsafe fn prelude_list_for(seg: *mut c_void) -> Option<*mut c_void> {
    let (already, io_copies, bufs, scales) = with(seg, |s| {
        if let Some(ref built) = s.prelude_built {
            return (Some(built.as_raw()), Vec::new(), Vec::new(), Vec::new());
        }
        let copies: Vec<(ID3D12Resource, ID3D12Resource, u64)> = s
            .io_blocks
            .iter()
            .filter(|b| b.used > 0)
            .map(|b| (b.vram.clone(), b.upload.clone(), b.used))
            .collect();
        if copies.is_empty() && s.prelude_scales.is_empty() {
            return (None, Vec::new(), Vec::new(), Vec::new());
        }
        (None, copies, s.prelude_bufs.clone(), s.prelude_scales.clone())
    });
    if let Some(p) = already {
        return Some(p);
    }
    if io_copies.is_empty() && scales.is_empty() {
        return None;
    }
    let run = (|| -> Result<ID3D12GraphicsCommandList, String> {
        let dev = super::real::list_device(seg).ok_or("no device for prelude")?;
        let ty = super::list_x::tracked_list_type(seg).unwrap_or(D3D12_COMMAND_LIST_TYPE_DIRECT);
        if ty != D3D12_COMMAND_LIST_TYPE_DIRECT && ty != D3D12_COMMAND_LIST_TYPE_COMPUTE {
            return Err("unsupported list type for prelude".into());
        }
        let dev_key = dev.as_raw() as usize;
        let reused = {
            let mut free = FREE_PRELUDE.lock().unwrap();
            free.iter().position(|(_, _, d, t)| *d == dev_key && *t == ty.0).map(|i| free.swap_remove(i))
        };
        let (alloc, list) = match reused {
            Some((a, l, _, _)) => {
                a.Reset().map_err(|e| format!("prelude allocator reset: {e}"))?;
                l.Reset(&a, None).map_err(|e| format!("prelude list reset: {e}"))?;
                (a, l)
            }
            None => {
                let a: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).map_err(|e| format!("prelude allocator create: {e}"))?;
                let _ = a.SetName(windows::core::w!("xcl prelude alloc"));
                let l: ID3D12GraphicsCommandList = dev.CreateCommandList(0, ty, &a, None).map_err(|e| format!("prelude list create: {e}"))?;
                let _ = l.SetName(windows::core::w!("xcl prelude"));
                (a, l)
            }
        };
        if !io_copies.is_empty() {
            let mut to_copy: Vec<D3D12_RESOURCE_BARRIER> = Vec::with_capacity(io_copies.len());
            let mut to_uav: Vec<D3D12_RESOURCE_BARRIER> = Vec::with_capacity(io_copies.len());
            for (vram, _, _) in &io_copies {
                to_copy.push(D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                    Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    Anonymous: D3D12_RESOURCE_BARRIER_0 {
                        Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                            pResource: std::mem::ManuallyDrop::new(Some(vram.clone())),
                            Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                            StateBefore: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                            StateAfter: D3D12_RESOURCE_STATE_COPY_DEST,
                        }),
                    },
                });
                to_uav.push(D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                    Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    Anonymous: D3D12_RESOURCE_BARRIER_0 {
                        Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                            pResource: std::mem::ManuallyDrop::new(Some(vram.clone())),
                            Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                            StateBefore: D3D12_RESOURCE_STATE_COPY_DEST,
                            StateAfter: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        }),
                    },
                });
            }
            list.ResourceBarrier(&to_copy);
            for (vram, upload, used) in &io_copies {
                list.CopyBufferRegion(vram, 0, upload, 0, *used);
            }
            list.ResourceBarrier(&to_uav);
            for b in to_copy.iter_mut().chain(to_uav.iter_mut()) {
                std::mem::ManuallyDrop::drop(&mut (*b.Anonymous.Transition).pResource);
            }
        }
        if !scales.is_empty() {
            let u = crate::gpu::uma::get().ok_or("no unified memory for prelude scale")?;
            let windows = super::real::uma_window_vas();
            let first = *windows.first().ok_or("unified memory is not open on the GPU")?;
            let shared_heap = if super::console_heap::window_views_ready() {
                super::console_heap::global_heap()
            } else {
                None
            };
            let mut pre_barriers: Vec<D3D12_RESOURCE_BARRIER> = Vec::with_capacity(bufs.len() + 1);
            pre_barriers.push(D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                        pResource: std::mem::ManuallyDrop::new(None),
                    }),
                },
            });
            for b in &bufs {
                pre_barriers.push(D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                    Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    Anonymous: D3D12_RESOURCE_BARRIER_0 {
                        Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                            pResource: std::mem::ManuallyDrop::new(Some(b.clone())),
                            Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                            StateBefore: D3D12_RESOURCE_STATE_COMMON,
                            StateAfter: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        }),
                    },
                });
            }
            list.ResourceBarrier(&pre_barriers);
            for b in &mut pre_barriers[1..] {
                std::mem::ManuallyDrop::drop(&mut (*b.Anonymous.Transition).pResource);
            }
            if let Some(ref heap) = shared_heap {
                list.SetDescriptorHeaps(&[Some(heap.clone())]);
            }
            let (rs, pso) = if shared_heap.is_some() {
                prelude_scale_heap_pipeline(&dev)?
            } else {
                prelude_scale_pipeline(&dev)?
            };
            list.SetComputeRootSignature(&rs);
            if shared_heap.is_none() {
                for k in 0..4usize {
                    let va = windows.get(k).copied().filter(|&v| v != 0).unwrap_or(first);
                    list.SetComputeRootUnorderedAccessView(2 + k as u32, va);
                }
            }
            list.SetPipelineState(&pso);
            for (buf_idx, buf) in bufs.iter().enumerate() {
                let batch: Vec<&PreludeScaleJob> = scales.iter().filter(|j| j.buf_idx == buf_idx).collect();
                if batch.is_empty() {
                    continue;
                }
                list.SetComputeRootUnorderedAccessView(1, buf.GetGPUVirtualAddress());
                for chunk in batch.chunks(65_535) {
                    let mut words = Vec::with_capacity(chunk.len() * 12);
                    for j in chunk {
                        words.extend_from_slice(&[
                            j.count, j.stride, j.mesh_offset, j.split, j.cap, j.src_win, j.src_off, j.dst_off, j.has_cnt, j.cnt_win, j.cnt_off, 0,
                        ]);
                    }
                    let bytes = std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), words.len() * 4);
                    let p = arena_take(seg, u, bytes.len().max(16), 256).ok_or("unified memory is full for prelude scale jobs")?;
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), p, bytes.len());
                    let jobs_va = uma_va_in(u, &windows, p as usize).unwrap_or(first);
                    list.SetComputeRootUnorderedAccessView(0, jobs_va);
                    list.Dispatch(chunk.len() as u32, 1, 1);
                }
            }
            let mut post_barriers: Vec<D3D12_RESOURCE_BARRIER> = Vec::with_capacity(bufs.len());
            for b in &bufs {
                post_barriers.push(D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                    Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    Anonymous: D3D12_RESOURCE_BARRIER_0 {
                        Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                            pResource: std::mem::ManuallyDrop::new(Some(b.clone())),
                            Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                            StateBefore: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                            StateAfter: D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                        }),
                    },
                });
            }
            list.ResourceBarrier(&post_barriers);
            for b in &mut post_barriers {
                std::mem::ManuallyDrop::drop(&mut (*b.Anonymous.Transition).pResource);
            }
        }
        list.Close().map_err(|e| format!("prelude list close: {e}"))?;
        let ret = list.clone();
        with(seg, |s| {
            s.prelude_built = Some(list.clone());
            s.prelude_owned = Some((alloc, list, dev_key, ty.0));
        });
        Ok(ret)
    })();
    match run {
        Ok(l) => Some(l.as_raw()),
        Err(e) => {
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 8 {
                crate::xlog_warn!("D3D12.X: prelude list failed: {e}");
            }
            None
        }
    }
}

const STATES_SHARDS: usize = 64;
static STATES: [Mutex<Option<HashMap<usize, State>>>; STATES_SHARDS] = [const { Mutex::new(None) }; STATES_SHARDS];

#[inline]
fn state_shard(list: usize) -> &'static Mutex<Option<HashMap<usize, State>>> {
    let h = (list >> 4).wrapping_mul(0x9E37_79B9_7F4A_7C15usize);
    &STATES[(h ^ (h >> 16)) & (STATES_SHARDS - 1)]
}

/// Invalidate cached PC ABI root-signature / UAV tracking on `list` when external code
/// (such as internal sync/blit passes or ambient stand-in dispatches) binds another root signature.
pub(super) fn invalidate_pc_abi(list: *mut c_void, gfx: bool, cs: bool) {
    with(list, |s| {
        if gfx {
            s.pc_abi_gfx = AbiBound::default();
        }
        if cs {
            s.pc_abi_cs = AbiBound::default();
        }
        s.heaps_ensured = false;
    });
}

unsafe fn ensure_heaps_cached(this: *mut c_void) {
    if with(this, |s| s.heaps_ensured) {
        return;
    }
    if super::list_x::ensure_view_heap(this) {
        with(this, |s| s.heaps_ensured = true);
    }
}

/// Note that `list` issued a GPU buffer write, UAV barrier, or wait: subsequent two-pass indirect
/// argument scaling must execute inline instead of hoisting into the prelude.
pub(super) fn note_gpu_write(list: *mut c_void) {
    with(list, |s| {
        s.prelude_blocked = true;
    });
}

/// Allow prelude hoisting on a freshly split continuation segment with no head fence wait.
pub(super) fn unblock_prelude(list: *mut c_void) {
    with(list, |s| {
        s.prelude_blocked = false;
    });
}

/// draws / dispatches of console code: with a translated pipeline, skipped, with a PC one
static COUNTS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// draws and dispatches of console code so far (`XCL_ACTIVITY`)
pub fn activity() -> (u64, u64) {
    (COUNTS[DRAWS].load(Ordering::Relaxed), COUNTS[DISPATCHES].load(Ordering::Relaxed))
}
const DRAWS: usize = 0;
const DRAWS_SKIPPED: usize = 1;
const DRAWS_PC: usize = 2;
const DISPATCHES: usize = 3;
const DISPATCHES_SKIPPED: usize = 4;

fn count(k: usize) {
    let n = COUNTS[k].fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_power_of_two() && n >= 64 {
        let c: Vec<u64> = COUNTS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        let (allocators, blocks) = console_list::pending_blocks();
        let (coherent_used, coherent_total, sparse_used) = crate::gpu::uma::get()
            .map_or((0, 0, 0), |u| (u.used() >> 20, u.size() >> 20, u.sparse_used() >> 20));
        crate::xlog_info!(
            "D3D12.X: console code so far: {} draws with translated pipelines ({} skipped, {} with PC pipelines), {} dispatches ({} skipped); coherent {coherent_used}/{coherent_total} MB, sparse {sparse_used} MB, {blocks} blocks held for {allocators} allocators",
            c[DRAWS], c[DRAWS_SKIPPED], c[DRAWS_PC], c[DISPATCHES], c[DISPATCHES_SKIPPED],
        );
    }
}

fn with<R>(list: *mut c_void, f: impl FnOnce(&mut State) -> R) -> R {
    f(state_shard(list as usize).lock().unwrap().get_or_insert_with(HashMap::new).entry(list as usize).or_default())
}

/// Resolve the PC backing allocator that actually owns `list`'s recording (`alloc_pool` can swap
/// a title allocator to a new PC allocator while recordings on the old backing are still in flight,
/// so UMA arena blocks and scratch buffers must stay pinned to the backing allocator until
/// `reset_pc_allocator` genuinely resets it).
fn backing_allocator(list: *mut c_void, allocator: usize) -> usize {
    if allocator == 0 || !super::alloc_pool::enabled() {
        return allocator;
    }
    super::alloc_pool::recording_backing(list as usize).unwrap_or_else(|| super::alloc_pool::current(allocator))
}

/// The list was created or reset onto `allocator` with `initial_pso`: the previous
/// recording's memory is the previous allocator's until that one is reset, and pipeline
/// and root signature state reset per D3D12 `CreateCommandList` / `Reset` semantics.
pub(super) fn reset(list: *mut c_void, allocator: usize, initial_pso: usize) {
    let backing = backing_allocator(list, allocator);
    let (old, blocks, res_bufs, io_blocks, prelude) = with(list, |s| {
        let old = std::mem::replace(&mut s.allocator, backing);
        s.arena = Default::default();
        s.pso = initial_pso;
        s.layout = None;
        s.args.clear();
        s.layout_gfx = None;
        s.args_gfx.clear();
        s.rs_gfx = 0;
        s.rs_cs = 0;
        s.heaps = [0; 2];
        s.uma_windows = None;
        s.dummy_table_p = 0;
        s.dummy_res_va = 0;
        s.args_gen_gfx = s.args_gen_gfx.wrapping_add(1);
        s.args_gen_cs = s.args_gen_cs.wrapping_add(1);
        s.io_cache_gfx = None;
        s.io_cache_cs = None;
        s.io_cache_tp_cs = None;
        s.io_cache_tp_ms = None;
        s.pc_abi_gfx = AbiBound::default();
        s.pc_abi_cs = AbiBound::default();
        s.heaps_ensured = false;
        s.pso_abi_cache = None;
        s.vram_io_dev = None;
        s.prelude_blocked = false;
        s.prelude_scratch = Default::default();
        s.prelude_bufs.clear();
        s.prelude_scales.clear();
        s.prelude_built = None;
        let prelude = s.prelude_owned.take();
        // (the scratch blocks are in `res_retired` already)
        (s.scratch, s.scratch_used, s.scratch_taken) = (None, 0, 0);
        let mut res_bufs = std::mem::take(&mut s.res_retired);
        if let Some(b) = s.res_buf.take() {
            res_bufs.push(b);
        }
        let io_blocks = std::mem::take(&mut s.io_blocks);
        (old, std::mem::take(&mut s.uma), res_bufs, io_blocks, prelude)
    });
    console_list::hold_until_reset(old, blocks);
    hold_res_until_reset(old, res_bufs);
    hold_io_until_reset(old, io_blocks);
    hold_prelude_until_reset(old, prelude);
}

/// Update `list`'s tracked backing allocator after `alloc_pool::reset_list` rotates `list` onto a
/// fresh PC command allocator (e.g., `KickoffX` or `record_deferred`), retiring any prior
/// recording buffers on the old allocator first without clearing pipeline/root bindings.
pub(super) fn rebind_allocator(list: *mut c_void, allocator: usize) {
    let backing = backing_allocator(list, allocator);
    let (old, blocks, res_bufs, io_blocks, prelude) = with(list, |s| {
        let old = std::mem::replace(&mut s.allocator, backing);
        s.arena = Default::default();
        s.uma_windows = None;
        s.dummy_table_p = 0;
        s.dummy_res_va = 0;
        s.io_cache_gfx = None;
        s.io_cache_cs = None;
        s.io_cache_tp_cs = None;
        s.io_cache_tp_ms = None;
        s.pc_abi_gfx = AbiBound::default();
        s.pc_abi_cs = AbiBound::default();
        s.heaps_ensured = false;
        s.pso_abi_cache = None;
        s.vram_io_dev = None;
        s.prelude_blocked = false;
        s.prelude_scratch = Default::default();
        s.prelude_bufs.clear();
        s.prelude_scales.clear();
        s.prelude_built = None;
        let prelude = s.prelude_owned.take();
        (s.scratch, s.scratch_used, s.scratch_taken) = (None, 0, 0);
        let mut res_bufs = std::mem::take(&mut s.res_retired);
        if let Some(b) = s.res_buf.take() {
            res_bufs.push(b);
        }
        let io_blocks = std::mem::take(&mut s.io_blocks);
        (old, std::mem::take(&mut s.uma), res_bufs, io_blocks, prelude)
    });
    console_list::hold_until_reset(old, blocks);
    hold_res_until_reset(old, res_bufs);
    hold_io_until_reset(old, io_blocks);
    hold_prelude_until_reset(old, prelude);
}

/// `ClearState(pso)`: resets pipeline and root-argument state back to creation defaults
/// with `pso` bound, keeping the current recording's allocator and UMA arena intact.
pub(super) fn clear_state(list: *mut c_void, pso: usize) {
    with(list, |s| {
        s.pso = pso;
        s.layout = None;
        s.args.clear();
        s.layout_gfx = None;
        s.args_gfx.clear();
        s.rs_gfx = 0;
        s.rs_cs = 0;
        s.heaps = [0; 2];
        s.args_gen_gfx = s.args_gen_gfx.wrapping_add(1);
        s.args_gen_cs = s.args_gen_cs.wrapping_add(1);
        s.io_cache_gfx = None;
        s.io_cache_cs = None;
        s.io_cache_tp_cs = None;
        s.io_cache_tp_ms = None;
        s.pc_abi_gfx = AbiBound::default();
        s.pc_abi_cs = AbiBound::default();
        s.heaps_ensured = false;
        s.pso_abi_cache = None;
    });
}

/// Carry bound pipeline and root-signature / root-argument state from `from` to `to` across a
/// deferred mid-list split (`to` records into `allocator` with its own fresh UMA arena while
/// `from` retains its previously allocated blocks until `allocator` resets).
pub(super) fn inherit_for_split(from: *mut c_void, to: *mut c_void, allocator: usize) {
    let backing = backing_allocator(to, allocator);
    let (layout, args, layout_gfx, args_gfx, rs_gfx, rs_cs, pso, heaps) = with(from, |s| {
        (s.layout.clone(), s.args.clone(), s.layout_gfx.clone(), s.args_gfx.clone(), s.rs_gfx, s.rs_cs, s.pso, s.heaps)
    });
    let (old, blocks, res_bufs, io_blocks, prelude) = with(to, |s| {
        let old = std::mem::replace(&mut s.allocator, backing);
        s.arena = Default::default();
        s.layout = layout;
        s.args = args;
        s.layout_gfx = layout_gfx;
        s.args_gfx = args_gfx;
        s.rs_gfx = rs_gfx;
        s.rs_cs = rs_cs;
        s.pso = pso;
        s.heaps = heaps;
        s.uma_windows = None;
        s.dummy_table_p = 0;
        s.dummy_res_va = 0;
        s.args_gen_gfx = s.args_gen_gfx.wrapping_add(1);
        s.args_gen_cs = s.args_gen_cs.wrapping_add(1);
        s.io_cache_gfx = None;
        s.io_cache_cs = None;
        s.io_cache_tp_cs = None;
        s.io_cache_tp_ms = None;
        s.pc_abi_gfx = AbiBound::default();
        s.pc_abi_cs = AbiBound::default();
        s.heaps_ensured = false;
        s.pso_abi_cache = None;
        s.vram_io_dev = None;
        s.prelude_blocked = true;
        s.prelude_scratch = Default::default();
        s.prelude_bufs.clear();
        s.prelude_scales.clear();
        s.prelude_built = None;
        let prelude = s.prelude_owned.take();
        (s.scratch, s.scratch_used, s.scratch_taken) = (None, 0, 0);
        let mut res_bufs = std::mem::take(&mut s.res_retired);
        if let Some(b) = s.res_buf.take() {
            res_bufs.push(b);
        }
        let io_blocks = std::mem::take(&mut s.io_blocks);
        (old, std::mem::take(&mut s.uma), res_bufs, io_blocks, prelude)
    });
    console_list::hold_until_reset(old, blocks);
    hold_res_until_reset(old, res_bufs);
    hold_io_until_reset(old, io_blocks);
    hold_prelude_until_reset(old, prelude);
}

/// A submit is about to wait for the pipeline `this` has bound.
pub(super) fn want_bound(this: *mut c_void) {
    super::pso_async::want(with(this, |s| s.pso));
}

/// A put-off dispatch that could not be recorded.
pub(super) fn dispatch_dropped() {
    count(DISPATCHES_SKIPPED);
}

/// `to` goes on in the arena `from` took its pieces from (`from` takes no more).
pub(super) fn pass_arena(from: *mut c_void, to: *mut c_void) {
    let arena = with(from, |s| std::mem::take(&mut s.arena));
    with(to, |s| s.arena = arena);
}

/// The list is gone.
pub(super) fn forget(list: *mut c_void) {
    if let Some(mut s) = state_shard(list as usize).lock().unwrap().as_mut().and_then(|m| m.remove(&(list as usize))) {
        console_list::hold_until_reset(s.allocator, s.uma);
        if let Some(b) = s.res_buf.take() {
            s.res_retired.push(b);
        }
        hold_res_until_reset(s.allocator, s.res_retired);
        hold_io_until_reset(s.allocator, s.io_blocks);
        hold_prelude_until_reset(s.allocator, s.prelude_owned.take());
    }
}

#[cfg(test)]
pub(crate) fn seed_reset_state_for_test(list: *mut c_void, allocator: usize, pso: usize, uma: Vec<usize>) {
    with(list, |s| {
        s.allocator = allocator;
        s.pso = pso;
        s.layout = Some(Arc::new(Layout::default()));
        s.args = vec![1, 2, 3, 4];
        s.layout_gfx = Some(Arc::new(Layout::default()));
        s.args_gfx = vec![5, 6, 7, 8];
        s.uma = uma;
    });
}

#[cfg(test)]
pub(crate) fn reset_state_for_test(list: *mut c_void) -> (usize, usize, Vec<u8>, Vec<u8>, Vec<usize>) {
    with(list, |s| (s.allocator, s.pso, s.args.clone(), s.args_gfx.clone(), s.uma.clone()))
}

#[cfg(test)]
pub(crate) fn remove_state_for_test(list: *mut c_void) {
    state_shard(list as usize).lock().unwrap().as_mut().and_then(|m| m.remove(&(list as usize)));
}

/// Allocate `n` bytes (aligned to `align`) from `this`'s UMA arena, falling back to a sparse
/// window block outside `STATES.lock()` when the coherent window is full (`#505`).
unsafe fn arena_take(this: *mut c_void, u: &crate::gpu::uma::Uma, n: usize, align: usize) -> Option<*mut u8> {
    if let Some(p) = with(this, |s| s.arena.take(u, n, align, &mut s.uma)) {
        return Some(p);
    }
    if !u.has_sparse() {
        return None;
    }
    let n = n.max(1);
    let align = align.max(16).next_power_of_two();
    let block_size = if n > crate::gpu::uma::ARENA_BLOCK / 2 { n } else { crate::gpu::uma::ARENA_BLOCK };
    let block_align = align.max(crate::gpu::uma::GRAIN);
    let (block, _) = super::sparse::alloc(block_size, block_align)?;
    static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        crate::xlog_info!("D3D12.X: coherent UMA window full – arena blocks falling back to sparse windows");
    }
    Some(with(this, |s| s.arena.install(block, block_size, n, align, &mut s.uma)))
}

/// SetDescriptorHeaps: the console addresses of `heaps`, for shaders that index them themselves
/// (`Layout::heaps`).
pub(super) fn set_heaps(list: *mut c_void, heaps: &[usize]) {
    let mut bases = [0u32; 2];
    for (sampler, base) in heaps.iter().filter_map(|&h| super::console_heap::base(h)) {
        bases[sampler as usize] = base;
    }
    with(list, |s| {
        s.heaps_ensured = false;
        if s.heaps != bases {
            s.heaps = bases;
            s.args_gen_gfx = s.args_gen_gfx.wrapping_add(1);
            s.args_gen_cs = s.args_gen_cs.wrapping_add(1);
        }
    });
}

/// bytes of parameter `i` at constant `offset` (of the compute or the graphics arguments)
fn put_in(list: *mut c_void, gfx: bool, i: u32, offset: u32, bytes: &[u8]) {
    with(list, |s| {
        let (layout, args, gen) = if gfx { (&s.layout_gfx, &mut s.args_gfx, &mut s.args_gen_gfx) } else { (&s.layout, &mut s.args, &mut s.args_gen_cs) };
        let Some(at) = layout.as_ref().and_then(|l| l.params.get(i as usize)).map(|p| match *p {
            Param::Constants { at, .. } => at as usize + 4 * offset as usize,
            Param::Descriptor { at } | Param::Table { at } => at as usize,
        }) else {
            return;
        };
        if let Some(t) = args.get_mut(at..at + bytes.len()) {
            if t != bytes {
                t.copy_from_slice(bytes);
                *gen = gen.wrapping_add(1);
            }
        }
    })
}

fn put(list: *mut c_void, i: u32, offset: u32, bytes: &[u8]) {
    put_in(list, false, i, offset, bytes)
}

/// #34 SetGraphicsRootSignature (PC #30)
pub(super) unsafe extern "system" fn set_graphics_root_signature(this: *mut c_void, rs: *mut c_void) {
    let layout = console_list::layout_of(rs as usize);
    with(this, |s| {
        // the same root signature again keeps the bound arguments (D3D12: only a different
        // one invalidates them)
        if !console_list::same_layout(&s.layout_gfx, &layout) {
            let need = layout.as_ref().map_or(0, |l| l.size as usize);
            s.args_gfx.clear();
            s.args_gfx.resize(need, 0);
            s.args_gen_gfx = s.args_gen_gfx.wrapping_add(1);
            s.layout_gfx = layout;
        }
        s.rs_gfx = rs as usize;
    });
}

/// #36 SetGraphicsRootDescriptorTable (PC #32)
pub(super) unsafe extern "system" fn set_graphics_root_table(this: *mut c_void, i: u32, handle: u64) {
    put_in(this, true, i, 0, &(handle as u32).to_le_bytes());
}

/// #38 SetGraphicsRoot32BitConstant (PC #34)
pub(super) unsafe extern "system" fn set_graphics_root_constant(this: *mut c_void, i: u32, value: u32, offset: u32) {
    put_in(this, true, i, offset, &value.to_le_bytes());
}

/// #40 SetGraphicsRoot32BitConstants (PC #36)
pub(super) unsafe extern "system" fn set_graphics_root_constants(this: *mut c_void, i: u32, n: u32, data: *const u32, offset: u32) {
    if !data.is_null() && n > 0 {
        // Windows x64 is little-endian: pass the dword slice bytes directly without heap allocation.
        let bytes = std::slice::from_raw_parts(data.cast::<u8>(), n as usize * 4);
        put_in(this, true, i, offset, bytes);
    }
}

/// #42 / #44 / #46 SetGraphicsRoot{ConstantBuffer,ShaderResource,UnorderedAccess}View (PC #38 /
/// #40 / #42)
pub(super) unsafe extern "system" fn set_graphics_root_descriptor(this: *mut c_void, i: u32, va: u64) {
    note_address(va);
    let va = readable_root(va);
    put_in(this, true, i, 0, &va.to_le_bytes());
}

/// The address a root descriptor at `va` gets (`sparse::gpu_view`): a root descriptor has no
/// size, so as much of the tile is taken as lies in the view. A read through it is in step with
/// the title's memory, a write is not (a root UAV in a GPU view stays what it was before).
fn readable_root(va: u64) -> u64 {
    for len in [0x1_0000usize, 0x1000] {
        if let Some(b) = super::sparse::gpu_view::base_for(va as usize, len) {
            return b as u64;
        }
    }
    va
}

/// `XCL_TRACE_ADDRESSES=1`: root descriptor addresses of console code – in unified memory
/// (translated shaders see them) or not – counted and logged
fn note_address(va: u64) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static IN: AtomicU64 = AtomicU64::new(0);
    static OUT: AtomicU64 = AtomicU64::new(0);
    if !*ON.get_or_init(|| std::env::var_os("XCL_TRACE_ADDRESSES").is_some()) {
        return;
    }
    let inside = crate::gpu::uma::get().is_some_and(|u| u.contains(va as usize));
    let (i, o) = if inside { (IN.fetch_add(1, Ordering::Relaxed) + 1, OUT.load(Ordering::Relaxed)) } else { (IN.load(Ordering::Relaxed), OUT.fetch_add(1, Ordering::Relaxed) + 1) };
    if (i + o).is_power_of_two() {
        crate::xlog_info!("D3D12.X: root descriptor addresses: {i} in unified memory, {o} outside (latest {va:#x})");
    }
}

/// command signatures that change root arguments (they need the title's root signature)
static ROOT_SIGNATURES: RwLock<Option<std::collections::HashSet<usize>>> = RwLock::new(None);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DispatchMeshSig {
    companion_dispatch: usize,
    byte_stride: u32,
    mesh_offset: u32,
}

static DISPATCH_MESH_SIGS: RwLock<Option<HashMap<usize, DispatchMeshSig>>> = RwLock::new(None);

pub(super) fn note_root_changing_signature(sig: usize) {
    ROOT_SIGNATURES.write().unwrap().get_or_insert_with(Default::default).insert(sig);
}

/// command signatures with a Dispatch argument (noted at every creation, so a signature made
/// at a freed one's address does not keep its kind)
static DISPATCH_SIGNATURES: RwLock<Option<std::collections::HashSet<usize>>> = RwLock::new(None);

pub(super) fn note_signature_dispatches(sig: usize, dispatches: bool) {
    let mut s = DISPATCH_SIGNATURES.write().unwrap();
    let s = s.get_or_insert_with(Default::default);
    if dispatches {
        s.insert(sig);
    } else {
        s.remove(&sig);
    }
}

/// command signatures whose arguments only draw (Draw, DrawIndexed, DispatchMesh): an
/// ExecuteIndirect of one leaves no state behind, so it can be recorded at the submit (noted at
/// every creation, as `DISPATCH_SIGNATURES`)
static DRAW_ONLY_SIGNATURES: RwLock<Option<std::collections::HashSet<usize>>> = RwLock::new(None);

pub(super) fn note_signature_draws_only(sig: usize, draws_only: bool) {
    let mut s = DRAW_ONLY_SIGNATURES.write().unwrap();
    let s = s.get_or_insert_with(Default::default);
    if draws_only {
        s.insert(sig);
    } else {
        s.remove(&sig);
    }
}

fn signature_draws_only(sig: usize) -> bool {
    DRAW_ONLY_SIGNATURES.read().unwrap().as_ref().is_some_and(|s| s.contains(&sig))
}

fn signature_dispatches(sig: usize) -> bool {
    DISPATCH_SIGNATURES.read().unwrap().as_ref().is_some_and(|s| s.contains(&sig))
}

pub(super) fn note_dispatch_mesh_signature(sig: usize, companion_dispatch: usize, byte_stride: u32, mesh_offset: u32) {
    DISPATCH_MESH_SIGS
        .write()
        .unwrap()
        .get_or_insert_with(HashMap::new)
        .insert(sig, DispatchMeshSig { companion_dispatch, byte_stride, mesh_offset });
}

fn dispatch_mesh_sig_of(sig: usize) -> Option<DispatchMeshSig> {
    DISPATCH_MESH_SIGS.read().unwrap().as_ref()?.get(&sig).copied()
}

unsafe fn emit_uav_barrier(list: &ID3D12GraphicsCommandList, res: Option<&ID3D12Resource>) {
    let mut b = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(res.cloned()),
            }),
        },
    };
    list.ResourceBarrier(&[std::ptr::read(&b)]);
    std::mem::ManuallyDrop::drop(&mut (*b.Anonymous.UAV).pResource);
}

unsafe fn emit_uav_and_transition(list: &ID3D12GraphicsCommandList, res: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) {
    let mut b = [
        D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: std::mem::ManuallyDrop::new(None),
                }),
            },
        },
        D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: std::mem::ManuallyDrop::new(Some(res.clone())),
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: before,
                    StateAfter: after,
                }),
            },
        },
    ];
    list.ResourceBarrier(&b);
    std::mem::ManuallyDrop::drop(&mut (*b[1].Anonymous.Transition).pResource);
}

unsafe fn emit_transition_and_uav(list: &ID3D12GraphicsCommandList, res: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) {
    let mut b = [
        D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: std::mem::ManuallyDrop::new(Some(res.clone())),
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: before,
                    StateAfter: after,
                }),
            },
        },
        D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: std::mem::ManuallyDrop::new(None),
                }),
            },
        },
    ];
    list.ResourceBarrier(&b);
    std::mem::ManuallyDrop::drop(&mut (*b[0].Anonymous.Transition).pResource);
}

unsafe fn emit_transition(list: &ID3D12GraphicsCommandList, res: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) {
    let mut b = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: std::mem::ManuallyDrop::new(Some(res.clone())),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    };
    list.ResourceBarrier(&[std::ptr::read(&b)]);
    std::mem::ManuallyDrop::drop(&mut (*b.Anonymous.Transition).pResource);
}

unsafe fn default_uav_buffer(dev: &ID3D12Device, bytes: u64, what: &str) -> Result<ID3D12Resource, String> {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
    };
    let mut res: Option<ID3D12Resource> = None;
    dev.CreateCommittedResource(
        &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_COMMON,
        None,
        &mut res,
    )
    .map_err(|e| format!("{what} ({bytes:#x} B): {e}"))?;
    res.ok_or_else(|| format!("null {what}"))
}

/// Groups of a two-pass dispatch that have a record: one record a group in `cap_bytes` of
/// records of `record_bytes`, and every group in X, the mesh pass's `split` times as many (at most
/// 65 535 a dimension). The compute pass writes record `gid.x` with no bound of its own.
fn record_groups(cap_bytes: u64, record_bytes: u64, split: u32) -> u32 {
    (cap_bytes / record_bytes.max(1)).min(65_535 / split.max(1) as u64) as u32
}

/// The ExecuteIndirects of a two-pass ExecuteIndirect of `max` commands, each of one command and
/// followed by a UAV barrier on the records: (command, compute pass). Each command's compute pass
/// writes its records from record 0, so the next one goes only after the mesh pass drew them.
fn two_pass_steps(max: u32) -> impl Iterator<Item = (u32, bool)> {
    (0..max).flat_map(|k| [(k, true), (k, false)])
}

/// The record buffer of this list for `need_bytes` (at most 128 MiB): buffer, address, bytes.
unsafe fn ensure_res_buffer(this: *mut c_void, need_bytes: u64) -> Result<(ID3D12Resource, u64, u64), String> {
    let need = need_bytes.max(4 << 20).next_multiple_of(1 << 20).min(128 << 20);
    if let Some((buf, cap, _dev)) = with(this, |s| s.res_buf.as_ref().filter(|(_, cap, _)| *cap >= need).cloned()) {
        let va = buf.GetGPUVirtualAddress();
        return Ok((buf, va, cap));
    }
    let dev = super::real::list_device(this).ok_or("no device for two-pass buffer")?;
    let dev_key = dev.as_raw() as usize;
    let picked = {
        let mut free = FREE_RES.lock().unwrap();
        free.iter().position(|(_, cap, d)| *d == dev_key && *cap >= need).map(|i| free.swap_remove(i))
    };
    let entry = match picked {
        Some(e) => e,
        None => (default_uav_buffer(&dev, need, "two-pass res buffer")?, need, dev_key),
    };
    let (buf, cap) = (entry.0.clone(), entry.1);
    let va = buf.GetGPUVirtualAddress();
    with(this, |s| {
        if let Some(old) = s.res_buf.replace(entry) {
            s.res_retired.push(old);
        }
    });
    Ok((buf, va, cap))
}

/// The `io` block of a two-pass stage: its user SGPRs (`compute_pass`: the MS layout) with the
/// argument table at `table_at`, and the high dword of unified memory's base, as the other binds
/// write it (the translated pair picks each access's memory window from it).
#[cfg(test)]
fn two_pass_io(layout: Option<&Layout>, args: &[u8], table_at: u64, compute_pass: bool, uma_base: usize, heaps: [u32; 2]) -> Vec<u32> {
    let mut io = match layout {
        Some(l) if compute_pass => console_list::with_heaps(l, None, console_list::user_data_ms(l, args, table_at), heaps),
        Some(l) => console_list::with_heaps(l, None, console_list::user_data_gfx(l, args, table_at), heaps),
        None if compute_pass => {
            let mut v = vec![0u32; 16];
            v[8] = table_at as u32;
            v[9] = (table_at >> 32) as u32;
            v
        }
        None => vec![table_at as u32, (table_at >> 32) as u32],
    };
    io.resize(crate::gpu::recomp::KERNEL_SGPRS, 0);
    io[crate::gpu::recomp::IO_UMA_BASE_HI] = (uma_base as u64 >> 32) as u32;
    io
}

/// Stack zero-allocation form of `two_pass_io`.
fn fill_two_pass_io(layout: Option<&Layout>, args: &[u8], table_at: u64, compute_pass: bool, uma_base: usize, heaps: [u32; 2]) -> [u32; crate::gpu::recomp::KERNEL_SGPRS] {
    let mut io = [0u32; crate::gpu::recomp::KERNEL_SGPRS];
    match layout {
        Some(l) if compute_pass => console_list::fill_user_data_ms(l, args, table_at, heaps, &mut io),
        Some(l) => console_list::fill_user_data(l, true, None, args, table_at, heaps, &mut io),
        None if compute_pass => {
            io[8] = table_at as u32;
            io[9] = (table_at >> 32) as u32;
        }
        None => {
            io[0] = table_at as u32;
            io[1] = (table_at >> 32) as u32;
        }
    }
    io[crate::gpu::recomp::IO_UMA_BASE_HI] = (uma_base as u64 >> 32) as u32;
    io
}

unsafe fn bind_two_pass_stage(
    this: *mut c_void,
    list: &ID3D12GraphicsCommandList,
    real_mesh_pso: usize,
    tp: &super::real::TwoPassPso,
    res_va: u64,
    compute_pass: bool,
) -> Result<(), String> {
    // the pair is translated with heap windows when they are on (`pso::mem_windows`): as `bind_abi`
    ensure_heaps_cached(this);
    let u = crate::gpu::uma::get().ok_or("no unified memory")?;
    let rs_raw = if compute_pass { tp.compute_rs } else { tp.mesh_rs };
    let rs_ptr = rs_raw as *mut c_void;
    let rs = ID3D12RootSignature::from_raw_borrowed(&rs_ptr).ok_or("two-pass root signature")?;
    let pso_raw = if compute_pass { tp.compute } else { real_mesh_pso } as *mut c_void;
    let pc = ID3D12PipelineState::from_raw_borrowed(&pso_raw).ok_or(if compute_pass { "two-pass compute pipeline" } else { "two-pass mesh pipeline" })?;

    let io_dev = with(this, |s| s.vram_io_dev.clone());
    let io_dev = match io_dev {
        Some(cached) => cached,
        None => {
            let d = vram_io_device(this);
            with(this, |s| s.vram_io_dev = Some(d.clone()));
            d
        }
    };
    let mut need_table_fallback: Option<usize> = None;
    let mut need_io_fallback: Option<[u32; crate::gpu::recomp::KERNEL_SGPRS]> = None;
    let try_fast = with(this, |s| -> Result<Option<(bool, u8, [u64; 6])>, String> {
        if s.uma_windows.as_ref().is_none_or(Vec::is_empty) {
            let w = super::real::uma_window_vas();
            if w.is_empty() {
                return Err("unified memory is not open on the GPU".into());
            }
            s.uma_windows = Some(w);
        }
        let windows = s.uma_windows.as_deref().unwrap();
        let first = windows[0];
        let cache = if compute_pass { s.io_cache_tp_cs } else { s.io_cache_tp_ms };
        let io_va = if let Some((gen, va)) = cache.filter(|&(g, _)| g == s.args_gen_gfx) {
            let _ = gen;
            va
        } else {
            let layout = s.layout_gfx.as_deref();
            let args = s.args_gfx.as_slice();
            let inline = layout.map_or(0, |l| if compute_pass { l.inline_ms } else { l.inline_gfx } as usize).min(args.len());
            let tail = &args[inline..];
            let table_p = if tail.is_empty() {
                if s.dummy_table_p == 0 {
                    let Some(p) = s.arena.take(u, 16, 256, &mut s.uma) else {
                        need_table_fallback = Some(16);
                        return Ok(None);
                    };
                    s.dummy_table_p = p as usize;
                }
                s.dummy_table_p
            } else {
                let Some(p) = s.arena.take(u, tail.len().max(16), 256, &mut s.uma) else {
                    need_table_fallback = Some(tail.len().max(16));
                    return Ok(None);
                };
                std::ptr::copy_nonoverlapping(tail.as_ptr(), p, tail.len());
                p as usize
            };
            let io = fill_two_pass_io(layout, args, table_p as u64, compute_pass, u.base(), s.heaps);
            let io_slice = std::slice::from_raw_parts(io.as_ptr().cast::<u8>(), std::mem::size_of_val(&io));
            let va = match io_dev.as_ref().and_then(|dev| alloc_vram_io_in(&mut s.io_blocks, s.allocator, s.prelude_built.is_some(), dev, io_slice)) {
                Some(va) => va,
                None => {
                    let Some(io_p) = s.arena.take(u, io_slice.len().max(16), 256, &mut s.uma) else {
                        need_io_fallback = Some(io);
                        return Ok(None);
                    };
                    std::ptr::copy_nonoverlapping(io_slice.as_ptr(), io_p, io_slice.len());
                    uma_va_in(u, windows, io_p as usize).unwrap_or(first)
                }
            };
            if compute_pass {
                s.io_cache_tp_cs = Some((s.args_gen_gfx, va));
            } else {
                s.io_cache_tp_ms = Some((s.args_gen_gfx, va));
            }
            va
        };
        let views = [
            io_va,
            res_va,
            windows.first().copied().filter(|&v| v != 0).unwrap_or(first),
            windows.get(1).copied().filter(|&v| v != 0).unwrap_or(first),
            windows.get(2).copied().filter(|&v| v != 0).unwrap_or(first),
            windows.get(3).copied().filter(|&v| v != 0).unwrap_or(first),
        ];
        let bound = if compute_pass { &mut s.pc_abi_cs } else { &mut s.pc_abi_gfx };
        let (set_rs, uav_mask) = bound.diff_and_update(rs_raw, views);
        Ok(Some((set_rs, uav_mask, views)))
    })?;
    let (set_rs, uav_mask, views) = match try_fast {
        Some(v) => v,
        None => {
            // Coherent UMA full: allocate from sparse fallback outside the shard lock.
            let (layout, args, heaps, windows, gen) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone(), s.heaps, s.uma_windows.clone().unwrap(), s.args_gen_gfx));
            let first = windows[0];
            let io = match need_io_fallback {
                Some(io) => io,
                None => {
                    let _ = need_table_fallback;
                    let inline = layout.as_ref().map_or(0, |l| if compute_pass { l.inline_ms } else { l.inline_gfx } as usize).min(args.len());
                    let tail = &args[inline..];
                    let p = arena_take(this, u, tail.len().max(16), 256).ok_or("unified memory is full")?;
                    std::ptr::copy_nonoverlapping(tail.as_ptr(), p, tail.len());
                    fill_two_pass_io(layout.as_deref(), &args, p as u64, compute_pass, u.base(), heaps)
                }
            };
            let io_slice = std::slice::from_raw_parts(io.as_ptr().cast::<u8>(), std::mem::size_of_val(&io));
            let io_p = arena_take(this, u, io_slice.len().max(16), 256).ok_or("unified memory is full")?;
            std::ptr::copy_nonoverlapping(io_slice.as_ptr(), io_p, io_slice.len());
            let io_va = uma_va_in(u, &windows, io_p as usize).unwrap_or(first);
            let views = [
                io_va,
                res_va,
                windows.first().copied().filter(|&v| v != 0).unwrap_or(first),
                windows.get(1).copied().filter(|&v| v != 0).unwrap_or(first),
                windows.get(2).copied().filter(|&v| v != 0).unwrap_or(first),
                windows.get(3).copied().filter(|&v| v != 0).unwrap_or(first),
            ];
            let (set_rs, uav_mask) = with(this, |s| {
                if s.args_gen_gfx == gen {
                    if compute_pass {
                        s.io_cache_tp_cs = Some((gen, io_va));
                    } else {
                        s.io_cache_tp_ms = Some((gen, io_va));
                    }
                }
                let bound = if compute_pass { &mut s.pc_abi_cs } else { &mut s.pc_abi_gfx };
                bound.diff_and_update(rs_raw, views)
            });
            (set_rs, uav_mask, views)
        }
    };
    emit_abi_bindings(list, !compute_pass, rs, set_rs, uav_mask, &views);
    list.SetPipelineState(pc);
    Ok(())
}

/// The pass that rewrites a two-pass ExecuteIndirect's arguments, on XCL's ABI: u0 = `count`,
/// `stride`, `offset` (of the DispatchMesh argument), `split`, `cap` (groups with a record), and
/// where the commands are (u2) and go (u3), in bytes. Each command goes out twice, for the compute
/// pass and then (at `count · stride`) for the mesh pass, with at most `cap` groups, all in X:
/// both passes find a group's record by `gid.x` alone. With a count buffer (u0 dword 7 set: the
/// count at dword 8's byte offset in u4), the commands past its count get no groups: they are run
/// one by one (`two_pass_steps`), so without one.
pub(super) const SCALE_ARGS_HLSL: &str = r#"RWByteAddressBuffer io : register(u0);
RWByteAddressBuffer src : register(u2);
RWByteAddressBuffer dst : register(u3);
RWByteAddressBuffer cnt : register(u4);
[numthreads(64, 1, 1)]
void main(uint tid : SV_DispatchThreadID) {
    uint count = io.Load(0), stride = io.Load(4), offset = io.Load(8), split = io.Load(12), cap = io.Load(16);
    if (tid >= count) return;
    uint live = io.Load(28) != 0u ? min(count, cnt.Load(io.Load(32))) : count;
    uint s = io.Load(20) + tid * stride, d = io.Load(24) + tid * stride, m = d + count * stride;
    for (uint i = 0u; i < stride; i += 4u) {
        uint w = src.Load(s + i);
        dst.Store(d + i, w);
        dst.Store(m + i, w);
    }
    uint3 g = src.Load3(s + offset);
    uint n = 0u;
    if (tid < live && g.x != 0u && g.y != 0u && g.z != 0u) {
        n = min(g.x, cap);
        n = g.y > cap / n ? cap : n * g.y;
        n = g.z > cap / n ? cap : n * g.z;
    }
    dst.Store3(d + offset, uint3(n, 1u, 1u));
    dst.Store3(m + offset, uint3(n * split, 1u, 1u));
}
"#;

/// The groups of an indirect dispatch (u2, `io` dword 0: their byte offset) as they go out (u3):
/// 0 x 0 x 0 when they are more than `io` dword 1 in all, or a count alone is (`MOST_GROUPS`). The
/// counts come from memory the GPU wrote; where that is not what the title meant they can be
/// anything, and billions of groups hold the GPU for seconds or until Windows resets the driver.
/// More than 65 535 groups in X with 1 in Y are folded into `65 535 x ceil(x / 65 535)` with the
/// console count written to `u4` at `io` dword 3 (`IO_GROUP_LIMIT * 4`, when non-zero); any other
/// count over D3D12's 65 535 is cut to it.
pub(super) const GUARD_ARGS_HLSL: &str = r#"RWByteAddressBuffer io : register(u0);
RWByteAddressBuffer src : register(u2);
RWByteAddressBuffer dst : register(u3);
RWByteAddressBuffer lim : register(u4);
[numthreads(1, 1, 1)]
void main() {
    uint at = io.Load(0), most = io.Load(4), to = io.Load(8), lim_at = io.Load(12);
    uint3 g = src.Load3(at);
    uint3 out_g = uint3(0u, 0u, 0u);
    uint limit = 0u;
    if (g.x != 0u && g.y != 0u && g.z != 0u && all(g <= most)) {
        if (g.x <= most / g.y) {
            if (g.x * g.y <= most / g.z) {
                if (g.x > 65535u && g.y == 1u && g.z <= 65535u) {
                    uint rows = (g.x - 1u) / 65535u + 1u;
                    if (rows <= 65535u) {
                        out_g = uint3(65535u, rows, g.z);
                        limit = g.x;
                    } else {
                        out_g = min(g, 65535u.xxx);
                    }
                } else {
                    out_g = min(g, 65535u.xxx);
                }
            }
        }
    }
    dst.Store3(to, out_g);
    if (lim_at != 0u) {
        lim.Store(lim_at, limit);
    }
}
"#;

/// Descriptor-heap + inline root-constants form of `SCALE_ARGS_HLSL`: reads `src` and `cnt`
/// through the shared descriptor heap's raw window UAVs (`MEM_HEAP_BASE + win`, so unmapped
/// sparse tiles safely read as zero instead of faulting on a raw root UAV) and passes all scalar
/// parameters inline in `b0` (zero UMA arena allocation).
pub(super) const SCALE_ARGS_HEAP_HLSL: &str = r#"cbuffer Args : register(b0) {
    uint count, stride, offset, split, cap, src_slot, src_off, has_cnt, cnt_slot, cnt_off;
};
RWByteAddressBuffer dst : register(u0);
[numthreads(64, 1, 1)]
void main(uint tid : SV_DispatchThreadID) {
    if (tid >= count) return;
    RWByteAddressBuffer src = ResourceDescriptorHeap[src_slot];
    uint live = count;
    if (has_cnt != 0u) {
        RWByteAddressBuffer cnt = ResourceDescriptorHeap[cnt_slot];
        live = min(count, cnt.Load(cnt_off));
    }
    uint s = src_off + tid * stride, d = tid * stride, m = d + count * stride;
    for (uint i = 0u; i < stride; i += 4u) {
        uint w = src.Load(s + i);
        dst.Store(d + i, w);
        dst.Store(m + i, w);
    }
    uint3 g = src.Load3(s + offset);
    uint n = 0u;
    if (tid < live && g.x != 0u && g.y != 0u && g.z != 0u) {
        n = min(g.x, cap);
        n = g.y > cap / n ? cap : n * g.y;
        n = g.z > cap / n ? cap : n * g.z;
    }
    dst.Store3(d + offset, uint3(n, 1u, 1u));
    dst.Store3(m + offset, uint3(n * split, 1u, 1u));
}
"#;

/// Descriptor-heap + inline root-constants form of `GUARD_ARGS_HLSL`.
pub(super) const GUARD_ARGS_HEAP_HLSL: &str = r#"cbuffer Args : register(b0) {
    uint src_slot, src_off, most, lim_off;
};
RWByteAddressBuffer dst : register(u0);
RWByteAddressBuffer lim : register(u1);
[numthreads(1, 1, 1)]
void main() {
    RWByteAddressBuffer src = ResourceDescriptorHeap[src_slot];
    uint3 g = src.Load3(src_off);
    uint3 out_g = uint3(0u, 0u, 0u);
    uint limit = 0u;
    if (g.x != 0u && g.y != 0u && g.z != 0u && all(g <= most)) {
        if (g.x <= most / g.y) {
            if (g.x * g.y <= most / g.z) {
                if (g.x > 65535u && g.y == 1u && g.z <= 65535u) {
                    uint rows = (g.x - 1u) / 65535u + 1u;
                    if (rows <= 65535u) {
                        out_g = uint3(65535u, rows, g.z);
                        limit = g.x;
                    } else {
                        out_g = min(g, 65535u.xxx);
                    }
                } else {
                    out_g = min(g, 65535u.xxx);
                }
            }
        }
    }
    dst.Store3(0, out_g);
    if (lim_off != 0u) {
        lim.Store(lim_off, limit);
    }
}
"#;

/// Batched prelude form of `SCALE_ARGS_HLSL` using 4 raw root UAV windows (`u2..u5`).
pub(super) const PRELUDE_SCALE_HLSL: &str = r#"RWByteAddressBuffer jobs : register(u0);
RWByteAddressBuffer dst  : register(u1);
RWByteAddressBuffer w0   : register(u2);
RWByteAddressBuffer w1   : register(u3);
RWByteAddressBuffer w2   : register(u4);
RWByteAddressBuffer w3   : register(u5);
uint load_u32(uint win, uint off) {
    if (win == 0u) return w0.Load(off);
    if (win == 1u) return w1.Load(off);
    if (win == 2u) return w2.Load(off);
    return w3.Load(off);
}
uint3 load_u32x3(uint win, uint off) {
    if (win == 0u) return w0.Load3(off);
    if (win == 1u) return w1.Load3(off);
    if (win == 2u) return w2.Load3(off);
    return w3.Load3(off);
}
[numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint b = gid.x * 48u;
    uint count   = jobs.Load(b);
    uint stride  = jobs.Load(b + 4u);
    uint offset  = jobs.Load(b + 8u);
    uint split   = jobs.Load(b + 12u);
    uint cap     = jobs.Load(b + 16u);
    uint src_win = jobs.Load(b + 20u);
    uint src_off = jobs.Load(b + 24u);
    uint dst_off = jobs.Load(b + 28u);
    uint has_cnt = jobs.Load(b + 32u);
    uint cnt_win = jobs.Load(b + 36u);
    uint cnt_off = jobs.Load(b + 40u);
    uint live = count;
    if (has_cnt != 0u) {
        live = min(count, load_u32(cnt_win, cnt_off));
    }
    for (uint tid = gtid.x; tid < count; tid += 64u) {
        uint s = src_off + tid * stride;
        uint d = dst_off + tid * stride;
        uint m = d + count * stride;
        for (uint i = 0u; i < stride; i += 4u) {
            uint w = load_u32(src_win, s + i);
            dst.Store(d + i, w);
            dst.Store(m + i, w);
        }
        uint3 g = load_u32x3(src_win, s + offset);
        uint n = 0u;
        if (tid < live && g.x != 0u && g.y != 0u && g.z != 0u) {
            n = min(g.x, cap);
            n = g.y > cap / n ? cap : n * g.y;
            n = g.z > cap / n ? cap : n * g.z;
        }
        dst.Store3(d + offset, uint3(n, 1u, 1u));
        dst.Store3(m + offset, uint3(n * split, 1u, 1u));
    }
}
"#;

/// Descriptor-heap form of `PRELUDE_SCALE_HLSL` (`ResourceDescriptorHeap[MEM_HEAP_BASE + win]`).
pub(super) const PRELUDE_SCALE_HEAP_HLSL: &str = r#"RWByteAddressBuffer jobs : register(u0);
RWByteAddressBuffer dst  : register(u1);
[numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint b = gid.x * 48u;
    uint count   = jobs.Load(b);
    uint stride  = jobs.Load(b + 4u);
    uint offset  = jobs.Load(b + 8u);
    uint split   = jobs.Load(b + 12u);
    uint cap     = jobs.Load(b + 16u);
    uint src_win = jobs.Load(b + 20u);
    uint src_off = jobs.Load(b + 24u);
    uint dst_off = jobs.Load(b + 28u);
    uint has_cnt = jobs.Load(b + 32u);
    uint cnt_win = jobs.Load(b + 36u);
    uint cnt_off = jobs.Load(b + 40u);
    RWByteAddressBuffer src = ResourceDescriptorHeap[NonUniformResourceIndex(4u + src_win)];
    uint live = count;
    if (has_cnt != 0u) {
        RWByteAddressBuffer cnt = ResourceDescriptorHeap[NonUniformResourceIndex(4u + cnt_win)];
        live = min(count, cnt.Load(cnt_off));
    }
    for (uint tid = gtid.x; tid < count; tid += 64u) {
        uint s = src_off + tid * stride;
        uint d = dst_off + tid * stride;
        uint m = d + count * stride;
        for (uint i = 0u; i < stride; i += 4u) {
            uint w = src.Load(s + i);
            dst.Store(d + i, w);
            dst.Store(m + i, w);
        }
        uint3 g = src.Load3(s + offset);
        uint n = 0u;
        if (tid < live && g.x != 0u && g.y != 0u && g.z != 0u) {
            n = min(g.x, cap);
            n = g.y > cap / n ? cap : n * g.y;
            n = g.z > cap / n ? cap : n * g.z;
        }
        dst.Store3(d + offset, uint3(n, 1u, 1u));
        dst.Store3(m + offset, uint3(n * split, 1u, 1u));
    }
}
"#;

/// `XCL_INDIRECT_PRELUDE=0`: run two-pass argument scaling inline instead of hoisting into the
/// batched `xcl prelude` command list.
fn indirect_prelude_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_INDIRECT_PRELUDE").map_or(true, |v| v.trim() != "0"))
}

/// Top byte bound of a raw window UAV created by `runner::raw_window_uav`.
const RAW_WINDOW_UAV_MAX: usize = (1usize << 32) - 32;

/// Map a 4-byte-aligned UMA host address + byte span to `(window_index, byte_offset)` for
/// prelude jobs (up to `MAX_HEAP_WINDOWS` when shared heap window views are active, else 4).
fn uma_win_off(u: &crate::gpu::uma::Uma, p: usize, span: usize) -> Option<(u32, u32)> {
    if p % 4 != 0 || span == 0 {
        return None;
    }
    let k = u.window_of(p)?;
    let heap_mode = super::console_heap::window_views_ready();
    let max_wins = if heap_mode { crate::gpu::uma::MAX_HEAP_WINDOWS } else { 4 };
    if k >= max_wins {
        return None;
    }
    let (base, size) = u.window(k);
    let off = p.checked_sub(base)?;
    let end = off.checked_add(span)?;
    let limit = if heap_mode { size.min(RAW_WINDOW_UAV_MAX) } else { size.min(u32::MAX as usize) };
    if end > limit {
        return None;
    }
    Some((k as u32, off as u32))
}

fn can_hoist_prelude(this: *mut c_void) -> bool {
    indirect_prelude_on()
        && super::title_list::wrapper_of(this).is_some()
        && matches!(
            super::list_x::tracked_list_type(this),
            Some(D3D12_COMMAND_LIST_TYPE_DIRECT | D3D12_COMMAND_LIST_TYPE_COMPUTE)
        )
        && with(this, |s| !s.prelude_blocked && s.allocator != 0 && s.prelude_built.is_none())
}

/// Map a 4-byte-aligned UMA host address + byte span to `(heap_slot, byte_offset)` within the
/// shared descriptor heap's window UAVs (`MEM_HEAP_BASE + win`), respecting `raw_window_uav`'s
/// 32-byte top margin on 4 GiB windows.
fn uma_heap_slot_off(u: &crate::gpu::uma::Uma, p: usize, span: usize) -> Option<(u32, u32)> {
    if p % 4 != 0 || span == 0 {
        return None;
    }
    let k = u.window_of(p)?;
    if k >= crate::gpu::uma::MAX_HEAP_WINDOWS {
        return None;
    }
    let (base, size) = u.window(k);
    let off = p.checked_sub(base)?;
    let end = off.checked_add(span)?;
    if end > size.min(RAW_WINDOW_UAV_MAX) {
        return None;
    }
    Some((crate::gpu::recomp::MEM_HEAP_BASE + k as u32, off as u32))
}

/// The groups of an indirect dispatch above which they are not the title's (`GUARD_ARGS_HLSL`):
/// a 4K target in groups of 2x2 pixels is 2.1 million.
const MOST_GROUPS: u32 = 1 << 22;

/// local: `XCL_INDIRECT_MOST_GROUPS` instead of `MOST_GROUPS`
fn most_groups() -> u32 {
    static V: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("XCL_INDIRECT_MOST_GROUPS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(MOST_GROUPS))
}

/// Align a GPU virtual address down to a 256-byte root UAV base and its 4-byte-aligned shader
/// byte offset (`RWByteAddressBuffer::Load` requires 4-byte alignment, and D3D12 root raw UAVs
/// require >= 16-byte alignment; UMA window buffers are 64 KiB-aligned so `va & !255` stays in
/// the same buffer and sparse tile). Rejects addresses `< 256` so the aligned base is never 0.
fn align_root_uav(va: u64) -> Result<(u64, u32), String> {
    if va < 256 || va % 4 != 0 {
        return Err(format!("indirect buffer address {va:#x} is invalid or not 4-byte aligned"));
    }
    Ok((va & !255, (va & 255) as u32))
}

type ArgsPipelines = Mutex<Option<HashMap<usize, (ID3D12RootSignature, ID3D12PipelineState)>>>;

unsafe fn heap_args_root_signature(dev: &ID3D12Device, num_consts: u32, uav_regs: &[u32]) -> Result<ID3D12RootSignature, String> {
    let mut params = Vec::with_capacity(1 + uav_regs.len());
    params.push(D3D12_ROOT_PARAMETER1 {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
        Anonymous: D3D12_ROOT_PARAMETER1_0 {
            Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: 0, Num32BitValues: num_consts },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    });
    for &reg in uav_regs {
        params.push(D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                    ShaderRegister: reg,
                    RegisterSpace: 0,
                    Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        });
    }
    let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
        Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
        Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
            Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                NumStaticSamplers: 0,
                pStaticSamplers: std::ptr::null(),
                Flags: D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED,
            },
        },
    };
    let mut blob: Option<windows::Win32::Graphics::Direct3D::ID3DBlob> = None;
    D3D12SerializeVersionedRootSignature(&desc, &mut blob, None).map_err(|e| format!("heap args root signature serialize: {e}"))?;
    let blob = blob.ok_or("heap args root signature blob")?;
    dev.CreateRootSignature(0, std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize()))
        .map_err(|e| format!("heap args root signature create: {e}"))
}

unsafe fn heap_args_pipeline(dev: &ID3D12Device, hlsl: &str, label: &str, num_consts: u32, uav_regs: &[u32], cache: &'static ArgsPipelines) -> Result<(ID3D12RootSignature, ID3D12PipelineState), String> {
    let key = dev.as_raw() as usize;
    if let Some(pair) = cache.lock().unwrap().as_ref().and_then(|m| m.get(&key).cloned()) {
        return Ok(pair);
    }
    let dxil = crate::gpu::recomp::dxc::compile_cached(hlsl, "main", "cs_6_6")?.0;
    let rs = heap_args_root_signature(dev, num_consts, uav_regs)?;
    let mut cdesc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: std::mem::ManuallyDrop::new(Some(rs.clone())),
        CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as *const c_void, BytecodeLength: dxil.len() },
        ..Default::default()
    };
    let res_pso: Result<ID3D12PipelineState, _> = dev.CreateComputePipelineState(&cdesc);
    std::mem::ManuallyDrop::drop(&mut cdesc.pRootSignature);
    let pso = res_pso.map_err(|e| format!("{label} heap pso create: {e}"))?;
    let _ = pso.SetName(&windows::core::HSTRING::from(label));
    cache.lock().unwrap().get_or_insert_with(HashMap::new).insert(key, (rs.clone(), pso.clone()));
    Ok((rs, pso))
}

unsafe fn guard_indirect_heap_pipeline(dev: &ID3D12Device) -> Result<(ID3D12RootSignature, ID3D12PipelineState), String> {
    static CACHE: ArgsPipelines = Mutex::new(None);
    heap_args_pipeline(dev, GUARD_ARGS_HEAP_HLSL, "xcl guard_dispatch_args", 4, &[0, 1], &CACHE)
}

unsafe fn scale_indirect_heap_pipeline(dev: &ID3D12Device) -> Result<(ID3D12RootSignature, ID3D12PipelineState), String> {
    static CACHE: ArgsPipelines = Mutex::new(None);
    heap_args_pipeline(dev, SCALE_ARGS_HEAP_HLSL, "xcl make_scaled_args_buffer", 10, &[0], &CACHE)
}

unsafe fn prelude_root_signature(dev: &ID3D12Device, num_uavs: usize, heap_indexed: bool) -> Result<ID3D12RootSignature, String> {
    let params: Vec<D3D12_ROOT_PARAMETER1> = (0..num_uavs)
        .map(|reg| D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                    ShaderRegister: reg as u32,
                    RegisterSpace: 0,
                    Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        })
        .collect();
    let flags = if heap_indexed {
        D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED
    } else {
        D3D12_ROOT_SIGNATURE_FLAG_NONE
    };
    let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
        Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
        Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
            Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                NumStaticSamplers: 0,
                pStaticSamplers: std::ptr::null(),
                Flags: flags,
            },
        },
    };
    let mut blob: Option<windows::Win32::Graphics::Direct3D::ID3DBlob> = None;
    D3D12SerializeVersionedRootSignature(&desc, &mut blob, None).map_err(|e| format!("prelude root signature serialize: {e}"))?;
    let blob = blob.ok_or("prelude root signature blob")?;
    dev.CreateRootSignature(0, std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize()))
        .map_err(|e| format!("prelude root signature create: {e}"))
}

unsafe fn prelude_scale_pipeline_in(
    dev: &ID3D12Device,
    hlsl: &str,
    num_uavs: usize,
    heap_indexed: bool,
    cache: &'static ArgsPipelines,
) -> Result<(ID3D12RootSignature, ID3D12PipelineState), String> {
    let key = dev.as_raw() as usize;
    if let Some(pair) = cache.lock().unwrap().as_ref().and_then(|m| m.get(&key).cloned()) {
        return Ok(pair);
    }
    let dxil = crate::gpu::recomp::dxc::compile_cached(hlsl, "main", "cs_6_6")?.0;
    let rs = prelude_root_signature(dev, num_uavs, heap_indexed)?;
    let mut cdesc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: std::mem::ManuallyDrop::new(Some(rs.clone())),
        CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as *const c_void, BytecodeLength: dxil.len() },
        ..Default::default()
    };
    let res_pso: Result<ID3D12PipelineState, _> = dev.CreateComputePipelineState(&cdesc);
    std::mem::ManuallyDrop::drop(&mut cdesc.pRootSignature);
    let pso = res_pso.map_err(|e| format!("prelude scale pso create: {e}"))?;
    let _ = pso.SetName(windows::core::w!("xcl prelude_scale_mesh_args"));
    cache.lock().unwrap().get_or_insert_with(HashMap::new).insert(key, (rs.clone(), pso.clone()));
    Ok((rs, pso))
}

unsafe fn prelude_scale_pipeline(dev: &ID3D12Device) -> Result<(ID3D12RootSignature, ID3D12PipelineState), String> {
    static CACHE: ArgsPipelines = Mutex::new(None);
    prelude_scale_pipeline_in(dev, PRELUDE_SCALE_HLSL, 6, false, &CACHE)
}

unsafe fn prelude_scale_heap_pipeline(dev: &ID3D12Device) -> Result<(ID3D12RootSignature, ID3D12PipelineState), String> {
    static CACHE: ArgsPipelines = Mutex::new(None);
    prelude_scale_pipeline_in(dev, PRELUDE_SCALE_HEAP_HLSL, 2, true, &CACHE)
}

/// Allocate `bytes` in `State::prelude_scratch` (disjoint from inline `scratch` so the prelude
/// leaves these buffers in `INDIRECT_ARGUMENT` for the entire main segment).
unsafe fn alloc_prelude_scratch(this: *mut c_void, dev: &ID3D12Device, bytes: u64) -> Result<(ID3D12Resource, u64, usize), String> {
    let bytes = bytes.next_multiple_of(256);
    if let Some(r) = with(this, |s| {
        let sl = &mut s.prelude_scratch;
        let (b, cap, _) = sl.buf.as_ref()?;
        (sl.used + bytes <= *cap).then(|| {
            sl.used += bytes;
            (b.clone(), sl.used - bytes, sl.buf_idx)
        })
    }) {
        return Ok(r);
    }
    let size = bytes.max(SCRATCH_BLOCK);
    let remaining = SCRATCH_CAP.saturating_sub(with(this, |s| s.scratch_taken));
    if size > remaining {
        return Err(format!("its rewritten arguments are past {} MiB this recording", SCRATCH_CAP >> 20));
    }
    let dev_key = dev.as_raw() as usize;
    let picked = {
        let mut free = FREE_RES.lock().unwrap();
        free.iter().position(|(_, cap, d)| *d == dev_key && *cap >= size && *cap <= remaining).map(|i| free.swap_remove(i))
    };
    let block = match picked {
        Some(e) => e,
        None => (default_uav_buffer(dev, size, "prelude two-pass argument scratch")?, size, dev_key),
    };
    let b = block.0.clone();
    let buf_idx = with(this, |s| {
        let idx = s.prelude_bufs.len();
        s.scratch_taken += block.1;
        s.prelude_bufs.push(block.0.clone());
        s.prelude_scratch = PreludeScratchSlot { buf: Some(block.clone()), used: bytes, buf_idx: idx };
        s.res_retired.push(block);
        idx
    });
    Ok((b, 0, buf_idx))
}

unsafe fn guard_indirect_pipeline(dev: &ID3D12Device) -> Result<(ID3D12RootSignature, ID3D12PipelineState), String> {
    static CACHE: Mutex<Option<HashMap<usize, (ID3D12RootSignature, ID3D12PipelineState)>>> = Mutex::new(None);
    let key = dev.as_raw() as usize;
    if let Some(pair) = CACHE.lock().unwrap().as_ref().and_then(|m| m.get(&key).cloned()) {
        return Ok(pair);
    }
    let dxil = crate::gpu::recomp::dxc::compile_cached(GUARD_ARGS_HLSL, "main", "cs_6_6")?.0;
    let rs = crate::gpu::recomp::pso::abi_root_signature(dev, false)?;
    let mut cdesc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: std::mem::ManuallyDrop::new(Some(rs.clone())),
        CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as _, BytecodeLength: dxil.len() },
        ..Default::default()
    };
    let pso: ID3D12PipelineState = dev.CreateComputePipelineState(&cdesc).map_err(|e| format!("guard_indirect_pipeline: {e}"))?;
    std::mem::ManuallyDrop::drop(&mut cdesc.pRootSignature);
    CACHE.lock().unwrap().get_or_insert_with(HashMap::new).insert(key, (rs.clone(), pso.clone()));
    Ok((rs, pso))
}

/// `XCL_INDIRECT_GUARD=1`: opt in to guarding and folding indirect dispatch group counts.
/// Default off: folding non-count indirect arguments triggers `0x887A0006` device removals on NVIDIA.
fn indirect_guard() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_INDIRECT_GUARD").is_ok_and(|v| v.trim() == "1"))
}

/// The arguments of one indirect dispatch at UMA address `src`, checked and >65 535 X-folded
/// (`GUARD_ARGS_HLSL`): writes the guarded/folded `(x, y, z)` to scratch (`INDIRECT_ARGUMENT`)
/// and writes `IO_GROUP_LIMIT` into `dispatch_io_va`. The list's compute root signature and
/// pipeline are the guard's after this.
unsafe fn guard_dispatch_args(this: *mut c_void, list: &ID3D12GraphicsCommandList, src: usize, dispatch_io_va: u64) -> Result<(ID3D12Resource, u64), String> {
    if super::list_x::tracked_list_type(this) == Some(D3D12_COMMAND_LIST_TYPE_BUNDLE) {
        return Err("command bundles cannot record indirect argument barriers".into());
    }
    let dev = super::real::list_device(this).ok_or("no device")?;
    let u = crate::gpu::uma::get().ok_or("no unified memory")?;
    super::real::uma_window_buffer(src, 12).ok_or("the indirect arguments are not in one memory window")?;
    let (dst, dst_off) = scratch_args(this, &dev, 256)?;
    let dst_va = dst.GetGPUVirtualAddress() + dst_off;
    let lim_off = (crate::gpu::recomp::IO_GROUP_LIMIT * 4) as u32;
    if super::console_heap::window_views_ready() {
        if let Some((src_slot, src_off)) = uma_heap_slot_off(u, src, 12) {
            let (rs, pso) = guard_indirect_heap_pipeline(&dev)?;
            let consts = [src_slot, src_off, most_groups(), lim_off];
            ensure_heaps_cached(this);
            // the title's writes of the arguments before this (`fresh_indirect_arguments`)
            emit_uav_and_transition(list, &dst, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            list.SetComputeRootSignature(&rs);
            list.SetPipelineState(&pso);
            list.SetComputeRoot32BitConstants(0, consts.len() as u32, consts.as_ptr() as *const c_void, 0);
            list.SetComputeRootUnorderedAccessView(1, dst_va);
            list.SetComputeRootUnorderedAccessView(2, dispatch_io_va);
            list.Dispatch(1, 1, 1);
            emit_transition_and_uav(list, &dst, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
            with(this, |s| s.pc_abi_cs = AbiBound::default());
            return Ok((dst, dst_off));
        }
    }
    let (rs, pso) = guard_indirect_pipeline(&dev)?;
    let src_va = super::real::uma_va_of(src).ok_or("unified memory is not open on the GPU")?;
    let (src_base, src_off) = align_root_uav(src_va)?;
    let consts = [src_off, most_groups(), 0, lim_off];
    let p = arena_take(this, u, std::mem::size_of_val(&consts), 256).ok_or("unified memory is full")?;
    std::ptr::copy_nonoverlapping(consts.as_ptr().cast::<u8>(), p, std::mem::size_of_val(&consts));
    let consts_va = super::real::uma_va_of(p as usize).ok_or("unified memory is not open on the GPU")?;
    // the title's writes of the arguments before this (`fresh_indirect_arguments`)
    emit_uav_and_transition(list, &dst, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    list.SetComputeRootSignature(&rs);
    list.SetPipelineState(&pso);
    list.SetComputeRootUnorderedAccessView(0, consts_va);
    list.SetComputeRootUnorderedAccessView(2, src_base);
    list.SetComputeRootUnorderedAccessView(3, dst_va);
    list.SetComputeRootUnorderedAccessView(4, dispatch_io_va);
    list.Dispatch(1, 1, 1);
    // Transition scratch to INDIRECT_ARGUMENT and flush the `IO_GROUP_LIMIT` store in `dispatch_io_va`.
    emit_transition_and_uav(list, &dst, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    with(this, |s| s.pc_abi_cs = AbiBound::default());
    Ok((dst, dst_off))
}
unsafe fn scale_indirect_pipeline(dev: &ID3D12Device) -> Result<(ID3D12RootSignature, ID3D12PipelineState), String> {
    static CACHE: Mutex<Option<HashMap<usize, (ID3D12RootSignature, ID3D12PipelineState)>>> = Mutex::new(None);
    let key = dev.as_raw() as usize;
    if let Some(pair) = CACHE.lock().unwrap().as_ref().and_then(|m| m.get(&key).cloned()) {
        return Ok(pair);
    }
    let dxil = crate::gpu::recomp::dxc::compile_cached(SCALE_ARGS_HLSL, "main", "cs_6_6")?.0;
    let rs = crate::gpu::recomp::pso::abi_root_signature(dev, false)?;
    let mut cdesc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: std::mem::ManuallyDrop::new(Some(rs.clone())),
        CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as *const c_void, BytecodeLength: dxil.len() },
        ..Default::default()
    };
    let res_pso: Result<ID3D12PipelineState, _> = dev.CreateComputePipelineState(&cdesc);
    std::mem::ManuallyDrop::drop(&mut cdesc.pRootSignature);
    let pso = res_pso.map_err(|e| format!("scale pso create: {e}"))?;
    CACHE.lock().unwrap().get_or_insert_with(HashMap::new).insert(key, (rs.clone(), pso.clone()));
    Ok((rs, pso))
}

/// `bytes` of this recording's scratch blocks (COMMON between uses): the block and the offset.
/// Past `SCRATCH_CAP` a recording gets no more.
unsafe fn scratch_args(this: *mut c_void, dev: &ID3D12Device, bytes: u64) -> Result<(ID3D12Resource, u64), String> {
    let bytes = bytes.next_multiple_of(256);
    if let Some(r) = with(this, |s| {
        let (b, cap, _) = s.scratch.as_ref()?;
        (s.scratch_used + bytes <= *cap).then(|| {
            s.scratch_used += bytes;
            (b.clone(), s.scratch_used - bytes)
        })
    }) {
        return Ok(r);
    }
    let size = bytes.max(SCRATCH_BLOCK);
    let remaining = SCRATCH_CAP.saturating_sub(with(this, |s| s.scratch_taken));
    if size > remaining {
        return Err(format!("its rewritten arguments are past {} MiB this recording", SCRATCH_CAP >> 20));
    }
    let dev_key = dev.as_raw() as usize;
    let picked = {
        let mut free = FREE_RES.lock().unwrap();
        free.iter().position(|(_, cap, d)| *d == dev_key && *cap >= size && *cap <= remaining).map(|i| free.swap_remove(i))
    };
    let block = match picked {
        Some(e) => e,
        None => (default_uav_buffer(dev, size, "two-pass argument scratch")?, size, dev_key),
    };
    let b = block.0.clone();
    with(this, |s| {
        s.scratch_taken += block.1;
        s.scratch_used = bytes;
        s.scratch = Some(block.clone());
        s.res_retired.push(block);
    });
    Ok((b, 0))
}

/// The root UAVs of the pass that rewrites a two-pass ExecuteIndirect's arguments
/// (`SCALE_ARGS_HLSL`), as (root slot, GPU address): every view its program declares gets one.
///
/// u4 holds the count and the program reads it only with a count buffer – but the root signature
/// was just set, so a slot left out holds nothing, and a driver may load from it ahead of the
/// test: a load at GPU address 0, which an NVIDIA GPU answers with a fault and a device removal
/// (0x887A0006, the first operation not done in the list: this pass's Dispatch – a retail
/// title's world, three times, each at the first such draw of a list). Without a count buffer
/// u4 names the constants, which are always there.
fn scale_views(consts: u64, src: u64, dst: u64, count: Option<u64>) -> [(u32, u64); 4] {
    [(0, consts), (2, src), (3, dst), (4, count.unwrap_or(consts))]
}

/// The arguments of `max` commands at UMA address `src`, rewritten for the two passes
/// (`SCALE_ARGS_HLSL`) into a scratch block: the block (INDIRECT_ARGUMENT until the caller puts
/// it back to COMMON) and where the compute pass's commands are; the mesh pass's follow them.
///
/// They are read through the memory window over them, as translated shaders read memory, not
/// through the title's buffer: its state on the PC is the title's and not known here. So is the
/// count (`count`: its UMA address, when the title gave a count buffer).
#[allow(clippy::too_many_arguments)]
unsafe fn make_scaled_args_buffer(this: *mut c_void, list: &ID3D12GraphicsCommandList, src: usize, count: Option<usize>, max: u32, stride: u32, mesh_offset: u32, split: u32, cap: u32) -> Result<(ID3D12Resource, u64, bool), String> {
    if super::list_x::tracked_list_type(this) == Some(D3D12_COMMAND_LIST_TYPE_BUNDLE) {
        return Err("command bundles cannot record indirect argument barriers".into());
    }
    let dev = super::real::list_device(this).ok_or("no device")?;
    let u = crate::gpu::uma::get().ok_or("no unified memory")?;
    let commands = (max as u64).saturating_mul(stride as u64);
    super::real::uma_window_buffer(src, commands as usize).ok_or("the indirect arguments are not in one memory window")?;
    if let Some(p) = count {
        super::real::uma_window_buffer(p, 4).ok_or("the indirect count is not in a memory window")?;
    }
    if can_hoist_prelude(this) {
        if let Some((src_win, src_off)) = uma_win_off(u, src, commands as usize) {
            let cnt_loc = match count {
                Some(p) => uma_win_off(u, p, 4).map(|(w, o)| (1u32, w, o)),
                None => Some((0u32, 0u32, 0u32)),
            };
            if let Some((has_cnt, cnt_win, cnt_off)) = cnt_loc {
                let (dst, dst_off, buf_idx) = alloc_prelude_scratch(this, &dev, commands.saturating_mul(2))?;
                if let Ok(dst_off32) = u32::try_from(dst_off) {
                    if dst_off32.checked_add((commands * 2) as u32).is_some() {
                        with(this, |s| {
                            s.prelude_scales.push(PreludeScaleJob {
                                buf_idx,
                                dst_off: dst_off32,
                                count: max,
                                stride,
                                mesh_offset,
                                split,
                                cap,
                                src_win,
                                src_off,
                                has_cnt,
                                cnt_win,
                                cnt_off,
                            });
                        });
                        return Ok((dst, dst_off, true));
                    }
                }
            }
        }
    }
    let (dst, dst_off) = scratch_args(this, &dev, commands.saturating_mul(2))?;
    let dst_va = dst.GetGPUVirtualAddress() + dst_off;
    if super::console_heap::window_views_ready() {
        if let Some((src_slot, src_off)) = uma_heap_slot_off(u, src, commands as usize) {
            let cnt_loc = match count {
                Some(p) => uma_heap_slot_off(u, p, 4).map(|(sl, off)| (1u32, sl, off)),
                None => Some((0u32, src_slot, 0u32)),
            };
            if let Some((has_cnt, cnt_slot, cnt_off)) = cnt_loc {
                let (rs, pso) = scale_indirect_heap_pipeline(&dev)?;
                let consts = [max, stride, mesh_offset, split, cap, src_slot, src_off, has_cnt, cnt_slot, cnt_off];
                ensure_heaps_cached(this);
                // the title's writes of the arguments before this (`fresh_indirect_arguments`)
                emit_uav_and_transition(list, &dst, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                list.SetComputeRootSignature(&rs);
                list.SetPipelineState(&pso);
                list.SetComputeRoot32BitConstants(0, consts.len() as u32, consts.as_ptr() as *const c_void, 0);
                list.SetComputeRootUnorderedAccessView(1, dst_va);
                list.Dispatch(max.div_ceil(64), 1, 1);
                emit_transition(list, &dst, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
                with(this, |s| s.pc_abi_cs = AbiBound::default());
                return Ok((dst, dst_off, false));
            }
        }
    }
    let (rs, pso) = scale_indirect_pipeline(&dev)?;
    let src_va = super::real::uma_va_of(src).ok_or("unified memory is not open on the GPU")?;
    let count_va = match count {
        Some(p) => Some(super::real::uma_va_of(p).ok_or("unified memory is not open on the GPU")?),
        None => None,
    };
    let consts = [max, stride, mesh_offset, split, cap, 0, 0, count_va.is_some() as u32, 0];
    let p = arena_take(this, u, std::mem::size_of_val(&consts), 256).ok_or("unified memory is full")?;
    std::ptr::copy_nonoverlapping(consts.as_ptr().cast::<u8>(), p, std::mem::size_of_val(&consts));
    let consts_va = super::real::uma_va_of(p as usize).ok_or("unified memory is not open on the GPU")?;
    // the title's writes of the arguments before this (`fresh_indirect_arguments`)
    emit_uav_barrier(list, None);
    emit_transition(list, &dst, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    list.SetComputeRootSignature(&rs);
    list.SetPipelineState(&pso);
    for (slot, va) in scale_views(consts_va, src_va, dst_va, count_va) {
        list.SetComputeRootUnorderedAccessView(slot, va);
    }
    list.Dispatch(max.div_ceil(64), 1, 1);
    emit_transition(list, &dst, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    with(this, |s| s.pc_abi_cs = AbiBound::default());
    Ok((dst, dst_off, false))
}

pub(super) unsafe extern "system" fn dispatch_mesh(this: *mut c_void, x: u32, y: u32, z: u32) {
    type PcDm = unsafe extern "system" fn(*mut c_void, u32, u32, u32);
    let pc_dm: PcDm = std::mem::transmute(super::real::orig_fn(this, 79));
    let raw_pso = with(this, |s| s.pso);
    let pso = super::pso_async::resolve_wait(raw_pso);
    if let Some(tp) = super::real::two_pass_of(pso) {
        let groups = x.saturating_mul(y).saturating_mul(z);
        if groups == 0 {
            return;
        }
        let Some(list) = ID3D12GraphicsCommandList::from_raw_borrowed(&this) else { return };
        let record_bytes = tp.record.record_dwords() as u64 * 4;
        let need = (groups as u64).saturating_mul(record_bytes);
        let run = (|| -> Result<(), String> {
            let (res_buf, res_va, cap) = ensure_res_buffer(this, need)?;
            let n = groups.min(record_groups(cap, record_bytes, tp.mesh_groups_per_record));
            if n == 0 {
                return Err(format!("a record of {record_bytes} B does not fit the record buffer"));
            }
            if n < groups {
                static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if N.fetch_add(1, Ordering::Relaxed) < 8 {
                    crate::xlog_warn!("D3D12.X: two-pass DispatchMesh of {groups} groups: {} past the record buffer dropped", groups - n);
                }
            }
            bind_two_pass_stage(this, list, pso, &tp, res_va, true)?;
            list.Dispatch(n, 1, 1);
            emit_uav_barrier(list, Some(&res_buf));
            bind_two_pass_stage(this, list, pso, &tp, res_va, false)?;
            pc_dm(this, n * tp.mesh_groups_per_record, 1, 1);
            emit_uav_barrier(list, Some(&res_buf));
            Ok(())
        })();
        match run {
            Ok(()) => {
                count(DRAWS);
                static LOGGED: AtomicU64 = AtomicU64::new(0);
                let n = LOGGED.fetch_add(1, Ordering::Relaxed) + 1;
                if n <= 8 || n.is_power_of_two() {
                    crate::xlog_info!("D3D12.X: two-pass DispatchMesh #{n} ({groups} groups, split {}, kind {})", tp.mesh_groups_per_record, tp.kind);
                }
            }
            Err(e) => {
                count(DRAWS_SKIPPED);
                static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if N.fetch_add(1, Ordering::Relaxed) < 8 {
                    crate::xlog_warn!("D3D12.X: two-pass DispatchMesh skipped: {e}");
                }
            }
        }
        return;
    }
    if super::pso_async::is_pending(raw_pso) {
        count(DRAWS_SKIPPED);
        return;
    }
    let rs = with(this, |s| {
        s.pc_abi_gfx = AbiBound::default();
        s.rs_gfx
    }) as *mut c_void;
    if let (Some(list), Some(r)) = (ID3D12GraphicsCommandList::from_raw_borrowed(&this), ID3D12RootSignature::from_raw_borrowed(&rs)) {
        list.SetGraphicsRootSignature(r);
    }
    count(DRAWS_PC);
    pc_dm(this, x, y, z);
}

/// command signatures of one Dispatch argument → their stride (a split program runs them as its
/// chain, `chained_indirect`)
static DISPATCH_ONLY: RwLock<Option<HashMap<usize, u32>>> = RwLock::new(None);

pub(super) fn note_dispatch_only_signature(sig: usize, stride: Option<u32>) {
    let mut s = DISPATCH_ONLY.write().unwrap();
    let s = s.get_or_insert_with(Default::default);
    match stride {
        Some(n) => s.insert(sig, n),
        None => s.remove(&sig),
    };
}

/// #59 ExecuteIndirect(signature, max count, arguments, offset, count buffer, offset): with a
/// translated pipeline, on XCL's ABI with the kept root arguments (signatures that change root
/// arguments themselves cannot be – skipped, logged)
pub(super) unsafe extern "system" fn execute_indirect(this: *mut c_void, sig: usize, max: u32, args: usize, args_off: u64, count_buf: usize, count_off: u64) {
    type F = unsafe extern "system" fn(*mut c_void, usize, u32, usize, u64, usize, u64);
    let pc: F = std::mem::transmute(super::real::orig_fn(this, 59));
    let raw_pso = with(this, |s| s.pso);
    let pending = super::pso_async::is_pending(raw_pso);
    let unbuilt_compute = signature_dispatches(sig) && pending && super::real::is_console_compute(raw_pso);
    // a graphics one too: the title's thread goes on to its next binds, whose builds then run on
    // the workers at once instead of one after another here (only a signature that just draws:
    // vertex / index buffer views and root arguments it sets would stay on the list after it)
    let unbuilt_graphics = signature_draws_only(sig) && pending && !super::real::is_console_compute(raw_pso);
    if (unbuilt_compute || unbuilt_graphics) && super::list_x::defer(this, super::list_x::Deferred::Indirect(sig, max, args, args_off, count_buf, count_off)) {
        return super::pso_async::need_by_submit(raw_pso);
    }
    let pso = super::pso_async::resolve_wait(raw_pso);
    // where the commands are in unified memory (a console buffer's offsets are the title's)
    let commands_at = super::real::console_buffer(args).map(|(p, n)| (p + args_off as usize, n.saturating_sub(args_off as usize)));
    let count_at = super::real::console_buffer(count_buf).filter(|&(_, n)| count_off as usize + 4 <= n).map(|(p, _)| p + count_off as usize);
    let (args_off_title, count_off_title) = (args_off, count_off);
    let (args_off, count_off) = (args_off + super::real::buffer_delta(args), count_off + super::real::buffer_delta(count_buf));
    if let Some(tp) = super::real::two_pass_of(pso) {
        if let Some(msig) = dispatch_mesh_sig_of(sig) {
            let Some(list) = ID3D12GraphicsCommandList::from_raw_borrowed(&this) else { return };
            let run = (|| -> Result<(), String> {
                if max == 0 || args == 0 {
                    return Ok(());
                }
                let commands = max as usize * msig.byte_stride as usize;
                let src = match commands_at {
                    Some((p, n)) if commands <= n => p,
                    _ => return Err("its arguments are not in a console buffer in unified memory".into()),
                };
                let record_bytes = tp.record.record_dwords() as u64 * 4;
                let (res_buf, res_va, cap) = ensure_res_buffer(this, 32 << 20)?;
                let cap = record_groups(cap, record_bytes, tp.mesh_groups_per_record);
                if cap == 0 {
                    return Err(format!("a record of {record_bytes} B does not fit the record buffer"));
                }
                let count = match (count_buf, count_at) {
                    (0, _) => None,
                    (_, Some(p)) => Some(p),
                    _ => return Err("its count is not in a console buffer in unified memory".into()),
                };
                // the groups past the record buffer, and the commands past the count, are dropped
                // on the GPU (`SCALE_ARGS_HLSL`, or hoisted into `xcl prelude`)
                let (scaled, at, hoisted) = make_scaled_args_buffer(this, list, src, count, max, msig.byte_stride, msig.mesh_offset, tp.mesh_groups_per_record, cap)?;
                let passes = (|| -> Result<(), String> {
                    for (k, compute) in two_pass_steps(max) {
                        bind_two_pass_stage(this, list, pso, &tp, res_va, compute)?;
                        let (s, first) = if compute { (msig.companion_dispatch, at) } else { (sig, at + commands as u64) };
                        pc(this, s, 1, scaled.as_raw() as usize, first + k as u64 * msig.byte_stride as u64, 0, 0);
                        emit_uav_barrier(list, Some(&res_buf));
                    }
                    Ok(())
                })();
                if !hoisted {
                    emit_transition(list, &scaled, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COMMON);
                }
                passes
            })();
            match run {
                Ok(()) => {
                    count(DRAWS);
                    static LOGGED: AtomicU64 = AtomicU64::new(0);
                    let n = LOGGED.fetch_add(1, Ordering::Relaxed) + 1;
                    if n <= 8 || n.is_power_of_two() {
                        crate::xlog_info!("D3D12.X: two-pass ExecuteIndirect(DispatchMesh) #{n} (max {max}, split {}, kind {})", tp.mesh_groups_per_record, tp.kind);
                    }
                }
                Err(e) => {
                    count(DRAWS_SKIPPED);
                    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                    if N.fetch_add(1, Ordering::Relaxed) < 8 {
                        crate::xlog_warn!("D3D12.X: two-pass ExecuteIndirect skipped: {e}");
                    }
                }
            }
            return;
        }
    }
    let (gfx, compute) = (super::real::is_translated_graphics(pso), super::real::is_translated_compute(pso));
    if !gfx && !compute {
        // A pipeline still being built (or whose build failed): its stand-in draws nothing, and
        // the list may hold XCL's ABI root signature from a translated draw before it – the PC
        // call with that pair is invalid (debug layer 953, "the currently set Root Signature
        // doesn't match the currently set Pipeline State Object"; a retail title's world then
        // hung the GPU, 0x887a0006). Skipped, like a draw or dispatch of such a pipeline.
        if super::pso_async::bound_unbuilt(raw_pso, pso) {
            count(DRAWS_SKIPPED);
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: ExecuteIndirect with a pipeline that is still being built skipped");
            }
            return;
        }
        // A finished two-pass MS/GS pipeline is a PC mesh shader pipeline (MS+PS) registered in
        // TWO_PASS rather than TRANSLATED_GRAPHICS. Falling through to raw PC ExecuteIndirect
        // with the title's graphics root signature (e.g. a non-DispatchMesh indirect draw or a
        // root-changing signature) invalidates the command list (Close → E_INVALIDARG).
        if let Some(tp) = super::real::two_pass_of(pso) {
            count(DRAWS_SKIPPED);
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!(
                    "D3D12.X: ExecuteIndirect with two-pass {} pipeline ({pso:#x}) skipped – PC pipeline is a mesh shader",
                    tp.kind
                );
            }
            return;
        }
        // a console compute program without a translation (its build failed): nothing to run,
        // as for Dispatch. The PC call would run it with the list's compute root signature, which
        // can still be XCL's ABI one from a translated dispatch before it (debug layer 953).
        // A native PC compute pipeline is not one of these and still goes to the PC call.
        // (`CONSOLE_COMPUTE` holds the title's object, which for a shared stand-in differs from
        // the driver pipeline `resolve_wait` returned.)
        if signature_dispatches(sig) && (super::real::is_console_compute(raw_pso) || super::real::is_console_compute(pso)) {
            count(DISPATCHES_SKIPPED);
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: ExecuteIndirect dispatch with a pipeline that has no translated console shader ({raw_pso:#x}) skipped");
            }
            return;
        }
        // a PC pipeline: the title's root signature, as for a draw (`graphics_bound`)
        let rs = with(this, |s| {
            s.pc_abi_gfx = AbiBound::default();
            s.rs_gfx
        }) as *mut c_void;
        if let (Some(list), Some(r)) = (ID3D12GraphicsCommandList::from_raw_borrowed(&this), ID3D12RootSignature::from_raw_borrowed(&rs)) {
            list.SetGraphicsRootSignature(r);
        }
        let t = super::gpu_profile::begin(this, pso, super::gpu_profile::Kind::PcIndirect, [max, 0, 0]);
        fresh_indirect_arguments(this, || pc(this, sig, max, args, args_off, count_buf, count_off));
        super::gpu_profile::end(this, t);
        return;
    }
    if ROOT_SIGNATURES.read().unwrap().as_ref().is_some_and(|s| s.contains(&sig)) {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 4 {
            crate::xlog_warn!("D3D12.X: ExecuteIndirect that changes root arguments with a translated pipeline skipped");
        }
        return;
    }
    let Some(list) = ID3D12GraphicsCommandList::from_raw_borrowed(&this) else { return };
    if compute {
        // a split program: its chain (it binds the ABI itself)
        let ei = || fresh_indirect_arguments(this, || pc(this, sig, max, args, args_off, count_buf, count_off));
        let t_chain = super::gpu_profile::begin(this, pso, super::gpu_profile::Kind::IndirectDispatch, [max, 0, 0]);
        match chained_indirect(this, list, pso, sig, max, (args, args_off_title, args_off), (count_buf, count_off_title, count_off), ei) {
            Ok(true) => {
                super::gpu_profile::end(this, t_chain);
                if super::ambient::has_compute_psos() {
                    let (layout, args, view_heap) = with(this, |s| (s.layout.clone(), s.args.clone(), s.heaps[0]));
                    super::ambient::after_compute_dispatch(this, pso, [0, 0, 0], layout.as_deref(), &args, view_heap);
                }
                note_gpu_write(this);
                return count(DISPATCHES);
            }
            Ok(false) => {
                super::gpu_profile::end(this, t_chain);
            }
            Err(e) => {
                super::gpu_profile::end(this, t_chain);
                count(DISPATCHES_SKIPPED);
                static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if N.fetch_add(1, Ordering::Relaxed) < 4 {
                    crate::xlog_warn!("D3D12.X: ExecuteIndirect of a split program skipped: {e}");
                }
                return;
            }
        }
    }
    // (before the dispatch's own root signature and pipeline are bound)
    let mut guarded = None;
    let mut preset_io = None;
    if compute && max == 1 && indirect_guard() && signature_dispatches(sig) {
        if let Some((p, _)) = commands_at.filter(|&(_, n)| n >= 12) {
            // Allocate a dedicated `io` buffer for this dispatch so `guard_dispatch_args` can
            // write `IO_GROUP_LIMIT` on the GPU without racing against other `io` buffers.
            match alloc_fresh_compute_io(this, pso) {
                Ok(io_va) => {
                    preset_io = Some(io_va);
                    match guard_dispatch_args(this, list, p, io_va) {
                        Ok(g) => guarded = Some(g),
                        Err(e) => {
                            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                                crate::xlog_warn!("D3D12.X: ExecuteIndirect's group counts are not checked: {e}");
                            }
                        }
                    }
                }
                Err(e) => {
                    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                    if N.fetch_add(1, Ordering::Relaxed) < 4 {
                        crate::xlog_warn!("D3D12.X: ExecuteIndirect's group counts are not checked: {e}");
                    }
                }
            }
        }
    }
    let (args, args_off) = guarded.as_ref().map_or((args, args_off), |(b, o)| (b.as_raw() as usize, *o));
    match bind_abi(this, list, pso, gfx, 0, preset_io) {
        Ok(()) => {
            if compute {
                match super::real::dispatch_chain::whole(pso) {
                    Ok(super::real::dispatch_chain::WholePso::Ready(whole)) => list.SetPipelineState(&whole),
                    Ok(super::real::dispatch_chain::WholePso::None) => {}
                    Ok(super::real::dispatch_chain::WholePso::Pending) => {
                        if let Some((b, _)) = guarded {
                            emit_transition(list, &b, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COMMON);
                        }
                        count(DISPATCHES_SKIPPED);
                        return;
                    }
                    Err(e) => {
                        if let Some((b, _)) = guarded {
                            emit_transition(list, &b, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COMMON);
                        }
                        crate::xlog_warn!("D3D12.X: ExecuteIndirect whole compute PSO: {e}");
                        count(DISPATCHES_SKIPPED);
                        return;
                    }
                }
            }
            count(if gfx { DRAWS } else { DISPATCHES });
            let t = super::gpu_profile::begin(this, pso, if gfx { super::gpu_profile::Kind::IndirectDraw } else { super::gpu_profile::Kind::IndirectDispatch }, [max, 0, 0]);
            fresh_indirect_arguments(this, || pc(this, sig, max, args, args_off, count_buf, count_off));
            super::gpu_profile::end(this, t);
            if let Some((b, _)) = guarded {
                emit_transition(list, &b, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COMMON);
            }
            if compute {
                if super::ambient::has_compute_psos() {
                    let (layout, args, view_heap) = with(this, |s| (s.layout.clone(), s.args.clone(), s.heaps[0]));
                    super::ambient::after_compute_dispatch(this, pso, [0, 0, 0], layout.as_deref(), &args, view_heap);
                }
                note_gpu_write(this);
            }
        }
        Err(e) => {
            if let Some((b, _)) = guarded {
                emit_transition(list, &b, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COMMON);
                if let Some(real) = ID3D12PipelineState::from_raw_borrowed(&(pso as *mut c_void)) {
                    list.SetPipelineState(real);
                }
            }
            count(if gfx { DRAWS_SKIPPED } else { DISPATCHES_SKIPPED });
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: ExecuteIndirect with a translated pipeline skipped: {e}");
            }
        }
    }
}

/// An ExecuteIndirect of split program `pso` with a command signature of one Dispatch, run as its
/// chain (`dispatch_chain::execute_indirect`); the buffers as (resource, title's offset, PC's).
/// Ok(false): nothing recorded – run it as the whole form.
#[allow(clippy::too_many_arguments)]
unsafe fn chained_indirect(this: *mut c_void, list: &ID3D12GraphicsCommandList, pso: usize, sig: usize, max: u32,
    (args, args_title, args_pc): (usize, u64, u64), (count_buf, count_title, count_pc): (usize, u64, u64), ei: impl FnOnce()) -> Result<bool, String> {
    let Some(stride) = DISPATCH_ONLY.read().unwrap().as_ref().and_then(|m| m.get(&sig).copied()) else { return Ok(false) };
    let Some(args_va) = indirect_address(args, args_title, args_pc) else { return Ok(false) };
    let count_va = match count_buf {
        0 => None,
        c => match indirect_address(c, count_title, count_pc) { Some(v) => Some(v), None => return Ok(false) },
    };
    // the arguments there now (the GPU's last ones) size the rounds
    let read = |res: usize, off: u64, n: usize| -> Option<Vec<u32>> {
        let (p, size) = super::real::console_buffer(res)?;
        let at = p + off as usize;
        (off as usize + 4 * n <= size && crate::gpu::shader::readable(at, 4 * n))
            .then(|| (0..n).map(|i| std::ptr::read_volatile((at as *const u32).add(i))).collect())
    };
    let n = if count_buf == 0 { max } else { read(count_buf, count_title, 1).map_or(max, |c| c[0].min(max)) };
    let seen: Vec<[u32; 3]> = (0..n.min(64)).filter_map(|k| read(args, args_title + u64::from(k * stride), 3)).map(|v| [v[0], v[1], v[2]]).collect();
    let predicated = super::list_x::predication_active(this);
    let r = super::real::dispatch_chain::execute_indirect(
        list,
        pso,
        with(this, |s| s.allocator),
        args_va,
        stride,
        max,
        count_va,
        &seen,
        predicated,
        || {
            // `pso::record_slicer` binds the slicer's compute root signature before invoking `bind`.
            invalidate_pc_abi(this, false, true);
            bind_abi(this, list, pso, false, 0, None)
        },
        ei,
    );
    if matches!(r, Ok(true)) {
        invalidate_pc_abi(this, false, true);
    }
    r
}

/// The GPU address of a title's buffer `res` at its offset `title` (a buffer in unified memory) or
/// `pc` (another)
unsafe fn indirect_address(res: usize, title: u64, pc: u64) -> Option<u64> {
    match super::real::console_buffer(res) {
        Some((p, _)) => super::real::uma_va_of(p + title as usize),
        None => Some(ID3D12Resource::from_raw_borrowed(&(res as *mut c_void))?.GetGPUVirtualAddress()).filter(|&va| va != 0).map(|va| va + pc),
    }
}

/// XCL's own buffer for `fresh_indirect_arguments` (device, buffer): COMMON between its uses
static ARGUMENT_ORDER: Mutex<Option<(usize, ID3D12Resource)>> = Mutex::new(None);

/// `op` (an ExecuteIndirect on `list`) reading the arguments and count the list wrote before it.
///
/// The console orders a dispatch that writes indirect arguments before the ExecuteIndirect that
/// reads them with a barrier that reaches the PC as a UAV barrier (`list_x::resource_barrier_x`).
/// NVIDIA's driver (RTX 5090) fetches an ExecuteIndirect's arguments without waiting for that
/// barrier: it ran the count that was there before the list in all but a few of 100 tries, also
/// when the dispatch wrote them through another resource over the same memory, as translated
/// shaders do through the memory window buffers (`execute_indirect_reads_the_arguments_written_before_it`).
/// A retail title's ExecuteIndirect after the dispatch that writes its group count then runs
/// whatever count the memory held before. A transition to INDIRECT_ARGUMENT makes the driver
/// wait for the writes – of any buffer, as measured there. The title's buffer is in the title's
/// state (its transitions reach the PC list), so a buffer of XCL's own takes it, COMMON ->
/// INDIRECT_ARGUMENT before and back after. Only on a list XCL made (`list_x::on_create`) and
/// not a bundle, which cannot hold barriers; its type comes from XCL's state, not a call to it.
unsafe fn fresh_indirect_arguments(list: *mut c_void, op: impl FnOnce()) {
    let tracked = super::list_x::tracked_list_type(list).is_some_and(|t| t != D3D12_COMMAND_LIST_TYPE_BUNDLE);
    let buffer = if tracked { ID3D12GraphicsCommandList::from_raw_borrowed(&list).and_then(|l| argument_order_buffer(l)) } else { None };
    let Some(buffer) = buffer else { return op() };
    let b = buffer.as_raw() as usize;
    super::list_x::transition(list, b, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    op();
    super::list_x::transition(list, b, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COMMON);
}

/// The device's buffer for `fresh_indirect_arguments` (made on first use)
unsafe fn argument_order_buffer(list: &ID3D12GraphicsCommandList) -> Option<ID3D12Resource> {
    let mut dev: Option<ID3D12Device> = None;
    list.GetDevice(&mut dev).ok()?;
    let dev = dev?;
    let mut g = ARGUMENT_ORDER.lock().unwrap();
    if let Some((_, b)) = g.as_ref().filter(|(d, _)| *d == dev.as_raw() as usize) {
        return Some(b.clone());
    }
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: 256,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut b: Option<ID3D12Resource> = None;
    if let Err(e) = dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut b) {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 4 {
            crate::xlog_warn!("D3D12.X: ExecuteIndirect: no buffer to order its arguments after the writes before it ({e}) – it may read old ones");
        }
        return None;
    }
    let b = b?;
    *g = Some((dev.as_raw() as usize, b.clone()));
    Some(b)
}

/// A DrawInstanced of the console's rect list (three corners of each rectangle, the fourth
/// implied) as two triangles a rectangle; false: not a rect list, or one this cannot expand
/// (the caller draws the three vertices as a triangle – half of each rectangle).
///
/// The PC has no rect lists. The fourth corner is made here, on the CPU, from the title's
/// vertex data: v3 = v0 + v2 − v1 (seen in a retail title's full-screen passes: (−1,1) (−1,−1)
/// (1,−1) with texture coordinates (0,0) (0,1) (1,1) – the missing corner is (1,1), (1,0)).
/// The vertex layout is not looked at: each 4-byte word that is the same in all three vertices
/// is kept (a packed colour), any other is taken as a float. Vertices a shader makes from the
/// vertex index alone (no vertex buffer) are not expanded.
unsafe fn draw_rects(this: *mut c_void, list: &ID3D12GraphicsCommandList, vertices: u32, instances: u32, first_vertex: u32, first_instance: u32) -> bool {
    if !cpu_rect_expansion(with(this, |s| s.pso)) {
        return false;
    }
    let Some(vbs) = super::list_x::rect_list_state(this) else { return false };
    let rects = (vertices / 3) as usize;
    if vbs.is_empty() || rects == 0 || rects > 4096 {
        return false;
    }
    let mut views = Vec::new();
    for &(slot, (va, bytes, stride), _) in &vbs {
        let (s, at) = (stride as usize, va as usize + first_vertex as usize * stride as usize);
        let len = rects * 3 * s;
        if va == 0 || s == 0 || s % 4 != 0 || first_vertex as usize * s + len > bytes as usize || !crate::gpu::shader::readable(at, len) {
            return false;
        }
        let p = allocate_upload_memory(this, rects * 6 * s, 16);
        if p.is_null() {
            return false;
        }
        let word = |v: usize, w: usize| (at as *const u8).add(v * s + w * 4).cast::<u32>().read_unaligned();
        for r in 0..rects {
            let out = p.add(r * 6 * s);
            let src = (at as *const u8).add(r * 3 * s);
            // v0 v1 v2, then v0 v2 v3
            std::ptr::copy_nonoverlapping(src, out, 3 * s);
            std::ptr::copy_nonoverlapping(src, out.add(3 * s), s);
            std::ptr::copy_nonoverlapping(src.add(2 * s), out.add(4 * s), s);
            for w in 0..s / 4 {
                let (a, b, c) = (word(r * 3, w), word(r * 3 + 1, w), word(r * 3 + 2, w));
                let d = if a == b && b == c { a } else { (f32::from_bits(a) + f32::from_bits(c) - f32::from_bits(b)).to_bits() };
                out.add(5 * s + w * 4).cast::<u32>().write_unaligned(d);
            }
        }
        views.push((slot, D3D12_VERTEX_BUFFER_VIEW { BufferLocation: super::list_x::pc_va_of(p as u64), SizeInBytes: (rects * 6 * s) as u32, StrideInBytes: stride }));
    }
    for (slot, v) in &views {
        list.IASetVertexBuffers(*slot, Some(&[*v]));
    }
    list.DrawInstanced(rects as u32 * 6, instances, 0, first_instance);
    for &(slot, _, pc) in &vbs {
        if let Some(v) = pc {
            list.IASetVertexBuffers(slot, Some(&[v]));
        }
    }
    static N: AtomicU64 = AtomicU64::new(0);
    if N.fetch_add(1, Ordering::Relaxed) == 0 {
        crate::xlog_info!("D3D12.X: rect lists are drawn as two triangles a rectangle, the fourth corner made on the CPU");
    }
    true
}

/// Whether a rect list drawn with the bound pipeline `pso` is expanded on the CPU (`draw_rects`):
/// not with the rect list GS (`real::has_rect_list_gs`), which completes each rect itself – both
/// would draw every rect twice (each CPU half is three corners of the rect).
fn cpu_rect_expansion(pso: usize) -> bool {
    !super::real::has_rect_list_gs(super::pso_async::resolve(pso))
}

/// #12 DrawInstanced
pub(super) unsafe extern "system" fn draw_instanced(this: *mut c_void, vertices: u32, instances: u32, first_vertex: u32, first_instance: u32) {
    DRAW_ARGS.with(|d| d.set((false, vertices, instances, first_vertex, 0)));
    if let Some(list) = graphics_bound(this) {
        let t = super::gpu_profile::begin(this, with(this, |s| s.pso), super::gpu_profile::Kind::Draw, [vertices, instances, 0]);
        if draw_rects(this, &list, vertices, instances, first_vertex, first_instance) {
            super::gpu_profile::end(this, t);
            if super::ambient::has_sky_psos() {
                let pso = super::pso_async::resolve(with(this, |s| s.pso));
                super::ambient::after_draw(this, pso);
            }
            return;
        }
        super::list_x::sync_title_buffers(this, None, first_vertex, vertices, if first_instance != 0 { instances.max(2) } else { instances });
        list.DrawInstanced(vertices, instances, first_vertex, first_instance);
        super::gpu_profile::end(this, t);
        if super::ambient::has_sky_psos() {
            let pso = super::pso_async::resolve(with(this, |s| s.pso));
            super::ambient::after_draw(this, pso);
        }
    }
}

thread_local! {
    /// the draw being bound: (indexed, count, instances, first index / vertex, base vertex) – for the frame trace
    static DRAW_ARGS: std::cell::Cell<(bool, u32, u32, u32, i32)> = const { std::cell::Cell::new((false, 0, 0, 0, 0)) };
}

/// #13 DrawIndexedInstanced
pub(super) unsafe extern "system" fn draw_indexed_instanced(this: *mut c_void, indices: u32, instances: u32, first_index: u32, base_vertex: i32, first_instance: u32) {
    if let Some(list) = prepare_indexed_draw(this, indices, instances, first_index, base_vertex) {
        super::list_x::sync_title_buffers(this, Some((first_index, indices, base_vertex)), 0, 0, if first_instance != 0 { instances.max(2) } else { instances });
        emit_prepared_indexed_draw(&list, indices, instances, first_index, base_vertex, first_instance);
        if super::ambient::has_sky_psos() {
            let pso = super::pso_async::resolve(with(this, |s| s.pso));
            super::ambient::after_draw(this, pso);
        }
    }
}

/// Prepare an indexed draw once, before callers make temporary native state changes.
pub(super) unsafe fn prepare_indexed_draw(
    this: *mut c_void,
    indices: u32,
    instances: u32,
    first_index: u32,
    base_vertex: i32,
) -> Option<ID3D12GraphicsCommandList> {
    DRAW_ARGS.with(|d| d.set((true, indices, instances, first_index, base_vertex)));
    graphics_bound(this)
}

/// Submit an indexed draw whose translated graphics state has already been prepared.
pub(super) unsafe fn emit_prepared_indexed_draw(
    list: &ID3D12GraphicsCommandList,
    indices: u32,
    instances: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
) {
    let this = list.as_raw();
    let t = super::gpu_profile::begin(this, with(this, |s| s.pso), super::gpu_profile::Kind::Draw, [indices, instances, 0]);
    list.DrawIndexedInstanced(indices, instances, first_index, base_vertex, first_instance);
    super::gpu_profile::end(this, t);
}

/// Before a draw: with a translated pipeline its ABI root signature and the kept graphics
/// arguments are bound (None: the draw is skipped – a console pipeline without a
/// translation, or no unified memory).
unsafe fn graphics_bound(this: *mut c_void) -> Option<ID3D12GraphicsCommandList> {
    let list = ID3D12GraphicsCommandList::from_raw_borrowed(&this)?.clone();
    let (pso, cached_gfx) = with(this, |s| (s.pso, s.pso_abi_cache.is_some_and(|c| c.query_pso == s.pso && c.is_gfx)));
    if !cached_gfx {
        // a pipeline still being built in the background: its stand-in, which draws nothing
        // (`resolve_wait` waits for it only up to `XCL_PSO_WAIT_MS`)
        let real = super::pso_async::resolve_wait(pso);
        if !super::real::is_translated_graphics(real) {
        // A finished two-pass MS/GS pipeline is a PC mesh shader pipeline (MS+PS) registered in
        // TWO_PASS rather than TRANSLATED_GRAPHICS. Binding the title's graphics RS and issuing
        // DrawInstanced / DrawIndexedInstanced against a mesh PSO invalidates the command list
        // (Close → E_INVALIDARG).
        if let Some(tp) = super::real::two_pass_of(real) {
            count(DRAWS_SKIPPED);
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!(
                    "D3D12.X: draw with two-pass {} pipeline ({real:#x}) skipped – PC pipeline is a mesh shader",
                    tp.kind
                );
            }
            return None;
        }
        count(DRAWS_PC);
        if super::present::trace_frame() {
            // which pipelines draw without a translation (first 12 distinct)
            static SEEN: Mutex<Vec<usize>> = Mutex::new(Vec::new());
            let mut g = SEEN.lock().unwrap();
            if !g.contains(&pso) && g.len() < 12 {
                g.push(pso);
                let real = super::pso_async::resolve_wait(pso);
                crate::xlog_info!(
                    "frame trace: draw with an untranslated pipeline {pso:#x} (resolves to {real:#x}; pending {}, two-pass {}, RT0 {:?})",
                    super::pso_async::is_pending(pso),
                    super::real::two_pass_of(real).is_some(),
                    super::present::rt0_resource(this as usize).map(|r| ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| { let d = x.GetDesc(); (d.Width, d.Height, d.Format.0) }))
                );
            }
        }
        // a PC pipeline (also a stand-in or fallback: made with the title's root signature):
        // the title's root signature, which nothing else binds on the PC list – a draw without
        // one is invalid (the debug layer reports it on every GPU; WARP faults on it)
        if super::pso_async::is_pending(pso) {
            super::list_x::note_dropped_draw(this);
        }
        let rs = with(this, |s| {
            s.pc_abi_gfx = AbiBound::default();
            s.rs_gfx
        }) as *mut c_void;
        if let Some(r) = ID3D12RootSignature::from_raw_borrowed(&rs) {
            list.SetGraphicsRootSignature(r);
        }
        return Some(list);
        }
    }
    if super::present::trace_frame() {
        trace_ui_draw(this, pso);
    }
    match bind_abi(this, &list, pso, true, 0, None) {
        Ok(()) => {
            count(DRAWS);
            Some(list)
        }
        Err(e) => {
            super::list_x::note_dropped_draw(this);
            count(DRAWS_SKIPPED);
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: draw with a translated pipeline skipped: {e}");
            }
            None
        }
    }
}

/// `XCL_TRACE_FRAME`: draws into a presented buffer (the UI) once the title has presented for a
/// while – their pipeline, root arguments and the resources their descriptor tables start with
/// (format, size, kept in step with title memory or not). The first 24.
/// `XCL_TRACE_FRAME`, after 300 presents: one line per distinct (VS, PS, render target) drawn –
/// the shaders' dump hashes, the target and the textures of the first descriptor table.
unsafe fn trace_draw_kinds(this: *mut c_void, pso: usize) {
    // keyed by the target's shape, not the object: a UI library's transient targets (Renoir's
    // layers) are new objects every frame and would fill the list at once
    static SEEN: Mutex<Vec<(u64, u64, Option<(u64, u32, i32, u32)>)>> = Mutex::new(Vec::new());
    if super::present::activity().0 < 300 {
        return;
    }
    let Some((vs, ps)) = super::real::pso_shaders(super::pso_async::resolve(pso)) else { return };
    let rt = super::present::rt0_resource(this as usize).unwrap_or(0);
    let desc = |r: usize| ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| { let d = x.GetDesc(); (d.Width, d.Height, d.Format.0, d.SampleDesc.Count) });
    {
        let mut g = SEEN.lock().unwrap();
        let key = (vs, ps, desc(rt));
        if g.len() >= 160 || g.contains(&key) {
            return;
        }
        g.push(key);
    }
    let (layout, args) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone()));
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    let mut tex = Vec::new();
    if let (Some(layout), Some(u)) = (layout, crate::gpu::uma::get()) {
        for p in layout.params.iter() {
            if let Param::Table { at } = *p {
                let t = word(at as usize) as usize;
                for k in 0..4usize {
                    if let Some(r) = super::console_heap::resource_at(u.base() + t + 32 * k) {
                        tex.push(format!("{:?}{}", desc(r), super::title_textures::address(r).map_or(String::new(), |a| format!(" at {a:#x}"))));
                    }
                }
            }
            if tex.len() >= 4 {
                break;
            }
        }
    }
    crate::xlog_info!(
        "frame trace: draws vs {vs:016x} ps {ps:016x} into {:?}{} – textures {}; {}",
        desc(rt),
        if super::present::is_presented(rt) { " (presented)" } else { "" },
        tex.join(", "),
        super::list_x::state_summary(this)
    );
}

/// `XCL_TRACE_SHADERS=hash,…` (dump hashes, hex): the draws traced with their root arguments
/// are those whose vertex or pixel shader is listed – into any target, at any time (the first
/// 24); without it, draws into a presented target (or any – `XCL_TRACE_UI_ANY`) after 300
/// presents.
fn traced_shaders() -> &'static [u64] {
    static S: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    S.get_or_init(|| std::env::var("XCL_TRACE_SHADERS").unwrap_or_default().split(',').filter_map(|h| u64::from_str_radix(h.trim(), 16).ok()).collect())
}

/// `XCL_TRACE_FRAME` with a watched target (`present::watch_rt`): draws whose descriptor
/// tables name it – that object, or another at its title address (the first 16).
unsafe fn trace_reads_of_watched(this: *mut c_void, pso: usize) {
    let Some(w) = super::present::watched_rt() else { return };
    let w_addr = super::real::title_address(w).or_else(|| super::title_textures::address(w));
    let Some(u) = crate::gpu::uma::get() else { return };
    let (layout, args) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone()));
    let Some(layout) = layout else { return };
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    for p in layout.params.iter() {
        let Param::Table { at } = *p else { continue };
        let t = word(at as usize) as usize;
        for k in 0..16usize {
            let Some(r) = super::console_heap::resource_at(u.base() + t + 32 * k) else { continue };
            let addr = super::real::title_address(r).or_else(|| super::title_textures::address(r));
            if super::present::is_watched(r) || (addr.is_some() && addr == w_addr) {
                let rt = super::present::rt0_resource(this as usize).unwrap_or(0);
                // each (source, target, shaders) once
                static SEEN: Mutex<Vec<(usize, usize, usize)>> = Mutex::new(Vec::new());
                {
                    let mut g = SEEN.lock().unwrap();
                    if g.len() >= 32 || g.contains(&(r, rt, pso)) {
                        return;
                    }
                    g.push((r, rt, pso));
                }
                if !super::present::is_presented(rt) {
                    super::present::follow(rt);
                }
                if let Some(dsv) = super::list_x::bound_dsv(this) {
                    super::list_x::watch_dsv(dsv);
                }
                let d = |r: usize| ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| { let d = x.GetDesc(); (d.Width, d.Height, d.Format.0, d.Flags.0) });
                crate::xlog_info!(
                    "frame trace: a draw (shaders {:x?}, {}) reads a watched target {r:#x} {:?} at {addr:x?} (slot {k}), drawing into {rt:#x} {:?}{} – {}",
                    super::real::pso_shaders(super::pso_async::resolve(pso)),
                    super::list_x::state_summary(this),
                    d(r),
                    d(rt),
                    if super::present::is_presented(rt) { " (presented)" } else { "" },
                    root_args_summary(this, 32)
                );
                let args = DRAW_ARGS.with(|d| d.get());
                crate::xlog_info!(
                    "frame trace: … that draw: {args:?} (indexed, count, instances, first, base vertex); input layout [{}]; {}",
                    super::real::pso_layout(super::pso_async::resolve(pso)).unwrap_or_default(),
                    super::list_x::geometry_summary(this, args)
                );
                return;
            }
        }
    }
}

/// `XCL_TRACE_FRAME`: render targets a draw into the watched target samples (a UI library's
/// layers) are dumped with it (`present::follow`).
unsafe fn follow_layer_sources(this: *mut c_void) {
    let (Some(w), Some(rt)) = (super::present::watched_rt(), super::present::rt0_resource(this as usize)) else { return };
    if rt != w {
        return;
    }
    let Some(u) = crate::gpu::uma::get() else { return };
    let (layout, args) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone()));
    let Some(layout) = layout else { return };
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    for p in layout.params.iter() {
        let Param::Table { at } = *p else { continue };
        let t = word(at as usize) as usize;
        for k in 0..4usize {
            let Some(r) = super::console_heap::resource_at(u.base() + t + 32 * k) else { continue };
            let layer = ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).is_some_and(|x| {
                let d = x.GetDesc();
                d.Flags.0 & 1 != 0 && d.Width >= 64 && d.Height >= 32 && d.SampleDesc.Count == 1
            });
            if layer && r != w {
                super::present::follow(r);
            }
        }
    }
}

unsafe fn trace_ui_draw(this: *mut c_void, pso: usize) {
    trace_draw_kinds(this, pso);
    trace_reads_of_watched(this, pso);
    follow_layer_sources(this);
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    // `XCL_TRACE_MANY=n`: n draws instead of 24 (a whole frame of a UI)
    static CAP: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    let cap = *CAP.get_or_init(|| std::env::var("XCL_TRACE_MANY").ok().and_then(|v| v.parse().ok()).unwrap_or(24));
    if N.load(std::sync::atomic::Ordering::Relaxed) >= cap {
        return;
    }
    let Some(rt) = super::present::rt0_resource(this as usize) else { return };
    let wanted = traced_shaders();
    if wanted.is_empty() {
        // draws into a presented target, or – `XCL_TRACE_UI_ANY` – into any target
        if super::present::activity().0 < 300 || (!super::present::is_presented(rt) && std::env::var_os("XCL_TRACE_UI_ANY").is_none()) {
            return;
        }
    } else if !super::real::pso_shaders(super::pso_async::resolve(pso)).is_some_and(|(v, p)| wanted.contains(&v) || wanted.contains(&p)) {
        return;
    } else {
        super::present::watch_rt(rt);
    }
    N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let sh = super::real::pso_shaders(super::pso_async::resolve(pso)).map(|(v, p)| format!("vs {v:016x} ps {p:016x}"));
    crate::xlog_info!("frame trace: UI draw (pipeline {pso:#x}, {}): {}", sh.unwrap_or_default(), root_args_summary(this, 16));
    if !wanted.is_empty() {
        // the picked shaders' geometry too (vertices dumped at the next present)
        let args = DRAW_ARGS.with(|d| d.get());
        crate::xlog_info!(
            "frame trace: … that draw: {args:?} (indexed, count, instances, first, base vertex); input layout [{}]; {}",
            super::real::pso_layout(super::pso_async::resolve(pso)).unwrap_or_default(),
            super::list_x::geometry_summary(this, args)
        );
    }
}

/// A draw's root arguments for the frame trace: constants, root buffers (their first words, as
/// the CPU sees them), the first slots of descriptor tables (the PC resources they name).
unsafe fn root_args_summary(this: *mut c_void, buffer_words: usize) -> String {
    let Some(u) = crate::gpu::uma::get() else { return String::new() };
    let (layout, args) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone()));
    let Some(layout) = layout else { return String::new() };
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    let mut what = Vec::new();
    for (i, p) in layout.params.iter().enumerate() {
        match *p {
            Param::Constants { at, count } => what.push(format!("p{i} constants {:x?}", (0..count.min(8) as usize).map(|k| word(at as usize + 4 * k)).collect::<Vec<_>>())),
            Param::Descriptor { at } => {
                let va = (word(at as usize) as u64 | (word(at as usize + 4) as u64) << 32) as usize;
                let w: Vec<u32> = if va != 0 && crate::gpu::shader::readable(va, 4 * buffer_words) { (0..buffer_words).map(|k| ((va + 4 * k) as *const u32).read_unaligned()).collect() } else { Vec::new() };
                what.push(format!("p{i} buffer {va:#x} {w:x?}"));
            }
            Param::Table { at } => {
                let t = word(at as usize) as usize;
                let slots: Vec<String> = (0..4usize)
                    .map(|k| {
                        let at = u.base() + t + 32 * k;
                        let w = super::console_heap::words_at((t + 32 * k) as u32);
                        match super::console_heap::resource_at(at) {
                            Some(r) => {
                                if super::title_textures::address(r).is_some() {
                                    super::title_textures::probe_gpu_content(r);
                                }
                                let d = ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| x.GetDesc());
                                format!("[{k}] {r:#x} {:?} title {:?} T# {:x?}", d.map(|d| (d.Dimension.0, d.Width, d.Height, d.Format.0)), super::title_textures::address(r).map(|a| format!("{a:#x}")), w.map(|w| w[..4].to_vec()))
                            }
                            None => format!("[{k}] - T# {:x?}", w.map(|w| w[..4].to_vec())),
                        }
                    })
                    .collect();
                what.push(format!("p{i} table {t:#x}: {}", slots.join(", ")));
            }
        }
    }
    what.join("; ")
}

/// #33 SetComputeRootSignature (PC #29)
pub(super) unsafe extern "system" fn set_compute_root_signature(this: *mut c_void, rs: *mut c_void) {
    let layout = console_list::layout_of(rs as usize);
    with(this, |s| {
        if !console_list::same_layout(&s.layout, &layout) {
            let need = layout.as_ref().map_or(0, |l| l.size as usize);
            s.args.clear();
            s.args.resize(need, 0);
            s.args_gen_cs = s.args_gen_cs.wrapping_add(1);
            s.layout = layout;
        }
        s.rs_cs = rs as usize;
    });
}

/// #35 SetComputeRootDescriptorTable (PC #31): the table's console address, 32 bits
pub(super) unsafe extern "system" fn set_compute_root_table(this: *mut c_void, i: u32, handle: u64) {
    put(this, i, 0, &(handle as u32).to_le_bytes());
}

/// #37 SetComputeRoot32BitConstant (PC #33)
pub(super) unsafe extern "system" fn set_compute_root_constant(this: *mut c_void, i: u32, value: u32, offset: u32) {
    put(this, i, offset, &value.to_le_bytes());
}

/// #39 SetComputeRoot32BitConstants (PC #35)
pub(super) unsafe extern "system" fn set_compute_root_constants(this: *mut c_void, i: u32, n: u32, data: *const u32, offset: u32) {
    if !data.is_null() && n > 0 {
        let bytes = std::slice::from_raw_parts(data.cast::<u8>(), n as usize * 4);
        put(this, i, offset, bytes);
    }
}

/// #41 / #43 / #45 SetComputeRoot{ConstantBuffer,ShaderResource,UnorderedAccess}View (PC #37 /
/// #39 / #41): the 64-bit GPU address
pub(super) unsafe extern "system" fn set_compute_root_descriptor(this: *mut c_void, i: u32, va: u64) {
    note_address(va);
    let va = readable_root(va);
    put(this, i, 0, &va.to_le_bytes());
}

/// #25 SetPipelineState: noted, then bound as on the PC
pub(super) unsafe extern "system" fn set_pipeline_state(this: *mut c_void, pso: *mut c_void) {
    with(this, |s| {
        if s.pso != pso as usize {
            s.pso = pso as usize;
            s.pso_abi_cache = None;
        }
    });
    super::real::bind_pipeline(this, pso);
}

/// #14 Dispatch: with a translated pipeline, on XCL's ABI with the kept root arguments
pub(super) unsafe extern "system" fn dispatch(this: *mut c_void, x: u32, y: u32, z: u32) {
    dispatch_flags(this, x, y, z, 0);
}

/// `DispatchX` flag: x, y, z are thread counts, not group counts; the console launches whole
/// groups and leaves the lanes past the count out of EXEC (0x8000, wave32, needs nothing here)
const DISPATCH_THREAD_DIMENSIONS: u32 = 0x20;

/// What a dispatch of `dims` with `flags` launches on pipeline `pso`: its groups and the thread
/// limits for `IO_THREAD_LIMIT` (0: that axis ends on a whole group). None: it cannot be run
/// as the console would (thread dimensions with a group shape not known).
fn dispatch_groups(pso: usize, dims: [u32; 3], flags: u32) -> Option<([u32; 3], [u32; 3])> {
    if flags & DISPATCH_THREAD_DIMENSIONS == 0 {
        return Some((dims, [0; 3]));
    }
    let threads = super::real::pso_threads(pso)?.map(|t| t.max(1));
    let groups = std::array::from_fn(|k| dims[k].div_ceil(threads[k]));
    let limits = std::array::from_fn(|k| if dims[k] % threads[k] != 0 { dims[k] } else { 0 });
    Some((groups, limits))
}

/// `Dispatch` / `DispatchX(x, y, z, flags)`.
pub(super) unsafe fn dispatch_flags(this: *mut c_void, x: u32, y: u32, z: u32, flags: u32) {
    let pso = with(this, |s| s.pso);
    if super::pso_async::is_pending(pso) && super::list_x::defer(this, super::list_x::Deferred::Dispatch(x, y, z, flags)) {
        return super::pso_async::need_by_submit(pso);
    }
    let pso = super::pso_async::resolve_wait(pso);
    let Some(list) = ID3D12GraphicsCommandList::from_raw_borrowed(&this) else { return };
    if !super::real::is_translated_compute(pso) {
        // a console pipeline without a translation (including a background build that
        // failed): nothing to run (its root arguments were not given to the PC list either)
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
            crate::xlog_warn!("D3D12.X: Dispatch with a pipeline that has no translated console shader ({pso:#x}) skipped");
        }
        count(DISPATCHES_SKIPPED);
        return;
    }
    let Some((groups, limits)) = dispatch_groups(pso, [x, y, z], flags) else {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
            crate::xlog_warn!("D3D12.X: DispatchX of {x}x{y}x{z} threads (flags {flags:#x}) with a pipeline whose group shape is not known ({pso:#x}) skipped");
        }
        count(DISPATCHES_SKIPPED);
        return;
    };
    if flags != 0 {
        // each kind of flags once: what titles pass, and what it launches
        static SEEN: Mutex<Vec<u32>> = Mutex::new(Vec::new());
        let first = { let mut g = SEEN.lock().unwrap(); !g.contains(&flags) && { g.push(flags); true } };
        if first {
            crate::xlog_info!("D3D12.X: DispatchX with flags {flags:#x}: {x}x{y}x{z} launched as {}x{}x{} groups (thread limits {limits:?})", groups[0], groups[1], groups[2]);
        }
    }
    with(this, |s| s.limits = limits);
    let done = translated_dispatch(this, list, pso, groups[0], groups[1], groups[2]);
    with(this, |s| s.limits = [0; 3]);
    match done {
        Ok(()) => count(DISPATCHES),
        Err(e) => {
            count(DISPATCHES_SKIPPED);
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: Dispatch with a translated pipeline skipped: {e}");
            }
        }
    }
}

/// `XCL_TRACE_FRAME`: a full-screen dispatch (enough groups to cover a 4K target in tiles of up
/// to 128 pixels) – what its root arguments say. A bindless title passes the indices of its
/// descriptors in root constants and constant buffers: every small value there is looked up
/// in the dispatch's descriptor tables, and the resources found (the pass's inputs and
/// outputs) are logged.
unsafe fn trace_fullscreen(this: *mut c_void, pso: usize, groups: [u32; 3]) {
    if groups[0] < 30 || groups[1] < 17 {
        return;
    }
    let Some(u) = crate::gpu::uma::get() else { return };
    let (layout, args) = with(this, |s| (s.layout.clone(), s.args.clone()));
    let Some(layout) = layout else { return };
    static SEEN: Mutex<Vec<usize>> = Mutex::new(Vec::new());
    {
        let mut g = SEEN.lock().unwrap();
        if g.contains(&pso) || g.len() >= 12 {
            return;
        }
        g.push(pso);
    }
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()));
    let mut tables: Vec<u32> = Vec::new();
    let mut values: Vec<u32> = Vec::new();
    let mut what: Vec<String> = Vec::new();
    for (i, p) in layout.params.iter().enumerate() {
        match *p {
            Param::Constants { at, count } => {
                let w: Vec<u32> = (0..count as usize).filter_map(|k| word(at as usize + 4 * k)).collect();
                what.push(format!("p{i} constants {w:x?}"));
                values.extend(&w);
            }
            Param::Descriptor { at } => {
                let lo = word(at as usize).unwrap_or(0) as u64;
                let hi = word(at as usize + 4).unwrap_or(0) as u64;
                let va = (lo | hi << 32) as usize;
                let n = 64usize;
                if va != 0 && crate::gpu::shader::readable(va, 4 * n) {
                    let w: Vec<u32> = (0..n).map(|k| ((va + 4 * k) as *const u32).read_unaligned()).collect();
                    what.push(format!("p{i} buffer {va:#x}: {w:x?}"));
                    values.extend(&w);
                } else {
                    what.push(format!("p{i} buffer {va:#x} (not readable)"));
                }
            }
            Param::Table { at } => {
                let v = word(at as usize).unwrap_or(0);
                what.push(format!("p{i} table {v:#x}"));
                tables.push(v);
            }
        }
    }
    let mut found: Vec<(u32, usize)> = Vec::new();
    for &v in &values {
        if v >= 1 << 20 {
            continue;
        }
        for &t in &tables {
            let slot = u.base() + t as usize + 32 * v as usize;
            if let Some(r) = super::console_heap::resource_at(slot) {
                if !found.iter().any(|x| x.1 == r) {
                    found.push((v, r));
                }
            }
        }
    }
    let res: Vec<String> = found
        .iter()
        .map(|&(v, r)| {
            let d = ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| x.GetDesc());
            format!("[{v}] {r:#x} {:?}{}", d.map(|d| (d.Dimension.0, d.Width, d.Height, d.Format.0)), if super::present::is_presented(r) { " PRESENTED" } else { "" })
        })
        .collect();
    crate::xlog_info!("frame trace: full-screen dispatch {groups:?}, pipeline {pso:#x}: {}", what.join("; "));
    crate::xlog_info!("frame trace: … its indices name: {}", if res.is_empty() { "nothing".to_string() } else { res.join("; ") });
}

/// #148 AllocateUploadMemory(size, alignment) → CPU/GPU pointer in unified memory (used by
/// `XboxBvhBuilder.dll` and other console middleware on default title command lists).
/// Freed automatically when the list's command allocator is reset.
pub(super) unsafe extern "system" fn allocate_upload_memory(this: *mut c_void, size: usize, align: usize) -> *mut u8 {
    let Some(u) = crate::gpu::uma::get() else {
        return std::ptr::null_mut();
    };
    let size = size.max(16);
    let align = align.max(16);
    if size <= crate::gpu::uma::ARENA_BLOCK / 2 {
        if let Some(p) = with(this, |s| s.arena.take(u, size, align, &mut s.uma)) {
            return p;
        }
    }
    if u.has_sparse() {
        if let Some((p, _)) = super::sparse::alloc(size, align) {
            with(this, |s| s.uma.push(p));
            return p as *mut u8;
        }
    }
    match u.alloc(size, align) {
        Some(p) => {
            with(this, |s| s.uma.push(p as usize));
            p
        }
        None => std::ptr::null_mut(),
    }
}

unsafe fn translated_dispatch(this: *mut c_void, list: &ID3D12GraphicsCommandList, pso: usize, x: u32, y: u32, z: u32) -> Result<(), String> {
    if super::present::trace_frame() {
        trace_fullscreen(this, pso, [x, y, z]);
    }
    // more groups in X than D3D12 takes: folded (`recomp::IO_GROUP_LIMIT`)
    let (limit, [x, y, z]) = fold(x, y, z);
    bind_abi(this, list, pso, false, limit, None)?;
    let t = super::gpu_profile::begin(this, pso, super::gpu_profile::Kind::Dispatch, [x, y, z]);
    let r = super::real::dispatch_chain::dispatch(list, pso, with(this, |s| s.allocator), [x, y, z]);
    super::gpu_profile::end(this, t);
    invalidate_pc_abi(this, false, true);
    note_gpu_write(this);
    if super::ambient::has_compute_psos() {
        let (layout, args, view_heap) = with(this, |s| (s.layout.clone(), s.args.clone(), s.heaps[0]));
        super::ambient::after_compute_dispatch(this, pso, [x, y, z], layout.as_deref(), &args, view_heap);
    }
    r
}

/// A dispatch of `x` x `y` x `z` groups as D3D12 takes it: (the count for
/// `recomp::IO_GROUP_LIMIT` - 0: not folded -, the groups to dispatch). More than 65 535 in X
/// with 1 in Y is folded into Y; any other count past 65 535 is clamped (logged once).
pub(super) fn fold(x: u32, y: u32, z: u32) -> (u32, [u32; 3]) {
    use crate::gpu::recomp::FOLD_X;
    if x <= FOLD_X && y <= 65_535 && z <= 65_535 {
        return (0, [x, y, z]);
    }
    if y == 1 && z <= 65_535 && x.div_ceil(FOLD_X) <= 65_535 {
        return (x, [FOLD_X, x.div_ceil(FOLD_X), z]);
    }
    static N: AtomicU64 = AtomicU64::new(0);
    if N.fetch_add(1, Ordering::Relaxed) < 4 {
        crate::xlog_warn!("D3D12.X: a dispatch of {x}x{y}x{z} groups cannot be folded – clamped to 65535 a dimension");
    }
    (0, [x.min(65_535), y.min(65_535), z.min(65_535)])
}

/// Allocate a dedicated compute `io` buffer in UMA for an indirect dispatch whose `IO_GROUP_LIMIT`
/// slot will be written on the GPU by `guard_dispatch_args`.
unsafe fn alloc_fresh_compute_io(this: *mut c_void, pso: usize) -> Result<u64, String> {
    let u = crate::gpu::uma::get().ok_or("no unified memory")?;
    let windows = super::real::uma_window_vas();
    let first = *windows.first().ok_or("unified memory is not open on the GPU")?;
    let real = super::pso_async::resolve_wait(pso);
    let user_sgprs = super::real::pso_user_sgprs(real);
    let (layout, args, heaps) = with(this, |s| (s.layout.clone(), s.args.clone(), s.heaps));
    let copy = |bytes: &[u8]| -> Result<usize, String> {
        let p = arena_take(this, u, bytes.len().max(16), 256).ok_or("unified memory is full")?;
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), p, bytes.len());
        Ok(p as usize)
    };
    let inline = layout.as_ref().map_or(0, |l| console_list::inline_for(l, user_sgprs) as usize).min(args.len());
    let table_at = copy(&args[inline..])?;
    let mut io = [0u32; crate::gpu::recomp::KERNEL_SGPRS];
    match layout.as_ref() {
        Some(l) => console_list::fill_user_data(l, false, user_sgprs, &args, table_at as u64, heaps, &mut io),
        None => {
            io[0] = table_at as u32;
            io[1] = (table_at >> 32) as u32;
        }
    }
    io[crate::gpu::recomp::IO_UMA_BASE_HI] = (u.base() >> 32) as u32;
    io[crate::gpu::recomp::IO_GROUP_LIMIT] = 0;
    let io_bytes = std::slice::from_raw_parts(io.as_ptr().cast::<u8>(), std::mem::size_of_val(&io));
    let io_at = copy(io_bytes)?;
    Ok(uma_va_in(u, &windows, io_at).unwrap_or(first))
}

/// XCL's ABI for a translated pipeline: its root signature, the user SGPRs (`io`, the kept
/// arguments the console's way – compute or graphics) and the argument table in unified
/// memory, u1 res, u2… the memory windows; then the pipeline. `group_limit`: a folded
/// dispatch's group count in X (`recomp::IO_GROUP_LIMIT`), 0 for any other.
unsafe fn bind_abi(this: *mut c_void, list: &ID3D12GraphicsCommandList, pso: usize, gfx: bool, group_limit: u32, preset_io_va: Option<u64>) -> Result<(), String> {
    let u = crate::gpu::uma::get().ok_or("no unified memory")?;
    let (need_heaps, cached_pso, io_dev) = with(this, |s| {
        let c = s.pso_abi_cache.filter(|c| c.query_pso == pso && c.is_gfx == gfx);
        (!s.heaps_ensured, c, s.vram_io_dev.clone())
    });
    if need_heaps {
        ensure_heaps_cached(this);
    }
    let (real, user_sgprs, rs_raw, store_pso_cache) = match cached_pso {
        Some(c) => (c.real_pso, c.user_sgprs, c.rs_raw, false),
        None => {
            // (only reached with a translated pipeline: built, `resolve_wait` returns it at once)
            let real = super::pso_async::resolve_wait(pso);
            let user_sgprs = if gfx { None } else { super::real::pso_user_sgprs(real) };
            let rs_raw = super::real::pso_root_signature(real).or_else(|| super::real::abi_rs_cached().map(|p| p as usize)).ok_or("no XCL ABI root signature")?;
            (real, user_sgprs, rs_raw, true)
        }
    };
    let rs_ptr = rs_raw as *mut c_void;
    let rs = ID3D12RootSignature::from_raw_borrowed(&rs_ptr).ok_or("root signature")?;
    let real_ptr = real as *mut c_void;
    let pc = ID3D12PipelineState::from_raw_borrowed(&real_ptr).ok_or("pipeline")?;

    let io_dev = match io_dev {
        Some(cached) => cached,
        None => {
            let d = vram_io_device(this);
            with(this, |s| s.vram_io_dev = Some(d.clone()));
            d
        }
    };

    let try_fast = with(this, |s| -> Result<Option<(bool, u8, [u64; 6])>, String> {
        if store_pso_cache {
            s.pso_abi_cache = Some(CachedPsoAbi {
                query_pso: pso,
                real_pso: real,
                rs_raw,
                user_sgprs,
                is_gfx: gfx,
            });
        }
        if s.uma_windows.as_ref().is_none_or(Vec::is_empty) {
            let w = super::real::uma_window_vas();
            if w.is_empty() {
                return Err("unified memory is not open on the GPU".into());
            }
            s.uma_windows = Some(w);
        }
        let windows = s.uma_windows.as_deref().unwrap();
        let first = windows[0];
        let limits = if gfx { [0; 3] } else { std::mem::take(&mut s.limits) };
        let can_cache = preset_io_va.is_none() && group_limit == 0 && limits == [0; 3];
        let io_va = if let Some(v) = preset_io_va {
            v
        } else if can_cache && gfx && s.io_cache_gfx.is_some_and(|(g, _)| g == s.args_gen_gfx) {
            s.io_cache_gfx.unwrap().1
        } else if can_cache && !gfx && s.io_cache_cs.is_some_and(|(g, u_s, _)| g == s.args_gen_cs && u_s == user_sgprs) {
            s.io_cache_cs.unwrap().2
        } else {
            let (layout, args) = if gfx { (s.layout_gfx.as_deref(), s.args_gfx.as_slice()) } else { (s.layout.as_deref(), s.args.as_slice()) };
            let inline = layout.map_or(0, |l| if gfx { l.inline_gfx } else { console_list::inline_for(l, user_sgprs) } as usize).min(args.len());
            let tail = &args[inline..];
            let table_p = if tail.is_empty() {
                if s.dummy_table_p == 0 {
                    let Some(p) = s.arena.take(u, 16, 256, &mut s.uma) else {
                        if !gfx {
                            s.limits = limits;
                        }
                        return Ok(None);
                    };
                    s.dummy_table_p = p as usize;
                }
                s.dummy_table_p
            } else {
                let Some(p) = s.arena.take(u, tail.len().max(16), 256, &mut s.uma) else {
                    if !gfx {
                        s.limits = limits;
                    }
                    return Ok(None);
                };
                std::ptr::copy_nonoverlapping(tail.as_ptr(), p, tail.len());
                p as usize
            };
            let mut io = [0u32; crate::gpu::recomp::KERNEL_SGPRS];
            match layout {
                Some(l) => console_list::fill_user_data(l, gfx, user_sgprs, args, table_p as u64, s.heaps, &mut io),
                None => {
                    io[0] = table_p as u32;
                    io[1] = (table_p >> 32) as u32;
                }
            }
            io[crate::gpu::recomp::IO_UMA_BASE_HI] = (u.base() >> 32) as u32;
            io[crate::gpu::recomp::IO_GROUP_LIMIT] = group_limit;
            if !gfx {
                for (k, l) in limits.into_iter().enumerate() {
                    io[crate::gpu::recomp::IO_THREAD_LIMIT[k]] = l;
                }
            }
            let io_slice = std::slice::from_raw_parts(io.as_ptr().cast::<u8>(), std::mem::size_of_val(&io));
            let va = match io_dev.as_ref().and_then(|dev| alloc_vram_io_in(&mut s.io_blocks, s.allocator, s.prelude_built.is_some(), dev, io_slice)) {
                Some(va) => va,
                None => {
                    let Some(io_p) = s.arena.take(u, io_slice.len().max(16), 256, &mut s.uma) else {
                        if !gfx {
                            s.limits = limits;
                        }
                        return Ok(None);
                    };
                    std::ptr::copy_nonoverlapping(io_slice.as_ptr(), io_p, io_slice.len());
                    uma_va_in(u, windows, io_p as usize).unwrap_or(first)
                }
            };
            if can_cache {
                if gfx {
                    s.io_cache_gfx = Some((s.args_gen_gfx, va));
                } else {
                    s.io_cache_cs = Some((s.args_gen_cs, user_sgprs, va));
                }
            }
            va
        };
        if s.dummy_res_va == 0 {
            static ZEROS: [u8; 256] = [0u8; 256];
            s.dummy_res_va = match io_dev.as_ref().and_then(|dev| alloc_vram_io_in(&mut s.io_blocks, s.allocator, s.prelude_built.is_some(), dev, &ZEROS)) {
                Some(va) => va,
                None => {
                    let Some(p) = s.arena.take(u, 256, 256, &mut s.uma) else {
                        return Ok(None);
                    };
                    std::ptr::write_bytes(p, 0, 256);
                    uma_va_in(u, windows, p as usize).unwrap_or(first)
                }
            };
        }
        let views = [
            io_va,
            s.dummy_res_va,
            windows.first().copied().filter(|&v| v != 0).unwrap_or(first),
            windows.get(1).copied().filter(|&v| v != 0).unwrap_or(first),
            windows.get(2).copied().filter(|&v| v != 0).unwrap_or(first),
            windows.get(3).copied().filter(|&v| v != 0).unwrap_or(first),
        ];
        let bound = if gfx { &mut s.pc_abi_gfx } else { &mut s.pc_abi_cs };
        let (set_rs, uav_mask) = bound.diff_and_update(rs_raw, views);
        Ok(Some((set_rs, uav_mask, views)))
    })?;

    let (set_rs, uav_mask, views) = match try_fast {
        Some(v) => v,
        None => {
            // Coherent UMA full: allocate via sparse fallback outside the shard lock.
            let windows = with(this, |s| s.uma_windows.clone().unwrap());
            let first = windows[0];
            let copy = |bytes: &[u8]| -> Result<usize, String> {
                let p = arena_take(this, u, bytes.len().max(16), 256).ok_or("unified memory is full")?;
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), p, bytes.len());
                Ok(p as usize)
            };
            let io_va = if let Some(v) = preset_io_va {
                v
            } else {
                let (layout, args, heaps, limits) = with(this, |s| {
                    let (layout, args) = if gfx { (&s.layout_gfx, &s.args_gfx) } else { (&s.layout, &s.args) };
                    let limits = if gfx { [0; 3] } else { std::mem::take(&mut s.limits) };
                    (layout.clone(), args.clone(), s.heaps, limits)
                });
                let inline = layout.as_ref().map_or(0, |l| if gfx { l.inline_gfx } else { console_list::inline_for(l, user_sgprs) } as usize).min(args.len());
                let table_at = copy(&args[inline..])?;
                let mut io = [0u32; crate::gpu::recomp::KERNEL_SGPRS];
                match layout.as_ref() {
                    Some(l) => console_list::fill_user_data(l, gfx, user_sgprs, &args, table_at as u64, heaps, &mut io),
                    None => {
                        io[0] = table_at as u32;
                        io[1] = (table_at >> 32) as u32;
                    }
                }
                io[crate::gpu::recomp::IO_UMA_BASE_HI] = (u.base() >> 32) as u32;
                io[crate::gpu::recomp::IO_GROUP_LIMIT] = group_limit;
                if !gfx {
                    for (k, l) in limits.into_iter().enumerate() {
                        io[crate::gpu::recomp::IO_THREAD_LIMIT[k]] = l;
                    }
                }
                let io_slice = std::slice::from_raw_parts(io.as_ptr().cast::<u8>(), std::mem::size_of_val(&io));
                uma_va_in(u, &windows, copy(io_slice)?).unwrap_or(first)
            };
            let res_va = match with(this, |s| s.dummy_res_va) {
                0 => {
                    let va = uma_va_in(u, &windows, copy(&[0u8; 256])?).unwrap_or(first);
                    with(this, |s| s.dummy_res_va = va);
                    va
                }
                va => va,
            };
            let views = [
                io_va,
                res_va,
                windows.first().copied().filter(|&v| v != 0).unwrap_or(first),
                windows.get(1).copied().filter(|&v| v != 0).unwrap_or(first),
                windows.get(2).copied().filter(|&v| v != 0).unwrap_or(first),
                windows.get(3).copied().filter(|&v| v != 0).unwrap_or(first),
            ];
            let (set_rs, uav_mask) = with(this, |s| {
                let bound = if gfx { &mut s.pc_abi_gfx } else { &mut s.pc_abi_cs };
                bound.diff_and_update(rs_raw, views)
            });
            (set_rs, uav_mask, views)
        }
    };
    emit_abi_bindings(list, gfx, rs, set_rs, uav_mask, &views);
    list.SetPipelineState(pc);
    Ok(())
}

#[cfg(test)]
pub(super) fn last_draw_args() -> (bool, u32, u32, u32, i32) {
    DRAW_ARGS.with(|d| d.get())
}

#[cfg(test)]
pub(super) fn counts_snapshot() -> [u64; 5] {
    std::array::from_fn(|k| COUNTS[k].load(Ordering::Relaxed))
}

/// The pipeline `list` has bound (as the title made it; 0: none).
pub(super) fn tracked_pso(list: *mut c_void) -> usize {
    state_shard(list as usize)
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|m| m.get(&(list as usize)))
        .map_or(0, |s| s.pso)
}

/// Tracked console `(graphics_root_signature, compute_root_signature)` on `list` for restoring
/// PC root signatures after an internal graphics/compute sync pass.
pub(super) fn tracked_root_sigs(list: *mut c_void) -> (usize, usize) {
    state_shard(list as usize)
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|m| m.get(&(list as usize)))
        .map_or((0, 0), |s| (s.rs_gfx, s.rs_cs))
}

#[cfg(test)]
mod tests {
    use super::super::{device_x, list_x, pso_async, real, title_list};
    use super::*;
    use std::sync::atomic::AtomicUsize;

    /// A pipeline with the rect list GS draws its rect lists whole: draw_rects leaves them alone.
    #[test]
    fn rect_lists_with_the_rect_list_gs_are_not_expanded_on_the_cpu() {
        let (with_gs, without) = (0x7E57_4000usize, 0x7E57_5000usize);
        real::register_rect_list_gs_for_test(with_gs);
        assert!(!cpu_rect_expansion(with_gs));
        assert!(cpu_rect_expansion(without) && cpu_rect_expansion(0));
        real::forget_pso_for_test(with_gs);
        assert!(cpu_rect_expansion(with_gs));
    }

    /// An ExecuteIndirect whose bound pipeline is still being built is skipped (its stand-in
    /// with XCL's ABI root signature on the list is the debug layer's 953 and a hung GPU);
    /// a built or failed one goes to the PC call. The gate `execute_indirect` uses.
    #[test]
    fn execute_indirect_with_a_pipeline_still_building_is_skipped() {
        let (building, built, failed) = (0x7E57_6001usize, 0x7E57_6002usize, 0x7E57_6003usize);
        // lazy submits: no worker starts them, so the states stay under the test's control
        pso_async::submit_lazy(building, false, Box::new(|| 0x1234), 0);
        pso_async::submit_lazy(built, false, Box::new(|| 0x1235), 0);
        pso_async::submit_lazy(failed, false, Box::new(|| 0), 0);
        assert!(pso_async::is_pending(building));
        assert_eq!(pso_async::wait_built(built), 0x1235);
        assert_eq!(pso_async::wait_built(failed), failed);
        assert!(!pso_async::is_pending(built) && !pso_async::is_pending(failed));
        // never resolved: still pending (the skip case), then forgotten
        assert!(pso_async::is_pending(building));
        pso_async::released(building);
        pso_async::released(built);
        pso_async::released(failed);
        assert!(!pso_async::is_pending(building));
    }

    /// A dispatch writes an ExecuteIndirect's group count through another buffer over the same
    /// memory (as translated shaders write through the memory windows), a UAV barrier (what a
    /// console barrier becomes), then the ExecuteIndirect: it runs the count just written, not
    /// the one the memory held before the list (`fresh_indirect_arguments`; without it an RTX
    /// 5090 ran the old count in nearly every try).
    #[test]
    fn execute_indirect_reads_the_arguments_written_before_it() {
        use crate::gpu::{fence, recomp::runner::skip_test};
        use std::mem::{transmute_copy, ManuallyDrop};
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
        use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory2, IDXGIAdapter1, IDXGIFactory6, DXGI_CREATE_FACTORY_FLAGS};

        const TRIES: u32 = 40;
        // mode 0: write (x, 1, 1) at a[0..12]; mode 1: count the groups
        let src = r#"
#define RS "RootConstants(num32BitConstants=2, b0), UAV(u0), UAV(u1)"
cbuffer C : register(b0) { uint mode; uint x; };
RWByteAddressBuffer a : register(u0);
RWByteAddressBuffer c : register(u1);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint t : SV_GroupIndex) {
    if (t != 0u) return;
    if (mode == 0u) a.Store3(0, uint3(x, 1u, 1u)); else c.InterlockedAdd(0, 1u);
}
"#;
        if crate::gpu::recomp::dxc::location().is_err() {
            skip_test("no DXC");
            return;
        }
        unsafe {
            let setup = || -> Result<(ID3D12Device, String), String> {
                let factory: IDXGIFactory6 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).map_err(|e| format!("DXGI factory unavailable: {e}"))?;
                let sel = crate::gpu::adapter::Selector::from_env_or("auto");
                let (adapter, name): (IDXGIAdapter1, String) = if sel == crate::gpu::adapter::Selector::Warp {
                    (factory.EnumWarpAdapter().map_err(|e| format!("WARP unavailable: {e}"))?, "WARP (software renderer)".into())
                } else {
                    let (adapter, desc) = crate::gpu::adapter::candidates(&factory, &sel).into_iter().next()
                        .ok_or_else(|| crate::gpu::adapter::no_match(&factory, &sel))?;
                    (adapter, crate::gpu::adapter::name(&desc))
                };
                let dev = crate::gpu::agility::create_device(&adapter, D3D_FEATURE_LEVEL_11_0).map_err(|e| format!("{name}: D3D12 device unavailable ({e})"))?;
                let mut sm = D3D12_FEATURE_DATA_SHADER_MODEL { HighestShaderModel: D3D_SHADER_MODEL_6_6 };
                dev.CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &mut sm as *mut _ as _, std::mem::size_of_val(&sm) as u32)
                    .map_err(|e| format!("{name}: cannot query SM 6.6 ({e})"))?;
                if sm.HighestShaderModel.0 < D3D_SHADER_MODEL_6_6.0 {
                    return Err(crate::gpu::agility::format_no_sm66_message(&name, crate::gpu::agility::state()));
                }
                Ok((dev, name))
            };
            let (dev, name) = match setup() {
                Ok(selected) => selected,
                Err(reason) => { skip_test(&reason); return; }
            };
            println!("ExecuteIndirect adapter: {name}");
            let dxil = crate::gpu::recomp::dxc::compile(src, "main", "cs_6_6", &[]).unwrap();
            let rs: ID3D12RootSignature = dev.CreateRootSignature(0, &dxil).unwrap();
            let cs = D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as _, BytecodeLength: dxil.len() };
            let pso: ID3D12PipelineState = dev.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC { pRootSignature: transmute_copy(&rs), CS: cs, ..Default::default() }).unwrap();
            let arg = D3D12_INDIRECT_ARGUMENT_DESC { Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH, ..Default::default() };
            let mut sig: Option<ID3D12CommandSignature> = None;
            dev.CreateCommandSignature(&D3D12_COMMAND_SIGNATURE_DESC { ByteStride: 12, NumArgumentDescs: 1, pArgumentDescs: &arg, NodeMask: 0 }, None, &mut sig).unwrap();
            let sig = sig.unwrap();
            let ty = D3D12_COMMAND_LIST_TYPE_COMPUTE;
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: ty, ..Default::default() }).unwrap();
            let done: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let desc = |width| D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: width,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            // the arguments and the buffer the dispatch writes them through: one memory
            let mut heap: Option<ID3D12Heap> = None;
            let heap_desc = D3D12_HEAP_DESC { SizeInBytes: 1 << 16, Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, Alignment: 0, Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS };
            dev.CreateHeap(&heap_desc, &mut heap).unwrap();
            let heap = heap.unwrap();
            let placed = || {
                let mut r: Option<ID3D12Resource> = None;
                dev.CreatePlacedResource(&heap, 0, &desc(1 << 16), D3D12_RESOURCE_STATE_COMMON, None, &mut r).unwrap();
                r.unwrap()
            };
            let (args, written) = (placed(), placed());
            // the group count, read on the CPU
            let cpu_heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_CUSTOM, CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, MemoryPoolPreference: D3D12_MEMORY_POOL_L0, ..Default::default() };
            let mut groups: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&cpu_heap, D3D12_HEAP_FLAG_NONE, &desc(256), D3D12_RESOURCE_STATE_COMMON, None, &mut groups).unwrap();
            let groups = groups.unwrap();
            let mut count = std::ptr::null_mut();
            groups.Map(0, None, Some(&mut count)).unwrap();
            let count = count as *mut u32;
            let uav_barrier = D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 { UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: ManuallyDrop::new(None) }) },
            };
            let run = |list: &ID3D12GraphicsCommandList, close: &dyn Fn()| {
                close();
                queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
                let v = done.GetCompletedValue() + 1;
                queue.Signal(&done, v).unwrap();
                fence::wait(&done, v, &dev, None, 10_000).unwrap();
            };
            let (mut old, mut wrong) = (0u32, Vec::new());
            for k in 0..TRIES {
                // each try writes a count the memory does not hold yet
                let new = if k % 2 == 0 { 4096 } else { 0 };
                let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
                let record = |x: u32| -> ID3D12GraphicsCommandList {
                    let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, ty, &alloc, &pso).unwrap();
                    list.SetComputeRootSignature(&rs);
                    list.SetComputeRootUnorderedAccessView(1, written.GetGPUVirtualAddress());
                    list.SetComputeRootUnorderedAccessView(2, groups.GetGPUVirtualAddress());
                    list.SetComputeRoot32BitConstants(0, 2, [0u32, x].as_ptr() as _, 0);
                    list.Dispatch(1, 1, 1);
                    list
                };
                let before = record(old);
                run(&before, &|| before.Close().unwrap());
                alloc.Reset().unwrap();
                *count = 0;
                let list = record(new);
                list.ResourceBarrier(&[uav_barrier.clone()]);
                list.SetComputeRoot32BitConstants(0, 2, [1u32, 0].as_ptr() as _, 0);
                // from here as XCL records a title's ExecuteIndirect: on the adopted list
                let raw = list.as_raw();
                real::adopt(raw, real::Kind::CommandList);
                list_x::on_create(raw, alloc.as_raw() as usize, 0, ty);
                type Ei = unsafe extern "system" fn(*mut c_void, usize, u32, usize, u64, usize, u64);
                type Close = unsafe extern "system" fn(*mut c_void) -> windows::core::HRESULT;
                let ei: Ei = std::mem::transmute(real::orig_fn(raw, 59));
                fresh_indirect_arguments(raw, || ei(raw, sig.as_raw() as usize, 1, args.as_raw() as usize, 0, 0, 0));
                let close: Close = std::mem::transmute(real::orig_fn(raw, 9));
                run(&list, &|| close(raw).unwrap());
                let ran = std::ptr::read_volatile(count);
                if ran != new {
                    wrong.push((k, ran));
                }
                old = new;
            }
            groups.Unmap(0, None);
            assert!(wrong.is_empty(), "{} of {TRIES} ExecuteIndirects ran another group count than the one written before them (try, groups): {wrong:?}", wrong.len());
        }
    }

    /// Minimal RTS0 v2 root signature blob with one 4-constant parameter.
    fn test_rts0_blob() -> Vec<u8> {
        let words: [u32; 12] = [
            2, 1, 24, 0, 0, 0, // version=2, num_params=1, params_offset=24
            1, 0, 36,          // param 0: type=Constants(1), visibility=0, payload_offset=36
            0, 0, 4,           // shader_reg=0, reg_space=0, num_32bit_values=4
        ];
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    #[test]
    fn reset_and_clear_state_track_initial_pso_and_clear_root_args() {
        let list = 0x7100_0001usize as *mut c_void;
        let rs_compute = 0x7100_0010usize;
        let rs_gfx = 0x7100_0020usize;
        let alloc1 = 0x7100_0100usize;
        let alloc2 = 0x7100_0200usize;
        let pso_init = 0x7100_1000usize;
        let pso_mid = 0x7100_2000usize;
        let pso_cleared = 0x7100_3000usize;
        let pso_reset = 0x7100_4000usize;

        let blob = test_rts0_blob();
        console_list::note_root_signature(rs_compute, &blob);
        console_list::note_root_signature(rs_gfx, &blob);

        // Initial CreateCommandList / reset seeds PSO and allocator with clean root state.
        reset(list, alloc1, pso_init);
        with(list, |s| {
            assert_eq!(s.pso, pso_init);
            assert_eq!(s.allocator, alloc1);
            assert!(s.layout.is_none() && s.args.is_empty());
            assert!(s.layout_gfx.is_none() && s.args_gfx.is_empty());
        });

        // Populate compute + graphics root signatures, constants, and a different PSO.
        unsafe {
            set_compute_root_signature(list, rs_compute as *mut c_void);
            set_compute_root_constant(list, 0, 0xDEAD_BEEF, 0);
            set_graphics_root_signature(list, rs_gfx as *mut c_void);
            set_graphics_root_constant(list, 0, 0xCAFE_BABE, 0);
        }
        with(list, |s| {
            s.pso = pso_mid;
            assert!(s.layout.is_some() && !s.args.is_empty());
            assert!(s.layout_gfx.is_some() && !s.args_gfx.is_empty());
        });

        // ClearState(pso_cleared) updates PSO and clears root signatures/args while keeping allocator.
        clear_state(list, pso_cleared);
        with(list, |s| {
            assert_eq!(s.pso, pso_cleared);
            assert_eq!(s.allocator, alloc1);
            assert!(s.layout.is_none() && s.args.is_empty());
            assert!(s.layout_gfx.is_none() && s.args_gfx.is_empty());
        });

        // ClearState(null) clears PSO to 0.
        clear_state(list, 0);
        assert_eq!(tracked_pso(list), 0);

        // Re-populate root args, then Reset(alloc2, pso_reset) switches allocator/PSO and clears root args.
        unsafe {
            set_compute_root_signature(list, rs_compute as *mut c_void);
            set_compute_root_constant(list, 0, 0x1234_5678, 0);
            set_graphics_root_signature(list, rs_gfx as *mut c_void);
            set_graphics_root_constant(list, 0, 0x8765_4321, 0);
        }
        reset(list, alloc2, pso_reset);
        with(list, |s| {
            assert_eq!(s.pso, pso_reset);
            assert_eq!(s.allocator, alloc2);
            assert!(s.layout.is_none() && s.args.is_empty());
            assert!(s.layout_gfx.is_none() && s.args_gfx.is_empty());
        });

        // Reset with null initial PSO clears stale PSO back to 0.
        reset(list, alloc2, 0);
        assert_eq!(tracked_pso(list), 0);

        forget(list);
    }

    #[test]
    fn the_same_root_signature_again_keeps_the_root_arguments() {
        let list = 0x7100_0301usize as *mut c_void;
        let (rs_a, rs_b) = (0x7100_0310usize, 0x7100_0320usize);
        console_list::note_root_signature(rs_a, &test_rts0_blob());
        console_list::note_root_signature(rs_b, &test_rts0_blob());
        let set = |gfx: bool, rs: usize| unsafe {
            if gfx { set_graphics_root_signature(list, rs as *mut c_void) } else { set_compute_root_signature(list, rs as *mut c_void) }
        };
        let args = |gfx: bool| with(list, |s| if gfx { s.args_gfx.clone() } else { s.args.clone() });
        reset(list, 0x7100_0330, 0);
        set(false, rs_a);
        set(true, rs_a);
        unsafe { set_compute_root_constant(list, 0, 0xABCD, 0) };
        assert!(args(false).iter().any(|b| *b != 0));
        assert!(args(true).iter().all(|b| *b == 0), "compute and graphics arguments are separate");
        // the same signature again: the arguments stay
        set(false, rs_a);
        assert_eq!(&args(false)[..4], &0xABCDu32.to_le_bytes());
        // graphics is independent: a different one there leaves compute alone
        unsafe { set_graphics_root_constant(list, 0, 0x1234, 0) };
        set(true, rs_b);
        assert!(args(true).iter().all(|b| *b == 0));
        assert_eq!(&args(false)[..4], &0xABCDu32.to_le_bytes());
        // a different signature clears
        set(false, rs_b);
        assert!(args(false).iter().all(|b| *b == 0));
        // a signature re-made at the same address is a different one
        unsafe { set_compute_root_constant(list, 0, 0xABCD, 0) };
        console_list::note_root_signature(rs_b, &test_rts0_blob());
        set(false, rs_b);
        assert!(args(false).iter().all(|b| *b == 0));
        // Reset and ClearState clear, so the signature set again afterwards starts empty
        unsafe { set_compute_root_constant(list, 0, 0xABCD, 0) };
        clear_state(list, 0);
        set(false, rs_b);
        assert!(args(false).iter().all(|b| *b == 0));
        unsafe { set_compute_root_constant(list, 0, 0xABCD, 0) };
        reset(list, 0x7100_0330, 0);
        set(false, rs_b);
        assert!(args(false).iter().all(|b| *b == 0));
        forget(list);
    }

    #[repr(C)]
    struct FakeAdoptedList {
        vt: *const usize,
        refs: AtomicUsize,
        dispatch_calls: AtomicUsize,
        set_compute_rs_calls: AtomicUsize,
        set_compute_uav_calls: AtomicUsize,
        set_pso_calls: AtomicUsize,
        execute_indirect_calls: AtomicUsize,
    }

    unsafe extern "system" fn fake_add_ref(this: *mut c_void) -> u32 {
        (*(this as *const FakeAdoptedList)).refs.fetch_add(1, Ordering::Relaxed) as u32 + 1
    }

    unsafe extern "system" fn fake_release(this: *mut c_void) -> u32 {
        (*(this as *const FakeAdoptedList)).refs.fetch_sub(1, Ordering::Relaxed) as u32 - 1
    }

    unsafe extern "system" fn fake_dispatch(this: *mut c_void, _x: u32, _y: u32, _z: u32) {
        (*(this as *const FakeAdoptedList)).dispatch_calls.fetch_add(1, Ordering::Relaxed);
    }

    unsafe extern "system" fn fake_set_pso(this: *mut c_void, _pso: *mut c_void) {
        (*(this as *const FakeAdoptedList)).set_pso_calls.fetch_add(1, Ordering::Relaxed);
    }

    unsafe extern "system" fn fake_set_compute_rs(this: *mut c_void, _rs: *mut c_void) {
        (*(this as *const FakeAdoptedList)).set_compute_rs_calls.fetch_add(1, Ordering::Relaxed);
    }

    unsafe extern "system" fn fake_set_compute_uav(this: *mut c_void, _slot: u32, _va: u64) {
        (*(this as *const FakeAdoptedList)).set_compute_uav_calls.fetch_add(1, Ordering::Relaxed);
    }

    unsafe extern "system" fn fake_execute_indirect(
        this: *mut c_void,
        _sig: usize,
        _max: u32,
        _args: usize,
        _args_off: u64,
        _count_buf: usize,
        _count_off: u64,
    ) {
        (*(this as *const FakeAdoptedList)).execute_indirect_calls.fetch_add(1, Ordering::Relaxed);
    }

    /// `DispatchX` with thread dimensions (flag 0x20, as the BVH builder's rehydration passes
    /// use it): the console launches whole groups of the pipeline's `numthreads` and leaves
    /// the lanes past the count out; without the flag the counts are groups. With the flag
    /// and a group shape not known, nothing is launched (the counts as groups was the bug).
    #[test]
    fn dispatch_x_with_thread_dimensions_launches_whole_groups() {
        let pso = 0x7300_0001usize;
        real::register_threads_for_test(pso, [64, 1, 1]);
        // 640 threads in groups of 64: 10 groups, the last one whole
        assert_eq!(dispatch_groups(pso, [640, 1, 1], 0x20), Some(([10, 1, 1], [0, 0, 0])));
        // 100 threads: 2 groups, and the lanes past 100 to leave out
        assert_eq!(dispatch_groups(pso, [100, 1, 1], 0x20), Some(([2, 1, 1], [100, 0, 0])));
        // the wave32 bit (0x8000) changes nothing here
        assert_eq!(dispatch_groups(pso, [100, 1, 1], 0x8020), Some(([2, 1, 1], [100, 0, 0])));
        // a 2-D group shape
        let pso2 = 0x7300_0002usize;
        real::register_threads_for_test(pso2, [8, 8, 1]);
        assert_eq!(dispatch_groups(pso2, [20, 16, 3], 0x20), Some(([3, 2, 3], [20, 0, 0])));
        // group counts (Dispatch, DispatchX without the flag) are launched as they are
        assert_eq!(dispatch_groups(pso, [7, 2, 1], 0), Some(([7, 2, 1], [0, 0, 0])));
        assert_eq!(dispatch_groups(pso, [7, 2, 1], 0x8000), Some(([7, 2, 1], [0, 0, 0])));
        // no threads: no groups
        assert_eq!(dispatch_groups(pso, [0, 1, 1], 0x20), Some(([0, 1, 1], [0, 0, 0])));
        // thread dimensions on a pipeline whose group shape is not known: not launched
        assert_eq!(dispatch_groups(0x7300_00FF, [100, 1, 1], 0x20), None);
        assert_eq!(dispatch_groups(0x7300_00FF, [100, 1, 1], 0), Some(([100, 1, 1], [0, 0, 0])));
    }

    #[test]
    fn async_compute_pso_success_and_failure_in_dispatch_and_execute_indirect() {
        let mut orig_vt = [0usize; 64];
        orig_vt[1] = fake_add_ref as usize;
        orig_vt[2] = fake_release as usize;
        orig_vt[14] = fake_dispatch as usize;
        orig_vt[25] = fake_set_pso as usize;
        orig_vt[29] = fake_set_compute_rs as usize;
        orig_vt[41] = fake_set_compute_uav as usize;
        orig_vt[59] = fake_execute_indirect as usize;

        // Extended vtable layout expected by `real::orig_fn`: entry -1 holds `orig_vt` pointer,
        // and entries 0..64 hold the COM method pointers used via `ID3D12GraphicsCommandList`.
        let mut ext_vt = vec![0usize; 65];
        ext_vt[0] = orig_vt.as_ptr() as usize;
        ext_vt[1..].copy_from_slice(&orig_vt);

        let fake = Box::new(FakeAdoptedList {
            // SAFETY: `ext_vt` has 65 elements so `.add(1)` points to slot 0 with `orig_vt` at `-1`.
            vt: unsafe { ext_vt.as_ptr().add(1) },
            refs: AtomicUsize::new(1),
            dispatch_calls: AtomicUsize::new(0),
            set_compute_rs_calls: AtomicUsize::new(0),
            set_compute_uav_calls: AtomicUsize::new(0),
            set_pso_calls: AtomicUsize::new(0),
            execute_indirect_calls: AtomicUsize::new(0),
        });
        let list_ptr = &*fake as *const FakeAdoptedList as *mut c_void;

        // 1. Background compute build SUCCEEDS: real PSO R is registered in TRANSLATED_COMPUTE,
        // stand-in S is NOT. ExecuteIndirect resolves S -> R and treats it as translated.
        let standin_ok = 0x7200_0001usize;
        let real_ok = 0x7200_0002usize;
        let sig_root_changing = 0x7200_00F0usize;
        note_root_changing_signature(sig_root_changing);

        pso_async::submit(
            standin_ok,
            true,
            Box::new(move || {
                real::register_translated_compute_for_test(real_ok);
                real_ok
            }),
        );
        assert_eq!(pso_async::wait_built(standin_ok), real_ok);
        assert!(!real::is_translated_compute(standin_ok));
        assert!(real::is_translated_compute(real_ok));

        reset(list_ptr, 0x7200_0100, standin_ok);
        unsafe {
            // With `sig_root_changing`, a translated compute PSO is caught by the root-changing
            // guard instead of falling through to `pc` (which happened when `R` wasn't registered).
            execute_indirect(list_ptr, sig_root_changing, 1, 0, 0, 0, 0);
        }
        assert_eq!(
            fake.execute_indirect_calls.load(Ordering::Relaxed),
            0,
            "ExecuteIndirect must recognize resolved async compute PSO R as translated"
        );

        // 2. Background compute build FAILS (returns 0): stand-in S_fail is NOT in
        // TRANSLATED_COMPUTE. Dispatch must skip cleanly without binding XCL's ABI RS/UAVs
        // onto S_fail, and ExecuteIndirect must not enter bind_abi either.
        let standin_fail = 0x7200_0003usize;
        pso_async::submit(standin_fail, true, Box::new(|| 0));
        assert_eq!(pso_async::wait_built(standin_fail), standin_fail);
        assert!(!real::is_translated_compute(standin_fail));

        reset(list_ptr, 0x7200_0100, standin_fail);
        let before = counts_snapshot();
        unsafe {
            dispatch(list_ptr, 1, 1, 1);
        }
        let after = counts_snapshot();
        assert_eq!(after[DISPATCHES], before[DISPATCHES]);
        assert!(after[DISPATCHES_SKIPPED] > before[DISPATCHES_SKIPPED]);
        assert_eq!(fake.set_compute_rs_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.set_compute_uav_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.set_pso_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.dispatch_calls.load(Ordering::Relaxed), 0);

        unsafe {
            execute_indirect(list_ptr, sig_root_changing, 1, 0, 0, 0, 0);
        }
        assert_eq!(fake.set_compute_rs_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.set_compute_uav_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.set_pso_calls.load(Ordering::Relaxed), 0);
        assert_eq!(
            fake.execute_indirect_calls.load(Ordering::Relaxed),
            1,
            "failed async compute stand-in must not enter translated ExecuteIndirect path"
        );

        // 3. The same failed stand-in (made from a console shader) with a Dispatch signature:
        // nothing to run, as for Dispatch – the PC call would pair the stand-in with the list's
        // compute root signature (XCL's ABI one after a translated dispatch: debug layer 953)
        let sig_dispatch = 0x7200_00F8usize;
        real::register_console_compute_for_test(standin_fail);
        note_signature_dispatches(sig_dispatch, true);
        let before = counts_snapshot();
        unsafe {
            execute_indirect(list_ptr, sig_dispatch, 1, 0, 0, 0, 0);
        }
        let after = counts_snapshot();
        assert_eq!(fake.execute_indirect_calls.load(Ordering::Relaxed), 1, "a dispatch of an untranslated console compute pipeline must not reach the PC ExecuteIndirect");
        assert!(after[DISPATCHES_SKIPPED] > before[DISPATCHES_SKIPPED]);

        // 4. A native PC compute pipeline (not made from a console shader) with the same Dispatch
        // signature still goes to the PC call, as on main
        let native = 0x7200_0004usize;
        reset(list_ptr, 0x7200_0100, native);
        unsafe {
            execute_indirect(list_ptr, sig_dispatch, 1, 0, 0, 0, 0);
        }
        assert_eq!(fake.execute_indirect_calls.load(Ordering::Relaxed), 2, "a native PC compute pipeline must still reach the PC ExecuteIndirect");

        // a signature made later at the same address without a Dispatch argument goes to the PC
        reset(list_ptr, 0x7200_0100, standin_fail);
        note_signature_dispatches(sig_dispatch, false);
        unsafe {
            execute_indirect(list_ptr, sig_dispatch, 1, 0, 0, 0, 0);
        }
        assert_eq!(fake.execute_indirect_calls.load(Ordering::Relaxed), 3);

        forget(list_ptr);
        let _ = pso_async::released(standin_ok);
        let _ = pso_async::released(standin_fail);
        real::forget_pso_for_test(real_ok);
        real::forget_pso_for_test(standin_fail);
        assert!(!real::is_console_compute(standin_fail), "a released pipeline's address does not stay a console one");
    }

    #[test]
    fn execute_indirect_skips_failed_shared_console_compute_standin() {
        let mut orig_vt = [0usize; 64];
        orig_vt[1] = fake_add_ref as usize;
        orig_vt[2] = fake_release as usize;
        orig_vt[14] = fake_dispatch as usize;
        orig_vt[25] = fake_set_pso as usize;
        orig_vt[29] = fake_set_compute_rs as usize;
        orig_vt[41] = fake_set_compute_uav as usize;
        orig_vt[59] = fake_execute_indirect as usize;

        let mut ext_vt = vec![0usize; 65];
        ext_vt[0] = orig_vt.as_ptr() as usize;
        ext_vt[1..].copy_from_slice(&orig_vt);

        let fake = Box::new(FakeAdoptedList {
            vt: unsafe { ext_vt.as_ptr().add(1) },
            refs: AtomicUsize::new(1),
            dispatch_calls: AtomicUsize::new(0),
            set_compute_rs_calls: AtomicUsize::new(0),
            set_compute_uav_calls: AtomicUsize::new(0),
            set_pso_calls: AtomicUsize::new(0),
            execute_indirect_calls: AtomicUsize::new(0),
        });
        let list_ptr = &*fake as *const FakeAdoptedList as *mut c_void;

        // With shared stand-ins (`standin::shell`), `dev_create_compute_pso` registers the
        // title-facing shell in `CONSOLE_COMPUTE`, while `pso_async::wait_built` returns the
        // shared driver pipeline when the background build fails.
        let shell_fail = 0x7210_0001usize;
        let shell_ok = 0x7210_0002usize;
        let shared_driver = 0x7210_0003usize;
        let real_ok = 0x7210_0004usize;
        let sig_dispatch = 0x7210_00F8usize;
        let sig_root_changing = 0x7210_00F0usize;
        note_signature_dispatches(sig_dispatch, true);
        note_root_changing_signature(sig_root_changing);

        pso_async::submit_lazy(shell_fail, true, Box::new(|| 0), 0);
        pso_async::shared(shell_fail, shared_driver);
        real::register_console_compute_for_test(shell_fail);

        pso_async::submit_lazy(
            shell_ok,
            true,
            Box::new(move || {
                real::register_translated_compute_for_test(real_ok);
                real_ok
            }),
            0,
        );
        pso_async::shared(shell_ok, shared_driver);
        real::register_console_compute_for_test(shell_ok);

        assert_eq!(pso_async::wait_built(shell_fail), shared_driver);
        assert!(!pso_async::is_pending(shell_fail));
        assert!(real::is_console_compute(shell_fail));
        assert!(!real::is_console_compute(shared_driver));

        // 1. Failed shared console compute stand-in with a Dispatch signature must be skipped,
        // never forwarded to PC ExecuteIndirect with the shared driver pipeline.
        reset(list_ptr, 0x7210_0100, shell_fail);
        let before = counts_snapshot();
        unsafe {
            execute_indirect(list_ptr, sig_dispatch, 1, 0, 0, 0, 0);
        }
        let after = counts_snapshot();
        assert_eq!(
            fake.execute_indirect_calls.load(Ordering::Relaxed),
            0,
            "failed shared console compute stand-in must not reach PC ExecuteIndirect"
        );
        assert!(after[DISPATCHES_SKIPPED] > before[DISPATCHES_SKIPPED]);

        // 2. Another shell sharing the same driver pipeline whose build succeeds still takes the
        // translated path (caught here by the root-changing guard rather than falling through to PC).
        // (built first: a bind does not build it, `resolve_wait`)
        assert_eq!(pso_async::wait_built(shell_ok), real_ok);
        reset(list_ptr, 0x7210_0100, shell_ok);
        unsafe {
            execute_indirect(list_ptr, sig_root_changing, 1, 0, 0, 0, 0);
        }
        assert_eq!(fake.execute_indirect_calls.load(Ordering::Relaxed), 0);

        // 3. Releasing `shell_fail` forgets its `CONSOLE_COMPUTE` entry so a native PC compute
        // pipeline later allocated at the same address still reaches PC ExecuteIndirect.
        let _ = pso_async::released(shell_fail);
        real::forget_pso_for_test(shell_fail);
        assert!(!real::is_console_compute(shell_fail));
        reset(list_ptr, 0x7210_0100, shell_fail);
        unsafe {
            execute_indirect(list_ptr, sig_dispatch, 1, 0, 0, 0, 0);
        }
        assert_eq!(
            fake.execute_indirect_calls.load(Ordering::Relaxed),
            1,
            "native PC pipeline at a released shell's address must still reach PC ExecuteIndirect"
        );

        forget(list_ptr);
        note_signature_dispatches(sig_dispatch, false);
        let _ = pso_async::released(shell_ok);
        real::forget_pso_for_test(shell_ok);
        real::forget_pso_for_test(real_ok);
    }

    #[test]
    fn two_pass_pso_skips_direct_draw_and_unmatched_execute_indirect() {
        #[repr(C)]
        struct FakeTwoPassList {
            vt: *const usize,
            refs: AtomicUsize,
            set_graphics_rs_calls: AtomicUsize,
            execute_indirect_calls: AtomicUsize,
        }
        unsafe extern "system" fn add_ref(this: *mut c_void) -> u32 {
            (*(this as *const FakeTwoPassList)).refs.fetch_add(1, Ordering::Relaxed) as u32 + 1
        }
        unsafe extern "system" fn release(this: *mut c_void) -> u32 {
            (*(this as *const FakeTwoPassList)).refs.fetch_sub(1, Ordering::Relaxed) as u32 - 1
        }
        unsafe extern "system" fn set_graphics_rs(this: *mut c_void, _rs: *mut c_void) {
            (*(this as *const FakeTwoPassList)).set_graphics_rs_calls.fetch_add(1, Ordering::Relaxed);
        }
        unsafe extern "system" fn exec_indirect(
            this: *mut c_void,
            _sig: usize,
            _max: u32,
            _args: usize,
            _args_off: u64,
            _count_buf: usize,
            _count_off: u64,
        ) {
            (*(this as *const FakeTwoPassList)).execute_indirect_calls.fetch_add(1, Ordering::Relaxed);
        }

        let mut orig_vt = [0usize; 64];
        orig_vt[1] = add_ref as usize;
        orig_vt[2] = release as usize;
        orig_vt[30] = set_graphics_rs as usize;
        orig_vt[59] = exec_indirect as usize;

        let mut ext_vt = vec![0usize; 65];
        ext_vt[0] = orig_vt.as_ptr() as usize;
        ext_vt[1..].copy_from_slice(&orig_vt);

        let fake = Box::new(FakeTwoPassList {
            vt: unsafe { ext_vt.as_ptr().add(1) },
            refs: AtomicUsize::new(1),
            set_graphics_rs_calls: AtomicUsize::new(0),
            execute_indirect_calls: AtomicUsize::new(0),
        });
        let list_ptr = &*fake as *const FakeTwoPassList as *mut c_void;

        // Register a finished two-pass GS/MS PSO behind an async stand-in (registered in TWO_PASS,
        // not TRANSLATED_GRAPHICS).
        let standin_tp = 0x7250_0001usize;
        let real_tp = 0x7250_0002usize;
        pso_async::submit(
            standin_tp,
            false,
            Box::new(move || {
                real::register_two_pass_for_test(real_tp, "GS");
                real_tp
            }),
        );
        assert_eq!(pso_async::wait_built(standin_tp), real_tp);
        assert!(real::two_pass_of(real_tp).is_some());
        assert!(!real::is_translated_graphics(real_tp));

        reset(list_ptr, 0x7250_0100, standin_tp);
        unsafe {
            // Seed a non-null title graphics RS so a fallthrough would visibly call SetGraphicsRootSignature.
            set_graphics_root_signature(list_ptr, list_ptr);
        }

        // 1. Direct draw (`graphics_bound`) must return None without binding the title's RS.
        let before = counts_snapshot();
        let bound = unsafe { graphics_bound(list_ptr) };
        let after = counts_snapshot();
        assert!(bound.is_none(), "two-pass MS/GS PSO must not fall through to PC DrawInstanced");
        assert!(after[DRAWS_SKIPPED] > before[DRAWS_SKIPPED]);
        assert_eq!(after[DRAWS_PC], before[DRAWS_PC]);
        assert_eq!(fake.set_graphics_rs_calls.load(Ordering::Relaxed), 0);

        // 2. Unmatched `ExecuteIndirect` must skip without binding the title's RS or calling PC ExecuteIndirect.
        let before = counts_snapshot();
        unsafe {
            execute_indirect(list_ptr, 0x7250_00F0, 1, 0, 0, 0, 0);
        }
        let after = counts_snapshot();
        assert!(after[DRAWS_SKIPPED] > before[DRAWS_SKIPPED]);
        assert_eq!(fake.set_graphics_rs_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.execute_indirect_calls.load(Ordering::Relaxed), 0);

        forget(list_ptr);
        let _ = pso_async::released(standin_tp);
        real::forget_pso_for_test(real_tp);
        assert!(real::two_pass_of(real_tp).is_none());
    }

    #[test]
    fn create_list_reset_and_clear_state_track_initial_pso_on_d3d12() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;

        if console_list::enabled() {
            return;
        }
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let dev = crate::gpu::agility::create_device(None, D3D_FEATURE_LEVEL_11_0).unwrap_or_else(|_| {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                crate::gpu::agility::create_device(&warp, D3D_FEATURE_LEVEL_11_0).expect("WARP D3D12 device")
            });
            real::adopt(dev.as_raw(), real::Kind::Device);

            // Build two real PC compute PSOs from the inbox dummy CS bytecode.
            let mut ser = None;
            D3D12SerializeRootSignature(&D3D12_ROOT_SIGNATURE_DESC::default(), D3D_ROOT_SIGNATURE_VERSION_1, &mut ser, None).expect("serialize RS");
            let ser = ser.unwrap();
            let rs: ID3D12RootSignature = dev
                .CreateRootSignature(0, std::slice::from_raw_parts(ser.GetBufferPointer() as *const u8, ser.GetBufferSize()))
                .expect("CreateRootSignature");
            let cs = real::dummy_cs_bytecode();
            let make_pso = || -> ID3D12PipelineState {
                let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::ManuallyDrop::new(Some(rs.clone())),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: cs.as_ptr() as *const c_void, BytecodeLength: cs.len() },
                    ..Default::default()
                };
                let pso = dev.CreateComputePipelineState(&desc).expect("CreateComputePipelineState");
                std::mem::ManuallyDrop::drop(&mut desc.pRootSignature);
                pso
            };
            let pso1 = make_pso();
            let pso2 = make_pso();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("alloc");

            // Stand-in synthetic key backed by real `pso1` in pso_async: CreateCommandList must
            // resolve it via resolve_for_bind for PC D3D12 while tracking `standin` in `bind`.
            let standin = 0x7300_0001usize;
            let pso1_raw = pso1.as_raw() as usize;
            pso_async::submit(standin, true, Box::new(move || pso1_raw));
            assert_eq!(pso_async::wait_built(standin), pso1_raw);

            let mut raw_list: *mut c_void = std::ptr::null_mut();
            let iid = windows_sys::core::GUID::from_u128(0x5b160d0f_ac1b_4185_8ba8_b3ae42a5a455);
            let hr = device_x::create_command_list_pc(
                dev.as_raw(),
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT.0 as u32,
                alloc.as_raw() as usize,
                standin,
                &iid,
                &mut raw_list,
            );
            assert!(hr >= 0 && !raw_list.is_null(), "create_command_list_pc failed: {hr:#x}");
            let backing = title_list::backing(raw_list).expect("wrapped in title_list");
            assert_eq!(tracked_pso(backing), standin, "CreateCommandList must seed bind::State.pso");

            // ClearState(pso2) updates bind::State.pso to pso2 and clears root args.
            let rs_key = 0x7300_0010usize;
            console_list::note_root_signature(rs_key, &test_rts0_blob());
            set_compute_root_signature(backing, rs_key as *mut c_void);
            set_compute_root_constant(backing, 0, 0xABCD, 0);
            let cl = ID3D12GraphicsCommandList::from_raw_borrowed(&backing).unwrap();
            cl.ClearState(&pso2);
            assert_eq!(tracked_pso(backing), pso2.as_raw() as usize, "ClearState must update bind::State.pso");
            with(backing, |s| assert!(s.layout.is_none() && s.args.is_empty()));

            // ClearState(None) clears bind::State.pso to 0.
            cl.ClearState(None);
            assert_eq!(tracked_pso(backing), 0, "ClearState(None) must clear bind::State.pso");

            // Close + reset_x(alloc, pso2) updates bind::State.pso to pso2 and clears root args.
            set_graphics_root_signature(backing, rs_key as *mut c_void);
            set_graphics_root_constant(backing, 0, 0x1111, 0);
            cl.Close().expect("Close");
            let hr = list_x::reset_x(backing, alloc.as_raw() as usize, pso2.as_raw() as usize);
            assert!(hr >= 0, "reset_x failed: {hr:#x}");
            assert_eq!(tracked_pso(backing), pso2.as_raw() as usize, "Reset must update bind::State.pso");
            with(backing, |s| assert!(s.layout_gfx.is_none() && s.args_gfx.is_empty()));

            // Release the wrapped title list cleanly (calls bind::forget + list_x::on_release).
            let release_fn: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*(*(raw_list as *const *const usize)).add(2));
            assert_eq!(release_fn(raw_list), 0);
            let _ = pso_async::released(standin);
        }
    }

    #[test]
    fn dispatch_mesh_signature_tracking_returns_companion_and_layout() {
        let sig = 0x7400_0001usize;
        let companion = 0x7400_0002usize;
        assert_eq!(dispatch_mesh_sig_of(sig), None);
        note_dispatch_mesh_signature(sig, companion, 28, 16);
        assert_eq!(
            dispatch_mesh_sig_of(sig),
            Some(DispatchMeshSig { companion_dispatch: companion, byte_stride: 28, mesh_offset: 16 })
        );
        DISPATCH_MESH_SIGS.write().unwrap().as_mut().unwrap().remove(&sig);
        assert_eq!(dispatch_mesh_sig_of(sig), None);
    }

    #[test]
    fn two_pass_io_carries_the_unified_memory_base() {
        use crate::gpu::recomp::{IO_UMA_BASE_HI, KERNEL_SGPRS};
        let base = 0x4_0000_0000usize;
        for compute_pass in [true, false] {
            let io = two_pass_io(None, &[], 0x4_1234_5600, compute_pass, base, [0, 0]);
            assert_eq!(io.len(), KERNEL_SGPRS);
            assert_eq!(io[IO_UMA_BASE_HI], 4, "compute pass {compute_pass}");
        }
        let io = two_pass_io(None, &[], 0x4_1234_5600, true, base, [0, 0]);
        assert_eq!((io[8], io[9]), (0x1234_5600, 4));
        let layout = Layout { params: Vec::new(), size: 0, inline: 0, inline_gfx: 0, inline_ms: 0, heaps: [true, true] };
        let ms_io = two_pass_io(Some(&layout), &[], 0x200, true, base, [0xAA00, 0xBB00]);
        assert_eq!((ms_io[16], ms_io[17]), (0xAA00, 0xBB00));
        let ps_io = two_pass_io(Some(&layout), &[], 0x200, false, base, [0xAA00, 0xBB00]);
        assert_eq!((ps_io[8], ps_io[9]), (0xAA00, 0xBB00));
    }

    #[test]
    fn two_pass_groups_stay_inside_the_record_buffer() {
        // #304's transform record: 11 908 dwords
        let record = 11_908 * 4;
        assert_eq!(record_groups(32 << 20, record, 1), 704);
        assert_eq!(record_groups(128 << 20, record, 1), 2817);
        // the mesh pass's groups, `split` a record, all in X
        assert_eq!(record_groups(1 << 30, 64, 7), 65_535 / 7);
        assert_eq!(record_groups(1 << 20, 2 << 20, 1), 0);
    }

    /// Every root view the argument pass's program declares (u0, u2, u3, u4) gets an address,
    /// also without a count buffer: u4 then names the constants. (A slot left out after the root
    /// signature is set holds nothing, and a load from it is a load at GPU address 0.)
    #[test]
    fn the_argument_pass_names_a_buffer_in_every_view_it_declares() {
        for register in ["u0", "u2", "u3", "u4"] {
            assert!(SCALE_ARGS_HLSL.contains(&format!("register({register})")), "{register}");
        }
        assert!(!SCALE_ARGS_HLSL.contains("register(u1)") && !SCALE_ARGS_HLSL.contains("register(u5)"));
        let with_count = scale_views(0x1000, 0x2000, 0x3000, Some(0x4000));
        assert_eq!(with_count, [(0, 0x1000), (2, 0x2000), (3, 0x3000), (4, 0x4000)]);
        let without = scale_views(0x1000, 0x2000, 0x3000, None);
        assert_eq!(without, [(0, 0x1000), (2, 0x2000), (3, 0x3000), (4, 0x1000)]);
        assert!(without.iter().all(|&(_, va)| va != 0));
    }

    /// `SCALE_ARGS_HLSL` on `commands` (dwords each, the DispatchMesh argument at dword 1):
    /// the compute pass's and the mesh pass's commands.
    fn scale_args(commands: &[[u32; 4]], split: u32, cap: u32) -> Option<(Vec<[u32; 4]>, Vec<[u32; 4]>)> {
        scale_args_counted(commands, split, cap, None)
    }

    /// `scale_args` with a count buffer holding `count`
    fn scale_args_counted(commands: &[[u32; 4]], split: u32, cap: u32, count: Option<u32>) -> Option<(Vec<[u32; 4]>, Vec<[u32; 4]>)> {
        use crate::gpu::recomp::runner;
        if runner::adapter().is_err() {
            runner::skip_test("no GPU");
            return None;
        }
        let dxil = crate::gpu::recomp::dxc::compile(SCALE_ARGS_HLSL, "main", "cs_6_6", &[]).unwrap();
        let n = commands.len();
        let (src, dst) = (0u32, 256u32);
        let mut mem: Vec<u32> = commands.iter().flatten().copied().collect();
        mem.resize(64 + 2 * 4 * n, 0xDEAD);
        // the count at dword 60 (u4 is the same memory in the runner)
        assert!(4 * n <= 60);
        mem[60] = count.unwrap_or(0);
        let io = [n as u32, 16, 4, split, cap, src, dst, count.is_some() as u32, 240];
        let out = runner::dispatch(&dxil, [(n as u32).div_ceil(64), 1, 1], &io, &mem).unwrap();
        let cmd = |k: usize| [out[64 + 4 * k], out[65 + 4 * k], out[66 + 4 * k], out[67 + 4 * k]];
        Some(((0..n).map(cmd).collect(), (n..2 * n).map(cmd).collect()))
    }

    /// Indirect group counts that are not the title's run nothing, the others go out as they are.
    #[test]
    fn gpu_indirect_dispatch_runs_no_garbage_group_counts() {
        use crate::gpu::recomp::runner;
        if runner::adapter().is_err() {
            runner::skip_test("no GPU");
            return;
        }
        let dxil = crate::gpu::recomp::dxc::compile(GUARD_ARGS_HLSL, "main", "cs_6_6", &[]).unwrap();
        let cases: [([u32; 3], [u32; 3]); 10] = [
            ([1, 1, 1], [1, 1, 1]),
            ([2560 / 8, 1440 / 4, 1], [320, 360, 1]),
            ([115_200, 1, 1], [65_535, 2, 1]),
            ([0, 5, 5], [0, 0, 0]),
            ([0xFF00_0000, 0xE3, 0xFF7F_0000], [0, 0, 0]),
            ([65_535, 65_535, 65_535], [0, 0, 0]),
            ([2048, 2048, 1], [2048, 2048, 1]),
            ([2048, 2049, 1], [0, 0, 0]),
            ([3, 1, 0xFFFF_FFFF], [0, 0, 0]),
            ([1 << 16, 1 << 16, 1], [0, 0, 0]),
        ];
        for (g, want) in cases {
            let mut mem = vec![0xDEADu32; 64];
            mem[4..7].copy_from_slice(&g);
            let out = runner::dispatch(&dxil, [1, 1, 1], &[16, MOST_GROUPS, 128, 0], &mem).unwrap();
            assert_eq!(&out[32..35], &want, "{g:?}");
        }
    }
    #[test]
    fn gpu_indirect_two_pass_groups_stay_inside_the_record_buffer() {
        let Some((compute, mesh)) = scale_args(&[[7, 100, 1, 1], [8, 3, 1, 1], [9, 0, 1, 1]], 2, 10) else { return };
        assert_eq!(compute, [[7, 10, 1, 1], [8, 3, 1, 1], [9, 0, 1, 1]]);
        assert_eq!(mesh, [[7, 20, 1, 1], [8, 6, 1, 1], [9, 0, 1, 1]]);
    }

    #[test]
    fn gpu_indirect_two_pass_groups_go_in_x() {
        let commands = [[1, 1, 2, 1], [2, 2, 3, 4], [3, 300, 300, 1], [4, 5, 0, 2], [5, 70_000, 70_000, 70_000], [6, 1000, 1, 1], [7, 5000, 1, 1], [8, 5000, 1, 0]];
        let Some((compute, mesh)) = scale_args(&commands, 3, 1000) else { return };
        assert_eq!(compute, [[1, 2, 1, 1], [2, 24, 1, 1], [3, 1000, 1, 1], [4, 0, 1, 1], [5, 1000, 1, 1], [6, 1000, 1, 1], [7, 1000, 1, 1], [8, 0, 1, 1]]);
        assert_eq!(mesh, [[1, 6, 1, 1], [2, 72, 1, 1], [3, 3000, 1, 1], [4, 0, 1, 1], [5, 3000, 1, 1], [6, 3000, 1, 1], [7, 3000, 1, 1], [8, 0, 1, 1]]);
    }

    #[test]
    fn freed_two_pass_buffers_kept_for_reuse_are_capped() {
        assert!(pool_keeps(0, 0, 128 << 20));
        assert!(pool_keeps(1, 128 << 20, 128 << 20));
        assert!(!pool_keeps(2, 256 << 20, 4 << 20));
        assert!(!pool_keeps(16, 0, 1 << 20));
    }

    #[test]
    fn gpu_two_pass_argument_scratch_is_shared_and_capped() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        let Ok(dev) = crate::gpu::agility::create_device(None, D3D_FEATURE_LEVEL_11_0) else {
            crate::gpu::recomp::runner::skip_test("no GPU");
            return;
        };
        let list = 0x7500_0001usize as *mut c_void;
        unsafe {
            // small rewrites share one block
            let (a, at_a) = scratch_args(list, &dev, 100).unwrap();
            let (b, at_b) = scratch_args(list, &dev, 300).unwrap();
            assert_eq!((a.as_raw(), at_a), (b.as_raw(), 0));
            assert_eq!(at_b, 256);
            // a recording takes at most SCRATCH_CAP of blocks
            let mut taken = 1;
            while scratch_args(list, &dev, SCRATCH_BLOCK).is_ok() {
                taken += 1;
                assert!(taken <= SCRATCH_CAP / SCRATCH_BLOCK, "past the cap");
            }
            assert!(with(list, |s| s.scratch_taken) <= SCRATCH_CAP);
            // the next recording starts again
            reset(list, 0, 0);
            assert!(scratch_args(list, &dev, SCRATCH_BLOCK).is_ok());
        }
        forget(list);
    }

    #[test]
    fn gpu_indirect_two_pass_commands_past_the_count_get_no_groups() {
        let commands = [[1, 3, 1, 1], [2, 2, 1, 1], [3, 4, 1, 1]];
        let Some((compute, mesh)) = scale_args_counted(&commands, 2, 1000, Some(2)) else { return };
        assert_eq!(compute, [[1, 3, 1, 1], [2, 2, 1, 1], [3, 0, 1, 1]]);
        assert_eq!(mesh, [[1, 6, 1, 1], [2, 4, 1, 1], [3, 0, 1, 1]]);
        let Some((compute, _)) = scale_args_counted(&commands, 2, 1000, Some(7)) else { return };
        assert_eq!(compute, [[1, 3, 1, 1], [2, 2, 1, 1], [3, 4, 1, 1]]);
    }

    /// Commands of one two-pass ExecuteIndirect run in `two_pass_steps` order on the GPU, with
    /// stand-ins for the two passes: the "compute pass" writes each group's record (from record 0)
    /// with its command's constant, the "mesh pass" copies the records it reads out by command.
    #[test]
    fn gpu_two_pass_indirect_commands_keep_their_own_records() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
        let src = r#"
#define RS "RootConstants(num32BitConstants=1, b0), UAV(u0), UAV(u1)"
cbuffer C : register(b0) { uint v; };
RWByteAddressBuffer rec : register(u0);
RWByteAddressBuffer o : register(u1);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void cs(uint3 g : SV_GroupID) { rec.Store(g.x * 4u, v); }
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void ms(uint3 g : SV_GroupID) { o.Store((v * 16u + g.x) * 4u, rec.Load(g.x * 4u)); }
"#;
        // (constant, groups) of each command
        let commands = [(1u32, 3u32), (2, 2), (3, 4)];
        unsafe {
            let Ok(dev) = crate::gpu::agility::create_device(None, D3D_FEATURE_LEVEL_11_0) else {
                crate::gpu::recomp::runner::skip_test("no GPU");
                return;
            };
            let cs = crate::gpu::recomp::dxc::compile(src, "cs", "cs_6_6", &[]).unwrap();
            let ms = crate::gpu::recomp::dxc::compile(src, "ms", "cs_6_6", &[]).unwrap();
            let rs: ID3D12RootSignature = dev.CreateRootSignature(0, &cs).unwrap();
            let pso = |dxil: &[u8]| -> ID3D12PipelineState {
                let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&rs),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as _, BytecodeLength: dxil.len() },
                    ..Default::default()
                };
                dev.CreateComputePipelineState(&desc).unwrap()
            };
            let (pass1, pass2) = (pso(&cs), pso(&ms));
            let args = [
                D3D12_INDIRECT_ARGUMENT_DESC {
                    Type: D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT,
                    Anonymous: D3D12_INDIRECT_ARGUMENT_DESC_0 { Constant: D3D12_INDIRECT_ARGUMENT_DESC_0_1 { RootParameterIndex: 0, DestOffsetIn32BitValues: 0, Num32BitValuesToSet: 1 } },
                },
                D3D12_INDIRECT_ARGUMENT_DESC { Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH, ..Default::default() },
            ];
            let mut sig: Option<ID3D12CommandSignature> = None;
            dev.CreateCommandSignature(&D3D12_COMMAND_SIGNATURE_DESC { ByteStride: 16, NumArgumentDescs: 2, pArgumentDescs: args.as_ptr(), NodeMask: 0 }, &rs, &mut sig).unwrap();
            let sig = sig.unwrap();
            let buffer = |bytes: u64, cpu: bool| -> ID3D12Resource {
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: bytes,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    ..Default::default()
                };
                let heap = if cpu {
                    D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_CUSTOM, CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, MemoryPoolPreference: D3D12_MEMORY_POOL_L0, ..Default::default() }
                } else {
                    D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }
                };
                let mut r: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut r).unwrap();
                r.unwrap()
            };
            let (indirect, rec, out) = (buffer(256, true), buffer(256, false), buffer(1024, true));
            let map = |b: &ID3D12Resource| -> *mut u32 {
                let mut p = std::ptr::null_mut();
                b.Map(0, None, Some(&mut p)).unwrap();
                p as *mut u32
            };
            let (a, o) = (map(&indirect), map(&out));
            for (k, &(v, x)) in commands.iter().enumerate() {
                std::ptr::copy_nonoverlapping([v, x, 1, 1].as_ptr(), a.add(4 * k), 4);
            }
            std::ptr::write_bytes(o, 0, 256);
            let ty = D3D12_COMMAND_LIST_TYPE_COMPUTE;
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: ty, ..Default::default() }).unwrap();
            let done: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, ty, &alloc, &pass1).unwrap();
            emit_transition(&list, &indirect, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
            list.SetComputeRootSignature(&rs);
            list.SetComputeRootUnorderedAccessView(1, rec.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(2, out.GetGPUVirtualAddress());
            for (k, compute) in two_pass_steps(commands.len() as u32) {
                list.SetPipelineState(if compute { &pass1 } else { &pass2 });
                list.ExecuteIndirect(&sig, 1, &indirect, k as u64 * 16, None::<&ID3D12Resource>, 0);
                emit_uav_barrier(&list, Some(&rec));
            }
            emit_transition(&list, &indirect, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COMMON);
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            queue.Signal(&done, 1).unwrap();
            crate::gpu::fence::wait(&done, 1, &dev, None, 10_000).unwrap();
            for &(v, x) in &commands {
                let got: Vec<u32> = (0..x).map(|g| *o.add((v * 16 + g) as usize)).collect();
                assert_eq!(got, vec![v; x as usize], "the records the mesh pass of command {v} read");
            }
            indirect.Unmap(0, None);
            out.Unmap(0, None);
        }
    }

    /// With heap windows the two-pass programs read the memory windows from the shared heap
    /// (`pso::mem_windows`): both passes' binds set it on a list that has none, as `bind_abi` does.
    /// (Heap windows are opt-in: run with `XCL_HEAP_WINDOWS=1`, else skipped.)
    #[test]
    fn two_pass_binds_set_the_shared_view_heap() {
        use crate::gpu::recomp::runner::skip_test;
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
        if !crate::gpu::recomp::pso::heap_windows() {
            skip_test("no heap windows (opt-in: XCL_HEAP_WINDOWS=1)");
            return;
        }
        #[repr(C)]
        struct FakeList {
            vt: *const usize,
            calls: Mutex<Vec<Vec<usize>>>,
        }
        unsafe extern "system" fn fake_set_heaps(this: *mut c_void, n: u32, heaps: *const usize) {
            let v = if heaps.is_null() { Vec::new() } else { std::slice::from_raw_parts(heaps, n as usize).to_vec() };
            (*(this as *const FakeList)).calls.lock().unwrap().push(v);
        }
        unsafe {
            let Ok(dev) = crate::gpu::agility::create_device(None, D3D_FEATURE_LEVEL_11_0) else {
                skip_test("no GPU");
                return;
            };
            // the shared heap with a window view, unless the backend made it already
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 1 << 16,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut window: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut window).unwrap();
            let window = window.unwrap();
            if super::super::console_heap::global_heap().is_none() && !super::super::console_heap::set_window_views(&dev, &[(window.as_raw() as usize, 1 << 16)]) {
                skip_test("no shared heap");
                return;
            }
            let g = super::super::console_heap::global_heap().unwrap().as_raw() as usize;
            let mut orig_vt = [0usize; 64];
            orig_vt[28] = fake_set_heaps as usize;
            let mut ext_vt = vec![0usize; 65];
            ext_vt[0] = orig_vt.as_ptr() as usize;
            ext_vt[1..].copy_from_slice(&orig_vt);
            let fake = Box::new(FakeList { vt: ext_vt.as_ptr().add(1), calls: Mutex::new(Vec::new()) });
            let this = &*fake as *const FakeList as *mut c_void;
            let list = ID3D12GraphicsCommandList::from_raw_borrowed(&this).unwrap();
            // no root signatures: the bind stops before it records anything else
            let tp = super::super::real::TwoPassPso { compute: 0, compute_rs: 0, mesh_rs: 0, record: Default::default(), mesh_groups_per_record: 1, kind: "MS" };
            for compute_pass in [true, false] {
                super::super::list_x::set_descriptor_heaps_x(this, 0, std::ptr::null());
                assert!(super::super::list_x::bound_heaps(this, false).is_empty());
                assert!(bind_two_pass_stage(this, list, 0, &tp, 0, compute_pass).is_err());
                assert_eq!(super::super::list_x::bound_heaps(this, false), vec![g], "compute pass {compute_pass}: the shared heap is bound");
                assert_eq!(fake.calls.lock().unwrap().last(), Some(&vec![g]), "compute pass {compute_pass}");
            }
            super::super::list_x::bound_heaps(this, true);
            forget(this);
        }
    }


    /// A reusable block must fit the recording's remaining budget, not just its request.
    #[test]
    fn gpu_two_pass_pooled_scratch_respects_remaining_budget() {
        use windows::Win32::Graphics::{Direct3D::D3D_FEATURE_LEVEL_12_0, Dxgi::*};
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().unwrap();
            let wanted = std::env::var("XCL_ADAPTER").unwrap_or_default();
            let mut chosen = None;
            for i in 0.. {
                let Ok(adapter) = factory.EnumAdapters1(i) else { break };
                let desc = adapter.GetDesc1().unwrap();
                let end = desc.Description.iter().position(|&c| c == 0).unwrap_or(desc.Description.len());
                let name = String::from_utf16_lossy(&desc.Description[..end]);
                if !wanted.is_empty() && !name.contains(&wanted) { continue; }
                if let Ok(dev) = crate::gpu::agility::create_device(&adapter, D3D_FEATURE_LEVEL_12_0) {
                    chosen = Some((name, dev));
                    break;
                }
            }
            let Some((name, dev)) = chosen else {
                crate::gpu::recomp::runner::skip_test("no selected D3D12 adapter");
                return;
            };
            println!("POOLED_SCRATCH_ADAPTER={name}");
            let key = dev.as_raw() as usize;
            // These are state keys only; scratch_args does not call through a list pointer.
            let (consumer, producer, empty) = (0x7500_1011usize as *mut c_void, 0x7500_1012usize as *mut c_void, 0x7500_1013usize as *mut c_void);
            let allocator = 0x7500_1014usize;
            let first = scratch_args(consumer, &dev, SCRATCH_BLOCK).unwrap();
            assert_eq!(with(consumer, |s| s.scratch_taken), SCRATCH_BLOCK);
            reset(producer, allocator, 0);
            let large = scratch_args(producer, &dev, SCRATCH_CAP).unwrap();
            assert_eq!(large.0.GetDesc().Width, SCRATCH_CAP);
            reset(producer, 0, 0);
            release_allocator_res(allocator);
            let second = scratch_args(consumer, &dev, SCRATCH_BLOCK).unwrap();
            let taken = with(consumer, |s| s.scratch_taken);
            println!("POOLED_SCRATCH taken={taken} second_bytes={} cap={SCRATCH_CAP}", second.0.GetDesc().Width);
            // If it was too large for consumer, the cached block remains reusable by an empty recording.
            let reused = if taken <= SCRATCH_CAP {
                let r = scratch_args(empty, &dev, SCRATCH_CAP).unwrap();
                assert_eq!(r.0.as_raw(), large.0.as_raw());
                assert_eq!(with(empty, |s| s.scratch_taken), SCRATCH_CAP);
                assert!(scratch_args(empty, &dev, 256).is_err());
                Some(r)
            } else { None };
            drop((first, second, large, reused));
            for list in [consumer, producer, empty] { forget(list); }
            FREE_RES.lock().unwrap().retain(|(_, _, d)| *d != key);
            assert!(taken <= SCRATCH_CAP, "pooled block exceeds recording budget: {taken} > {SCRATCH_CAP}");
        }
    }
    #[test]
    fn root_uav_alignment_splits_unaligned_offsets_and_rejects_sub_dword() {
        assert_eq!(align_root_uav(0x1_0000_0000).unwrap(), (0x1_0000_0000, 0));
        // 12-byte D3D12_DISPATCH_ARGUMENTS stride: not 16-byte aligned, split into 256-byte base + byte offset
        assert_eq!(align_root_uav(0x1_0000_000c).unwrap(), (0x1_0000_0000, 12));
        assert_eq!(align_root_uav(0x1_0000_0018).unwrap(), (0x1_0000_0000, 24));
        assert_eq!(align_root_uav(0x1_0000_01fc).unwrap(), (0x1_0000_0100, 252));
        assert!(align_root_uav(0).is_err());
        assert!(align_root_uav(0xfc).is_err());
        assert_eq!(align_root_uav(0x100).unwrap(), (0x100, 0));
        assert!(align_root_uav(0x1_0000_0001).is_err());
        assert!(align_root_uav(0x1_0000_0002).is_err());
        assert!(align_root_uav(0x1_0000_0003).is_err());
    }

    #[test]
    fn rebind_allocator_retires_old_blocks_and_preserves_bindings() {
        let list = 0x7500_2011usize as *mut c_void;
        let old_alloc = 0x7500_2012usize;
        let new_alloc = 0x7500_2013usize;
        seed_reset_state_for_test(list, old_alloc, 0xCAFE, vec![0xDEAD_0000]);
        rebind_allocator(list, new_alloc);
        let (alloc, pso, args, args_gfx, uma) = reset_state_for_test(list);
        assert_eq!(alloc, new_alloc);
        assert_eq!(pso, 0xCAFE);
        assert_eq!(args, vec![1, 2, 3, 4]);
        assert_eq!(args_gfx, vec![5, 6, 7, 8]);
        assert!(uma.is_empty());
        remove_state_for_test(list);
        super::console_list::allocator_reset(old_alloc as *mut c_void);
    }

    #[test]
    fn io_pool_budget_and_abi_diff_behave_as_expected() {
        assert!(io_pool_keeps(0, IO_BLOCK_BYTES));
        assert!(io_pool_keeps(FREE_IO_CAP - IO_BLOCK_BYTES, IO_BLOCK_BYTES));
        assert!(!io_pool_keeps(FREE_IO_CAP, IO_BLOCK_BYTES));
        let mut b = AbiBound::default();
        let v1 = [0x100, 0x200, 0x300, 0x400, 0x500, 0x600];
        assert_eq!(b.diff_and_update(1, v1), (true, 0x3F));
        assert_eq!(b.diff_and_update(1, v1), (false, 0x00));
        let mut v2 = v1;
        v2[0] = 0x180;
        assert_eq!(b.diff_and_update(1, v2), (false, 0x01));
        assert_eq!(b.diff_and_update(2, v2), (true, 0x3F));
    }

    #[test]
    fn gpu_vram_io_prelude_copy_populates_vram_slab() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
        unsafe {
            let Ok(dev) = crate::gpu::agility::create_device(None, D3D_FEATURE_LEVEL_12_0) else {
                crate::gpu::recomp::runner::skip_test("no GPU");
                return;
            };
            let Some(mut blk) = create_io_block(&dev, IO_BLOCK_BYTES) else {
                crate::gpu::recomp::runner::skip_test("no io block");
                return;
            };
            let words: [u32; 8] = [0x1122_3344, 0x5566_7788, 0xAABB_CCDD, 0xDEAD_BEEF, 1, 2, 3, 4];
            let bytes = std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), std::mem::size_of_val(&words));
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), blk.mapped as *mut u8, bytes.len());
            blk.used = 256;
            let rb_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 256,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut rb: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &rb_desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut rb).unwrap();
            let rb = rb.unwrap();
            let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let f: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let a: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let l: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &a, None).unwrap();
            emit_transition(&l, &blk.vram, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
            l.CopyBufferRegion(&blk.vram, 0, &blk.upload, 0, blk.used);
            emit_transition(&l, &blk.vram, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
            l.CopyBufferRegion(&rb, 0, &blk.vram, 0, blk.used);
            l.Close().unwrap();
            q.ExecuteCommandLists(&[Some(l.cast().unwrap())]);
            q.Signal(&f, 1).unwrap();
            crate::gpu::fence::wait(&f, 1, &dev, None, 10_000).unwrap();
            let mut p = std::ptr::null_mut();
            rb.Map(0, None, Some(&mut p)).unwrap();
            let got = std::slice::from_raw_parts(p as *const u32, words.len()).to_vec();
            rb.Unmap(0, None);
            assert_eq!(got, words);
        }
    }

    #[test]
    fn prelude_hoist_blocks_on_gpu_write_and_clears_uma_windows_on_reset() {
        let a = 0x7710_1001usize as *mut c_void;
        let b = 0x7710_1002usize as *mut c_void;
        reset(a, 0x7710_A001, 0);
        with(a, |s| {
            s.uma_windows = Some(vec![0x1000]);
            assert!(!s.prelude_blocked);
        });
        note_gpu_write(a);
        with(a, |s| {
            assert!(s.prelude_blocked);
        });
        inherit_for_split(a, b, 0x7710_A001);
        with(b, |s| {
            assert!(s.prelude_blocked);
            assert!(s.uma_windows.is_none());
        });
        unblock_prelude(b);
        with(b, |s| assert!(!s.prelude_blocked));
        reset(a, 0x7710_A002, 0);
        with(a, |s| {
            assert!(!s.prelude_blocked);
            assert!(s.uma_windows.is_none());
            assert!(s.prelude_scales.is_empty());
        });
        forget(a);
        forget(b);
    }

    #[test]
    fn gpu_prelude_scale_shader_transforms_batch() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
        if crate::gpu::recomp::dxc::location().is_err() {
            crate::gpu::recomp::runner::skip_test("no DXC");
            return;
        }
        unsafe {
            let Ok(dev) = crate::gpu::agility::create_device(None, D3D_FEATURE_LEVEL_12_0) else {
                crate::gpu::recomp::runner::skip_test("no GPU");
                return;
            };
            let (rs, pso) = prelude_scale_pipeline(&dev).expect("prelude_scale_pipeline");
            let mk_buf = |bytes: u64, heap: D3D12_HEAP_TYPE, flags: D3D12_RESOURCE_FLAGS, state: D3D12_RESOURCE_STATES| -> ID3D12Resource {
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: bytes,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    Flags: flags,
                    ..Default::default()
                };
                let mut res: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut res).unwrap();
                res.unwrap()
            };
            let up = mk_buf(4096, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            let w0 = mk_buf(4096, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
            let jobs_buf = mk_buf(256, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
            let dst = mk_buf(4096, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
            let rb = mk_buf(256, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

            // Upload two 12-byte DispatchMesh commands at w0+0: (3, 2, 1) and (5, 1, 1); job at up+1024 (split=4, cap=100).
            let mut p = std::ptr::null_mut();
            up.Map(0, None, Some(&mut p)).unwrap();
            let u32s = std::slice::from_raw_parts_mut(p as *mut u32, 1024);
            u32s[0..6].copy_from_slice(&[3, 2, 1, 5, 1, 1]);
            // Job record (12 u32s at byte offset 1024 = u32 index 256): count=2, stride=12, offset=0, split=4, cap=100, src_win=0, src_off=0, dst_off=0, has_cnt=0
            u32s[256..268].copy_from_slice(&[2, 12, 0, 4, 100, 0, 0, 0, 0, 0, 0, 0]);
            up.Unmap(0, None);

            let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COMPUTE, ..Default::default() }).unwrap();
            let f: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let a: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE).unwrap();
            let l: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &a, None).unwrap();
            emit_transition(&l, &w0, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
            emit_transition(&l, &jobs_buf, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
            l.CopyBufferRegion(&w0, 0, &up, 0, 24);
            l.CopyBufferRegion(&jobs_buf, 0, &up, 1024, 48);
            emit_transition(&l, &w0, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            emit_transition(&l, &jobs_buf, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            emit_transition(&l, &dst, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            l.SetComputeRootSignature(&rs);
            l.SetPipelineState(&pso);
            l.SetComputeRootUnorderedAccessView(0, jobs_buf.GetGPUVirtualAddress());
            l.SetComputeRootUnorderedAccessView(1, dst.GetGPUVirtualAddress());
            for k in 0..4u32 {
                l.SetComputeRootUnorderedAccessView(2 + k, w0.GetGPUVirtualAddress());
            }
            l.Dispatch(1, 1, 1);
            emit_transition(&l, &dst, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
            l.CopyBufferRegion(&rb, 0, &dst, 0, 48);
            l.Close().unwrap();
            q.ExecuteCommandLists(&[Some(l.cast().unwrap())]);
            q.Signal(&f, 1).unwrap();
            crate::gpu::fence::wait(&f, 1, &dev, None, 10_000).unwrap();

            let mut out = std::ptr::null_mut();
            rb.Map(0, None, Some(&mut out)).unwrap();
            let got = std::slice::from_raw_parts(out as *const u32, 12).to_vec();
            rb.Unmap(0, None);
            // Compute pass commands (at 0..24): (6,1,1), (5,1,1); mesh pass commands (at 24..48): (24,1,1), (20,1,1)
            assert_eq!(got, vec![6, 1, 1, 5, 1, 1, 24, 1, 1, 20, 1, 1]);
        }
    }

    #[test]
    fn dispatches_past_the_d3d12_limit_are_folded_or_clamped() {
        assert_eq!(fold(64, 2, 1), (0, [64, 2, 1]));
        assert_eq!(fold(65_535, 1, 1), (0, [65_535, 1, 1]));
        assert_eq!(fold(72_512, 1, 1), (72_512, [65_535, 2, 1]));
        assert_eq!(fold(115_200, 1, 1), (115_200, [65_535, 2, 1]));
        assert_eq!(fold(131_071, 1, 3), (131_071, [65_535, 3, 3]));
        assert_eq!(fold(70_000, 2, 1), (0, [65_535, 2, 1]));
        for n in [65_536u32, 72_512, 75_072, 115_200, 200_000] {
            let (limit, [x, y, _]) = fold(n, 1, 1);
            assert_eq!(limit, n);
            assert!(x * y >= n && x * y - n < x, "{n}: {x}x{y}");
        }
    }
}
