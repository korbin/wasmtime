use std::ffi::c_void;
use wasmtime_core::error::Result;

/// See docs on [crate::clear_cache] for a description of what this function is trying to do.
///
/// 32-bit ARM Linux exposes cache maintenance to user space through the
/// ARM-private `cacheflush` system call, which cleans the data cache and
/// invalidates the instruction cache for the given range on all cores.
pub(crate) fn clear_cache(ptr: *const c_void, len: usize) -> Result<()> {
    // https://github.com/torvalds/linux/blob/master/arch/arm/include/uapi/asm/unistd.h
    const ARM_NR_BASE: libc::c_long = 0x0f0000;
    const ARM_NR_CACHEFLUSH: libc::c_long = ARM_NR_BASE + 2;
    let start = ptr as usize;
    let end = start + len;
    // Note that only `-1` indicates failure: some emulators (e.g. QEMU's
    // user-mode emulation) treat this system call as a no-op that leaves the
    // first argument in the return register.
    match unsafe { libc::syscall(ARM_NR_CACHEFLUSH, start, end, 0) } {
        -1 => Err(std::io::Error::last_os_error().into()),
        _ => Ok(()),
    }
}
