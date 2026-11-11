//! 32-bit ARM (A32) definitions of architecture-specific functions in
//! Wasmtime.

#[inline]
pub fn get_stack_pointer() -> usize {
    let stack_pointer: usize;
    unsafe {
        core::arch::asm!(
            "mov {}, sp",
            out(reg) stack_pointer,
            options(nostack, nomem),
        );
    }
    stack_pointer
}

pub unsafe fn get_next_older_pc_from_fp(fp: usize) -> usize {
    // The frame record is `[fp] = saved fp, [fp + 4] = saved lr`.
    unsafe { *(fp as *mut usize).offset(1) }
}

pub unsafe fn resume_to_exception_handler(
    pc: usize,
    sp: usize,
    fp: usize,
    payload1: usize,
    payload2: usize,
) -> ! {
    unsafe {
        core::arch::asm!(
            "mov sp, r2",
            "mov r11, r3",
            "bx r12",
            in("r0") payload1,
            in("r1") payload2,
            in("r2") sp,
            in("r3") fp,
            in("r12") pc,
            options(nostack, nomem, noreturn),
        );
    }
}

// The current frame pointer points to the next older frame pointer.
pub const NEXT_OLDER_FP_FROM_FP_OFFSET: usize = 0;

// SP of caller is FP in callee plus size of FP/return address pair.
pub const NEXT_OLDER_SP_FROM_FP_OFFSET: usize = 8;

pub fn assert_fp_is_aligned(fp: usize) {
    assert_eq!(fp % 8, 0, "stack should always be aligned to 8");
}
