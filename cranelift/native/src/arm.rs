//! Host feature detection for 32-bit ARM Linux using the auxiliary vector.

use cranelift_codegen::settings::Configurable;

// From the Linux kernel's `arch/arm/include/uapi/asm/hwcap.h`.
const HWCAP_NEON: libc::c_ulong = 1 << 12;
const HWCAP_VFPV4: libc::c_ulong = 1 << 16;
const HWCAP_IDIVA: libc::c_ulong = 1 << 17;

pub fn hwcap_detect(isa_builder: &mut dyn Configurable) -> Result<(), &'static str> {
    let hwcap = unsafe { libc::getauxval(libc::AT_HWCAP) };
    if hwcap & HWCAP_IDIVA != 0 {
        isa_builder.enable("has_idiv").unwrap();
    }
    if hwcap & HWCAP_VFPV4 != 0 {
        isa_builder.enable("has_vfp4").unwrap();
    }
    if hwcap & HWCAP_NEON != 0 {
        isa_builder.enable("has_neon").unwrap();
    }
    if platform_is_armv8() {
        isa_builder.enable("has_armv8").unwrap();
    }
    Ok(())
}

/// Whether the platform string in the auxiliary vector (e.g. "v7l" or "v8l")
/// names an ARMv8 (or later) core. A 32-bit process on an ARMv8 core sees
/// "v8l", whether the kernel is 32-bit or 64-bit.
fn platform_is_armv8() -> bool {
    let platform = unsafe { libc::getauxval(libc::AT_PLATFORM) } as *const libc::c_char;
    if platform.is_null() {
        return false;
    }
    let platform = unsafe { core::ffi::CStr::from_ptr(platform) }.to_bytes();
    match platform {
        [b'v', rest @ ..] => {
            let digits: Vec<u8> = rest
                .iter()
                .copied()
                .take_while(u8::is_ascii_digit)
                .collect();
            core::str::from_utf8(&digits)
                .ok()
                .and_then(|d| d.parse::<u32>().ok())
                .is_some_and(|v| v >= 8)
        }
        _ => false,
    }
}
