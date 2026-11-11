use crate::cdsl::isa::TargetIsa;
use crate::cdsl::settings::SettingGroupBuilder;

pub(crate) fn define() -> TargetIsa {
    let mut settings = SettingGroupBuilder::new("arm32");

    // The baseline target is ARMv7-A in ARM (A32) state with VFPv3 and the
    // hard-float procedure call standard (AAPCS-VFP), matching
    // `armv7-unknown-linux-gnueabihf`.

    settings.add_bool(
        "has_idiv",
        "Has hardware integer division (SDIV/UDIV) in ARM state.",
        "This is the ARMv7VE `idiva` extension, present on e.g. Cortex-A7, \
         Cortex-A15 and Cortex-A17, but absent on Cortex-A8 and Cortex-A9.",
        false,
    );
    settings.add_bool(
        "has_vfp4",
        "Has VFPv4 (fused multiply-add instructions VFMA/VFMS).",
        "",
        false,
    );
    settings.add_bool(
        "has_neon",
        "Has the Advanced SIMD (NEON) extension.",
        "",
        false,
    );
    settings.add_bool(
        "has_armv8",
        "Is an ARMv8-A core in AArch32 state.",
        "This adds e.g. the VRINT floating-point rounding instructions. \
         32-bit Linux userlands commonly run on ARMv8 cores (e.g. \
         Cortex-A53/A72).",
        false,
    );

    TargetIsa::new("arm32", settings.build())
}
