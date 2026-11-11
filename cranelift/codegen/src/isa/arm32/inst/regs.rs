//! ARMv7 register definitions.
//!
//! Integer registers `r0`..`r15` live in `RegClass::Int` with their hardware
//! number as the encoding. Floating-point values live in the double-precision
//! registers `d0`..`d15` (`RegClass::Float`); a single-precision value held
//! in `dN` lives in its low half `s(2N)`.

use crate::machinst::{Reg, Writable};
use alloc::format;
use alloc::string::String;
use regalloc2::{PReg, RegClass};

/// Get a reference to an integer register.
pub const fn rreg(num: u8) -> Reg {
    Reg::from_real_reg(preg(num))
}

/// Get a physical integer register.
pub const fn preg(num: u8) -> PReg {
    assert!(num < 16);
    PReg::new(num as usize, RegClass::Int)
}

/// Get a reference to a double-precision register.
pub const fn dreg(num: u8) -> Reg {
    Reg::from_real_reg(pdreg(num))
}

/// Get a physical double-precision register.
pub const fn pdreg(num: u8) -> PReg {
    assert!(num < 32);
    PReg::new(num as usize, RegClass::Float)
}

/// Get a reference to a quad (vector) register `q<num>`.
pub const fn qreg(num: u8) -> Reg {
    Reg::from_real_reg(pqreg(num))
}

/// Get a physical quad (vector) register.
pub const fn pqreg(num: u8) -> PReg {
    assert!(num < 16);
    PReg::new(num as usize, RegClass::Vector)
}

/// The frame pointer (`r11`).
pub const fn fp_reg() -> Reg {
    rreg(11)
}

/// Writable frame pointer.
pub fn writable_fp_reg() -> Writable<Reg> {
    Writable::from_reg(fp_reg())
}

/// The intra-procedure-call scratch register (`r12`), reserved as a
/// temporary for instruction sequences during emission.
pub const fn ip_reg() -> Reg {
    rreg(12)
}

/// Writable `ip`.
pub fn writable_ip_reg() -> Writable<Reg> {
    Writable::from_reg(ip_reg())
}

/// The stack pointer (`r13`).
pub const fn stack_reg() -> Reg {
    rreg(13)
}

/// Writable stack pointer.
pub fn writable_stack_reg() -> Writable<Reg> {
    Writable::from_reg(stack_reg())
}

/// The link register (`r14`).
pub const fn link_reg() -> Reg {
    rreg(14)
}

/// Writable link register.
pub fn writable_link_reg() -> Writable<Reg> {
    Writable::from_reg(link_reg())
}

/// Hardware encoding of a core register.
pub fn machreg_to_gpr(r: Reg) -> u32 {
    let real = r.to_real_reg().expect("register must be allocated");
    debug_assert_eq!(real.class(), RegClass::Int);
    u32::from(real.hw_enc())
}

/// Hardware number of a double register.
pub fn machreg_to_dreg(r: Reg) -> u32 {
    let real = r.to_real_reg().expect("register must be allocated");
    debug_assert_eq!(real.class(), RegClass::Float);
    u32::from(real.hw_enc())
}

/// Pretty-print a register.
pub fn show_reg(r: Reg) -> String {
    match r.to_real_reg() {
        Some(real) => match real.class() {
            RegClass::Int => match real.hw_enc() {
                11 => "fp".into(),
                12 => "ip".into(),
                13 => "sp".into(),
                14 => "lr".into(),
                15 => "pc".into(),
                n => format!("r{n}"),
            },
            RegClass::Float => format!("d{}", real.hw_enc()),
            RegClass::Vector => format!("q{}", real.hw_enc()),
        },
        None => format!("{r:?}"),
    }
}

/// Pretty-print a register as a single-precision register.
pub fn show_sreg(r: Reg) -> String {
    match r.to_real_reg() {
        Some(real) if real.class() == RegClass::Float => format!("s{}", real.hw_enc() * 2),
        _ => format!("{r:?}.s"),
    }
}

/// Pretty-print a register for the given floating-point size.
pub fn show_freg(r: Reg, is_f64: bool) -> String {
    if is_f64 { show_reg(r) } else { show_sreg(r) }
}
