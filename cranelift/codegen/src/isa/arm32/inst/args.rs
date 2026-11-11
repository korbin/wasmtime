//! ARMv7 instruction operand definitions: condition codes, immediates,
//! shifts and addressing modes.

use crate::ir::condcodes::{FloatCC, IntCC};
use crate::isa::arm32::inst::EmitState;
use crate::isa::arm32::inst::regs::show_reg;
use crate::isa::arm32::lower::isle::generated_code::ShiftOp;
use crate::machinst::{MachInstEmitState, OperandVisitor, OperandVisitorImpl, Reg, StackAMode};
use alloc::format;
use alloc::string::String;
use core::fmt;

/// A condition code, as encoded in bits 31:28 of every A32 instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Cond {
    /// Equal (Z set).
    Eq = 0,
    /// Not equal (Z clear).
    Ne = 1,
    /// Unsigned higher or same (C set).
    Hs = 2,
    /// Unsigned lower (C clear).
    Lo = 3,
    /// Minus / negative (N set).
    Mi = 4,
    /// Plus / positive or zero (N clear).
    Pl = 5,
    /// Overflow (V set).
    Vs = 6,
    /// No overflow (V clear).
    Vc = 7,
    /// Unsigned higher (C set and Z clear).
    Hi = 8,
    /// Unsigned lower or same (C clear or Z set).
    Ls = 9,
    /// Signed greater than or equal (N == V).
    Ge = 10,
    /// Signed less than (N != V).
    Lt = 11,
    /// Signed greater than (Z clear and N == V).
    Gt = 12,
    /// Signed less than or equal (Z set or N != V).
    Le = 13,
}

impl Cond {
    /// Encoding of this condition in bits 31:28.
    pub fn bits(self) -> u32 {
        self as u32
    }

    /// The inverse condition.
    pub fn invert(self) -> Cond {
        match self {
            Cond::Eq => Cond::Ne,
            Cond::Ne => Cond::Eq,
            Cond::Hs => Cond::Lo,
            Cond::Lo => Cond::Hs,
            Cond::Mi => Cond::Pl,
            Cond::Pl => Cond::Mi,
            Cond::Vs => Cond::Vc,
            Cond::Vc => Cond::Vs,
            Cond::Hi => Cond::Ls,
            Cond::Ls => Cond::Hi,
            Cond::Ge => Cond::Lt,
            Cond::Lt => Cond::Ge,
            Cond::Gt => Cond::Le,
            Cond::Le => Cond::Gt,
        }
    }

    /// Condition for an integer comparison.
    pub fn from_intcc(cc: IntCC) -> Cond {
        match cc {
            IntCC::Equal => Cond::Eq,
            IntCC::NotEqual => Cond::Ne,
            IntCC::SignedGreaterThanOrEqual => Cond::Ge,
            IntCC::SignedGreaterThan => Cond::Gt,
            IntCC::SignedLessThanOrEqual => Cond::Le,
            IntCC::SignedLessThan => Cond::Lt,
            IntCC::UnsignedGreaterThanOrEqual => Cond::Hs,
            IntCC::UnsignedGreaterThan => Cond::Hi,
            IntCC::UnsignedLessThanOrEqual => Cond::Ls,
            IntCC::UnsignedLessThan => Cond::Lo,
        }
    }

    /// Conditions for a floating-point comparison after `vcmp; vmrs`. The
    /// result holds if either condition holds.
    ///
    /// The flags after a VFP comparison are:
    ///
    /// | relation  | N | Z | C | V |
    /// |-----------|---|---|---|---|
    /// | equal     | 0 | 1 | 1 | 0 |
    /// | less      | 1 | 0 | 0 | 0 |
    /// | greater   | 0 | 0 | 1 | 0 |
    /// | unordered | 0 | 0 | 1 | 1 |
    pub fn from_floatcc(cc: FloatCC) -> (Cond, Option<Cond>) {
        match cc {
            FloatCC::Ordered => (Cond::Vc, None),
            FloatCC::Unordered => (Cond::Vs, None),
            FloatCC::Equal => (Cond::Eq, None),
            FloatCC::NotEqual => (Cond::Ne, None),
            FloatCC::OrderedNotEqual => (Cond::Mi, Some(Cond::Gt)),
            FloatCC::UnorderedOrEqual => (Cond::Eq, Some(Cond::Vs)),
            FloatCC::LessThan => (Cond::Mi, None),
            FloatCC::LessThanOrEqual => (Cond::Ls, None),
            FloatCC::GreaterThan => (Cond::Gt, None),
            FloatCC::GreaterThanOrEqual => (Cond::Ge, None),
            FloatCC::UnorderedOrLessThan => (Cond::Lt, None),
            FloatCC::UnorderedOrLessThanOrEqual => (Cond::Le, None),
            FloatCC::UnorderedOrGreaterThan => (Cond::Hi, None),
            FloatCC::UnorderedOrGreaterThanOrEqual => (Cond::Pl, None),
        }
    }

    /// Assembly suffix for this condition.
    pub fn suffix(self) -> &'static str {
        match self {
            Cond::Eq => "eq",
            Cond::Ne => "ne",
            Cond::Hs => "hs",
            Cond::Lo => "lo",
            Cond::Mi => "mi",
            Cond::Pl => "pl",
            Cond::Vs => "vs",
            Cond::Vc => "vc",
            Cond::Hi => "hi",
            Cond::Ls => "ls",
            Cond::Ge => "ge",
            Cond::Lt => "lt",
            Cond::Gt => "gt",
            Cond::Le => "le",
        }
    }
}

/// An A32 "modified immediate": an 8-bit value rotated right by an even
/// amount.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModImm {
    /// The 8-bit value.
    imm8: u8,
    /// The rotation divided by two (0..16).
    rot: u8,
}

impl ModImm {
    /// Try to encode `value` as a modified immediate.
    pub fn maybe_from_u32(value: u32) -> Option<ModImm> {
        for rot in 0..16u8 {
            // The value is `imm8 ror (2 * rot)`, so `imm8 = value rol (2 * rot)`.
            let v = value.rotate_left(2 * u32::from(rot));
            if v <= 0xff {
                return Some(ModImm { imm8: v as u8, rot });
            }
        }
        None
    }

    /// The 12-bit encoding (`rot:imm8`).
    pub fn bits(self) -> u32 {
        (u32::from(self.rot) << 8) | u32::from(self.imm8)
    }

    /// The value represented.
    pub fn value(self) -> u32 {
        u32::from(self.imm8).rotate_right(2 * u32::from(self.rot))
    }
}

impl fmt::Display for ModImm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.value())
    }
}

/// A shift applied to the register operand of a data-processing instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShiftOpAndAmt {
    /// The shift operation.
    pub op: ShiftOp,
    /// The amount, in `0..32`. An amount of zero (with `Lsl`) means "no
    /// shift". `Lsr` and `Asr` by 32 are encoded as amount 0.
    pub amt: u8,
}

impl ShiftOpAndAmt {
    /// No shift.
    pub fn none() -> Self {
        ShiftOpAndAmt {
            op: ShiftOp::Lsl,
            amt: 0,
        }
    }

    /// A shift by an immediate amount.
    pub fn new(op: ShiftOp, amt: u8) -> Self {
        assert!(amt > 0 && amt < 32 || amt == 0 && op == ShiftOp::Lsl);
        ShiftOpAndAmt { op, amt }
    }

    /// Bits 11:4 of the instruction (`imm5:type:0`).
    pub fn bits(self) -> u32 {
        (u32::from(self.amt) << 7) | (shift_op_bits(self.op) << 5)
    }

    /// Whether this shift is the identity.
    pub fn is_none(self) -> bool {
        self.amt == 0
    }
}

/// The 2-bit `type` field for a shift operation.
pub fn shift_op_bits(op: ShiftOp) -> u32 {
    match op {
        ShiftOp::Lsl => 0b00,
        ShiftOp::Lsr => 0b01,
        ShiftOp::Asr => 0b10,
        ShiftOp::Ror => 0b11,
    }
}

/// Assembly mnemonic for a shift operation.
pub fn shift_op_name(op: ShiftOp) -> &'static str {
    match op {
        ShiftOp::Lsl => "lsl",
        ShiftOp::Lsr => "lsr",
        ShiftOp::Asr => "asr",
        ShiftOp::Ror => "ror",
    }
}

impl fmt::Display for ShiftOpAndAmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_none() {
            Ok(())
        } else {
            write!(f, ", {} #{}", shift_op_name(self.op), self.amt)
        }
    }
}

/// A memory addressing mode.
#[derive(Clone, Copy, Debug)]
pub enum AMode {
    /// A register plus a signed offset. Offsets that do not fit the
    /// instruction's immediate field are materialized in `ip` at emission.
    RegOffset {
        /// The base register.
        base: Reg,
        /// The offset in bytes.
        offset: i32,
    },
    /// A register plus another register shifted left: `[base, index, lsl
    /// #shift]`. Only valid for word/byte loads and stores; for others the
    /// address is computed in `ip` first.
    RegReg {
        /// The base register.
        base: Reg,
        /// The index register.
        index: Reg,
        /// Left shift amount applied to the index.
        shift: u8,
    },
    /// An offset from the stack pointer.
    SPOffset(i32),
    /// An offset into the slot area (spill slots and stack slots), which sits
    /// just above the outgoing argument area. Resolved at emission time.
    SlotOffset(i32),
    /// An offset into the incoming argument area, measured from the top of
    /// that area. Resolved at emission time.
    IncomingArg(i32),
}

impl AMode {
    /// Add the registers referenced by this addressing mode to `collector`.
    pub(crate) fn get_operands(&mut self, collector: &mut impl OperandVisitor) {
        match self {
            AMode::RegOffset { base, .. } => collector.reg_use(base),
            AMode::RegReg { base, index, .. } => {
                collector.reg_use(base);
                collector.reg_use(index);
            }
            AMode::SPOffset(..) | AMode::SlotOffset(..) | AMode::IncomingArg(..) => {}
        }
    }

    /// Resolve stack-relative modes into a base register and an offset.
    /// Returns `None` for `RegReg`.
    pub(crate) fn resolve(&self, state: &EmitState) -> Option<(Reg, i64)> {
        use crate::isa::arm32::inst::regs::stack_reg;
        match *self {
            AMode::RegOffset { base, offset } => Some((base, i64::from(offset))),
            AMode::RegReg { .. } => None,
            AMode::SPOffset(off) => Some((stack_reg(), i64::from(off))),
            AMode::SlotOffset(off) => {
                let fl = state.frame_layout();
                Some((
                    stack_reg(),
                    i64::from(off) + i64::from(fl.outgoing_args_size),
                ))
            }
            AMode::IncomingArg(off) => {
                let fl = state.frame_layout();
                let sp_offset = fl.tail_args_size
                    + fl.setup_area_size
                    + fl.clobber_size
                    + fl.fixed_frame_storage_size
                    + fl.outgoing_args_size;
                Some((stack_reg(), i64::from(sp_offset) - i64::from(off)))
            }
        }
    }

    /// Pretty-print, resolving stack offsets with the given state.
    pub(crate) fn pretty_print(&self, state: &EmitState) -> String {
        match self {
            AMode::RegReg { base, index, shift } => {
                if *shift == 0 {
                    format!("[{}, {}]", show_reg(*base), show_reg(*index))
                } else {
                    format!(
                        "[{}, {}, lsl #{}]",
                        show_reg(*base),
                        show_reg(*index),
                        shift
                    )
                }
            }
            _ => {
                let (base, off) = self.resolve(state).unwrap();
                if off == 0 {
                    format!("[{}]", show_reg(base))
                } else {
                    format!("[{}, #{}]", show_reg(base), off)
                }
            }
        }
    }
}

impl From<StackAMode> for AMode {
    fn from(stack: StackAMode) -> AMode {
        match stack {
            StackAMode::IncomingArg(offset, stack_args_size) => {
                AMode::IncomingArg(i32::try_from(i64::from(stack_args_size) - offset).unwrap())
            }
            StackAMode::OutgoingArg(offset) => AMode::SPOffset(i32::try_from(offset).unwrap()),
            StackAMode::Slot(offset) => AMode::SlotOffset(i32::try_from(offset).unwrap()),
        }
    }
}

/// The 8-bit VFPv3 immediate encoding (`abcdefgh`) of a floating-point
/// constant, given its raw bits, if it is representable: the value must be
/// `±(16 + efgh) / 16 * 2^n` for `n` in `[-3, 4]`.
pub fn vfp_imm8(bits: u64, is64: bool) -> Option<u8> {
    let (exp_bits, frac_bits) = if is64 { (11u32, 52u32) } else { (8, 23) };
    let total = 1 + exp_bits + frac_bits;
    if (total < 64 && bits >> total != 0) || bits & ((1u64 << (frac_bits - 4)) - 1) != 0 {
        return None;
    }
    let sign = bits >> (total - 1);
    let exp = (bits >> frac_bits) & ((1 << exp_bits) - 1);
    let frac = (bits >> (frac_bits - 4)) & 0xf;
    // The exponent must be `NOT(b) : Replicate(b, exp_bits - 3) : cd`.
    let b = (exp >> (exp_bits - 2)) & 1;
    if exp >> (exp_bits - 1) == b {
        return None;
    }
    let reps = (exp >> 2) & ((1 << (exp_bits - 3)) - 1);
    if reps != if b == 1 { (1 << (exp_bits - 3)) - 1 } else { 0 } {
        return None;
    }
    let cd = exp & 3;
    Some(((sign << 7) | (b << 6) | (cd << 4) | frac) as u8)
}

/// The value of an 8-bit VFPv3 floating-point immediate.
pub fn vfp_expand_imm8(imm: u8) -> f64 {
    let imm = u64::from(imm);
    let b = (imm >> 6) & 1;
    let exp = ((b ^ 1) << 10) | (if b == 1 { 0xff } else { 0 } << 2) | ((imm >> 4) & 3);
    f64::from_bits(((imm >> 7) << 63) | (exp << 52) | ((imm & 0xf) << 48))
}
