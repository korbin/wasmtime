//! This module defines ARMv7-specific machine instruction types.

use crate::binemit::{Addend, CodeOffset, Reloc};
use crate::ir::types::{F32, F64, I8, I8X16, I16, I32, I64};
use crate::ir::{MemFlagsData, Type};
use crate::isa::FunctionAlignment;
use crate::machinst::*;
use crate::{CodegenError, CodegenResult};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use regalloc2::{OperandKind, OperandPos, PRegSet, RegClass};
use smallvec::SmallVec;

pub mod args;
pub use self::args::*;
pub mod regs;
pub use self::regs::*;
pub mod emit;
pub use self::emit::*;
pub mod neon;
pub mod unwind;

#[cfg(test)]
mod emit_tests;

pub use crate::ir::AtomicRmwOp;
pub use crate::isa::arm32::lower::isle::generated_code::{
    AluOp, BitOp, CmpOp, FpuCvtOp, FpuOp1, FpuOp2, FpuOp3, FpuRoundMode, FpuSize, LoadOp,
    MInst as Inst, MovOp, MulAccOp, MulLongOp, ShiftOp, StoreOp,
};
pub use crate::isa::arm32::lower::isle::generated_code::{
    VecAluOp, VecMiscOp, VecNarrowOp, VecShiftOp,
};

use crate::isa::arm32::abi::Arm32MachineDeps;

/// Additional information for `return_call[_ind]` instructions, left out of
/// line to lower the size of the `Inst` enum.
#[derive(Clone, Debug)]
pub struct ReturnCallInfo<T> {
    /// Where this call is going.
    pub dest: T,
    /// Arguments to the call instruction.
    pub uses: CallArgList,
    /// The size of the new stack frame's stack arguments.
    pub new_stack_arg_size: u32,
}

impl FpuCvtOp {
    /// The register sizes of the source and destination operands; an integer
    /// is held in an S register, like an `f32`.
    pub fn sizes(self) -> (FpuSize, FpuSize) {
        use FpuSize::{F32, F64};
        match self {
            FpuCvtOp::F32ToF64 => (F32, F64),
            FpuCvtOp::F64ToF32 => (F64, F32),
            FpuCvtOp::F32ToS32 | FpuCvtOp::F32ToU32 => (F32, F32),
            FpuCvtOp::F64ToS32 | FpuCvtOp::F64ToU32 => (F64, F32),
            FpuCvtOp::S32ToF32 | FpuCvtOp::U32ToF32 => (F32, F32),
            FpuCvtOp::S32ToF64 | FpuCvtOp::U32ToF64 => (F32, F64),
        }
    }
}

impl FpuSize {
    /// Whether this is double precision.
    pub fn is_f64(self) -> bool {
        self == FpuSize::F64
    }

    /// The size from a type.
    pub fn from_ty(ty: Type) -> FpuSize {
        match ty {
            F32 => FpuSize::F32,
            F64 => FpuSize::F64,
            _ => panic!("not a float type: {ty}"),
        }
    }
}

impl Inst {
    /// Generic constructor for a load of a type that lives in a single
    /// register.
    pub fn gen_load(into_reg: Writable<Reg>, mem: AMode, ty: Type, flags: MemFlagsData) -> Inst {
        match ty {
            I8 => Inst::Load {
                op: LoadOp::Ldrb,
                rd: into_reg,
                mem,
                flags,
            },
            I16 => Inst::Load {
                op: LoadOp::Ldrh,
                rd: into_reg,
                mem,
                flags,
            },
            I32 => Inst::Load {
                op: LoadOp::Ldr,
                rd: into_reg,
                mem,
                flags,
            },
            F32 | F64 => Inst::FpuLoad {
                size: FpuSize::from_ty(ty),
                rd: into_reg,
                mem,
                flags,
            },
            _ if ty.is_vector() && ty.bits() == 128 => Inst::VecLoad {
                rd: into_reg,
                mem,
                flags,
            },
            _ => panic!("gen_load: unsupported type {ty}"),
        }
    }

    /// Generic constructor for a store of a type that lives in a single
    /// register.
    pub fn gen_store(mem: AMode, from_reg: Reg, ty: Type, flags: MemFlagsData) -> Inst {
        match ty {
            I8 => Inst::Store {
                op: StoreOp::Strb,
                rt: from_reg,
                mem,
                flags,
            },
            I16 => Inst::Store {
                op: StoreOp::Strh,
                rt: from_reg,
                mem,
                flags,
            },
            I32 => Inst::Store {
                op: StoreOp::Str,
                rt: from_reg,
                mem,
                flags,
            },
            F32 | F64 => Inst::FpuStore {
                size: FpuSize::from_ty(ty),
                rt: from_reg,
                mem,
                flags,
            },
            _ if ty.is_vector() && ty.bits() == 128 => Inst::VecStore {
                rt: from_reg,
                mem,
                flags,
            },
            _ => panic!("gen_store: unsupported type {ty}"),
        }
    }

    /// Load a 32-bit constant into a register, using `movw`/`movt` or a
    /// single `mov`/`mvn` when possible.
    pub fn load_constant32(rd: Writable<Reg>, value: u32) -> SmallVec<[Inst; 2]> {
        let mut insts = SmallVec::new();
        if let Some(imm) = ModImm::maybe_from_u32(value) {
            insts.push(Inst::MovImm {
                op: MovOp::Mov,
                rd,
                imm,
            });
        } else if let Some(imm) = ModImm::maybe_from_u32(!value) {
            insts.push(Inst::MovImm {
                op: MovOp::Mvn,
                rd,
                imm,
            });
        } else {
            insts.push(Inst::MovW {
                rd,
                imm: value as u16,
            });
            if value >> 16 != 0 {
                insts.push(Inst::MovT {
                    rd,
                    rn: rd.to_reg(),
                    imm: (value >> 16) as u16,
                });
            }
        }
        insts
    }
}

//=============================================================================
// Register operand collection.

/// The register list of a lane load or store: `qN.dK[x]`, or `qN.dK` for a
/// 64-bit lane.
fn lane_operand(size: u8, q: String, lane: u8) -> String {
    if size == 3 {
        format!("{q}.d{lane}")
    } else {
        let per_d = 8 >> size;
        format!("{q}.d{}[{}]", lane / per_d, lane % per_d)
    }
}

/// Only `d0`-`d15` have single-precision halves (`d(n)` holds `s(2n)`), so a
/// float register that an instruction accesses as an S register must be one
/// of them; the register environment may also hand out `d16`-`d31` (see
/// `create_reg_env` in `abi.rs`).
const SREG_LIMIT: usize = 16;

fn s_operand(
    collector: &mut impl OperandVisitor,
    reg: &mut Reg,
    kind: OperandKind,
    pos: OperandPos,
) {
    collector.reg_limited(reg, SREG_LIMIT, kind, pos);
}

/// A use of a float register accessed as an S register.
fn s_use(collector: &mut impl OperandVisitor, reg: &mut Reg) {
    s_operand(collector, reg, OperandKind::Use, OperandPos::Early);
}

/// A def of a float register accessed as an S register.
fn s_def(collector: &mut impl OperandVisitor, reg: &mut Writable<Reg>) {
    s_operand(collector, reg.reg_mut(), OperandKind::Def, OperandPos::Late);
}

/// An early def of a float register accessed as an S register.
fn s_early_def(collector: &mut impl OperandVisitor, reg: &mut Writable<Reg>) {
    s_operand(
        collector,
        reg.reg_mut(),
        OperandKind::Def,
        OperandPos::Early,
    );
}

/// A use of a float register holding a value of the given size.
fn f_use(collector: &mut impl OperandVisitor, reg: &mut Reg, size: FpuSize) {
    if size.is_f64() {
        collector.reg_use(reg);
    } else {
        s_use(collector, reg);
    }
}

/// A def of a float register holding a value of the given size.
fn f_def(collector: &mut impl OperandVisitor, reg: &mut Writable<Reg>, size: FpuSize) {
    if size.is_f64() {
        collector.reg_def(reg);
    } else {
        s_def(collector, reg);
    }
}

/// An early def of a float register holding a value of the given size.
fn f_early_def(collector: &mut impl OperandVisitor, reg: &mut Writable<Reg>, size: FpuSize) {
    if size.is_f64() {
        collector.reg_early_def(reg);
    } else {
        s_early_def(collector, reg);
    }
}

fn arm32_get_operands(inst: &mut Inst, collector: &mut impl OperandVisitor) {
    match inst {
        Inst::Nop0 | Inst::Nop4 => {}
        Inst::AluRRR { rd, rn, rm, .. } => {
            collector.reg_use(rn);
            collector.reg_use(rm);
            collector.reg_def(rd);
        }
        Inst::AluRRRShift { rd, rn, rm, rs, .. } => {
            collector.reg_use(rn);
            collector.reg_use(rm);
            collector.reg_use(rs);
            collector.reg_def(rd);
        }
        Inst::AluRRImm { rd, rn, .. } => {
            collector.reg_use(rn);
            collector.reg_def(rd);
        }
        Inst::MovRR { rd, rm, .. } => {
            collector.reg_use(rm);
            collector.reg_def(rd);
        }
        Inst::MovRRShift { rd, rm, rs, .. } => {
            collector.reg_use(rm);
            collector.reg_use(rs);
            collector.reg_def(rd);
        }
        Inst::MovImm { rd, .. } | Inst::MovW { rd, .. } => {
            collector.reg_def(rd);
        }
        Inst::MovT { rd, rn, .. } => {
            collector.reg_use(rn);
            collector.reg_reuse_def(rd, 0);
        }
        Inst::Mov { rd, rm } => {
            collector.reg_use(rm);
            collector.reg_def(rd);
        }
        Inst::CmpRR { rn, rm, .. } | Inst::CCmpRR { rn, rm, .. } => {
            collector.reg_use(rn);
            collector.reg_use(rm);
        }
        Inst::CmpRImm { rn, .. } | Inst::CCmpRImm { rn, .. } => {
            collector.reg_use(rn);
        }
        Inst::CSel { rd, rn, rm, .. } => {
            collector.reg_use(rn);
            collector.reg_use(rm);
            collector.reg_reuse_def(rd, 1);
        }
        Inst::CSelImm { rd, rm, .. } => {
            collector.reg_use(rm);
            collector.reg_reuse_def(rd, 0);
        }
        Inst::CSet { rd, .. } | Inst::CSetm { rd, .. } => {
            collector.reg_def(rd);
        }
        Inst::Mul { rd, rn, rm } => {
            collector.reg_use(rn);
            collector.reg_use(rm);
            collector.reg_def(rd);
        }
        Inst::MulAcc { rd, rn, rm, ra, .. } => {
            collector.reg_use(rn);
            collector.reg_use(rm);
            collector.reg_use(ra);
            collector.reg_def(rd);
        }
        Inst::MulLong {
            rd_lo,
            rd_hi,
            rn,
            rm,
            ..
        } => {
            collector.reg_use(rn);
            collector.reg_use(rm);
            collector.reg_def(rd_lo);
            collector.reg_def(rd_hi);
        }
        Inst::Umaal {
            rd_lo,
            rd_hi,
            ra_lo,
            ra_hi,
            rn,
            rm,
        } => {
            collector.reg_use(ra_lo);
            collector.reg_use(ra_hi);
            collector.reg_use(rn);
            collector.reg_use(rm);
            collector.reg_reuse_def(rd_lo, 0);
            collector.reg_reuse_def(rd_hi, 1);
        }
        Inst::Div { rd, rn, rm, .. } => {
            collector.reg_use(rn);
            collector.reg_use(rm);
            collector.reg_def(rd);
        }
        Inst::UDivMod32 {
            rd_quot,
            rd_rem,
            rn,
            rm,
        } => {
            collector.reg_use(rn);
            collector.reg_use(rm);
            collector.reg_early_def(rd_quot);
            collector.reg_early_def(rd_rem);
        }
        Inst::UDivMod64 {
            quot_lo,
            quot_hi,
            rem_lo,
            rem_hi,
            n_lo,
            n_hi,
            d_lo,
            d_hi,
            ..
        } => {
            collector.reg_use(n_lo);
            collector.reg_use(n_hi);
            collector.reg_use(d_lo);
            collector.reg_use(d_hi);
            collector.reg_early_def(quot_lo);
            collector.reg_early_def(quot_hi);
            collector.reg_early_def(rem_lo);
            collector.reg_early_def(rem_hi);
        }
        Inst::BitRR { rd, rm, .. } => {
            collector.reg_use(rm);
            collector.reg_def(rd);
        }
        Inst::Shift64 {
            rd_lo,
            rd_hi,
            rn_lo,
            rn_hi,
            amt,
            tmp,
            ..
        } => {
            collector.reg_use(rn_lo);
            collector.reg_use(rn_hi);
            collector.reg_use(amt);
            collector.reg_early_def(rd_lo);
            collector.reg_early_def(rd_hi);
            collector.reg_early_def(tmp);
        }
        Inst::Rot64 {
            rd_lo,
            rd_hi,
            rn_lo,
            rn_hi,
            amt,
            t1,
            t2,
            ..
        } => {
            collector.reg_use(rn_lo);
            collector.reg_use(rn_hi);
            collector.reg_use(amt);
            collector.reg_early_def(rd_lo);
            collector.reg_early_def(rd_hi);
            collector.reg_early_def(t1);
            collector.reg_early_def(t2);
        }
        Inst::Load { rd, mem, .. } => {
            mem.get_operands(collector);
            collector.reg_def(rd);
        }
        Inst::Store { rt, mem, .. } => {
            mem.get_operands(collector);
            collector.reg_use(rt);
        }
        Inst::LoadAddr { rd, mem } => {
            mem.get_operands(collector);
            collector.reg_def(rd);
        }
        Inst::FpuRR { size, rd, rn, .. } | Inst::FpuRint { size, rd, rn, .. } => {
            f_use(collector, rn, *size);
            f_def(collector, rd, *size);
        }
        Inst::VecRint { rd, rn, tmp, .. } => {
            collector.reg_use(rn);
            collector.reg_def(rd);
            s_early_def(collector, tmp);
        }
        Inst::FpuMovImm { size, rd, .. } | Inst::FpuLoadConst { size, rd, .. } => {
            f_def(collector, rd, *size);
        }
        Inst::FpuZero { rd } | Inst::VecLoadConst { rd, .. } | Inst::VecZero { rd } => {
            collector.reg_def(rd);
        }
        Inst::FpuRRR {
            size, rd, rn, rm, ..
        } => {
            f_use(collector, rn, *size);
            f_use(collector, rm, *size);
            f_def(collector, rd, *size);
        }
        Inst::FpuRRRR {
            size,
            rd,
            ra,
            rn,
            rm,
            ..
        } => {
            // A reused input takes the def's register, which can't be
            // limited; emission handles `f32` accumulators in `d16`-`d31`.
            collector.reg_use(ra);
            f_use(collector, rn, *size);
            f_use(collector, rm, *size);
            collector.reg_reuse_def(rd, 0);
        }
        Inst::FpuCmp { size, rn, rm } => {
            f_use(collector, rn, *size);
            f_use(collector, rm, *size);
        }
        Inst::FpuCvt { op, rd, rn } => {
            let (src, dst) = op.sizes();
            f_use(collector, rn, src);
            f_def(collector, rd, dst);
        }
        Inst::FpuCSel { rd, rn, rm, .. } => {
            // Moves whole D registers, for either precision.
            collector.reg_use(rn);
            collector.reg_use(rm);
            collector.reg_reuse_def(rd, 1);
        }
        Inst::FpuMinMax {
            size,
            rd,
            rn,
            rm,
            tmp,
            ..
        } => {
            f_use(collector, rn, *size);
            f_use(collector, rm, *size);
            f_early_def(collector, rd, *size);
            collector.reg_early_def(tmp);
        }
        Inst::FpuRound {
            size,
            rd,
            rn,
            tmp1,
            tmp2,
            ftmp,
            ..
        } => {
            f_use(collector, rn, *size);
            f_early_def(collector, rd, *size);
            collector.reg_early_def(tmp1);
            collector.reg_early_def(tmp2);
            f_early_def(collector, ftmp, *size);
        }
        Inst::FpuLoad { size, rd, mem, .. } => {
            mem.get_operands(collector);
            f_def(collector, rd, *size);
        }
        Inst::FpuStore { size, rt, mem, .. } => {
            mem.get_operands(collector);
            f_use(collector, rt, *size);
        }
        Inst::FpuLoadUnaligned { rd, mem, .. } => {
            mem.get_operands(collector);
            collector.reg_def(rd);
        }
        Inst::FpuStoreUnaligned32 { rt, mem, .. } => {
            mem.get_operands(collector);
            collector.reg_use(rt);
        }
        Inst::MovToFpu32 { rd, rn } => {
            collector.reg_use(rn);
            s_def(collector, rd);
        }
        Inst::MovFromFpu32 { rd, rn } => {
            s_use(collector, rn);
            collector.reg_def(rd);
        }
        Inst::MovToFpu64 { rd, lo, hi } => {
            collector.reg_use(lo);
            collector.reg_use(hi);
            collector.reg_def(rd);
        }
        Inst::MovFromFpu64 { rd_lo, rd_hi, rn } => {
            collector.reg_use(rn);
            collector.reg_def(rd_lo);
            collector.reg_def(rd_hi);
        }
        Inst::FpuMove { rd, rn } => {
            collector.reg_use(rn);
            collector.reg_def(rd);
        }
        Inst::Jump { .. } | Inst::CondBr { .. } | Inst::TrapIf { .. } => {}
        Inst::Udf { .. } | Inst::Bkpt => {}
        Inst::BrTable { index, .. } => {
            collector.reg_use(index);
        }
        Inst::Call { info } => {
            let CallInfo { uses, defs, .. } = &mut **info;
            for CallArgPair { vreg, preg } in uses {
                collector.reg_fixed_use(vreg, *preg);
            }
            for CallRetPair { vreg, location } in defs {
                match location {
                    RetLocation::Reg(preg, ..) => collector.reg_fixed_def(vreg, *preg),
                    RetLocation::Stack(..) => collector.any_def(vreg),
                }
            }
            collector.reg_clobbers(info.clobbers);
            if let Some(try_call_info) = &mut info.try_call_info {
                try_call_info.collect_operands(collector);
            }
        }
        Inst::CallInd { info } => {
            let CallInfo {
                dest, uses, defs, ..
            } = &mut **info;
            collector.reg_use(dest);
            for CallArgPair { vreg, preg } in uses {
                collector.reg_fixed_use(vreg, *preg);
            }
            for CallRetPair { vreg, location } in defs {
                match location {
                    RetLocation::Reg(preg, ..) => collector.reg_fixed_def(vreg, *preg),
                    RetLocation::Stack(..) => collector.any_def(vreg),
                }
            }
            collector.reg_clobbers(info.clobbers);
            if let Some(try_call_info) = &mut info.try_call_info {
                try_call_info.collect_operands(collector);
            }
        }
        Inst::ReturnCall { info } => {
            for CallArgPair { vreg, preg } in &mut info.uses {
                collector.reg_fixed_use(vreg, *preg);
            }
        }
        Inst::ReturnCallInd { info } => {
            collector.reg_use(&mut info.dest);
            for CallArgPair { vreg, preg } in &mut info.uses {
                collector.reg_fixed_use(vreg, *preg);
            }
        }
        Inst::Args { args } => {
            for ArgPair { vreg, preg } in args {
                collector.reg_fixed_def(vreg, *preg);
            }
        }
        Inst::Rets { rets } => {
            for RetPair { vreg, preg } in rets {
                collector.reg_fixed_use(vreg, *preg);
            }
        }
        Inst::Ret => {}
        Inst::LoadExtName { rd, .. }
        | Inst::LoadExtNameNear { rd, .. }
        | Inst::LabelAddress { rd, .. } => {
            collector.reg_def(rd);
        }
        Inst::MovFromPReg { rd, rm } => {
            debug_assert!([fp_reg(), stack_reg()].contains(&Reg::from(*rm)));
            collector.reg_def(rd);
        }
        Inst::Fence => {}
        Inst::LoadExclusive { rd, addr, .. } => {
            collector.reg_use(addr);
            collector.reg_def(rd);
        }
        Inst::AtomicRmwLoop {
            rd,
            addr,
            operand,
            tmp1,
            tmp2,
            ..
        } => {
            collector.reg_use(addr);
            collector.reg_use(operand);
            collector.reg_early_def(rd);
            collector.reg_early_def(tmp1);
            collector.reg_early_def(tmp2);
        }
        Inst::AtomicCasLoop {
            rd,
            addr,
            expected,
            replacement,
            tmp,
            ..
        } => {
            collector.reg_use(addr);
            collector.reg_use(expected);
            collector.reg_use(replacement);
            collector.reg_early_def(rd);
            collector.reg_early_def(tmp);
        }
        Inst::AtomicRmwLoop64 {
            rd_lo,
            rd_hi,
            addr,
            operand_lo,
            operand_hi,
            ..
        } => {
            collector.reg_fixed_use(addr, rreg(7));
            collector.reg_fixed_use(operand_lo, rreg(2));
            collector.reg_fixed_use(operand_hi, rreg(3));
            collector.reg_fixed_def(rd_lo, rreg(0));
            collector.reg_fixed_def(rd_hi, rreg(1));
            collector.reg_clobbers(PRegSet::empty().with(preg(4)).with(preg(5)).with(preg(6)));
        }
        Inst::LoadAcquire { rd, addr, .. } => {
            collector.reg_use(addr);
            collector.reg_def(rd);
        }
        Inst::StoreRelease { rt, addr, .. } => {
            collector.reg_use(addr);
            collector.reg_use(rt);
        }
        Inst::AtomicLoad64 {
            rd_lo, rd_hi, addr, ..
        } => {
            collector.reg_use(addr);
            collector.reg_fixed_def(rd_lo, rreg(0));
            collector.reg_fixed_def(rd_hi, rreg(1));
        }
        Inst::AtomicCasLoop64 {
            rd_lo,
            rd_hi,
            addr,
            expected_lo,
            expected_hi,
            replacement_lo,
            replacement_hi,
            ..
        } => {
            collector.reg_fixed_use(addr, rreg(7));
            collector.reg_fixed_use(expected_lo, rreg(2));
            collector.reg_fixed_use(expected_hi, rreg(3));
            collector.reg_fixed_use(replacement_lo, rreg(4));
            collector.reg_fixed_use(replacement_hi, rreg(5));
            collector.reg_fixed_def(rd_lo, rreg(0));
            collector.reg_fixed_def(rd_hi, rreg(1));
            collector.reg_clobbers(PRegSet::empty().with(preg(6)));
        }
        Inst::Push { .. }
        | Inst::Pop { .. }
        | Inst::VPush { .. }
        | Inst::VPop { .. }
        | Inst::SpAdjust { .. }
        | Inst::StackProbeLoop { .. } => {}
        Inst::Unwind { .. } => {}
        Inst::DummyUse { reg } => {
            collector.reg_use(reg);
        }
        Inst::EmitIsland { .. } | Inst::SequencePoint => {}

        Inst::VecMove { rd, rn } => {
            collector.reg_use(rn);
            collector.reg_def(rd);
        }
        Inst::VecRRR { rd, rn, rm, .. } | Inst::VecMulLong { rd, rn, rm, .. } => {
            collector.reg_use(rn);
            collector.reg_use(rm);
            collector.reg_def(rd);
        }
        // Emitted as two instructions, one per D half.
        Inst::VecNarrow { rd, rn, rm, .. } | Inst::VecPairwise { rd, rn, rm, .. } => {
            collector.reg_use(rn);
            collector.reg_use(rm);
            collector.reg_early_def(rd);
        }
        Inst::VecExt { rd, rn, rm, .. } => {
            collector.reg_use(rn);
            collector.reg_use(rm);
            collector.reg_def(rd);
        }
        Inst::VecBsl { rd, ri, rn, rm } => {
            collector.reg_use(ri);
            collector.reg_use(rn);
            collector.reg_use(rm);
            collector.reg_reuse_def(rd, 0);
        }
        Inst::VecSraSelf { rd, rn, .. } => {
            collector.reg_use(rn);
            collector.reg_reuse_def(rd, 0);
        }
        Inst::VecRR { rd, rn, .. }
        | Inst::VecShiftImm { rd, rn, .. }
        | Inst::VecWiden { rd, rn, .. }
        | Inst::VecDupLane { rd, rn, .. }
        | Inst::VecDup { rd, rn, .. }
        | Inst::VecMovFromLane { rd, rn, .. }
        | Inst::VecToFpu { rd, rn, .. }
        | Inst::VecOrrHalves { rd, rn } => {
            collector.reg_use(rn);
            collector.reg_def(rd);
        }
        Inst::VecMovToLane { rd, ri, rn, .. } | Inst::VecFromFpu { rd, ri, rn, .. } => {
            collector.reg_use(ri);
            collector.reg_use(rn);
            collector.reg_reuse_def(rd, 0);
        }
        Inst::VecFromGprs { rd, r0, r1, r2, r3 } => {
            collector.reg_use(r0);
            collector.reg_use(r1);
            collector.reg_use(r2);
            collector.reg_use(r3);
            collector.reg_def(rd);
        }
        Inst::VecToFpu32 { rd, rn, tmp, .. } => {
            collector.reg_use(rn);
            s_def(collector, rd);
            s_early_def(collector, tmp);
        }
        Inst::VecFromFpu32 {
            rd, ri, rn, tmp, ..
        } => {
            collector.reg_use(ri);
            s_use(collector, rn);
            collector.reg_reuse_def(rd, 0);
            s_early_def(collector, tmp);
        }
        Inst::VecLoadLane { rd, ri, mem, .. } => {
            // The reused input comes first, so its index doesn't depend on
            // the addressing mode.
            collector.reg_use(ri);
            mem.get_operands(collector);
            collector.reg_reuse_def(rd, 0);
        }
        Inst::VecStoreLane { rt, mem, .. } => {
            mem.get_operands(collector);
            collector.reg_use(rt);
        }
        Inst::VecLoad { rd, mem, .. }
        | Inst::VecLoadLow { rd, mem, .. }
        | Inst::VecLoadSplat { rd, mem, .. } => {
            mem.get_operands(collector);
            collector.reg_def(rd);
        }
        Inst::VecStore { rt, mem, .. } => {
            mem.get_operands(collector);
            collector.reg_use(rt);
        }
        Inst::VecTbl {
            rd,
            tbl1,
            tbl2,
            idx,
        } => {
            match tbl2 {
                None => collector.reg_use(tbl1),
                Some(tbl2) => {
                    // The table must occupy consecutive D registers.
                    collector.reg_fixed_use(tbl1, qreg(14));
                    collector.reg_fixed_use(tbl2, qreg(15));
                }
            }
            collector.reg_use(idx);
            collector.reg_early_def(rd);
        }
        Inst::VecFpuRRR {
            rd, rn, rm, t1, t2, ..
        } => {
            collector.reg_use(rn);
            collector.reg_use(rm);
            collector.reg_def(rd);
            s_early_def(collector, t1);
            s_early_def(collector, t2);
        }
        Inst::VecFpuRR { rd, rn, t1, .. } => {
            collector.reg_use(rn);
            collector.reg_def(rd);
            s_early_def(collector, t1);
        }
        Inst::VecFpuFma {
            rd,
            ra,
            rn,
            rm,
            t1,
            t2,
            t3,
            ..
        } => {
            collector.reg_use(ra);
            collector.reg_use(rn);
            collector.reg_use(rm);
            collector.reg_def(rd);
            s_early_def(collector, t1);
            s_early_def(collector, t2);
            s_early_def(collector, t3);
        }
        Inst::VecFpuCmp {
            rd,
            rn,
            rm,
            t1,
            t2,
            gtmp,
            ..
        }
        | Inst::VecFpuMinMax {
            rd,
            rn,
            rm,
            t1,
            t2,
            gtmp,
            ..
        } => {
            collector.reg_use(rn);
            collector.reg_use(rm);
            collector.reg_def(rd);
            s_early_def(collector, t1);
            s_early_def(collector, t2);
            collector.reg_early_def(gtmp);
        }
        Inst::PopcntNeon {
            rd, lo, hi, tmp, ..
        } => {
            collector.reg_use(lo);
            collector.reg_use(hi);
            collector.reg_def(rd);
            collector.reg_early_def(tmp);
        }
        Inst::VecMulI64 { rd, rn, rm, tb } => {
            collector.reg_use(rn);
            collector.reg_use(rm);
            collector.reg_early_def(rd);
            collector.reg_early_def(tb);
        }
        Inst::VecCmpF32 {
            rd, rn, rm, ta, tb, ..
        }
        | Inst::VecMinMaxF32 {
            rd, rn, rm, ta, tb, ..
        } => {
            collector.reg_use(rn);
            collector.reg_use(rm);
            collector.reg_early_def(rd);
            collector.reg_early_def(ta);
            collector.reg_early_def(tb);
        }
        Inst::VecRoundF64 {
            rd,
            rn,
            qa,
            qb,
            qc,
            t,
            gtmp,
            ..
        } => {
            collector.reg_use(rn);
            collector.reg_early_def(rd);
            collector.reg_early_def(qa);
            collector.reg_early_def(qb);
            collector.reg_early_def(qc);
            collector.reg_early_def(t);
            collector.reg_early_def(gtmp);
        }
        Inst::VecRoundF32 {
            rd,
            rn,
            qa,
            qb,
            t1,
            t2,
            gtmp,
            ..
        } => {
            collector.reg_use(rn);
            collector.reg_early_def(rd);
            collector.reg_early_def(qa);
            collector.reg_early_def(qb);
            s_early_def(collector, t1);
            s_early_def(collector, t2);
            collector.reg_early_def(gtmp);
        }
        Inst::VecFpuRound {
            rd,
            rn,
            t1,
            t2,
            t3,
            gtmp1,
            gtmp2,
            ..
        } => {
            collector.reg_use(rn);
            collector.reg_def(rd);
            s_early_def(collector, t1);
            s_early_def(collector, t2);
            s_early_def(collector, t3);
            collector.reg_early_def(gtmp1);
            collector.reg_early_def(gtmp2);
        }
    }
}

impl MachInst for Inst {
    type LabelUse = LabelUse;
    type ABIMachineSpec = Arm32MachineDeps;

    // `udf #0`
    const TRAP_OPCODE: &'static [u8] = &0xe7f0_00f0u32.to_le_bytes();

    fn get_operands(&mut self, collector: &mut impl OperandVisitor) {
        arm32_get_operands(self, collector);
    }

    fn is_move(&self) -> Option<(Writable<Reg>, Reg)> {
        match self {
            Inst::Mov { rd, rm } => Some((*rd, *rm)),
            Inst::FpuMove { rd, rn } => Some((*rd, *rn)),
            Inst::VecMove { rd, rn } => Some((*rd, *rn)),
            _ => None,
        }
    }

    fn is_included_in_clobbers(&self) -> bool {
        !matches!(self, Inst::Args { .. })
    }

    fn is_trap(&self) -> bool {
        matches!(self, Inst::Udf { .. })
    }

    fn is_args(&self) -> bool {
        matches!(self, Inst::Args { .. })
    }

    fn call_type(&self) -> CallType {
        match self {
            Inst::Call { .. } | Inst::CallInd { .. } => CallType::Regular,
            Inst::ReturnCall { .. } | Inst::ReturnCallInd { .. } => CallType::TailCall,
            _ => CallType::None,
        }
    }

    fn is_term(&self) -> MachTerminator {
        match self {
            Inst::Rets { .. } => MachTerminator::Ret,
            Inst::ReturnCall { .. } | Inst::ReturnCallInd { .. } => MachTerminator::RetCall,
            Inst::Jump { .. } | Inst::CondBr { .. } | Inst::BrTable { .. } => {
                MachTerminator::Branch
            }
            Inst::Call { info } if info.try_call_info.is_some() => MachTerminator::Branch,
            Inst::CallInd { info } if info.try_call_info.is_some() => MachTerminator::Branch,
            _ => MachTerminator::None,
        }
    }

    fn is_mem_access(&self) -> bool {
        matches!(
            self,
            Inst::Load { .. }
                | Inst::Store { .. }
                | Inst::FpuLoad { .. }
                | Inst::FpuStore { .. }
                | Inst::FpuLoadUnaligned { .. }
                | Inst::FpuStoreUnaligned32 { .. }
                | Inst::LoadExclusive { .. }
                | Inst::AtomicRmwLoop { .. }
                | Inst::AtomicCasLoop { .. }
                | Inst::AtomicRmwLoop64 { .. }
                | Inst::AtomicCasLoop64 { .. }
                | Inst::AtomicLoad64 { .. }
                | Inst::LoadAcquire { .. }
                | Inst::StoreRelease { .. }
                | Inst::VecLoad { .. }
                | Inst::VecLoadLow { .. }
                | Inst::VecLoadSplat { .. }
                | Inst::VecLoadLane { .. }
                | Inst::VecStoreLane { .. }
                | Inst::VecStore { .. }
        )
    }

    fn gen_move(to_reg: Writable<Reg>, from_reg: Reg, ty: Type) -> Inst {
        let bits = ty.bits();
        assert!(bits <= 128);
        match to_reg.to_reg().class() {
            RegClass::Int => {
                assert!(bits <= 32);
                Inst::Mov {
                    rd: to_reg,
                    rm: from_reg,
                }
            }
            RegClass::Float => Inst::FpuMove {
                rd: to_reg,
                rn: from_reg,
            },
            RegClass::Vector => Inst::VecMove {
                rd: to_reg,
                rn: from_reg,
            },
        }
    }

    fn is_safepoint(&self) -> bool {
        matches!(self, Inst::Call { .. } | Inst::CallInd { .. })
    }

    fn gen_dummy_use(reg: Reg) -> Inst {
        Inst::DummyUse { reg }
    }

    fn gen_nop(preferred_size: usize) -> Inst {
        if preferred_size == 0 {
            return Inst::Nop0;
        }
        assert!(preferred_size >= 4);
        Inst::Nop4
    }

    fn gen_nop_units() -> Vec<Vec<u8>> {
        vec![0xe320_f000u32.to_le_bytes().to_vec()]
    }

    fn rc_for_type(ty: &Type) -> CodegenResult<(&[RegClass], &[Type])> {
        match *ty {
            I8 => Ok((&[RegClass::Int], &[I8])),
            I16 => Ok((&[RegClass::Int], &[I16])),
            I32 => Ok((&[RegClass::Int], &[I32])),
            I64 => Ok((&[RegClass::Int, RegClass::Int], &[I32, I32])),
            F32 => Ok((&[RegClass::Float], &[F32])),
            F64 => Ok((&[RegClass::Float], &[F64])),
            _ if ty.is_vector() && ty.bits() == 128 => Ok((&[RegClass::Vector], &[I8X16])),
            _ => Err(CodegenError::Unsupported(format!(
                "Unexpected SSA-value type on arm32: {ty}"
            ))),
        }
    }

    fn canonical_type_for_rc(rc: RegClass) -> Type {
        match rc {
            RegClass::Int => I32,
            RegClass::Float => F64,
            RegClass::Vector => I8X16,
        }
    }

    fn gen_jump(target: MachLabel) -> Inst {
        Inst::Jump { dest: target }
    }

    fn worst_case_size() -> CodeOffset {
        // The longest non-island-emitting sequences are the software
        // division loops and the floating-point rounding sequence; this
        // bound is checked in emission.
        512
    }

    fn worst_case_island_growth() -> CodeOffset {
        // At most one deferred trap (4 bytes) and a few branch fixups with
        // 12-byte veneers per instruction.
        64
    }

    fn function_alignment() -> FunctionAlignment {
        FunctionAlignment {
            minimum: 4,
            preferred: 16,
        }
    }
}

//=============================================================================
// Pretty-printing of instructions.

fn pretty_print_try_call(info: &TryCallInfo) -> String {
    format!(
        "; b {:?}; catch [{}]",
        info.continuation,
        info.pretty_print_dests()
    )
}

fn alu_op_name(op: AluOp) -> &'static str {
    match op {
        AluOp::And => "and",
        AluOp::Eor => "eor",
        AluOp::Sub => "sub",
        AluOp::Rsb => "rsb",
        AluOp::Add => "add",
        AluOp::Adc => "adc",
        AluOp::Sbc => "sbc",
        AluOp::Rsc => "rsc",
        AluOp::Orr => "orr",
        AluOp::Bic => "bic",
    }
}

fn vec_alu_op_name(op: VecAluOp) -> &'static str {
    match op {
        VecAluOp::Add => "add.i",
        VecAluOp::Sub => "sub.i",
        VecAluOp::Mul => "mul.i",
        VecAluOp::QAddS => "qadd.s",
        VecAluOp::QAddU => "qadd.u",
        VecAluOp::QSubS => "qsub.s",
        VecAluOp::QSubU => "qsub.u",
        VecAluOp::RHAddU => "rhadd.u",
        VecAluOp::MinS => "min.s",
        VecAluOp::MinU => "min.u",
        VecAluOp::MaxS => "max.s",
        VecAluOp::MaxU => "max.u",
        VecAluOp::Ceq => "ceq.i",
        VecAluOp::CgtS => "cgt.s",
        VecAluOp::CgtU => "cgt.u",
        VecAluOp::CgeS => "cge.s",
        VecAluOp::CgeU => "cge.u",
        VecAluOp::Tst => "tst.",
        VecAluOp::And => "and.",
        VecAluOp::Bic => "bic.",
        VecAluOp::Orr => "orr.",
        VecAluOp::Orn => "orn.",
        VecAluOp::Eor => "eor.",
        VecAluOp::ShlS => "shl.s",
        VecAluOp::ShlU => "shl.u",
        VecAluOp::QRDMulH => "qrdmulh.s",
        VecAluOp::PAdd => "padd.i",
        VecAluOp::PMinU => "pmin.u",
        VecAluOp::PMaxU => "pmax.u",
    }
}

fn cmp_op_name(op: CmpOp) -> &'static str {
    match op {
        CmpOp::Tst => "tst",
        CmpOp::Teq => "teq",
        CmpOp::Cmp => "cmp",
        CmpOp::Cmn => "cmn",
    }
}

fn bit_op_name(op: BitOp) -> &'static str {
    match op {
        BitOp::Clz => "clz",
        BitOp::Rbit => "rbit",
        BitOp::Rev => "rev",
        BitOp::Rev16 => "rev16",
        BitOp::Revsh => "revsh",
        BitOp::Sxtb => "sxtb",
        BitOp::Sxth => "sxth",
        BitOp::Uxtb => "uxtb",
        BitOp::Uxth => "uxth",
    }
}

fn load_op_name(op: LoadOp) -> &'static str {
    match op {
        LoadOp::Ldr => "ldr",
        LoadOp::Ldrb => "ldrb",
        LoadOp::Ldrsb => "ldrsb",
        LoadOp::Ldrh => "ldrh",
        LoadOp::Ldrsh => "ldrsh",
    }
}

fn store_op_name(op: StoreOp) -> &'static str {
    match op {
        StoreOp::Str => "str",
        StoreOp::Strb => "strb",
        StoreOp::Strh => "strh",
    }
}

fn fpu_suffix(size: FpuSize) -> &'static str {
    match size {
        FpuSize::F32 => "f32",
        FpuSize::F64 => "f64",
    }
}

fn rint_suffix(mode: FpuRoundMode) -> &'static str {
    match mode {
        FpuRoundMode::Ceil => "p",
        FpuRoundMode::Floor => "m",
        FpuRoundMode::Trunc => "z",
        FpuRoundMode::Nearest => "n",
    }
}

fn reg_mask_list(mask: u16) -> String {
    let regs: Vec<String> = (0..16u8)
        .filter(|i| mask & (1 << i) != 0)
        .map(|i| show_reg(rreg(i)))
        .collect();
    regs.join(", ")
}

impl Inst {
    fn print_with_state(&self, state: &mut EmitState) -> String {
        let r = |reg: Reg| show_reg(reg);
        let w = |reg: Writable<Reg>| show_reg(reg.to_reg());
        let f = |reg: Reg, size: FpuSize| show_freg(reg, size.is_f64());
        let fw = |reg: Writable<Reg>, size: FpuSize| show_freg(reg.to_reg(), size.is_f64());
        let s_suffix = |s: bool| if s { "s" } else { "" };
        match self {
            Inst::Nop0 => "nop-zero-len".to_string(),
            Inst::Nop4 => "nop".to_string(),
            Inst::AluRRR {
                op,
                s,
                rd,
                rn,
                rm,
                shift,
            } => format!(
                "{}{} {}, {}, {}{}",
                alu_op_name(*op),
                s_suffix(*s),
                w(*rd),
                r(*rn),
                r(*rm),
                shift
            ),
            Inst::AluRRRShift {
                op,
                s,
                rd,
                rn,
                rm,
                shift_op,
                rs,
            } => format!(
                "{}{} {}, {}, {}, {} {}",
                alu_op_name(*op),
                s_suffix(*s),
                w(*rd),
                r(*rn),
                r(*rm),
                shift_op_name(*shift_op),
                r(*rs)
            ),
            Inst::AluRRImm { op, s, rd, rn, imm } => format!(
                "{}{} {}, {}, {}",
                alu_op_name(*op),
                s_suffix(*s),
                w(*rd),
                r(*rn),
                imm
            ),
            Inst::MovRR {
                op,
                s,
                rd,
                rm,
                shift,
            } => {
                let name = match (op, shift.is_none()) {
                    (MovOp::Mov, true) => "mov".to_string(),
                    (MovOp::Mov, false) => {
                        return format!(
                            "{}{} {}, {}, #{}",
                            shift_op_name(shift.op),
                            s_suffix(*s),
                            w(*rd),
                            r(*rm),
                            shift.amt
                        );
                    }
                    (MovOp::Mvn, _) => "mvn".to_string(),
                };
                format!("{}{} {}, {}{}", name, s_suffix(*s), w(*rd), r(*rm), shift)
            }
            Inst::MovRRShift {
                rd,
                rm,
                shift_op,
                rs,
            } => format!(
                "{} {}, {}, {}",
                shift_op_name(*shift_op),
                w(*rd),
                r(*rm),
                r(*rs)
            ),
            Inst::MovImm { op, rd, imm } => {
                let name = match op {
                    MovOp::Mov => "mov",
                    MovOp::Mvn => "mvn",
                };
                format!("{} {}, {}", name, w(*rd), imm)
            }
            Inst::MovW { rd, imm } => format!("movw {}, #{}", w(*rd), imm),
            Inst::MovT { rd, rn, imm } => {
                let rd = w(*rd);
                let rn = r(*rn);
                if rd == rn {
                    format!("movt {rd}, #{imm}")
                } else {
                    format!("movt {rd}, #{imm} ; rn={rn}")
                }
            }
            Inst::Mov { rd, rm } => format!("mov {}, {}", w(*rd), r(*rm)),
            Inst::CmpRR { op, rn, rm, shift } => {
                format!("{} {}, {}{}", cmp_op_name(*op), r(*rn), r(*rm), shift)
            }
            Inst::CCmpRR { op, cond, rn, rm } => format!(
                "{}{} {}, {}",
                cmp_op_name(*op),
                cond.suffix(),
                r(*rn),
                r(*rm)
            ),
            Inst::CmpRImm { op, rn, imm } => format!("{} {}, {}", cmp_op_name(*op), r(*rn), imm),
            Inst::CCmpRImm { op, cond, rn, imm } => {
                format!("{}{} {}, {}", cmp_op_name(*op), cond.suffix(), r(*rn), imm)
            }
            Inst::CSel { rd, cond, rn, rm } => {
                format!("csel {}, {}, {}, {}", w(*rd), r(*rn), r(*rm), cond.suffix())
            }
            Inst::CSelImm {
                op,
                rd,
                cond,
                imm,
                rm,
            } => {
                let inv = match op {
                    MovOp::Mov => "",
                    MovOp::Mvn => "~",
                };
                format!(
                    "csel {}, {}{}, {}, {}",
                    w(*rd),
                    inv,
                    imm,
                    r(*rm),
                    cond.suffix()
                )
            }
            Inst::CSet { rd, cond } => format!("cset {}, {}", w(*rd), cond.suffix()),
            Inst::CSetm { rd, cond } => format!("csetm {}, {}", w(*rd), cond.suffix()),
            Inst::Mul { rd, rn, rm } => format!("mul {}, {}, {}", w(*rd), r(*rn), r(*rm)),
            Inst::MulAcc { op, rd, rn, rm, ra } => {
                let name = match op {
                    MulAccOp::Mla => "mla",
                    MulAccOp::Mls => "mls",
                };
                format!("{} {}, {}, {}, {}", name, w(*rd), r(*rn), r(*rm), r(*ra))
            }
            Inst::MulLong {
                op,
                rd_lo,
                rd_hi,
                rn,
                rm,
            } => {
                let name = match op {
                    MulLongOp::Umull => "umull",
                    MulLongOp::Smull => "smull",
                };
                format!(
                    "{} {}, {}, {}, {}",
                    name,
                    w(*rd_lo),
                    w(*rd_hi),
                    r(*rn),
                    r(*rm)
                )
            }
            Inst::Umaal {
                rd_lo,
                rd_hi,
                rn,
                rm,
                ..
            } => format!("umaal {}, {}, {}, {}", w(*rd_lo), w(*rd_hi), r(*rn), r(*rm)),
            Inst::Div { signed, rd, rn, rm } => format!(
                "{} {}, {}, {}",
                if *signed { "sdiv" } else { "udiv" },
                w(*rd),
                r(*rn),
                r(*rm)
            ),
            Inst::UDivMod32 {
                rd_quot,
                rd_rem,
                rn,
                rm,
            } => format!(
                "udivmod32 {}, {}, {}, {}",
                w(*rd_quot),
                w(*rd_rem),
                r(*rn),
                r(*rm)
            ),
            Inst::UDivMod64 {
                quot_lo,
                quot_hi,
                rem_lo,
                rem_hi,
                n_lo,
                n_hi,
                d_lo,
                d_hi,
                ..
            } => format!(
                "udivmod64 {}:{}, {}:{}, {}:{}, {}:{}",
                w(*quot_lo),
                w(*quot_hi),
                w(*rem_lo),
                w(*rem_hi),
                r(*n_lo),
                r(*n_hi),
                r(*d_lo),
                r(*d_hi),
            ),
            Inst::BitRR { op, rd, rm } => format!("{} {}, {}", bit_op_name(*op), w(*rd), r(*rm)),
            Inst::Shift64 {
                op,
                rd_lo,
                rd_hi,
                rn_lo,
                rn_hi,
                amt,
                tmp,
            } => format!(
                "{}64 {}:{}, {}:{}, {}, tmp={}",
                shift_op_name(*op),
                w(*rd_lo),
                w(*rd_hi),
                r(*rn_lo),
                r(*rn_hi),
                r(*amt),
                w(*tmp)
            ),
            Inst::Rot64 {
                left,
                rd_lo,
                rd_hi,
                rn_lo,
                rn_hi,
                amt,
                t1,
                t2,
            } => format!(
                "{}64 {}:{}, {}:{}, {}, tmps={}, {}",
                if *left { "rotl" } else { "rotr" },
                w(*rd_lo),
                w(*rd_hi),
                r(*rn_lo),
                r(*rn_hi),
                r(*amt),
                w(*t1),
                w(*t2)
            ),
            Inst::Load { op, rd, mem, .. } => format!(
                "{} {}, {}",
                load_op_name(*op),
                w(*rd),
                mem.pretty_print(state)
            ),
            Inst::Store { op, rt, mem, .. } => format!(
                "{} {}, {}",
                store_op_name(*op),
                r(*rt),
                mem.pretty_print(state)
            ),
            Inst::LoadAddr { rd, mem } => {
                format!("load_addr {}, {}", w(*rd), mem.pretty_print(state))
            }
            Inst::FpuZero { rd } => format!("vmov.i32 {}, #0", show_freg(rd.to_reg(), true)),
            Inst::FpuRint { mode, size, rd, rn } => format!(
                "vrint{}.{} {}, {}",
                rint_suffix(*mode),
                fpu_suffix(*size),
                fw(*rd, *size),
                f(*rn, *size)
            ),
            Inst::VecRint {
                mode,
                size,
                rd,
                rn,
                tmp,
            } => format!(
                "vrint{}.{} {}, {}, tmp={}",
                rint_suffix(*mode),
                fpu_suffix(*size),
                w(*rd),
                r(*rn),
                show_freg(tmp.to_reg(), true)
            ),
            Inst::FpuLoadConst { size, rd, constant } => {
                format!("vldr {}, [pc, {:?}]", fw(*rd, *size), constant)
            }
            Inst::VecLoadConst { rd, lo, hi } => {
                format!("vldr {}, [pc, {:?}:{:?}]", w(*rd), lo, hi)
            }
            Inst::VecZero { rd } => format!("vmov.i32 {}, #0", w(*rd)),
            Inst::FpuMovImm { size, rd, imm } => format!(
                "vmov.{} {}, #{:?}",
                fpu_suffix(*size),
                fw(*rd, *size),
                vfp_expand_imm8(*imm)
            ),
            Inst::FpuRR { op, size, rd, rn } => {
                let name = match op {
                    FpuOp1::Abs => "vabs",
                    FpuOp1::Neg => "vneg",
                    FpuOp1::Sqrt => "vsqrt",
                    FpuOp1::Mov => "vmov",
                };
                format!(
                    "{}.{} {}, {}",
                    name,
                    fpu_suffix(*size),
                    fw(*rd, *size),
                    f(*rn, *size)
                )
            }
            Inst::FpuRRR {
                op,
                size,
                rd,
                rn,
                rm,
            } => {
                let name = match op {
                    FpuOp2::Add => "vadd",
                    FpuOp2::Sub => "vsub",
                    FpuOp2::Mul => "vmul",
                    FpuOp2::Div => "vdiv",
                };
                format!(
                    "{}.{} {}, {}, {}",
                    name,
                    fpu_suffix(*size),
                    fw(*rd, *size),
                    f(*rn, *size),
                    f(*rm, *size)
                )
            }
            Inst::FpuRRRR {
                op,
                size,
                rd,
                ra,
                rn,
                rm,
            } => {
                let name = match op {
                    FpuOp3::Fma => "vfma",
                    FpuOp3::Fms => "vfms",
                    FpuOp3::Fnma => "vfnma",
                    FpuOp3::Fnms => "vfnms",
                };
                let hw = |r: Reg| r.to_real_reg().map(|r| u32::from(r.hw_enc()));
                match (size, hw(rd.to_reg()), hw(*rn), hw(*rm)) {
                    // See emission: an `f32` accumulator without an S half.
                    (FpuSize::F32, Some(d), Some(n), Some(m)) if d >= 16 => {
                        let t = (0..3).find(|t| *t != n && *t != m).unwrap();
                        format!(
                            "vswp d{t}, d{d} ; {name}.f32 s{}, s{}, s{} ; vswp d{t}, d{d}",
                            2 * t,
                            2 * n,
                            2 * m
                        )
                    }
                    _ => format!(
                        "{}.{} {}, {}, {} ; ra={}",
                        name,
                        fpu_suffix(*size),
                        fw(*rd, *size),
                        f(*rn, *size),
                        f(*rm, *size),
                        f(*ra, *size)
                    ),
                }
            }
            Inst::FpuCmp { size, rn, rm } => format!(
                "vcmp.{} {}, {} ; vmrs APSR_nzcv, fpscr",
                fpu_suffix(*size),
                f(*rn, *size),
                f(*rm, *size)
            ),
            Inst::FpuCvt { op, rd, rn } => {
                let (name, rd, rn) = match op {
                    FpuCvtOp::F32ToF64 => ("vcvt.f64.f32", r(rd.to_reg()), show_sreg(*rn)),
                    FpuCvtOp::F64ToF32 => ("vcvt.f32.f64", show_sreg(rd.to_reg()), r(*rn)),
                    FpuCvtOp::F32ToS32 => ("vcvt.s32.f32", show_sreg(rd.to_reg()), show_sreg(*rn)),
                    FpuCvtOp::F32ToU32 => ("vcvt.u32.f32", show_sreg(rd.to_reg()), show_sreg(*rn)),
                    FpuCvtOp::F64ToS32 => ("vcvt.s32.f64", show_sreg(rd.to_reg()), r(*rn)),
                    FpuCvtOp::F64ToU32 => ("vcvt.u32.f64", show_sreg(rd.to_reg()), r(*rn)),
                    FpuCvtOp::S32ToF32 => ("vcvt.f32.s32", show_sreg(rd.to_reg()), show_sreg(*rn)),
                    FpuCvtOp::U32ToF32 => ("vcvt.f32.u32", show_sreg(rd.to_reg()), show_sreg(*rn)),
                    FpuCvtOp::S32ToF64 => ("vcvt.f64.s32", r(rd.to_reg()), show_sreg(*rn)),
                    FpuCvtOp::U32ToF64 => ("vcvt.f64.u32", r(rd.to_reg()), show_sreg(*rn)),
                };
                format!("{name} {rd}, {rn}")
            }
            Inst::FpuCSel {
                size,
                rd,
                cond,
                rn,
                rm,
            } => format!(
                "vcsel.{} {}, {}, {}, {}",
                fpu_suffix(*size),
                fw(*rd, *size),
                f(*rn, *size),
                f(*rm, *size),
                cond.suffix()
            ),
            Inst::FpuMinMax {
                is_min,
                size,
                rd,
                rn,
                rm,
                tmp,
            } => format!(
                "{}.{} {}, {}, {}, tmp={}",
                if *is_min { "vmin" } else { "vmax" },
                fpu_suffix(*size),
                fw(*rd, *size),
                f(*rn, *size),
                f(*rm, *size),
                w(*tmp)
            ),
            Inst::FpuRound {
                mode,
                size,
                rd,
                rn,
                tmp1,
                tmp2,
                ftmp,
            } => format!(
                "vround_{:?}.{} {}, {}, tmps={}, {}, {}",
                mode,
                fpu_suffix(*size),
                fw(*rd, *size),
                f(*rn, *size),
                w(*tmp1),
                w(*tmp2),
                w(*ftmp)
            ),
            Inst::FpuLoad { size, rd, mem, .. } => {
                format!("vldr {}, {}", fw(*rd, *size), mem.pretty_print(state))
            }
            Inst::FpuStore { size, rt, mem, .. } => {
                format!("vstr {}, {}", f(*rt, *size), mem.pretty_print(state))
            }
            Inst::FpuLoadUnaligned { size, rd, mem, .. } => match size {
                FpuSize::F64 => {
                    format!("vld1.8 {{{}}}, {}", fw(*rd, *size), mem.pretty_print(state))
                }
                FpuSize::F32 => format!(
                    "vld1.32 {{{}[0]}}, {}",
                    show_freg(rd.to_reg(), true),
                    mem.pretty_print(state)
                ),
            },
            Inst::FpuStoreUnaligned32 { rt, mem, .. } => format!(
                "vst1.32 {{{}[0]}}, {}",
                show_freg(*rt, true),
                mem.pretty_print(state)
            ),
            Inst::MovToFpu32 { rd, rn } => format!("vmov {}, {}", show_sreg(rd.to_reg()), r(*rn)),
            Inst::MovFromFpu32 { rd, rn } => format!("vmov {}, {}", w(*rd), show_sreg(*rn)),
            Inst::MovToFpu64 { rd, lo, hi } => {
                format!("vmov {}, {}, {}", w(*rd), r(*lo), r(*hi))
            }
            Inst::MovFromFpu64 { rd_lo, rd_hi, rn } => {
                format!("vmov {}, {}, {}", w(*rd_lo), w(*rd_hi), r(*rn))
            }
            Inst::FpuMove { rd, rn } => format!("vmov.f64 {}, {}", w(*rd), r(*rn)),
            Inst::Jump { dest } => format!("b {}", dest.to_string()),
            Inst::CondBr {
                taken,
                not_taken,
                cond,
            } => format!(
                "b{} {} ; b {}",
                cond.suffix(),
                taken.to_string(),
                not_taken.to_string()
            ),
            Inst::TrapIf { cond, trap_code } => {
                format!("udf{} ; trap={}", cond.suffix(), trap_code)
            }
            Inst::Udf { trap_code } => format!("udf #0 ; trap={trap_code}"),
            Inst::Bkpt => "bkpt #0".to_string(),
            Inst::BrTable { index, targets } => {
                let targets: Vec<String> = targets.iter().map(|l| l.to_string()).collect();
                format!("br_table {}, [{}]", r(*index), targets.join(", "))
            }
            Inst::Call { info } => {
                let try_call = info
                    .try_call_info
                    .as_ref()
                    .map(|tci| pretty_print_try_call(tci))
                    .unwrap_or_default();
                format!("bl {}{}", info.dest.display(None), try_call)
            }
            Inst::CallInd { info } => {
                let try_call = info
                    .try_call_info
                    .as_ref()
                    .map(|tci| pretty_print_try_call(tci))
                    .unwrap_or_default();
                format!("blx {}{}", r(info.dest), try_call)
            }
            Inst::ReturnCall { info } => {
                let mut s = format!(
                    "return_call {:?} new_stack_arg_size:{}",
                    info.dest, info.new_stack_arg_size
                );
                for ret in &info.uses {
                    s.push_str(&format!(" {}={}", r(ret.preg), r(ret.vreg)));
                }
                s
            }
            Inst::ReturnCallInd { info } => {
                let mut s = format!(
                    "return_call_ind {} new_stack_arg_size:{}",
                    r(info.dest),
                    info.new_stack_arg_size
                );
                for ret in &info.uses {
                    s.push_str(&format!(" {}={}", r(ret.preg), r(ret.vreg)));
                }
                s
            }
            Inst::Args { args } => {
                let mut s = "args".to_string();
                for arg in args.iter() {
                    s.push_str(&format!(" {}={}", r(arg.preg), w(arg.vreg)));
                }
                s
            }
            Inst::Rets { rets } => {
                let mut s = "rets".to_string();
                for ret in rets.iter() {
                    s.push_str(&format!(" {}={}", r(ret.preg), r(ret.vreg)));
                }
                s
            }
            Inst::Ret => "bx lr".to_string(),
            Inst::LoadExtName { rd, name, offset } => {
                format!("ldr {}, ={}+{}", w(*rd), name.display(None), offset)
            }
            Inst::LoadExtNameNear { rd, name, offset } => {
                format!("ldr {}, =pcrel({}+{})", w(*rd), name.display(None), offset)
            }
            Inst::LabelAddress { rd, label } => {
                format!("label_address {}, {}", w(*rd), label.to_string())
            }
            Inst::MovFromPReg { rd, rm } => {
                format!("mov {}, {}", w(*rd), r(Reg::from(*rm)))
            }
            Inst::Fence => "dmb ish".to_string(),
            Inst::LoadExclusive { size, rd, addr } => {
                let suffix = match size {
                    1 => "b",
                    2 => "h",
                    _ => "",
                };
                format!("ldrex{} {}, [{}]", suffix, w(*rd), r(*addr))
            }
            Inst::AtomicRmwLoop {
                op,
                size,
                rd,
                addr,
                operand,
                tmp1,
                tmp2,
                ..
            } => format!(
                "atomic_rmw_loop_{:?}_{} {}, [{}], {}, tmps={}, {}",
                op,
                size,
                w(*rd),
                r(*addr),
                r(*operand),
                w(*tmp1),
                w(*tmp2)
            ),
            Inst::AtomicCasLoop {
                size,
                rd,
                addr,
                expected,
                replacement,
                tmp,
                ..
            } => format!(
                "atomic_cas_loop_{} {}, [{}], {}, {}, tmp={}",
                size,
                w(*rd),
                r(*addr),
                r(*expected),
                r(*replacement),
                w(*tmp)
            ),
            Inst::AtomicRmwLoop64 { op, addr, .. } => {
                format!("atomic_rmw_loop64_{:?} r0:r1, [{}], r2:r3", op, r(*addr))
            }
            Inst::AtomicCasLoop64 { addr, .. } => {
                format!("atomic_cas_loop64 r0:r1, [{}], r2:r3, r4:r5", r(*addr))
            }
            Inst::LoadAcquire { size, rd, addr, .. } => {
                let sfx = match size {
                    1 => "b",
                    2 => "h",
                    _ => "",
                };
                format!("lda{} {}, [{}]", sfx, w(*rd), r(*addr))
            }
            Inst::StoreRelease { size, rt, addr, .. } => {
                let sfx = match size {
                    1 => "b",
                    2 => "h",
                    _ => "",
                };
                format!("stl{} {}, [{}]", sfx, r(*rt), r(*addr))
            }
            Inst::AtomicLoad64 { addr, .. } => {
                format!("ldrexd r0, r1, [{}] ; clrex ; dmb ish", r(*addr))
            }
            Inst::Push { mask } => format!("push {{{}}}", reg_mask_list(*mask)),
            Inst::Pop { mask } => format!("pop {{{}}}", reg_mask_list(*mask)),
            Inst::VPush { first, count } => {
                format!("vpush {{d{}-d{}}}", first, first + count - 1)
            }
            Inst::VPop { first, count } => format!("vpop {{d{}-d{}}}", first, first + count - 1),
            Inst::SpAdjust { amount } => {
                if *amount >= 0 {
                    format!("add sp, sp, #{amount}")
                } else {
                    format!("sub sp, sp, #{}", -i64::from(*amount))
                }
            }
            Inst::StackProbeLoop {
                guard_size,
                probe_count,
            } => format!("stack_probe_loop guard_size={guard_size} probe_count={probe_count}"),
            Inst::Unwind { inst } => format!("unwind {inst:?}"),
            Inst::DummyUse { reg } => format!("dummy_use {}", r(*reg)),
            Inst::EmitIsland { needed_space } => format!("emit_island {needed_space}"),
            Inst::SequencePoint => "sequence_point".to_string(),
            Inst::VecMove { rd, rn } => format!("vmov {}, {}", w(*rd), r(*rn)),
            Inst::VecRRR {
                op,
                size,
                rd,
                rn,
                rm,
            } => format!(
                "v{}{} {}, {}, {}",
                vec_alu_op_name(*op),
                8u32 << size,
                w(*rd),
                r(*rn),
                r(*rm)
            ),
            Inst::VecBsl { rd, rn, rm, .. } => {
                format!("vbsl {}, {}, {}", w(*rd), r(*rn), r(*rm))
            }
            Inst::VecRR { op, size, rd, rn } => {
                format!("v{:?}.{} {}, {}", op, 8u32 << size, w(*rd), r(*rn)).to_lowercase()
            }
            Inst::VecSraSelf { size, rd, rn, amt } => {
                format!("vsra.u{} {}, {}, #{}", 8u32 << size, w(*rd), r(*rn), amt)
            }
            Inst::VecShiftImm {
                op,
                size,
                rd,
                rn,
                amt,
            } => format!(
                "v{:?}.{} {}, {}, #{}",
                op,
                8u32 << size,
                w(*rd),
                r(*rn),
                amt
            )
            .to_lowercase(),
            Inst::VecNarrow {
                op,
                size,
                rd,
                rn,
                rm,
            } => format!(
                "v{:?}.{} {}, {}, {}",
                op,
                16u32 << size,
                w(*rd),
                r(*rn),
                r(*rm)
            )
            .to_lowercase(),
            Inst::VecWiden {
                signed,
                high,
                size,
                rd,
                rn,
            } => format!(
                "vmovl.{}{} {}, {}.{}",
                if *signed { "s" } else { "u" },
                8u32 << size,
                w(*rd),
                r(*rn),
                if *high { "hi" } else { "lo" }
            ),
            Inst::VecMulLong {
                signed,
                high,
                size,
                rd,
                rn,
                rm,
            } => format!(
                "vmull.{}{} {}, {}.{h}, {}.{h}",
                if *signed { "s" } else { "u" },
                8u32 << size,
                w(*rd),
                r(*rn),
                r(*rm),
                h = if *high { "hi" } else { "lo" }
            ),
            Inst::VecPairwise {
                op,
                size,
                rd,
                rn,
                rm,
            } => format!(
                "v{}{} {}, {}, {}",
                vec_alu_op_name(*op),
                8u32 << size,
                w(*rd),
                r(*rn),
                r(*rm)
            ),
            Inst::VecDup { size, rd, rn } => {
                format!("vdup.{} {}, {}", 8u32 << size, w(*rd), r(*rn))
            }
            Inst::VecDupLane { size, rd, rn, lane } => {
                format!("vdup.{} {}, {}[{}]", 8u32 << size, w(*rd), r(*rn), lane)
            }
            Inst::VecMovToLane {
                size, rd, rn, lane, ..
            } => format!("vmov.{} {}[{}], {}", 8u32 << size, w(*rd), lane, r(*rn)),
            Inst::VecMovFromLane {
                size,
                signed,
                rd,
                rn,
                lane,
            } => format!(
                "vmov.{}{} {}, {}[{}]",
                if *signed { "s" } else { "u" },
                8u32 << size,
                w(*rd),
                r(*rn),
                lane
            ),
            Inst::VecFromGprs { rd, r0, r1, r2, r3 } => format!(
                "vmov {}, {}, {}, {}, {}",
                w(*rd),
                r(*r0),
                r(*r1),
                r(*r2),
                r(*r3)
            ),
            Inst::VecToFpu { rd, rn, half } => {
                format!("vmov.f64 {}, {}.d{}", w(*rd), r(*rn), half)
            }
            Inst::VecOrrHalves { rd, rn } => {
                format!("vorr {}, {}.d0, {}.d1", w(*rd), r(*rn), r(*rn))
            }
            Inst::VecFromFpu { rd, rn, half, .. } => {
                format!("vmov.f64 {}.d{}, {}", w(*rd), half, r(*rn))
            }
            Inst::VecToFpu32 { rd, rn, lane, tmp } => format!(
                "vmov.f32 {}, {}[{}] ; tmp={}",
                show_sreg(rd.to_reg()),
                r(*rn),
                lane,
                w(*tmp)
            ),
            Inst::VecFromFpu32 {
                rd, rn, lane, tmp, ..
            } => format!(
                "vmov.f32 {}[{}], {} ; tmp={}",
                w(*rd),
                lane,
                show_sreg(*rn),
                w(*tmp)
            ),
            Inst::VecLoad { rd, mem, .. } => {
                format!("vld1.8 {{{}}}, {}", w(*rd), mem.pretty_print(state))
            }
            Inst::VecLoadLow {
                size,
                zero_rest,
                rd,
                mem,
                ..
            } => {
                let zero = if *zero_rest {
                    format!("vmov.i32 {}, #0 ; ", w(*rd))
                } else {
                    String::new()
                };
                let lane = if *size == 3 { "8" } else { "32" };
                let elem = if *size == 3 { "" } else { "[0]" };
                format!(
                    "{zero}vld1.{lane} {{{}.d0{elem}}}, {}",
                    w(*rd),
                    mem.pretty_print(state)
                )
            }
            Inst::VecLoadLane {
                size,
                rd,
                lane,
                mem,
                ..
            } => format!(
                "vld1.{} {{{}}}, {}",
                if *size == 3 { 8 } else { 8u32 << size },
                lane_operand(*size, w(*rd), *lane),
                mem.pretty_print(state)
            ),
            Inst::VecStoreLane {
                size,
                rt,
                lane,
                mem,
                ..
            } => format!(
                "vst1.{} {{{}}}, {}",
                if *size == 3 { 8 } else { 8u32 << size },
                lane_operand(*size, r(*rt), *lane),
                mem.pretty_print(state)
            ),
            Inst::VecLoadSplat { size, rd, mem, .. } => format!(
                "vld1.{} {{{}.d0[], {}.d1[]}}, {}",
                8u32 << size,
                w(*rd),
                w(*rd),
                mem.pretty_print(state)
            ),
            Inst::VecStore { rt, mem, .. } => {
                format!("vst1.8 {{{}}}, {}", r(*rt), mem.pretty_print(state))
            }
            Inst::VecTbl {
                rd,
                tbl1,
                tbl2,
                idx,
            } => match tbl2 {
                Some(tbl2) => format!(
                    "vtbl.8 {}, {{{}, {}}}, {}",
                    w(*rd),
                    r(*tbl1),
                    r(*tbl2),
                    r(*idx)
                ),
                None => format!("vtbl.8 {}, {{{}}}, {}", w(*rd), r(*tbl1), r(*idx)),
            },
            Inst::VecExt { rd, rn, rm, imm } => {
                format!("vext.8 {}, {}, {}, #{}", w(*rd), r(*rn), r(*rm), imm)
            }
            Inst::VecFpuRRR {
                op,
                size,
                rd,
                rn,
                rm,
                ..
            } => format!(
                "v{:?}.{}x {}, {}, {}",
                op,
                fpu_suffix(*size),
                w(*rd),
                r(*rn),
                r(*rm)
            )
            .to_lowercase(),
            Inst::VecFpuRR {
                op, size, rd, rn, ..
            } => format!("v{:?}.{}x {}, {}", op, fpu_suffix(*size), w(*rd), r(*rn)).to_lowercase(),
            Inst::VecFpuFma {
                size,
                rd,
                ra,
                rn,
                rm,
                ..
            } => format!(
                "vfma.{}x {}, {}, {}, {}",
                fpu_suffix(*size),
                w(*rd),
                r(*ra),
                r(*rn),
                r(*rm)
            ),
            Inst::VecFpuCmp {
                cond,
                size,
                rd,
                rn,
                rm,
                ..
            } => format!(
                "vfcmp.{}.{}x {}, {}, {}",
                cond,
                fpu_suffix(*size),
                w(*rd),
                r(*rn),
                r(*rm)
            ),
            Inst::VecFpuMinMax {
                is_min,
                size,
                rd,
                rn,
                rm,
                ..
            } => format!(
                "{}.{}x {}, {}, {}",
                if *is_min { "vfmin" } else { "vfmax" },
                fpu_suffix(*size),
                w(*rd),
                r(*rn),
                r(*rm)
            ),
            Inst::VecFpuRound {
                mode, size, rd, rn, ..
            } => format!(
                "vround_{:?}.{}x {}, {}",
                mode,
                fpu_suffix(*size),
                w(*rd),
                r(*rn)
            )
            .to_lowercase(),
            Inst::PopcntNeon {
                rd,
                lo,
                hi,
                tmp,
                is64,
            } => {
                let input = if *is64 {
                    format!("{}, {}", r(*lo), r(*hi))
                } else {
                    r(*lo)
                };
                format!("popcnt_neon {}, {} ; tmp {}", w(*rd), input, w(*tmp))
            }
            Inst::VecMulI64 { rd, rn, rm, tb } => format!(
                "vmul.i64x {}, {}, {} ; tmp {}",
                w(*rd),
                r(*rn),
                r(*rm),
                w(*tb)
            ),
            Inst::VecCmpF32 {
                cond,
                rd,
                rn,
                rm,
                ta,
                tb,
            } => format!(
                "vcmp_{}.f32x {}, {}, {} ; tmps {}, {}",
                cond,
                w(*rd),
                r(*rn),
                r(*rm),
                w(*ta),
                w(*tb)
            ),
            Inst::VecMinMaxF32 {
                is_min,
                rd,
                rn,
                rm,
                ta,
                tb,
            } => format!(
                "v{}.f32x {}, {}, {} ; tmps {}, {}",
                if *is_min { "min" } else { "max" },
                w(*rd),
                r(*rn),
                r(*rm),
                w(*ta),
                w(*tb)
            ),
            Inst::VecRoundF64 {
                mode,
                rd,
                rn,
                qa,
                qb,
                qc,
                t,
                gtmp,
            } => format!(
                "vround_{:?}.f64x {}, {} ; tmps {}, {}, {}, {}, {}",
                mode,
                w(*rd),
                r(*rn),
                w(*qa),
                w(*qb),
                w(*qc),
                w(*t),
                w(*gtmp)
            )
            .to_lowercase(),
            Inst::VecRoundF32 {
                mode,
                rd,
                rn,
                qa,
                qb,
                t1,
                t2,
                gtmp,
            } => format!(
                "vround_{:?}.f32x {}, {} ; tmps {}, {}, {}, {}, {}",
                mode,
                w(*rd),
                r(*rn),
                w(*qa),
                w(*qb),
                w(*t1),
                w(*t2),
                w(*gtmp)
            )
            .to_lowercase(),
        }
    }
}

//=============================================================================
// Label fixups and jump veneers.

/// Different forms of label references for different instruction formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LabelUse {
    /// 24-bit branch offset (`b`, `bl`, `b<cond>`), in words, relative to
    /// the instruction address plus 8. Range is +/- 32 MiB.
    Branch24,
    /// A 32-bit data word holding the offset from the word itself to the
    /// label.
    PCRel32,
    /// A `movw rX, #lo; movt rX, #hi` pair holding the 32-bit offset from
    /// the PC value read by an `add` instruction immediately following the
    /// pair (i.e. the use offset plus 16).
    MovwMovtPcRel,
    /// The offset field of a `vldr` (8 bits, in words, with a sign bit),
    /// relative to the instruction address plus 8. Range is +/- 1020 bytes.
    Vldr8,
}

impl MachInstLabelUse for LabelUse {
    const ALIGN: CodeOffset = 4;

    // Branches reach +/- 32 MiB, but constant-pool references (`vldr`) only
    // +/- 1 KiB.
    const PRECISE_ISLAND_VENEER_ESTIMATE: bool = true;

    fn max_pos_range(self) -> CodeOffset {
        match self {
            // The branch offset is relative to `use + 8`.
            LabelUse::Branch24 => ((1 << 25) - 4) + 8,
            LabelUse::PCRel32 | LabelUse::MovwMovtPcRel => 0x7fff_ffff,
            LabelUse::Vldr8 => 1020 + 8,
        }
    }

    fn max_neg_range(self) -> CodeOffset {
        match self {
            LabelUse::Branch24 => (1 << 25) - 8,
            LabelUse::PCRel32 | LabelUse::MovwMovtPcRel => 0x7fff_ffff,
            LabelUse::Vldr8 => 1020 - 8,
        }
    }

    fn patch_size(self) -> CodeOffset {
        match self {
            LabelUse::Branch24 | LabelUse::PCRel32 | LabelUse::Vldr8 => 4,
            LabelUse::MovwMovtPcRel => 8,
        }
    }

    fn patch(self, buffer: &mut [u8], use_offset: CodeOffset, label_offset: CodeOffset) {
        let pc_rel = (label_offset as i64) - (use_offset as i64);
        debug_assert!(pc_rel <= self.max_pos_range() as i64);
        debug_assert!(pc_rel >= -(self.max_neg_range() as i64));
        match self {
            LabelUse::Branch24 => {
                let off = pc_rel - 8;
                debug_assert_eq!(off & 3, 0);
                let insn = u32::from_le_bytes(buffer[0..4].try_into().unwrap());
                let imm24 = ((off >> 2) as u32) & 0x00ff_ffff;
                let insn = (insn & 0xff00_0000) | imm24;
                buffer[0..4].copy_from_slice(&insn.to_le_bytes());
            }
            LabelUse::PCRel32 => {
                let insn = u32::from_le_bytes(buffer[0..4].try_into().unwrap());
                let value = insn.wrapping_add(pc_rel as u32);
                buffer[0..4].copy_from_slice(&value.to_le_bytes());
            }
            LabelUse::MovwMovtPcRel => {
                let off = (pc_rel - 16) as u32;
                let movw = u32::from_le_bytes(buffer[0..4].try_into().unwrap());
                let movt = u32::from_le_bytes(buffer[4..8].try_into().unwrap());
                let movw = (movw & 0xfff0_f000) | enc_imm16(off & 0xffff);
                let movt = (movt & 0xfff0_f000) | enc_imm16(off >> 16);
                buffer[0..4].copy_from_slice(&movw.to_le_bytes());
                buffer[4..8].copy_from_slice(&movt.to_le_bytes());
            }
            LabelUse::Vldr8 => {
                let off = pc_rel - 8;
                debug_assert_eq!(off & 3, 0);
                let insn = u32::from_le_bytes(buffer[0..4].try_into().unwrap());
                let u = u32::from(off >= 0);
                let imm8 = (off.unsigned_abs() / 4) as u32;
                debug_assert!(imm8 <= 0xff);
                let insn = (insn & !((1 << 23) | 0xff)) | (u << 23) | imm8;
                buffer[0..4].copy_from_slice(&insn.to_le_bytes());
            }
        }
    }

    fn supports_veneer(self) -> bool {
        matches!(self, LabelUse::Branch24)
    }

    fn veneer_size(self) -> CodeOffset {
        match self {
            LabelUse::Branch24 => 12,
            LabelUse::PCRel32 | LabelUse::MovwMovtPcRel | LabelUse::Vldr8 => unreachable!(),
        }
    }

    fn worst_case_veneer_size() -> CodeOffset {
        12
    }

    fn generate_veneer(
        self,
        buffer: &mut [u8],
        veneer_offset: CodeOffset,
    ) -> (CodeOffset, LabelUse) {
        match self {
            LabelUse::Branch24 => {
                // movw ip, #0
                // movt ip, #0
                // add pc, pc, ip
                let ip = 12;
                let movw = enc_movw(ip, 0);
                let movt = enc_movt(ip, 0);
                let add = 0xe08f_f00c; // add pc, pc, ip
                buffer[0..4].copy_from_slice(&movw.to_le_bytes());
                buffer[4..8].copy_from_slice(&movt.to_le_bytes());
                buffer[8..12].copy_from_slice(&u32::to_le_bytes(add));
                (veneer_offset, LabelUse::MovwMovtPcRel)
            }
            LabelUse::PCRel32 | LabelUse::MovwMovtPcRel | LabelUse::Vldr8 => {
                panic!("unsupported veneer")
            }
        }
    }

    fn from_reloc(reloc: Reloc, addend: Addend) -> Option<LabelUse> {
        match (reloc, addend) {
            (Reloc::Arm32Call, 0) => Some(LabelUse::Branch24),
            // The generic PC-relative 32-bit data relocation, as used by
            // `LoadExtNameNear`.
            (Reloc::X86PCRel4, 0) => Some(LabelUse::PCRel32),
            _ => None,
        }
    }
}

/// Encode a 16-bit immediate into the `imm4:imm12` fields of `movw`/`movt`.
pub(crate) fn enc_imm16(imm: u32) -> u32 {
    debug_assert!(imm <= 0xffff);
    ((imm & 0xf000) << 4) | (imm & 0x0fff)
}

/// `movw rd, #imm16`
pub(crate) fn enc_movw(rd: u32, imm: u32) -> u32 {
    0xe300_0000 | (rd << 12) | enc_imm16(imm)
}

/// `movt rd, #imm16`
pub(crate) fn enc_movt(rd: u32, imm: u32) -> u32 {
    0xe340_0000 | (rd << 12) | enc_imm16(imm)
}
