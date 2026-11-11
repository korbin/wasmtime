//! NEON (Advanced SIMD) and lane-wise VFP emission for 128-bit vectors.
//!
//! Vectors live in the Q registers `q8`-`q15`, i.e. the D registers
//! `d16`-`d31`, which never alias the scalar floating-point registers
//! `d0`-`d15`. Floating-point lane arithmetic uses scalar VFP instructions
//! (directly on `d16`-`d31` for `f64` lanes, and via temporaries in `d0`-`d15`
//! for `f32` lanes, which need S registers), because NEON floating-point
//! arithmetic always flushes denormals to zero.

use crate::ir::condcodes::FloatCC;
use crate::isa::arm32::inst::*;

const COND_AL: u32 = 0xe;
const IP: u32 = 12;

/// The D register number of the low half of a Q register.
fn qd(r: Reg) -> u32 {
    let real = r.to_real_reg().expect("register must be allocated");
    debug_assert_eq!(real.class(), RegClass::Vector);
    2 * u32::from(real.hw_enc())
}

fn qdw(r: Writable<Reg>) -> u32 {
    qd(r.to_reg())
}

/// D register number of a float-class register.
fn fd(r: Reg) -> u32 {
    machreg_to_dreg(r)
}

fn fdw(r: Writable<Reg>) -> u32 {
    machreg_to_dreg(r.to_reg())
}

fn gp(r: Reg) -> u32 {
    machreg_to_gpr(r)
}

fn gpw(r: Writable<Reg>) -> u32 {
    machreg_to_gpr(r.to_reg())
}

/// Register fields for a D register number (0-31) in the `Vd:D` style.
fn dfields(d: u32) -> (u32, u32) {
    (d & 0xf, d >> 4)
}

/// Register fields for an S register number (0-31).
fn sfields(s: u32) -> (u32, u32) {
    (s >> 1, s & 1)
}

/// Fill in the `D`/`Vd`, `N`/`Vn` and `M`/`Vm` fields of a NEON/VFP encoding
/// for D registers.
fn with_d_regs(base: u32, d: u32, n: u32, m: u32) -> u32 {
    let (vd, dd) = dfields(d);
    let (vn, nn) = dfields(n);
    let (vm, mm) = dfields(m);
    base | (dd << 22) | (vn << 16) | (vd << 12) | (nn << 7) | (mm << 5) | vm
}

/// A VFP data-processing instruction on arbitrary S registers.
pub(crate) fn enc_vfp_s(cond: u32, base: u32, d: u32, n: u32, m: u32) -> u32 {
    let (vd, dd) = sfields(d);
    let (vn, nn) = sfields(n);
    let (vm, mm) = sfields(m);
    (cond << 28) | base | (dd << 22) | (vn << 16) | (vd << 12) | (nn << 7) | (mm << 5) | vm
}

/// A VFP data-processing instruction on arbitrary D registers (`sz = 1`).
pub(crate) fn enc_vfp_d(cond: u32, base: u32, d: u32, n: u32, m: u32) -> u32 {
    (cond << 28) | (1 << 8) | with_d_regs(base, d, n, m)
}

const VADD: u32 = 0x0e30_0a00;
const VSUB: u32 = 0x0e30_0a40;
const VMUL: u32 = 0x0e20_0a00;
const VDIV: u32 = 0x0e80_0a00;
const VFMA: u32 = 0x0ea0_0a00;
const VMOV: u32 = 0x0eb0_0a40;
const VABS: u32 = 0x0eb0_0ac0;
const VNEG: u32 = 0x0eb1_0a40;
const VSQRT: u32 = 0x0eb1_0ac0;
const VCMP: u32 = 0x0eb4_0a40;
const VMRS_APSR: u32 = 0xeef1_fa10;
// Conversions between f32 and s32 in S registers: `vcvtr` rounds in the FPSCR
// mode, `vcvt` (to integer) toward zero.
const VCVT_TO_S32: u32 = 0x0ebd_0ac0;
const VCVTR_TO_S32: u32 = 0x0ebd_0a40;
const VCVT_FROM_S32: u32 = 0x0eb8_0ac0;
// NEON on Q registers (with `with_d_regs`).
const VBIT: u32 = 0xf320_0150;
const VBIF: u32 = 0xf330_0150;
const VBSL: u32 = 0xf310_0150;
const VAND: u32 = 0xf200_0150;
const VEOR: u32 = 0xf300_0150;
const VCGT_S32: u32 = 0xf220_0340;
const VCGE_S32: u32 = 0xf220_0350;
const VMOVN_I64: u32 = 0xf3ba_0200;
const VMLAL_U32: u32 = 0xf3a0_0800;
const VCEQ_I32: u32 = 0xf320_0850;
const VSUB_I32: u32 = 0xf320_0840;
const VACGE_F32: u32 = 0xf300_0e50;
const VCEQ_F32: u32 = 0xf200_0e40;
const VADD_F32: u32 = 0xf200_0d40;

/// `vorr.i32 qd, #(imm << 16)`.
fn vorr_i32_byte2(d: u32, imm: u32) -> u32 {
    let (vd, dd) = dfields(d);
    0xf280_0550
        | (((imm >> 7) & 1) << 24)
        | (((imm >> 4) & 7) << 16)
        | (imm & 0xf)
        | (dd << 22)
        | (vd << 12)
}

/// `vmov.i32 qd, #(imm << 24)`.
fn vmov_i32_byte3(d: u32, imm: u32) -> u32 {
    let (vd, dd) = dfields(d);
    0xf280_0650
        | (((imm >> 7) & 1) << 24)
        | (((imm >> 4) & 7) << 16)
        | (imm & 0xf)
        | (dd << 22)
        | (vd << 12)
}

fn fpu_op2_base(op: FpuOp2) -> u32 {
    match op {
        FpuOp2::Add => VADD,
        FpuOp2::Sub => VSUB,
        FpuOp2::Mul => VMUL,
        FpuOp2::Div => VDIV,
    }
}

fn fpu_op1_base(op: FpuOp1) -> u32 {
    match op {
        FpuOp1::Abs => VABS,
        FpuOp1::Neg => VNEG,
        FpuOp1::Sqrt => VSQRT,
        FpuOp1::Mov => VMOV,
    }
}

/// `vmov.f64 dd, dm` for arbitrary D registers.
fn vmov_d(d: u32, m: u32) -> u32 {
    enc_vfp_d(COND_AL, VMOV, d, 0, m)
}

/// `vmov.f32 sd, sm` for arbitrary S registers.
fn vmov_s(d: u32, m: u32) -> u32 {
    enc_vfp_s(COND_AL, VMOV, d, 0, m)
}

/// `vmov.32 dd[x], rt` (core to scalar).
fn vmov_to_scalar32(cond: u32, d: u32, x: u32, rt: u32) -> u32 {
    let (vd, dd) = dfields(d);
    (cond << 28) | 0x0e00_0b10 | (x << 21) | (vd << 16) | (rt << 12) | (dd << 7)
}

/// `vmov.32 rt, dd[x]` (scalar to core).
fn vmov_from_scalar32(rt: u32, d: u32, x: u32) -> u32 {
    let (vd, dd) = dfields(d);
    (COND_AL << 28) | 0x0e10_0b10 | (x << 21) | (vd << 16) | (rt << 12) | (dd << 7)
}

/// `vmov rt, s<n>`
fn vmov_core_from_s(rt: u32, s: u32) -> u32 {
    let (vn, n) = sfields(s);
    (COND_AL << 28) | 0x0e10_0a10 | (vn << 16) | (rt << 12) | (n << 7)
}

/// `vmov dm, rt, rt2`
fn vmov_d_from_core(d: u32, rt: u32, rt2: u32) -> u32 {
    let (vm, m) = dfields(d);
    0xec40_0b10 | (rt2 << 16) | (rt << 12) | (m << 5) | vm
}

/// The 3-same base encoding for an ALU op: (U, A, B, size-override).
fn alu_fields(op: VecAluOp) -> (u32, u32, u32, Option<u32>) {
    match op {
        VecAluOp::QAddS => (0, 0b0000, 1, None),
        VecAluOp::QAddU => (1, 0b0000, 1, None),
        VecAluOp::RHAddU => (1, 0b0001, 0, None),
        VecAluOp::And => (0, 0b0001, 1, Some(0b00)),
        VecAluOp::Bic => (0, 0b0001, 1, Some(0b01)),
        VecAluOp::Orr => (0, 0b0001, 1, Some(0b10)),
        VecAluOp::Orn => (0, 0b0001, 1, Some(0b11)),
        VecAluOp::Eor => (1, 0b0001, 1, Some(0b00)),
        VecAluOp::QSubS => (0, 0b0010, 1, None),
        VecAluOp::QSubU => (1, 0b0010, 1, None),
        VecAluOp::CgtS => (0, 0b0011, 0, None),
        VecAluOp::CgtU => (1, 0b0011, 0, None),
        VecAluOp::CgeS => (0, 0b0011, 1, None),
        VecAluOp::CgeU => (1, 0b0011, 1, None),
        VecAluOp::ShlS => (0, 0b0100, 0, None),
        VecAluOp::ShlU => (1, 0b0100, 0, None),
        VecAluOp::MaxS => (0, 0b0110, 0, None),
        VecAluOp::MaxU => (1, 0b0110, 0, None),
        VecAluOp::MinS => (0, 0b0110, 1, None),
        VecAluOp::MinU => (1, 0b0110, 1, None),
        VecAluOp::Add => (0, 0b1000, 0, None),
        VecAluOp::Sub => (1, 0b1000, 0, None),
        VecAluOp::Tst => (0, 0b1000, 1, None),
        VecAluOp::Ceq => (1, 0b1000, 1, None),
        VecAluOp::Mul => (0, 0b1001, 1, None),
        VecAluOp::PMaxU => (1, 0b1010, 0, None),
        VecAluOp::PMinU => (1, 0b1010, 1, None),
        VecAluOp::QRDMulH => (1, 0b1011, 0, None),
        VecAluOp::PAdd => (0, 0b1011, 1, None),
    }
}

/// A "three registers of the same length" instruction on D registers
/// (`q = false`) or Q registers (`q = true`, register numbers are D numbers
/// of the low halves).
pub(crate) fn enc_3same(op: VecAluOp, size: u8, q: bool, d: u32, n: u32, m: u32) -> u32 {
    let (u, a, b, size_override) = alu_fields(op);
    let size = size_override.unwrap_or(u32::from(size));
    let base = 0xf200_0000 | (u << 24) | (size << 20) | (a << 8) | (u32::from(q) << 6) | (b << 4);
    with_d_regs(base, d, n, m)
}

/// A "two registers, miscellaneous" instruction on Q registers.
pub(crate) fn enc_2misc(op: VecMiscOp, size: u8, d: u32, m: u32) -> u32 {
    let (size, a, b4) = match op {
        VecMiscOp::Rev64 => (u32::from(size), 0b00, 0b0000),
        VecMiscOp::Rev32 => (u32::from(size), 0b00, 0b0001),
        VecMiscOp::Rev16 => (u32::from(size), 0b00, 0b0010),
        VecMiscOp::PAddlS => (u32::from(size), 0b00, 0b0100),
        VecMiscOp::PAddlU => (u32::from(size), 0b00, 0b0101),
        VecMiscOp::Cnt => (0, 0b00, 0b1010),
        VecMiscOp::Mvn => (0, 0b00, 0b1011),
        VecMiscOp::Abs => (u32::from(size), 0b01, 0b0110),
        VecMiscOp::Neg => (u32::from(size), 0b01, 0b0111),
        VecMiscOp::CeqZ => {
            debug_assert!(size < 3);
            (u32::from(size), 0b01, 0b0010)
        }
        VecMiscOp::CvtF32S32 => (0b10, 0b11, 0b1100),
        VecMiscOp::CvtF32U32 => (0b10, 0b11, 0b1101),
        VecMiscOp::CvtS32F32 => (0b10, 0b11, 0b1110),
        VecMiscOp::CvtU32F32 => (0b10, 0b11, 0b1111),
    };
    let base = 0xf3b0_0000 | (size << 18) | (a << 16) | (b4 << 7) | (1 << 6);
    with_d_regs(base, d, 0, m)
}

/// Shift by immediate on Q registers.
pub(crate) fn enc_shift_imm(op: VecShiftOp, size: u8, d: u32, m: u32, amt: u8) -> u32 {
    let esize = 8u32 << size;
    let amt = u32::from(amt);
    debug_assert!(amt < esize);
    let (u, opc, imm) = match op {
        VecShiftOp::Shl => (0, 0b0101, esize + amt),
        VecShiftOp::ShrS => (0, 0b0000, 2 * esize - amt),
        VecShiftOp::ShrU => (1, 0b0000, 2 * esize - amt),
    };
    enc_shift_imm_fields(u, opc, imm, size, d, m)
}

/// `vsra.u<size> qd, qm, #amt` (unsigned shift right and accumulate).
fn enc_sra_u(size: u8, d: u32, m: u32, amt: u8) -> u32 {
    let esize = 8u32 << size;
    let amt = u32::from(amt);
    debug_assert!(amt >= 1 && amt <= esize);
    enc_shift_imm_fields(1, 0b0001, 2 * esize - amt, size, d, m)
}

fn enc_shift_imm_fields(u: u32, opc: u32, imm: u32, size: u8, d: u32, m: u32) -> u32 {
    // For 64-bit lanes, `L` is set and `imm6` holds the low 6 bits.
    let (l, imm6) = if size == 3 { (1, imm & 0x3f) } else { (0, imm) };
    let base = 0xf280_0010 | (u << 24) | (imm6 << 16) | (opc << 8) | (l << 7) | (1 << 6);
    with_d_regs(base, d, 0, m)
}

/// `vld1/vst1.<size> {dd[x]}, [base]` for lane `lane` of the Q register
/// whose first D register is `q`, or `vld1/vst1.8 {dd}, [base]` for a 64-bit
/// lane (`size` 3).
fn enc_lane_mem(load: bool, size: u8, q: u32, lane: u8, base: u32) -> u32 {
    let lane = u32::from(lane);
    let l = u32::from(load);
    if size == 3 {
        debug_assert!(lane < 2);
        let (vd, dd) = dfields(q + lane);
        return 0xf400_070f | (dd << 22) | (l << 21) | (base << 16) | (vd << 12);
    }
    let size = u32::from(size);
    let per_d = 8 >> size;
    debug_assert!(lane < 2 * per_d);
    let (vd, dd) = dfields(q + lane / per_d);
    let index_align = (lane % per_d) << (size + 1);
    0xf480_000f
        | (dd << 22)
        | (l << 21)
        | (base << 16)
        | (vd << 12)
        | (size << 10)
        | (index_align << 4)
}

/// Emit a vector instruction. Returns `false` if `inst` is not a vector
/// instruction.
pub(crate) fn emit_vec(inst: &Inst, sink: &mut MachBuffer<Inst>, state: &EmitState) -> bool {
    let put = |sink: &mut MachBuffer<Inst>, w: u32| sink.put4(w);
    match inst {
        Inst::VecMove { rd, rn } => {
            if rd.to_reg() != *rn {
                put(
                    sink,
                    enc_3same(VecAluOp::Orr, 0, true, qdw(*rd), qd(*rn), qd(*rn)),
                );
            }
        }
        Inst::VecRRR {
            op,
            size,
            rd,
            rn,
            rm,
        } => {
            // For `vshl` (register) the shift amounts are in the `Vn`
            // operand: `vshl qd, qm, qn` shifts `qm`.
            let (n, m) = match op {
                VecAluOp::ShlS | VecAluOp::ShlU => (qd(*rm), qd(*rn)),
                _ => (qd(*rn), qd(*rm)),
            };
            put(sink, enc_3same(*op, *size, true, qdw(*rd), n, m));
        }
        Inst::VecBsl { rd, ri, rn, rm } => {
            debug_assert_eq!(rd.to_reg(), *ri);
            // vbsl: rd = (rd & rn) | (~rd & rm)
            let base = 0xf310_0110 | (1 << 6);
            put(sink, with_d_regs(base, qdw(*rd), qd(*rn), qd(*rm)));
        }
        Inst::VecRR { op, size, rd, rn } => {
            put(sink, enc_2misc(*op, *size, qdw(*rd), qd(*rn)));
        }
        Inst::VecSraSelf { size, rd, rn, amt } => {
            debug_assert_eq!(rd.to_reg(), *rn);
            let d = qdw(*rd);
            put(sink, enc_sra_u(*size, d, d, *amt));
        }
        Inst::VecShiftImm {
            op,
            size,
            rd,
            rn,
            amt,
        } => {
            if *amt == 0 {
                if rd.to_reg() != *rn {
                    put(
                        sink,
                        enc_3same(VecAluOp::Orr, 0, true, qdw(*rd), qd(*rn), qd(*rn)),
                    );
                }
            } else {
                put(sink, enc_shift_imm(*op, *size, qdw(*rd), qd(*rn), *amt));
            }
        }
        Inst::VecNarrow {
            op,
            size,
            rd,
            rn,
            rm,
        } => {
            // The source size field is the destination lane size.
            let b4 = match op {
                VecNarrowOp::QMovun => 0b0100_1,
                VecNarrowOp::QMovnS => 0b0101_0,
                VecNarrowOp::QMovnU => 0b0101_1,
            };
            let enc = |d: u32, m: u32| {
                let base = 0xf3b2_0000 | (u32::from(*size) << 18) | (b4 << 6);
                with_d_regs(base, d, 0, m)
            };
            let d = qdw(*rd);
            put(sink, enc(d, qd(*rn)));
            put(sink, enc(d + 1, qd(*rm)));
        }
        Inst::VecWiden {
            signed,
            high,
            size,
            rd,
            rn,
        } => {
            // vmovl = vshll #0: imm3 = 1 << size in bits 21-19.
            let u = u32::from(!*signed);
            let imm3 = 1u32 << size;
            let base = 0xf280_0a10 | (u << 24) | (imm3 << 19);
            let m = qd(*rn) + u32::from(*high);
            put(sink, with_d_regs(base, qdw(*rd), 0, m));
        }
        Inst::VecMulLong {
            signed,
            high,
            size,
            rd,
            rn,
            rm,
        } => {
            let u = u32::from(!*signed);
            let base = 0xf280_0c00 | (u << 24) | (u32::from(*size) << 20);
            let h = u32::from(*high);
            put(sink, with_d_regs(base, qdw(*rd), qd(*rn) + h, qd(*rm) + h));
        }
        Inst::VecPairwise {
            op,
            size,
            rd,
            rn,
            rm,
        } => {
            // rd.lo = op(rn.lo, rn.hi); rd.hi = op(rm.lo, rm.hi). Compute the
            // high half first if it would otherwise be clobbered.
            let d = qdw(*rd);
            let (n, m) = (qd(*rn), qd(*rm));
            if d == m {
                put(sink, enc_3same(*op, *size, false, d + 1, m, m + 1));
                put(sink, enc_3same(*op, *size, false, d, n, n + 1));
            } else {
                put(sink, enc_3same(*op, *size, false, d, n, n + 1));
                put(sink, enc_3same(*op, *size, false, d + 1, m, m + 1));
            }
        }
        Inst::VecDup { size, rd, rn } => {
            let (b, e) = match size {
                0 => (1, 0),
                1 => (0, 1),
                _ => (0, 0),
            };
            let (vd, dd) = dfields(qdw(*rd));
            put(
                sink,
                0xee80_0b10
                    | (b << 22)
                    | (1 << 21)
                    | (vd << 16)
                    | (gp(*rn) << 12)
                    | (dd << 7)
                    | (e << 5),
            );
        }
        Inst::VecDupLane { size, rd, rn, lane } => {
            let lanes_per_d = 8u8 >> size;
            let m = qd(*rn) + u32::from(lane / lanes_per_d);
            let x = u32::from(lane % lanes_per_d);
            let imm4 = match size {
                0 => (x << 1) | 1,
                1 => (x << 2) | 0b10,
                _ => (x << 3) | 0b100,
            };
            let base = 0xf3b0_0c00 | (imm4 << 16) | (1 << 6);
            put(sink, with_d_regs(base, qdw(*rd), 0, m));
        }
        Inst::VecMovToLane {
            size,
            rd,
            ri,
            rn,
            lane,
        } => {
            debug_assert_eq!(rd.to_reg(), *ri);
            let lanes_per_d = 8u8 >> size;
            let d = qdw(*rd) + u32::from(lane / lanes_per_d);
            let x = u32::from(lane % lanes_per_d);
            let (opc1, opc2) = match size {
                0 => (0b10 | (x >> 2), x & 0b11),
                1 => (x >> 1, ((x & 1) << 1) | 1),
                _ => (x, 0),
            };
            let (vd, dd) = dfields(d);
            put(
                sink,
                0xee00_0b10 | (opc1 << 21) | (vd << 16) | (gp(*rn) << 12) | (dd << 7) | (opc2 << 5),
            );
        }
        Inst::VecMovFromLane {
            size,
            signed,
            rd,
            rn,
            lane,
        } => {
            let lanes_per_d = 8u8 >> size;
            let d = qd(*rn) + u32::from(lane / lanes_per_d);
            let x = u32::from(lane % lanes_per_d);
            let (opc1, opc2) = match size {
                0 => (0b10 | (x >> 2), x & 0b11),
                1 => (x >> 1, ((x & 1) << 1) | 1),
                _ => (x, 0),
            };
            let u = u32::from(!*signed && *size < 2);
            let (vn, nn) = dfields(d);
            put(
                sink,
                0xee10_0b10
                    | (u << 23)
                    | (opc1 << 21)
                    | (vn << 16)
                    | (gpw(*rd) << 12)
                    | (nn << 7)
                    | (opc2 << 5),
            );
        }
        Inst::VecFromGprs { rd, r0, r1, r2, r3 } => {
            let d = qdw(*rd);
            put(sink, vmov_d_from_core(d, gp(*r0), gp(*r1)));
            put(sink, vmov_d_from_core(d + 1, gp(*r2), gp(*r3)));
        }
        Inst::VecOrrHalves { rd, rn } => {
            let n = qd(*rn);
            put(sink, enc_3same(VecAluOp::Orr, 0, false, fdw(*rd), n, n + 1));
        }
        Inst::VecToFpu { rd, rn, half } => {
            put(sink, vmov_d(fdw(*rd), qd(*rn) + u32::from(*half)));
        }
        Inst::VecFromFpu { rd, ri, rn, half } => {
            debug_assert_eq!(rd.to_reg(), *ri);
            put(sink, vmov_d(qdw(*rd) + u32::from(*half), fd(*rn)));
        }
        Inst::VecToFpu32 { rd, rn, lane, tmp } => {
            // Copy the containing D half into `tmp`, then the lane into the
            // low half of `rd`.
            let t = fdw(*tmp);
            put(sink, vmov_d(t, qd(*rn) + u32::from(lane / 2)));
            put(sink, vmov_s(2 * fdw(*rd), 2 * t + u32::from(lane % 2)));
        }
        Inst::VecFromFpu32 {
            rd,
            ri,
            rn,
            lane,
            tmp,
        } => {
            debug_assert_eq!(rd.to_reg(), *ri);
            let t = fdw(*tmp);
            let d = qdw(*rd) + u32::from(lane / 2);
            put(sink, vmov_d(t, d));
            put(sink, vmov_s(2 * t + u32::from(lane % 2), 2 * fd(*rn)));
            put(sink, vmov_d(d, t));
        }
        Inst::VecLoadConst { rd, lo, hi } => {
            // vldr d(2n), [pc, #lo] ; vldr d(2n+1), [pc, #hi]
            let d = qdw(*rd);
            for (i, c) in [(0, lo), (1, hi)] {
                let label = sink.get_label_for_constant(*c);
                sink.use_label_at_offset(sink.cur_offset(), label, LabelUse::Vldr8);
                let dd = d + i;
                put(sink, 0xed9f_0b00 | ((dd >> 4) << 22) | ((dd & 0xf) << 12));
            }
        }
        Inst::VecRint {
            mode,
            size,
            rd,
            rn,
            tmp,
        } => {
            let (d, n) = (qdw(*rd), qd(*rn));
            if size.is_f64() {
                // VFP can access d16-d31 directly.
                for h in 0..2 {
                    put(sink, super::emit::enc_vrint(*mode, true, d + h, n + h));
                }
            } else {
                // Each D half goes through `tmp`, whose S halves are
                // accessible: vmov.f64 tmp, n ; vrint s, s (x2) ;
                // vmov.f64 d, tmp.
                let t = fdw(*tmp);
                for h in 0..2 {
                    put(sink, vmov_d(t, n + h));
                    put(sink, super::emit::enc_vrint(*mode, false, 2 * t, 2 * t));
                    put(
                        sink,
                        super::emit::enc_vrint(*mode, false, 2 * t + 1, 2 * t + 1),
                    );
                    put(sink, vmov_d(d + h, t));
                }
            }
        }
        Inst::VecZero { rd } => {
            // vmov.i32 qd, #0
            let d = qdw(*rd);
            put(sink, 0xf280_0050 | ((d >> 4) << 22) | ((d & 0xf) << 12));
        }
        Inst::VecLoadLow {
            size,
            zero_rest,
            rd,
            mem,
            flags,
        } => {
            let d = qdw(*rd);
            if *zero_rest {
                // vmov.i32 qd, #0
                put(sink, 0xf280_0050 | ((d >> 4) << 22) | ((d & 0xf) << 12));
            }
            let base = vec_mem_base(sink, state, mem);
            if let Some(trap_code) = flags.trap_code() {
                sink.add_trap(trap_code);
            }
            let (vd, dd) = dfields(d);
            let word = match size {
                // vld1.8 {dd}, [base]
                3 => 0xf420_070f,
                // vld1.32 {dd[0]}, [base]
                2 => 0xf4a0_080f,
                _ => unreachable!(),
            };
            put(sink, word | (dd << 22) | (base << 16) | (vd << 12));
        }
        Inst::VecLoadLane {
            size,
            rd,
            ri,
            lane,
            mem,
            flags,
        } => {
            debug_assert_eq!(rd.to_reg(), *ri);
            let base = vec_mem_base(sink, state, mem);
            if let Some(trap_code) = flags.trap_code() {
                sink.add_trap(trap_code);
            }
            put(sink, enc_lane_mem(true, *size, qdw(*rd), *lane, base));
        }
        Inst::VecStoreLane {
            size,
            rt,
            lane,
            mem,
            flags,
        } => {
            let base = vec_mem_base(sink, state, mem);
            if let Some(trap_code) = flags.trap_code() {
                sink.add_trap(trap_code);
            }
            put(sink, enc_lane_mem(false, *size, qd(*rt), *lane, base));
        }
        Inst::VecLoadSplat {
            size,
            rd,
            mem,
            flags,
        } => {
            debug_assert!(*size < 3);
            let base = vec_mem_base(sink, state, mem);
            if let Some(trap_code) = flags.trap_code() {
                sink.add_trap(trap_code);
            }
            // vld1.<size> {dd[], dd+1[]}, [base]: single element to all lanes
            // of two registers (`T` set).
            let (vd, dd) = dfields(qdw(*rd));
            put(
                sink,
                0xf4a0_0c2f | (dd << 22) | (base << 16) | (vd << 12) | (u32::from(*size) << 6),
            );
        }
        Inst::VecLoad { rd, mem, flags } => {
            let base = vec_mem_base(sink, state, mem);
            if let Some(trap_code) = flags.trap_code() {
                sink.add_trap(trap_code);
            }
            // vld1.8 {dd, dd+1}, [base]
            let (vd, dd) = dfields(qdw(*rd));
            put(sink, 0xf420_0a0f | (dd << 22) | (base << 16) | (vd << 12));
        }
        Inst::VecStore { rt, mem, flags } => {
            let base = vec_mem_base(sink, state, mem);
            if let Some(trap_code) = flags.trap_code() {
                sink.add_trap(trap_code);
            }
            let (vd, dd) = dfields(qd(*rt));
            put(sink, 0xf400_0a0f | (dd << 22) | (base << 16) | (vd << 12));
        }
        Inst::VecTbl {
            rd,
            tbl1,
            tbl2,
            idx,
        } => {
            let (n, len) = match tbl2 {
                None => (qd(*tbl1), 1),
                Some(tbl2) => {
                    debug_assert_eq!(qd(*tbl2), qd(*tbl1) + 2);
                    (qd(*tbl1), 3)
                }
            };
            let d = qdw(*rd);
            let m = qd(*idx);
            // vtbl.8 dd, {dn..}, dm (twice, for each half of the index).
            // `rd` is an early def, so it doesn't overlap the table or index.
            let base = 0xf3b0_0800 | (len << 8);
            put(sink, with_d_regs(base, d, n, m));
            put(sink, with_d_regs(base, d + 1, n, m + 1));
        }
        Inst::VecExt { rd, rn, rm, imm } => {
            let base = 0xf2b0_0000 | (u32::from(*imm) << 8) | (1 << 6);
            put(sink, with_d_regs(base, qdw(*rd), qd(*rn), qd(*rm)));
        }
        Inst::VecFpuRRR {
            op,
            size,
            rd,
            rn,
            rm,
            t1,
            t2,
        } => {
            let base = fpu_op2_base(*op);
            let (d, n, m) = (qdw(*rd), qd(*rn), qd(*rm));
            for h in 0..2 {
                if size.is_f64() {
                    put(sink, enc_vfp_d(COND_AL, base, d + h, n + h, m + h));
                } else {
                    let (a, b) = (fdw(*t1), fdw(*t2));
                    put(sink, vmov_d(a, n + h));
                    put(sink, vmov_d(b, m + h));
                    put(sink, enc_vfp_s(COND_AL, base, 2 * a, 2 * a, 2 * b));
                    put(
                        sink,
                        enc_vfp_s(COND_AL, base, 2 * a + 1, 2 * a + 1, 2 * b + 1),
                    );
                    put(sink, vmov_d(d + h, a));
                }
            }
        }
        Inst::VecFpuRR {
            op,
            size,
            rd,
            rn,
            t1,
        } => {
            let base = fpu_op1_base(*op);
            let (d, n) = (qdw(*rd), qd(*rn));
            for h in 0..2 {
                if size.is_f64() {
                    put(sink, enc_vfp_d(COND_AL, base, d + h, 0, n + h));
                } else {
                    let a = fdw(*t1);
                    put(sink, vmov_d(a, n + h));
                    put(sink, enc_vfp_s(COND_AL, base, 2 * a, 0, 2 * a));
                    put(sink, enc_vfp_s(COND_AL, base, 2 * a + 1, 0, 2 * a + 1));
                    put(sink, vmov_d(d + h, a));
                }
            }
        }
        Inst::VecFpuFma {
            size,
            rd,
            ra,
            rn,
            rm,
            t1,
            t2,
            t3,
        } => {
            let (d, a, n, m) = (qdw(*rd), qd(*ra), qd(*rn), qd(*rm));
            let (x, y, z) = (fdw(*t1), fdw(*t2), fdw(*t3));
            for h in 0..2 {
                put(sink, vmov_d(x, a + h));
                if size.is_f64() {
                    put(sink, enc_vfp_d(COND_AL, VFMA, x, n + h, m + h));
                } else {
                    put(sink, vmov_d(y, n + h));
                    put(sink, vmov_d(z, m + h));
                    put(sink, enc_vfp_s(COND_AL, VFMA, 2 * x, 2 * y, 2 * z));
                    put(
                        sink,
                        enc_vfp_s(COND_AL, VFMA, 2 * x + 1, 2 * y + 1, 2 * z + 1),
                    );
                }
                put(sink, vmov_d(d + h, x));
            }
        }
        Inst::VecFpuCmp {
            cond,
            size,
            rd,
            rn,
            rm,
            t1,
            t2,
            gtmp,
        } => {
            let (c1, c2) = Cond::from_floatcc(*cond);
            let (d, n, m) = (qdw(*rd), qd(*rn), qd(*rm));
            let g = gpw(*gtmp);
            // Lane masks are computed into `g` and inserted into `rd`; since
            // `rd` is an early def, it doesn't overlap the inputs.
            for h in 0..2 {
                let lanes: &[u32] = if size.is_f64() { &[0] } else { &[0, 1] };
                if !size.is_f64() {
                    put(sink, vmov_d(fdw(*t1), n + h));
                    put(sink, vmov_d(fdw(*t2), m + h));
                }
                for &l in lanes {
                    if size.is_f64() {
                        put(sink, enc_vfp_d(COND_AL, VCMP, n + h, 0, m + h));
                    } else {
                        put(
                            sink,
                            enc_vfp_s(COND_AL, VCMP, 2 * fdw(*t1) + l, 0, 2 * fdw(*t2) + l),
                        );
                    }
                    put(sink, VMRS_APSR);
                    // g = cond ? -1 : 0
                    put(sink, enc_mov_imm(COND_AL, g, 0));
                    put(sink, enc_mvn_imm(c1.bits(), g, 0));
                    if let Some(c2) = c2 {
                        put(sink, enc_mvn_imm(c2.bits(), g, 0));
                    }
                    if size.is_f64() {
                        put(sink, vmov_to_scalar32(COND_AL, d + h, 0, g));
                        put(sink, vmov_to_scalar32(COND_AL, d + h, 1, g));
                    } else {
                        put(sink, vmov_to_scalar32(COND_AL, d + h, l, g));
                    }
                }
            }
        }
        Inst::VecFpuMinMax {
            is_min,
            size,
            rd,
            rn,
            rm,
            t1,
            t2,
            gtmp,
        } => {
            // Run the scalar min/max sequence on each lane.
            let (d, n, m) = (qdw(*rd), qd(*rn), qd(*rm));
            for lane in 0..(if size.is_f64() { 2 } else { 4 }) {
                lane_to_fpu(sink, *size, *t1, n, lane);
                lane_to_fpu(sink, *size, *t2, m, lane);
                Inst::FpuMinMax {
                    is_min: *is_min,
                    size: *size,
                    rd: *t1,
                    rn: t1.to_reg(),
                    rm: t2.to_reg(),
                    tmp: *gtmp,
                }
                .emit(sink, &EmitInfo::dummy(), &mut state.clone());
                fpu_to_lane(sink, *size, d, lane, *t1, *gtmp);
            }
        }
        Inst::PopcntNeon {
            rd,
            lo,
            hi,
            tmp,
            is64,
        } => {
            let t = fdw(*tmp);
            if *is64 {
                // vmov dt, lo, hi
                put(sink, vmov_d_from_core(t, gp(*lo), gp(*hi)));
            } else {
                put(sink, vmov_to_scalar32(COND_AL, t, 0, gp(*lo)));
            }
            // Count the bits of each byte and add them up pairwise (on D
            // registers); for a 32-bit input, lane 0 only covers its bytes.
            let d_form = |w: u32| w & !(1 << 6);
            put(sink, d_form(enc_2misc(VecMiscOp::Cnt, 0, t, t)));
            put(sink, d_form(enc_2misc(VecMiscOp::PAddlU, 0, t, t)));
            put(sink, d_form(enc_2misc(VecMiscOp::PAddlU, 1, t, t)));
            if *is64 {
                put(sink, d_form(enc_2misc(VecMiscOp::PAddlU, 2, t, t)));
            }
            put(sink, vmov_from_scalar32(gpw(*rd), t, 0));
        }
        Inst::VecMulI64 { rd, rn, rm, tb } => {
            let (a, b, d, t) = (qd(*rn), qd(*rm), qdw(*rd), qdw(*tb));
            // d = [a.lo * b.hi + a.hi * b.lo] << 32 in each 64-bit lane.
            put(sink, enc_2misc(VecMiscOp::Rev64, 2, d, b));
            put(sink, enc_3same(VecAluOp::Mul, 2, true, d, d, a));
            put(sink, enc_2misc(VecMiscOp::PAddlU, 2, d, d));
            put(sink, enc_shift_imm(VecShiftOp::Shl, 3, d, d, 32));
            // d += a.lo * b.lo (64-bit products).
            put(sink, with_d_regs(VMOVN_I64, t, 0, a));
            put(sink, with_d_regs(VMOVN_I64, t + 1, 0, b));
            put(sink, with_d_regs(VMLAL_U32, d, t, t + 1));
        }
        Inst::VecCmpF32 {
            cond,
            rd,
            rn,
            rm,
            ta,
            tb,
        } => {
            let (x, y, d) = (qd(*rn), qd(*rm), qdw(*rd));
            let (a, b) = (qdw(*ta), qdw(*tb));
            // Order-preserving integer keys (see `VecMinMaxF32`), with
            // negative keys moved up by one so that -0 and +0 are equal:
            // key(x) in `b` and key(y) in `a`.
            for (s, k, v) in [(a, b, x), (d, a, y)] {
                put(sink, enc_shift_imm(VecShiftOp::ShrS, 2, s, v, 31));
                put(sink, enc_shift_imm(VecShiftOp::ShrU, 2, k, s, 1));
                put(sink, with_d_regs(VEOR, k, k, v));
                put(sink, with_d_regs(VSUB_I32, k, k, s));
            }
            let (kx, ky) = (b, a);
            let (word, invert) = match cond {
                FloatCC::Equal => (with_d_regs(VCEQ_I32, d, kx, ky), false),
                FloatCC::NotEqual => (with_d_regs(VCEQ_I32, d, kx, ky), true),
                FloatCC::LessThan => (with_d_regs(VCGT_S32, d, ky, kx), false),
                FloatCC::LessThanOrEqual => (with_d_regs(VCGE_S32, d, ky, kx), false),
                FloatCC::GreaterThan => (with_d_regs(VCGT_S32, d, kx, ky), false),
                FloatCC::GreaterThanOrEqual => (with_d_regs(VCGE_S32, d, kx, ky), false),
                _ => unreachable!("unsupported condition {cond}"),
            };
            put(sink, word);
            // Ordered comparisons are false for lanes with a NaN input
            // (`ne`, their complement, true).
            put(sink, with_d_regs(VCEQ_F32, a, x, x));
            put(sink, with_d_regs(VCEQ_F32, b, y, y));
            put(sink, with_d_regs(VAND, a, a, b));
            put(sink, with_d_regs(VAND, d, d, a));
            if invert {
                put(sink, enc_2misc(VecMiscOp::Mvn, 0, d, d));
            }
        }
        Inst::VecMinMaxF32 {
            is_min,
            rd,
            rn,
            rm,
            ta,
            tb,
        } => {
            let (x, y, d) = (qd(*rn), qd(*rm), qdw(*rd));
            let (a, b) = (qdw(*ta), qdw(*tb));
            // Order-preserving integer keys: negative values have their
            // magnitude bits flipped.
            for (t, v) in [(a, x), (b, y)] {
                put(sink, enc_shift_imm(VecShiftOp::ShrS, 2, t, v, 31));
                put(sink, enc_shift_imm(VecShiftOp::ShrU, 2, t, t, 1));
                put(sink, with_d_regs(VEOR, t, t, v));
            }
            // d = (min ? key(y) > key(x) : key(x) > key(y)) ? x : y; equal
            // keys are equal bit patterns.
            let (p, q) = if *is_min { (b, a) } else { (a, b) };
            put(sink, with_d_regs(VCGT_S32, d, p, q));
            put(sink, with_d_regs(VBSL, d, x, y));
            // Lanes where either input is NaN get the canonical NaN.
            put(sink, with_d_regs(VCEQ_F32, a, x, x));
            put(sink, with_d_regs(VCEQ_F32, b, y, y));
            put(sink, with_d_regs(VAND, a, a, b));
            put(sink, with_d_regs(VADD_F32, b, x, y));
            put(sink, with_d_regs(VBIF, d, b, a));
        }
        Inst::VecRoundF64 {
            mode,
            rd,
            rn,
            qa,
            qb,
            qc,
            t,
            gtmp,
        } => {
            let (x, d) = (qd(*rn), qdw(*rd));
            let (a, b, c, t) = (qdw(*qa), qdw(*qb), qdw(*qc), fdw(*t));
            // a = 2^52 and c = the sign mask, in both 64-bit lanes.
            put(sink, vmov_i32_byte3(a, 0x43));
            put(sink, vorr_i32_byte2(a, 0x30));
            put(sink, enc_shift_imm(VecShiftOp::Shl, 3, a, a, 32));
            put(sink, vmov_i32_byte3(c, 0x80));
            put(sink, enc_shift_imm(VecShiftOp::Shl, 3, c, c, 32));
            // b = 2^52 with the sign of x.
            put(sink, enc_3same(VecAluOp::Orr, 0, true, b, a, a));
            put(sink, with_d_regs(VBIT, b, x, c));
            let set_rmode = mode != &FpuRoundMode::Nearest;
            if set_rmode {
                super::emit::emit_set_rounding_mode(sink, *mode, gpw(*gtmp));
            }
            for h in 0..2 {
                // d = (x + b) - b, rounded in the current mode; but x itself
                // if |x| >= 2^52 (integral or infinite; NaN compares false
                // and keeps the quieted sum).
                put(sink, enc_vfp_d(COND_AL, VADD, d + h, x + h, b + h));
                put(sink, enc_vfp_d(COND_AL, VSUB, d + h, d + h, b + h));
                put(sink, enc_vfp_d(COND_AL, VABS, t, 0, x + h));
                put(sink, enc_vfp_d(COND_AL, VCMP, t, 0, a + h));
                put(sink, VMRS_APSR);
                put(sink, enc_vfp_d(Cond::Ge.bits(), VMOV, d + h, 0, x + h));
            }
            if set_rmode {
                put(sink, super::emit::enc_vmsr(gpw(*gtmp)));
            }
            // The result has the sign of the input (also when zero).
            put(sink, with_d_regs(VBIT, d, x, c));
        }
        Inst::VecRoundF32 {
            mode,
            rd,
            rn,
            qa,
            qb,
            t1,
            t2,
            gtmp,
        } => {
            let (x, d) = (qd(*rn), qdw(*rd));
            let (a, b) = (qdw(*qa), qdw(*qb));
            let (t1, t2) = (fdw(*t1), fdw(*t2));
            // Convert each lane to an integer and back with VFP, which (unlike
            // NEON) handles denormals: `vcvtr` rounds in the FPSCR mode,
            // `vcvt` toward zero. In-range lanes are below 2^23 in magnitude,
            // so the conversions are exact.
            put(sink, vmov_d(t1, x));
            put(sink, vmov_d(t2, x + 1));
            let set_rmode = matches!(mode, FpuRoundMode::Ceil | FpuRoundMode::Floor);
            if set_rmode {
                super::emit::emit_set_rounding_mode(sink, *mode, gpw(*gtmp));
            }
            let to_int = if *mode == FpuRoundMode::Trunc {
                VCVT_TO_S32
            } else {
                VCVTR_TO_S32
            };
            for s in [2 * t1, 2 * t1 + 1, 2 * t2, 2 * t2 + 1] {
                put(sink, enc_vfp_s(COND_AL, to_int, s, 0, s));
                put(sink, enc_vfp_s(COND_AL, VCVT_FROM_S32, s, 0, s));
            }
            if set_rmode {
                put(sink, super::emit::enc_vmsr(gpw(*gtmp)));
            }
            put(sink, vmov_d(a, t1));
            put(sink, vmov_d(a + 1, t2));
            // The result has the sign of the input (also when zero).
            put(sink, vmov_i32_byte3(b, 0x80));
            put(sink, with_d_regs(VBIT, a, x, b));
            // Inputs with `|x| >= 2^23` are integral (or infinite) already.
            // (NaN compares false here, and denormals are flushed to zero by
            // the NEON comparison, which doesn't matter.)
            put(sink, vmov_i32_byte3(b, 0x4b));
            put(sink, with_d_regs(VACGE_F32, b, x, b));
            put(sink, with_d_regs(VBIT, a, x, b));
            // NaN lanes: `x + x` is the canonical NaN, as NEON always produces
            // the default NaN.
            put(sink, with_d_regs(VCEQ_F32, b, x, x));
            put(sink, with_d_regs(VADD_F32, d, x, x));
            put(sink, with_d_regs(VBIT, d, a, b));
        }
        Inst::VecFpuRound {
            mode,
            size,
            rd,
            rn,
            t1,
            t2,
            t3,
            gtmp1,
            gtmp2,
        } => {
            let (d, n) = (qdw(*rd), qd(*rn));
            for lane in 0..(if size.is_f64() { 2 } else { 4 }) {
                lane_to_fpu(sink, *size, *t1, n, lane);
                Inst::FpuRound {
                    mode: *mode,
                    size: *size,
                    rd: *t2,
                    rn: t1.to_reg(),
                    tmp1: *gtmp1,
                    tmp2: *gtmp2,
                    ftmp: *t3,
                }
                .emit(sink, &EmitInfo::dummy(), &mut state.clone());
                fpu_to_lane(sink, *size, d, lane, *t2, *gtmp1);
            }
        }
        _ => return false,
    }
    true
}

/// Copy a float lane of the vector whose low D register is `n` into the
/// float register `t` (an f32 lane goes into the low S half).
fn lane_to_fpu(sink: &mut MachBuffer<Inst>, size: FpuSize, t: Writable<Reg>, n: u32, lane: u32) {
    let t = fdw(t);
    if size.is_f64() {
        sink.put4(vmov_d(t, n + lane));
    } else {
        sink.put4(vmov_d(t, n + lane / 2));
        if lane % 2 == 1 {
            sink.put4(vmov_s(2 * t, 2 * t + 1));
        }
    }
}

/// Write the float register `t` into a lane of the vector whose low D
/// register is `d`, using `g` as a core temporary for f32 lanes.
fn fpu_to_lane(
    sink: &mut MachBuffer<Inst>,
    size: FpuSize,
    d: u32,
    lane: u32,
    t: Writable<Reg>,
    g: Writable<Reg>,
) {
    let t = fdw(t);
    if size.is_f64() {
        sink.put4(vmov_d(d + lane, t));
    } else {
        let g = gpw(g);
        sink.put4(vmov_core_from_s(g, 2 * t));
        sink.put4(vmov_to_scalar32(COND_AL, d + lane / 2, lane % 2, g));
    }
}

/// Compute the address for a vector load/store into a register (`ip` when
/// an offset is involved) and return its number.
/// The base register for a NEON access to `mem` (which has no offset field),
/// computing the address into `ip` if needed.
pub(crate) fn vec_mem_base(sink: &mut MachBuffer<Inst>, state: &EmitState, mem: &AMode) -> u32 {
    match mem {
        AMode::RegReg { base, index, shift } => {
            let sh = if *shift == 0 {
                ShiftOpAndAmt::none()
            } else {
                ShiftOpAndAmt::new(ShiftOp::Lsl, *shift)
            };
            sink.put4(enc_dp_reg(
                COND_AL,
                0b0100,
                false,
                gp(*base),
                IP,
                gp(*index),
                sh,
            ));
            IP
        }
        _ => {
            let (base, offset) = mem.resolve(state).unwrap();
            if offset == 0 {
                gp(base)
            } else {
                emit_add_imm_chunks(sink, IP, gp(base), offset);
                IP
            }
        }
    }
}

fn enc_mov_imm(cond: u32, rd: u32, imm: u32) -> u32 {
    enc_dp_imm(
        cond,
        0b1101,
        false,
        0,
        rd,
        ModImm::maybe_from_u32(imm).unwrap(),
    )
}

fn enc_mvn_imm(cond: u32, rd: u32, imm: u32) -> u32 {
    enc_dp_imm(
        cond,
        0b1111,
        false,
        0,
        rd,
        ModImm::maybe_from_u32(imm).unwrap(),
    )
}
