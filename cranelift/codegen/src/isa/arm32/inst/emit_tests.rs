//! Encoding tests for the arm32 backend.
//!
//! The expected encodings were produced with the GNU assembler
//! (`arm-linux-gnueabihf-as -march=armv7ve -mfpu=neon-vfpv4`) from the
//! assembly text given for each case.

use crate::ir::condcodes::FloatCC;
use crate::ir::{MemFlagsData, TrapCode};
use crate::isa::arm32::inst::*;
use crate::isa::arm32::settings as arm32_settings;
use crate::settings;
use alloc::vec::Vec;

fn r(n: u8) -> Reg {
    rreg(n)
}

fn w(n: u8) -> Writable<Reg> {
    Writable::from_reg(rreg(n))
}

fn d(n: u8) -> Reg {
    dreg(n)
}

fn wd(n: u8) -> Writable<Reg> {
    Writable::from_reg(dreg(n))
}

fn q(n: u8) -> Reg {
    qreg(n)
}

fn wq(n: u8) -> Writable<Reg> {
    Writable::from_reg(qreg(n))
}

fn imm(v: u32) -> ModImm {
    ModImm::maybe_from_u32(v).unwrap()
}

fn sh(op: ShiftOp, amt: u8) -> ShiftOpAndAmt {
    ShiftOpAndAmt::new(op, amt)
}

fn mem(base: u8, offset: i32) -> AMode {
    AMode::RegOffset {
        base: r(base),
        offset,
    }
}

fn flags() -> MemFlagsData {
    MemFlagsData::trusted()
}

/// Test cases: the instruction, the GNU assembler text that produces the
/// expected encoding, and the expected little-endian instruction words.
fn cases() -> Vec<(Inst, &'static str, &'static [u32])> {
    let none = ShiftOpAndAmt::none();
    vec![
        // Data processing.
        (
            Inst::AluRRR {
                op: AluOp::Add,
                s: false,
                rd: w(0),
                rn: r(1),
                rm: r(2),
                shift: none,
            },
            "add r0, r1, r2",
            &[0xe0810002],
        ),
        (
            Inst::AluRRR {
                op: AluOp::Sub,
                s: true,
                rd: w(3),
                rn: r(4),
                rm: r(5),
                shift: none,
            },
            "subs r3, r4, r5",
            &[0xe0543005],
        ),
        (
            Inst::AluRRR {
                op: AluOp::Adc,
                s: false,
                rd: w(10),
                rn: r(9),
                rm: r(8),
                shift: none,
            },
            "adc r10, r9, r8",
            &[0xe0a9a008],
        ),
        (
            Inst::AluRRR {
                op: AluOp::Sbc,
                s: true,
                rd: w(7),
                rn: r(6),
                rm: r(5),
                shift: none,
            },
            "sbcs r7, r6, r5",
            &[0xe0d67005],
        ),
        (
            Inst::AluRRR {
                op: AluOp::Rsb,
                s: false,
                rd: w(1),
                rn: r(2),
                rm: r(3),
                shift: sh(ShiftOp::Lsl, 3),
            },
            "rsb r1, r2, r3, lsl #3",
            &[0xe0621183],
        ),
        (
            Inst::AluRRR {
                op: AluOp::Rsc,
                s: false,
                rd: w(1),
                rn: r(2),
                rm: r(3),
                shift: none,
            },
            "rsc r1, r2, r3",
            &[0xe0e21003],
        ),
        (
            Inst::AluRRR {
                op: AluOp::And,
                s: false,
                rd: w(0),
                rn: r(1),
                rm: r(2),
                shift: sh(ShiftOp::Lsr, 1),
            },
            "and r0, r1, r2, lsr #1",
            &[0xe00100a2],
        ),
        (
            Inst::AluRRR {
                op: AluOp::Orr,
                s: true,
                rd: w(0),
                rn: r(1),
                rm: r(2),
                shift: none,
            },
            "orrs r0, r1, r2",
            &[0xe1910002],
        ),
        (
            Inst::AluRRR {
                op: AluOp::Eor,
                s: false,
                rd: w(0),
                rn: r(1),
                rm: r(2),
                shift: sh(ShiftOp::Asr, 31),
            },
            "eor r0, r1, r2, asr #31",
            &[0xe0210fc2],
        ),
        (
            Inst::AluRRR {
                op: AluOp::Bic,
                s: false,
                rd: w(0),
                rn: r(1),
                rm: r(2),
                shift: sh(ShiftOp::Ror, 7),
            },
            "bic r0, r1, r2, ror #7",
            &[0xe1c103e2],
        ),
        (
            Inst::AluRRRShift {
                op: AluOp::Orr,
                s: false,
                rd: w(0),
                rn: r(1),
                rm: r(2),
                shift_op: ShiftOp::Lsr,
                rs: r(3),
            },
            "orr r0, r1, r2, lsr r3",
            &[0xe1810332],
        ),
        (
            Inst::AluRRImm {
                op: AluOp::Add,
                s: false,
                rd: w(0),
                rn: r(1),
                imm: imm(255),
            },
            "add r0, r1, #255",
            &[0xe28100ff],
        ),
        (
            Inst::AluRRImm {
                op: AluOp::Sub,
                s: true,
                rd: w(12),
                rn: r(12),
                imm: imm(1),
            },
            "subs ip, ip, #1",
            &[0xe25cc001],
        ),
        (
            Inst::AluRRImm {
                op: AluOp::Rsb,
                s: false,
                rd: w(0),
                rn: r(1),
                imm: imm(0),
            },
            "rsb r0, r1, #0",
            &[0xe2610000],
        ),
        (
            Inst::AluRRImm {
                op: AluOp::Bic,
                s: false,
                rd: w(0),
                rn: r(1),
                imm: imm(0x8000_0000),
            },
            "bic r0, r1, #0x80000000",
            &[0xe3c10102],
        ),
        (
            Inst::AluRRImm {
                op: AluOp::And,
                s: false,
                rd: w(0),
                rn: r(1),
                imm: imm(0xf000_000f),
            },
            "and r0, r1, #0xf000000f",
            &[0xe20102ff],
        ),
        (
            Inst::MovRR {
                op: MovOp::Mov,
                s: false,
                rd: w(0),
                rm: r(1),
                shift: none,
            },
            "mov r0, r1",
            &[0xe1a00001],
        ),
        (
            Inst::MovRR {
                op: MovOp::Mov,
                s: true,
                rd: w(0),
                rm: r(1),
                shift: sh(ShiftOp::Lsl, 16),
            },
            "lsls r0, r1, #16",
            &[0xe1b00801],
        ),
        (
            Inst::MovRR {
                op: MovOp::Mov,
                s: false,
                rd: w(0),
                rm: r(1),
                shift: sh(ShiftOp::Asr, 31),
            },
            "asr r0, r1, #31",
            &[0xe1a00fc1],
        ),
        (
            Inst::MovRR {
                op: MovOp::Mvn,
                s: false,
                rd: w(0),
                rm: r(1),
                shift: none,
            },
            "mvn r0, r1",
            &[0xe1e00001],
        ),
        (
            Inst::MovRRShift {
                rd: w(0),
                rm: r(1),
                shift_op: ShiftOp::Ror,
                rs: r(2),
            },
            "ror r0, r1, r2",
            &[0xe1a00271],
        ),
        (
            Inst::MovImm {
                op: MovOp::Mov,
                rd: w(0),
                imm: imm(0xff00),
            },
            "mov r0, #0xff00",
            &[0xe3a00cff],
        ),
        (
            Inst::MovImm {
                op: MovOp::Mvn,
                rd: w(0),
                imm: imm(0),
            },
            "mvn r0, #0",
            &[0xe3e00000],
        ),
        (
            Inst::MovW {
                rd: w(5),
                imm: 0xbeef,
            },
            "movw r5, #0xbeef",
            &[0xe30b5eef],
        ),
        (
            Inst::MovT {
                rd: w(5),
                rn: r(5),
                imm: 0xdead,
            },
            "movt r5, #0xdead",
            &[0xe34d5ead],
        ),
        (
            Inst::Mov {
                rd: w(11),
                rm: r(13),
            },
            "mov fp, sp",
            &[0xe1a0b00d],
        ),
        (
            Inst::CmpRR {
                op: CmpOp::Cmp,
                rn: r(0),
                rm: r(1),
                shift: none,
            },
            "cmp r0, r1",
            &[0xe1500001],
        ),
        (
            Inst::CmpRR {
                op: CmpOp::Tst,
                rn: r(0),
                rm: r(1),
                shift: sh(ShiftOp::Asr, 31),
            },
            "tst r0, r1, asr #31",
            &[0xe1100fc1],
        ),
        (
            Inst::CCmpRR {
                op: CmpOp::Cmp,
                cond: Cond::Eq,
                rn: r(0),
                rm: r(2),
            },
            "cmpeq r0, r2",
            &[0x01500002],
        ),
        (
            Inst::CmpRImm {
                op: CmpOp::Cmn,
                rn: r(3),
                imm: imm(1),
            },
            "cmn r3, #1",
            &[0xe3730001],
        ),
        (
            Inst::CCmpRImm {
                op: CmpOp::Cmp,
                cond: Cond::Eq,
                rn: r(3),
                imm: imm(0),
            },
            "cmpeq r3, #0",
            &[0x03530000],
        ),
        (
            Inst::CSel {
                rd: w(0),
                cond: Cond::Lt,
                rn: r(1),
                rm: r(2),
            },
            "mov r0, r2\nmovlt r0, r1",
            &[0xe1a00002, 0xb1a00001],
        ),
        (
            Inst::CSel {
                rd: w(1),
                cond: Cond::Lt,
                rn: r(1),
                rm: r(2),
            },
            "movge r1, r2",
            &[0xa1a01002],
        ),
        (
            Inst::CSel {
                rd: w(2),
                cond: Cond::Hi,
                rn: r(1),
                rm: r(2),
            },
            "movhi r2, r1",
            &[0x81a02001],
        ),
        (
            Inst::CSelImm {
                op: MovOp::Mov,
                rd: w(4),
                cond: Cond::Hi,
                imm: imm(0),
                rm: r(4),
            },
            "movhi r4, #0",
            &[0x83a04000],
        ),
        (
            Inst::CSelImm {
                op: MovOp::Mov,
                rd: w(10),
                cond: Cond::Eq,
                imm: imm(0xff00),
                rm: r(10),
            },
            "moveq r10, #0xff00",
            &[0x03a0acff],
        ),
        (
            Inst::CSelImm {
                op: MovOp::Mvn,
                rd: w(3),
                cond: Cond::Hs,
                imm: imm(19),
                rm: r(3),
            },
            "mvnhs r3, #19",
            &[0x23e03013],
        ),
        (
            Inst::CSet {
                rd: w(0),
                cond: Cond::Ne,
            },
            "mov r0, #0\nmovne r0, #1",
            &[0xe3a00000, 0x13a00001],
        ),
        (
            Inst::CSetm {
                rd: w(0),
                cond: Cond::Vs,
            },
            "mov r0, #0\nmvnvs r0, #0",
            &[0xe3a00000, 0x63e00000],
        ),
        // Multiplication and division.
        (
            Inst::Mul {
                rd: w(0),
                rn: r(1),
                rm: r(2),
            },
            "mul r0, r1, r2",
            &[0xe0000291],
        ),
        (
            Inst::MulAcc {
                op: MulAccOp::Mla,
                rd: w(0),
                rn: r(1),
                rm: r(2),
                ra: r(3),
            },
            "mla r0, r1, r2, r3",
            &[0xe0203291],
        ),
        (
            Inst::MulAcc {
                op: MulAccOp::Mls,
                rd: w(0),
                rn: r(1),
                rm: r(2),
                ra: r(3),
            },
            "mls r0, r1, r2, r3",
            &[0xe0603291],
        ),
        (
            Inst::MulLong {
                op: MulLongOp::Umull,
                rd_lo: w(0),
                rd_hi: w(1),
                rn: r(2),
                rm: r(3),
            },
            "umull r0, r1, r2, r3",
            &[0xe0810392],
        ),
        (
            Inst::MulLong {
                op: MulLongOp::Smull,
                rd_lo: w(4),
                rd_hi: w(5),
                rn: r(6),
                rm: r(7),
            },
            "smull r4, r5, r6, r7",
            &[0xe0c54796],
        ),
        (
            Inst::Umaal {
                rd_lo: w(0),
                rd_hi: w(1),
                ra_lo: r(0),
                ra_hi: r(1),
                rn: r(2),
                rm: r(3),
            },
            "umaal r0, r1, r2, r3",
            &[0xe0410392],
        ),
        (
            Inst::Div {
                signed: true,
                rd: w(0),
                rn: r(1),
                rm: r(2),
            },
            "sdiv r0, r1, r2",
            &[0xe710f211],
        ),
        (
            Inst::Div {
                signed: false,
                rd: w(0),
                rn: r(1),
                rm: r(2),
            },
            "udiv r0, r1, r2",
            &[0xe730f211],
        ),
        // Bit operations.
        (
            Inst::BitRR {
                op: BitOp::Clz,
                rd: w(0),
                rm: r(1),
            },
            "clz r0, r1",
            &[0xe16f0f11],
        ),
        (
            Inst::BitRR {
                op: BitOp::Rbit,
                rd: w(0),
                rm: r(1),
            },
            "rbit r0, r1",
            &[0xe6ff0f31],
        ),
        (
            Inst::BitRR {
                op: BitOp::Rev,
                rd: w(0),
                rm: r(1),
            },
            "rev r0, r1",
            &[0xe6bf0f31],
        ),
        (
            Inst::BitRR {
                op: BitOp::Rev16,
                rd: w(0),
                rm: r(1),
            },
            "rev16 r0, r1",
            &[0xe6bf0fb1],
        ),
        (
            Inst::BitRR {
                op: BitOp::Revsh,
                rd: w(0),
                rm: r(1),
            },
            "revsh r0, r1",
            &[0xe6ff0fb1],
        ),
        (
            Inst::BitRR {
                op: BitOp::Sxtb,
                rd: w(0),
                rm: r(1),
            },
            "sxtb r0, r1",
            &[0xe6af0071],
        ),
        (
            Inst::BitRR {
                op: BitOp::Sxth,
                rd: w(0),
                rm: r(1),
            },
            "sxth r0, r1",
            &[0xe6bf0071],
        ),
        (
            Inst::BitRR {
                op: BitOp::Uxtb,
                rd: w(0),
                rm: r(1),
            },
            "uxtb r0, r1",
            &[0xe6ef0071],
        ),
        (
            Inst::BitRR {
                op: BitOp::Uxth,
                rd: w(0),
                rm: r(1),
            },
            "uxth r0, r1",
            &[0xe6ff0071],
        ),
        // Loads and stores.
        (
            Inst::Load {
                op: LoadOp::Ldr,
                rd: w(0),
                mem: mem(1, 4095),
                flags: flags(),
            },
            "ldr r0, [r1, #4095]",
            &[0xe5910fff],
        ),
        (
            Inst::Load {
                op: LoadOp::Ldr,
                rd: w(0),
                mem: mem(1, -4),
                flags: flags(),
            },
            "ldr r0, [r1, #-4]",
            &[0xe5110004],
        ),
        (
            Inst::Load {
                op: LoadOp::Ldrb,
                rd: w(0),
                mem: mem(1, 1),
                flags: flags(),
            },
            "ldrb r0, [r1, #1]",
            &[0xe5d10001],
        ),
        (
            Inst::Load {
                op: LoadOp::Ldrh,
                rd: w(0),
                mem: mem(1, 254),
                flags: flags(),
            },
            "ldrh r0, [r1, #254]",
            &[0xe1d10fbe],
        ),
        (
            Inst::Load {
                op: LoadOp::Ldrsh,
                rd: w(0),
                mem: mem(1, -2),
                flags: flags(),
            },
            "ldrsh r0, [r1, #-2]",
            &[0xe15100f2],
        ),
        (
            Inst::Load {
                op: LoadOp::Ldrsb,
                rd: w(0),
                mem: mem(1, 3),
                flags: flags(),
            },
            "ldrsb r0, [r1, #3]",
            &[0xe1d100d3],
        ),
        (
            Inst::Load {
                op: LoadOp::Ldr,
                rd: w(0),
                mem: mem(1, 0x12345),
                flags: flags(),
            },
            "add ip, r1, #0x12000\nldr r0, [ip, #0x345]",
            &[0xe281ca12, 0xe59c0345],
        ),
        (
            Inst::Load {
                op: LoadOp::Ldrh,
                rd: w(0),
                mem: mem(1, 0x1234),
                flags: flags(),
            },
            "add ip, r1, #0x1200\nldrh r0, [ip, #0x34]",
            &[0xe281cc12, 0xe1dc03b4],
        ),
        (
            Inst::Load {
                op: LoadOp::Ldr,
                rd: w(0),
                mem: AMode::RegReg {
                    base: r(1),
                    index: r(2),
                    shift: 2,
                },
                flags: flags(),
            },
            "ldr r0, [r1, r2, lsl #2]",
            &[0xe7910102],
        ),
        (
            Inst::Load {
                op: LoadOp::Ldrh,
                rd: w(0),
                mem: AMode::RegReg {
                    base: r(1),
                    index: r(2),
                    shift: 0,
                },
                flags: flags(),
            },
            "ldrh r0, [r1, r2]",
            &[0xe19100b2],
        ),
        (
            Inst::Store {
                op: StoreOp::Str,
                rt: r(0),
                mem: mem(13, 8),
                flags: flags(),
            },
            "str r0, [sp, #8]",
            &[0xe58d0008],
        ),
        (
            Inst::Store {
                op: StoreOp::Strb,
                rt: r(0),
                mem: mem(1, -1),
                flags: flags(),
            },
            "strb r0, [r1, #-1]",
            &[0xe5410001],
        ),
        (
            Inst::Store {
                op: StoreOp::Strh,
                rt: r(0),
                mem: mem(1, 2),
                flags: flags(),
            },
            "strh r0, [r1, #2]",
            &[0xe1c100b2],
        ),
        (
            Inst::Store {
                op: StoreOp::Str,
                rt: r(0),
                mem: mem(1, -0x2004),
                flags: flags(),
            },
            "sub ip, r1, #0x2000\nstr r0, [ip, #-4]",
            &[0xe241ca02, 0xe50c0004],
        ),
        (
            Inst::LoadAddr {
                rd: w(0),
                mem: mem(13, 0x1004),
            },
            "add r0, sp, #4\nadd r0, r0, #0x1000",
            &[0xe28d0004, 0xe2800a01],
        ),
        // VFP.
        (
            Inst::FpuRRR {
                op: FpuOp2::Add,
                size: FpuSize::F64,
                rd: wd(0),
                rn: d(1),
                rm: d(15),
            },
            "vadd.f64 d0, d1, d15",
            &[0xee310b0f],
        ),
        (
            Inst::FpuRRR {
                op: FpuOp2::Sub,
                size: FpuSize::F32,
                rd: wd(1),
                rn: d(2),
                rm: d(3),
            },
            "vsub.f32 s2, s4, s6",
            &[0xee321a43],
        ),
        (
            Inst::FpuRRR {
                op: FpuOp2::Mul,
                size: FpuSize::F64,
                rd: wd(8),
                rn: d(9),
                rm: d(10),
            },
            "vmul.f64 d8, d9, d10",
            &[0xee298b0a],
        ),
        (
            Inst::FpuRRR {
                op: FpuOp2::Div,
                size: FpuSize::F32,
                rd: wd(15),
                rn: d(14),
                rm: d(13),
            },
            "vdiv.f32 s30, s28, s26",
            &[0xee8efa0d],
        ),
        (
            Inst::FpuRRRR {
                op: FpuOp3::Fma,
                size: FpuSize::F64,
                rd: wd(0),
                ra: d(0),
                rn: d(1),
                rm: d(2),
            },
            "vfma.f64 d0, d1, d2",
            &[0xeea10b02],
        ),
        (
            Inst::FpuRRRR {
                op: FpuOp3::Fms,
                size: FpuSize::F32,
                rd: wd(0),
                ra: d(0),
                rn: d(1),
                rm: d(2),
            },
            "vfms.f32 s0, s2, s4",
            &[0xeea10a42],
        ),
        (
            Inst::FpuRRRR {
                op: FpuOp3::Fma,
                size: FpuSize::F32,
                rd: wd(20),
                ra: d(20),
                rn: d(1),
                rm: d(2),
            },
            "vswp d0, d20 ; vfma.f32 s0, s2, s4 ; vswp d0, d20",
            &[0xf3b20024, 0xeea10a02, 0xf3b20024],
        ),
        (
            Inst::FpuRRRR {
                op: FpuOp3::Fnms,
                size: FpuSize::F32,
                rd: wd(31),
                ra: d(31),
                rn: d(0),
                rm: d(1),
            },
            "vswp d2, d31 ; vfnms.f32 s4, s0, s2 ; vswp d2, d31",
            &[0xf3b2202f, 0xee902a01, 0xf3b2202f],
        ),
        (
            Inst::FpuRRRR {
                op: FpuOp3::Fma,
                size: FpuSize::F64,
                rd: wd(20),
                ra: d(20),
                rn: d(17),
                rm: d(30),
            },
            "vfma.f64 d20, d17, d30",
            &[0xeee14bae],
        ),
        (
            Inst::FpuRRRR {
                op: FpuOp3::Fnma,
                size: FpuSize::F64,
                rd: wd(0),
                ra: d(0),
                rn: d(1),
                rm: d(2),
            },
            "vfnma.f64 d0, d1, d2",
            &[0xee910b42],
        ),
        (
            Inst::FpuRRRR {
                op: FpuOp3::Fnms,
                size: FpuSize::F64,
                rd: wd(0),
                ra: d(0),
                rn: d(1),
                rm: d(2),
            },
            "vfnms.f64 d0, d1, d2",
            &[0xee910b02],
        ),
        (
            Inst::FpuLoadUnaligned {
                size: FpuSize::F64,
                rd: wd(5),
                mem: mem(3, 0),
                flags: flags(),
            },
            "vld1.8 {d5}, [r3]",
            &[0xf423570f],
        ),
        (
            Inst::FpuLoadUnaligned {
                size: FpuSize::F64,
                rd: wd(15),
                mem: mem(14, 16),
                flags: flags(),
            },
            "add ip, lr, #16\nvld1.8 {d15}, [ip]",
            &[0xe28ec010, 0xf42cf70f],
        ),
        (
            Inst::FpuLoadUnaligned {
                size: FpuSize::F32,
                rd: wd(9),
                mem: mem(1, 0),
                flags: flags(),
            },
            "vld1.32 {d9[0]}, [r1]",
            &[0xf4a1980f],
        ),
        (
            Inst::FpuStoreUnaligned32 {
                rt: d(14),
                mem: mem(10, 0),
                flags: flags(),
            },
            "vst1.32 {d14[0]}, [r10]",
            &[0xf48ae80f],
        ),
        (
            Inst::FpuRint {
                mode: FpuRoundMode::Ceil,
                size: FpuSize::F32,
                rd: wd(0),
                rn: d(1),
            },
            "vrintp.f32 s0, s2",
            &[0xfeba0a41],
        ),
        (
            Inst::FpuRint {
                mode: FpuRoundMode::Floor,
                size: FpuSize::F64,
                rd: wd(15),
                rn: d(3),
            },
            "vrintm.f64 d15, d3",
            &[0xfebbfb43],
        ),
        (
            Inst::FpuRint {
                mode: FpuRoundMode::Trunc,
                size: FpuSize::F64,
                rd: wd(7),
                rn: d(9),
            },
            "vrintz.f64 d7, d9",
            &[0xeeb67bc9],
        ),
        (
            Inst::FpuRint {
                mode: FpuRoundMode::Nearest,
                size: FpuSize::F32,
                rd: wd(14),
                rn: d(5),
            },
            "vrintn.f32 s28, s10",
            &[0xfeb9ea45],
        ),
        (
            Inst::VecRint {
                mode: FpuRoundMode::Nearest,
                size: FpuSize::F64,
                rd: wq(9),
                rn: q(14),
                tmp: wd(0),
            },
            "vrintn.f64 d18, d28\nvrintn.f64 d19, d29",
            &[0xfef92b6c, 0xfef93b6d],
        ),
        (
            Inst::VecRint {
                mode: FpuRoundMode::Ceil,
                size: FpuSize::F32,
                rd: wq(8),
                rn: q(10),
                tmp: wd(3),
            },
            "vmov.f64 d3, d20\nvrintp.f32 s6, s6\nvrintp.f32 s7, s7\nvmov.f64 d16, d3\nvmov.f64 d3, d21\nvrintp.f32 s6, s6\nvrintp.f32 s7, s7\nvmov.f64 d17, d3",
            &[
                0xeeb03b64, 0xfeba3a43, 0xfefa3a63, 0xeef00b43, 0xeeb03b65, 0xfeba3a43, 0xfefa3a63,
                0xeef01b43,
            ],
        ),
        (
            Inst::FpuZero { rd: wd(0) },
            "vmov.i32 d0, #0",
            &[0xf2800010],
        ),
        (
            Inst::FpuZero { rd: wd(15) },
            "vmov.i32 d15, #0",
            &[0xf280f010],
        ),
        (
            Inst::FpuMovImm {
                size: FpuSize::F64,
                rd: wd(0),
                imm: vfp_imm8(1.0f64.to_bits(), true).unwrap(),
            },
            "vmov.f64 d0, #1.0",
            &[0xeeb70b00],
        ),
        (
            Inst::FpuMovImm {
                size: FpuSize::F64,
                rd: wd(15),
                imm: vfp_imm8((-0.125f64).to_bits(), true).unwrap(),
            },
            "vmov.f64 d15, #-0.125",
            &[0xeebcfb00],
        ),
        (
            Inst::FpuMovImm {
                size: FpuSize::F64,
                rd: wd(7),
                imm: vfp_imm8(31.0f64.to_bits(), true).unwrap(),
            },
            "vmov.f64 d7, #31.0",
            &[0xeeb37b0f],
        ),
        (
            Inst::FpuMovImm {
                size: FpuSize::F32,
                rd: wd(1),
                imm: vfp_imm8(u64::from(0.5f32.to_bits()), false).unwrap(),
            },
            "vmov.f32 s2, #0.5",
            &[0xeeb61a00],
        ),
        (
            Inst::FpuMovImm {
                size: FpuSize::F32,
                rd: wd(15),
                imm: vfp_imm8(u64::from((-10.0f32).to_bits()), false).unwrap(),
            },
            "vmov.f32 s30, #-10.0",
            &[0xeebafa04],
        ),
        (
            Inst::FpuRR {
                op: FpuOp1::Abs,
                size: FpuSize::F64,
                rd: wd(0),
                rn: d(1),
            },
            "vabs.f64 d0, d1",
            &[0xeeb00bc1],
        ),
        (
            Inst::FpuRR {
                op: FpuOp1::Neg,
                size: FpuSize::F32,
                rd: wd(0),
                rn: d(1),
            },
            "vneg.f32 s0, s2",
            &[0xeeb10a41],
        ),
        (
            Inst::FpuRR {
                op: FpuOp1::Sqrt,
                size: FpuSize::F64,
                rd: wd(3),
                rn: d(4),
            },
            "vsqrt.f64 d3, d4",
            &[0xeeb13bc4],
        ),
        (
            Inst::FpuMove {
                rd: wd(1),
                rn: d(9),
            },
            "vmov.f64 d1, d9",
            &[0xeeb01b49],
        ),
        (
            Inst::FpuCmp {
                size: FpuSize::F64,
                rn: d(0),
                rm: d(1),
            },
            "vcmp.f64 d0, d1\nvmrs APSR_nzcv, fpscr",
            &[0xeeb40b41, 0xeef1fa10],
        ),
        (
            Inst::FpuCmp {
                size: FpuSize::F32,
                rn: d(2),
                rm: d(3),
            },
            "vcmp.f32 s4, s6\nvmrs APSR_nzcv, fpscr",
            &[0xeeb42a43, 0xeef1fa10],
        ),
        (
            Inst::FpuCvt {
                op: FpuCvtOp::F32ToF64,
                rd: wd(0),
                rn: d(1),
            },
            "vcvt.f64.f32 d0, s2",
            &[0xeeb70ac1],
        ),
        (
            Inst::FpuCvt {
                op: FpuCvtOp::F64ToF32,
                rd: wd(0),
                rn: d(1),
            },
            "vcvt.f32.f64 s0, d1",
            &[0xeeb70bc1],
        ),
        (
            Inst::FpuCvt {
                op: FpuCvtOp::F64ToS32,
                rd: wd(0),
                rn: d(1),
            },
            "vcvt.s32.f64 s0, d1",
            &[0xeebd0bc1],
        ),
        (
            Inst::FpuCvt {
                op: FpuCvtOp::F64ToU32,
                rd: wd(0),
                rn: d(1),
            },
            "vcvt.u32.f64 s0, d1",
            &[0xeebc0bc1],
        ),
        (
            Inst::FpuCvt {
                op: FpuCvtOp::F32ToS32,
                rd: wd(0),
                rn: d(1),
            },
            "vcvt.s32.f32 s0, s2",
            &[0xeebd0ac1],
        ),
        (
            Inst::FpuCvt {
                op: FpuCvtOp::F32ToU32,
                rd: wd(0),
                rn: d(1),
            },
            "vcvt.u32.f32 s0, s2",
            &[0xeebc0ac1],
        ),
        (
            Inst::FpuCvt {
                op: FpuCvtOp::S32ToF64,
                rd: wd(0),
                rn: d(1),
            },
            "vcvt.f64.s32 d0, s2",
            &[0xeeb80bc1],
        ),
        (
            Inst::FpuCvt {
                op: FpuCvtOp::U32ToF64,
                rd: wd(0),
                rn: d(1),
            },
            "vcvt.f64.u32 d0, s2",
            &[0xeeb80b41],
        ),
        (
            Inst::FpuCvt {
                op: FpuCvtOp::S32ToF32,
                rd: wd(0),
                rn: d(1),
            },
            "vcvt.f32.s32 s0, s2",
            &[0xeeb80ac1],
        ),
        (
            Inst::FpuCvt {
                op: FpuCvtOp::U32ToF32,
                rd: wd(0),
                rn: d(1),
            },
            "vcvt.f32.u32 s0, s2",
            &[0xeeb80a41],
        ),
        (
            Inst::FpuCSel {
                size: FpuSize::F64,
                rd: wd(0),
                cond: Cond::Gt,
                rn: d(1),
                rm: d(2),
            },
            "vmov.f64 d0, d2\nvmovgt.f64 d0, d1",
            &[0xeeb00b42, 0xceb00b41],
        ),
        (
            Inst::FpuLoad {
                size: FpuSize::F64,
                rd: wd(0),
                mem: mem(1, 1020),
                flags: flags(),
            },
            "vldr d0, [r1, #1020]",
            &[0xed910bff],
        ),
        (
            Inst::FpuLoad {
                size: FpuSize::F32,
                rd: wd(7),
                mem: mem(1, -8),
                flags: flags(),
            },
            "vldr s14, [r1, #-8]",
            &[0xed117a02],
        ),
        (
            Inst::FpuStore {
                size: FpuSize::F64,
                rt: d(15),
                mem: mem(13, 16),
                flags: flags(),
            },
            "vstr d15, [sp, #16]",
            &[0xed8dfb04],
        ),
        (
            Inst::FpuStore {
                size: FpuSize::F64,
                rt: d(0),
                mem: mem(13, 0x1000),
                flags: flags(),
            },
            "add ip, sp, #0x1000\nvstr d0, [ip]",
            &[0xe28dca01, 0xed8c0b00],
        ),
        (
            Inst::MovToFpu32 {
                rd: wd(3),
                rn: r(4),
            },
            "vmov s6, r4",
            &[0xee034a10],
        ),
        (
            Inst::MovFromFpu32 { rd: w(4), rn: d(3) },
            "vmov r4, s6",
            &[0xee134a10],
        ),
        (
            Inst::MovToFpu64 {
                rd: wd(5),
                lo: r(0),
                hi: r(1),
            },
            "vmov d5, r0, r1",
            &[0xec410b15],
        ),
        (
            Inst::MovFromFpu64 {
                rd_lo: w(2),
                rd_hi: w(3),
                rn: d(5),
            },
            "vmov r2, r3, d5",
            &[0xec532b15],
        ),
        // Control flow and misc.
        (Inst::Ret, "bx lr", &[0xe12fff1e]),
        (
            Inst::Udf {
                trap_code: TrapCode::STACK_OVERFLOW,
            },
            "udf #0",
            &[0xe7f000f0],
        ),
        (Inst::Bkpt, "bkpt #0", &[0xe1200070]),
        (Inst::Fence, "dmb ish", &[0xf57ff05b]),
        (Inst::Nop4, "nop", &[0xe320f000]),
        (
            Inst::Push {
                mask: (1 << 11) | (1 << 14),
            },
            "push {fp, lr}",
            &[0xe92d4800],
        ),
        (
            Inst::Pop {
                mask: 0b0000_0111_1111_0000,
            },
            "pop {r4-r10}",
            &[0xe8bd07f0],
        ),
        (
            Inst::Pop {
                mask: (1 << 4) | (1 << 5) | (1 << 11) | (1 << 15),
            },
            "pop {r4, r5, fp, pc}",
            &[0xe8bd8830],
        ),
        (
            Inst::VPush { first: 8, count: 8 },
            "vpush {d8-d15}",
            &[0xed2d8b10],
        ),
        (
            Inst::VPop { first: 8, count: 2 },
            "vpop {d8-d9}",
            &[0xecbd8b04],
        ),
        (
            Inst::SpAdjust { amount: -16 },
            "sub sp, sp, #16",
            &[0xe24dd010],
        ),
        (
            Inst::SpAdjust { amount: 0x10010 },
            "add sp, sp, #16\nadd sp, sp, #0x10000",
            &[0xe28dd010, 0xe28dd801],
        ),
        (
            Inst::LoadExclusive {
                size: 4,
                rd: w(0),
                addr: r(1),
            },
            "ldrex r0, [r1]",
            &[0xe1910f9f],
        ),
        (
            Inst::LoadExclusive {
                size: 1,
                rd: w(0),
                addr: r(1),
            },
            "ldrexb r0, [r1]",
            &[0xe1d10f9f],
        ),
        (
            Inst::LoadExclusive {
                size: 2,
                rd: w(0),
                addr: r(1),
            },
            "ldrexh r0, [r1]",
            &[0xe1f10f9f],
        ),
        // NEON.
        (
            Inst::VecMove {
                rd: wq(8),
                rn: q(9),
            },
            "vorr q8, q9, q9",
            &[0xf26201f2],
        ),
        (
            Inst::VecRRR {
                op: VecAluOp::Add,
                size: 0,
                rd: wq(8),
                rn: q(9),
                rm: q(10),
            },
            "vadd.i8 q8, q9, q10",
            &[0xf24208e4],
        ),
        (
            Inst::VecRRR {
                op: VecAluOp::Sub,
                size: 3,
                rd: wq(8),
                rn: q(9),
                rm: q(10),
            },
            "vsub.i64 q8, q9, q10",
            &[0xf37208e4],
        ),
        (
            Inst::VecRRR {
                op: VecAluOp::Mul,
                size: 2,
                rd: wq(15),
                rn: q(14),
                rm: q(13),
            },
            "vmul.i32 q15, q14, q13",
            &[0xf26ce9fa],
        ),
        (
            Inst::VecRRR {
                op: VecAluOp::QAddS,
                size: 1,
                rd: wq(8),
                rn: q(9),
                rm: q(10),
            },
            "vqadd.s16 q8, q9, q10",
            &[0xf25200f4],
        ),
        (
            Inst::VecRRR {
                op: VecAluOp::QSubU,
                size: 0,
                rd: wq(8),
                rn: q(9),
                rm: q(10),
            },
            "vqsub.u8 q8, q9, q10",
            &[0xf34202f4],
        ),
        (
            Inst::VecRRR {
                op: VecAluOp::QSubS,
                size: 3,
                rd: wq(8),
                rn: q(9),
                rm: q(10),
            },
            "vqsub.s64 q8, q9, q10",
            &[0xf27202f4],
        ),
        (
            Inst::VecRRR {
                op: VecAluOp::RHAddU,
                size: 1,
                rd: wq(8),
                rn: q(9),
                rm: q(10),
            },
            "vrhadd.u16 q8, q9, q10",
            &[0xf35201e4],
        ),
        (
            Inst::VecRRR {
                op: VecAluOp::MinS,
                size: 2,
                rd: wq(8),
                rn: q(9),
                rm: q(10),
            },
            "vmin.s32 q8, q9, q10",
            &[0xf26206f4],
        ),
        (
            Inst::VecRRR {
                op: VecAluOp::MaxU,
                size: 0,
                rd: wq(8),
                rn: q(9),
                rm: q(10),
            },
            "vmax.u8 q8, q9, q10",
            &[0xf34206e4],
        ),
        (
            Inst::VecRRR {
                op: VecAluOp::Ceq,
                size: 1,
                rd: wq(8),
                rn: q(9),
                rm: q(10),
            },
            "vceq.i16 q8, q9, q10",
            &[0xf35208f4],
        ),
        (
            Inst::VecRRR {
                op: VecAluOp::CgtS,
                size: 2,
                rd: wq(8),
                rn: q(9),
                rm: q(10),
            },
            "vcgt.s32 q8, q9, q10",
            &[0xf26203e4],
        ),
        (
            Inst::VecRRR {
                op: VecAluOp::CgeU,
                size: 0,
                rd: wq(8),
                rn: q(9),
                rm: q(10),
            },
            "vcge.u8 q8, q9, q10",
            &[0xf34203f4],
        ),
        (
            Inst::VecRRR {
                op: VecAluOp::And,
                size: 0,
                rd: wq(8),
                rn: q(9),
                rm: q(10),
            },
            "vand q8, q9, q10",
            &[0xf24201f4],
        ),
        (
            Inst::VecRRR {
                op: VecAluOp::Bic,
                size: 0,
                rd: wq(8),
                rn: q(9),
                rm: q(10),
            },
            "vbic q8, q9, q10",
            &[0xf25201f4],
        ),
        (
            Inst::VecRRR {
                op: VecAluOp::Eor,
                size: 0,
                rd: wq(8),
                rn: q(9),
                rm: q(10),
            },
            "veor q8, q9, q10",
            &[0xf34201f4],
        ),
        (
            Inst::VecRRR {
                op: VecAluOp::ShlU,
                size: 2,
                rd: wq(8),
                rn: q(9),
                rm: q(10),
            },
            "vshl.u32 q8, q9, q10",
            &[0xf36404e2],
        ),
        (
            Inst::VecRRR {
                op: VecAluOp::ShlS,
                size: 3,
                rd: wq(8),
                rn: q(9),
                rm: q(10),
            },
            "vshl.s64 q8, q9, q10",
            &[0xf27404e2],
        ),
        (
            Inst::VecRRR {
                op: VecAluOp::QRDMulH,
                size: 1,
                rd: wq(8),
                rn: q(9),
                rm: q(10),
            },
            "vqrdmulh.s16 q8, q9, q10",
            &[0xf3520be4],
        ),
        (
            Inst::VecBsl {
                rd: wq(8),
                ri: q(8),
                rn: q(9),
                rm: q(10),
            },
            "vbsl q8, q9, q10",
            &[0xf35201f4],
        ),
        (
            Inst::VecRR {
                op: VecMiscOp::Neg,
                size: 2,
                rd: wq(8),
                rn: q(9),
            },
            "vneg.s32 q8, q9",
            &[0xf3f903e2],
        ),
        (
            Inst::VecRR {
                op: VecMiscOp::Abs,
                size: 0,
                rd: wq(8),
                rn: q(9),
            },
            "vabs.s8 q8, q9",
            &[0xf3f10362],
        ),
        (
            Inst::VecRR {
                op: VecMiscOp::Cnt,
                size: 0,
                rd: wq(8),
                rn: q(9),
            },
            "vcnt.8 q8, q9",
            &[0xf3f00562],
        ),
        (
            Inst::VecRR {
                op: VecMiscOp::Mvn,
                size: 0,
                rd: wq(8),
                rn: q(9),
            },
            "vmvn q8, q9",
            &[0xf3f005e2],
        ),
        (
            Inst::VecRR {
                op: VecMiscOp::Rev64,
                size: 2,
                rd: wq(8),
                rn: q(9),
            },
            "vrev64.32 q8, q9",
            &[0xf3f80062],
        ),
        (
            Inst::VecRR {
                op: VecMiscOp::PAddlS,
                size: 0,
                rd: wq(8),
                rn: q(9),
            },
            "vpaddl.s8 q8, q9",
            &[0xf3f00262],
        ),
        (
            Inst::VecRR {
                op: VecMiscOp::PAddlU,
                size: 1,
                rd: wq(8),
                rn: q(9),
            },
            "vpaddl.u16 q8, q9",
            &[0xf3f402e2],
        ),
        (
            Inst::VecRR {
                op: VecMiscOp::CvtF32S32,
                size: 2,
                rd: wq(8),
                rn: q(9),
            },
            "vcvt.f32.s32 q8, q9",
            &[0xf3fb0662],
        ),
        (
            Inst::VecRR {
                op: VecMiscOp::CvtU32F32,
                size: 2,
                rd: wq(8),
                rn: q(9),
            },
            "vcvt.u32.f32 q8, q9",
            &[0xf3fb07e2],
        ),
        (
            Inst::VecSraSelf {
                size: 0,
                rd: wq(8),
                rn: q(8),
                amt: 7,
            },
            "vsra.u8 q8, q8, #7",
            &[0xf3c90170],
        ),
        (
            Inst::VecSraSelf {
                size: 1,
                rd: wq(9),
                rn: q(9),
                amt: 7,
            },
            "vsra.u16 q9, q9, #7",
            &[0xf3d92172],
        ),
        (
            Inst::VecSraSelf {
                size: 2,
                rd: wq(12),
                rn: q(12),
                amt: 14,
            },
            "vsra.u32 q12, q12, #14",
            &[0xf3f28178],
        ),
        (
            Inst::VecSraSelf {
                size: 3,
                rd: wq(15),
                rn: q(15),
                amt: 28,
            },
            "vsra.u64 q15, q15, #28",
            &[0xf3e4e1fe],
        ),
        (
            Inst::VecSraSelf {
                size: 3,
                rd: wq(8),
                rn: q(8),
                amt: 31,
            },
            "vsra.u64 q8, q8, #31",
            &[0xf3e101f0],
        ),
        (
            Inst::VecSraSelf {
                size: 2,
                rd: wq(10),
                rn: q(10),
                amt: 1,
            },
            "vsra.u32 q10, q10, #1",
            &[0xf3ff4174],
        ),
        (
            Inst::VecSraSelf {
                size: 1,
                rd: wq(11),
                rn: q(11),
                amt: 16,
            },
            "vsra.u16 q11, q11, #16",
            &[0xf3d06176],
        ),
        (
            Inst::VecShiftImm {
                op: VecShiftOp::Shl,
                size: 1,
                rd: wq(8),
                rn: q(9),
                amt: 3,
            },
            "vshl.i16 q8, q9, #3",
            &[0xf2d30572],
        ),
        (
            Inst::VecShiftImm {
                op: VecShiftOp::ShrS,
                size: 2,
                rd: wq(8),
                rn: q(9),
                amt: 31,
            },
            "vshr.s32 q8, q9, #31",
            &[0xf2e10072],
        ),
        (
            Inst::VecShiftImm {
                op: VecShiftOp::ShrU,
                size: 0,
                rd: wq(8),
                rn: q(9),
                amt: 1,
            },
            "vshr.u8 q8, q9, #1",
            &[0xf3cf0072],
        ),
        (
            Inst::VecShiftImm {
                op: VecShiftOp::ShrS,
                size: 3,
                rd: wq(8),
                rn: q(9),
                amt: 63,
            },
            "vshr.s64 q8, q9, #63",
            &[0xf2c100f2],
        ),
        (
            Inst::VecShiftImm {
                op: VecShiftOp::Shl,
                size: 3,
                rd: wq(8),
                rn: q(9),
                amt: 5,
            },
            "vshl.i64 q8, q9, #5",
            &[0xf2c505f2],
        ),
        (
            Inst::VecNarrow {
                op: VecNarrowOp::QMovnS,
                size: 0,
                rd: wq(8),
                rn: q(9),
                rm: q(10),
            },
            "vqmovn.s16 d16, q9\nvqmovn.s16 d17, q10",
            &[0xf3f202a2, 0xf3f212a4],
        ),
        (
            Inst::VecNarrow {
                op: VecNarrowOp::QMovun,
                size: 1,
                rd: wq(8),
                rn: q(9),
                rm: q(10),
            },
            "vqmovun.s32 d16, q9\nvqmovun.s32 d17, q10",
            &[0xf3f60262, 0xf3f61264],
        ),
        (
            Inst::VecNarrow {
                op: VecNarrowOp::QMovnU,
                size: 1,
                rd: wq(8),
                rn: q(9),
                rm: q(10),
            },
            "vqmovn.u32 d16, q9\nvqmovn.u32 d17, q10",
            &[0xf3f602e2, 0xf3f612e4],
        ),
        (
            Inst::VecWiden {
                signed: true,
                high: false,
                size: 0,
                rd: wq(8),
                rn: q(9),
            },
            "vmovl.s8 q8, d18",
            &[0xf2c80a32],
        ),
        (
            Inst::VecWiden {
                signed: false,
                high: true,
                size: 2,
                rd: wq(8),
                rn: q(9),
            },
            "vmovl.u32 q8, d19",
            &[0xf3e00a33],
        ),
        (
            Inst::VecMulLong {
                signed: true,
                high: true,
                size: 1,
                rd: wq(8),
                rn: q(9),
                rm: q(10),
            },
            "vmull.s16 q8, d19, d21",
            &[0xf2d30ca5],
        ),
        (
            Inst::VecMulLong {
                signed: false,
                high: false,
                size: 2,
                rd: wq(8),
                rn: q(9),
                rm: q(10),
            },
            "vmull.u32 q8, d18, d20",
            &[0xf3e20ca4],
        ),
        (
            Inst::VecPairwise {
                op: VecAluOp::PAdd,
                size: 1,
                rd: wq(8),
                rn: q(9),
                rm: q(10),
            },
            "vpadd.i16 d16, d18, d19\nvpadd.i16 d17, d20, d21",
            &[0xf2520bb3, 0xf2541bb5],
        ),
        (
            Inst::VecDup {
                size: 0,
                rd: wq(8),
                rn: r(1),
            },
            "vdup.8 q8, r1",
            &[0xeee01b90],
        ),
        (
            Inst::VecDup {
                size: 1,
                rd: wq(9),
                rn: r(2),
            },
            "vdup.16 q9, r2",
            &[0xeea22bb0],
        ),
        (
            Inst::VecDup {
                size: 2,
                rd: wq(10),
                rn: r(3),
            },
            "vdup.32 q10, r3",
            &[0xeea43b90],
        ),
        (
            Inst::VecDupLane {
                size: 2,
                rd: wq(8),
                rn: q(9),
                lane: 3,
            },
            "vdup.32 q8, d19[1]",
            &[0xf3fc0c63],
        ),
        (
            Inst::VecMovToLane {
                size: 0,
                rd: wq(8),
                ri: q(8),
                rn: r(1),
                lane: 13,
            },
            "vmov.8 d17[5], r1",
            &[0xee611bb0],
        ),
        (
            Inst::VecMovToLane {
                size: 1,
                rd: wq(8),
                ri: q(8),
                rn: r(1),
                lane: 3,
            },
            "vmov.16 d16[3], r1",
            &[0xee201bf0],
        ),
        (
            Inst::VecMovToLane {
                size: 2,
                rd: wq(8),
                ri: q(8),
                rn: r(1),
                lane: 2,
            },
            "vmov.32 d17[0], r1",
            &[0xee011b90],
        ),
        (
            Inst::VecMovFromLane {
                size: 0,
                signed: false,
                rd: w(1),
                rn: q(9),
                lane: 9,
            },
            "vmov.u8 r1, d19[1]",
            &[0xeed31bb0],
        ),
        (
            Inst::VecMovFromLane {
                size: 1,
                signed: true,
                rd: w(1),
                rn: q(9),
                lane: 5,
            },
            "vmov.s16 r1, d19[1]",
            &[0xee131bf0],
        ),
        (
            Inst::VecMovFromLane {
                size: 2,
                signed: false,
                rd: w(1),
                rn: q(9),
                lane: 1,
            },
            "vmov.32 r1, d18[1]",
            &[0xee321b90],
        ),
        (
            Inst::VecFromGprs {
                rd: wq(8),
                r0: r(0),
                r1: r(1),
                r2: r(2),
                r3: r(3),
            },
            "vmov d16, r0, r1\nvmov d17, r2, r3",
            &[0xec410b30, 0xec432b31],
        ),
        (
            Inst::VecRR {
                op: VecMiscOp::CeqZ,
                size: 0,
                rd: wq(8),
                rn: q(9),
            },
            "vceq.i8 q8, q9, #0",
            &[0xf3f10162],
        ),
        (
            Inst::VecRR {
                op: VecMiscOp::CeqZ,
                size: 1,
                rd: wq(10),
                rn: q(10),
            },
            "vceq.i16 q10, q10, #0",
            &[0xf3f54164],
        ),
        (
            Inst::VecRR {
                op: VecMiscOp::CeqZ,
                size: 2,
                rd: wq(15),
                rn: q(8),
            },
            "vceq.i32 q15, q8, #0",
            &[0xf3f9e160],
        ),
        (
            Inst::VecRR {
                op: VecMiscOp::Rev32,
                size: 0,
                rd: wq(8),
                rn: q(9),
            },
            "vrev32.8 q8, q9",
            &[0xf3f000e2],
        ),
        (
            Inst::VecRR {
                op: VecMiscOp::Rev32,
                size: 1,
                rd: wq(10),
                rn: q(10),
            },
            "vrev32.16 q10, q10",
            &[0xf3f440e4],
        ),
        (
            Inst::VecRR {
                op: VecMiscOp::Rev16,
                size: 0,
                rd: wq(15),
                rn: q(8),
            },
            "vrev16.8 q15, q8",
            &[0xf3f0e160],
        ),
        (
            Inst::VecRR {
                op: VecMiscOp::Rev64,
                size: 2,
                rd: wq(12),
                rn: q(13),
            },
            "vrev64.32 q12, q13",
            &[0xf3f8806a],
        ),
        (
            Inst::VecRoundF32 {
                mode: FpuRoundMode::Floor,
                rd: wq(8),
                rn: q(9),
                qa: wq(10),
                qb: wq(11),
                t1: wd(0),
                t2: wd(1),
                gtmp: w(4),
            },
            "vmov.f64 d0, d18 ; vmov.f64 d1, d19 ; vmrs r4, fpscr ; bic ip, r4, #0xc00000 ; orr ip, ip, #0x800000 ; vmsr fpscr, ip ; vcvtr.s32.f32 s0, s0 ; vcvt.f32.s32 s0, s0 ; vcvtr.s32.f32 s1, s1 ; vcvt.f32.s32 s1, s1 ; vcvtr.s32.f32 s2, s2 ; vcvt.f32.s32 s2, s2 ; vcvtr.s32.f32 s3, s3 ; vcvt.f32.s32 s3, s3 ; vmsr fpscr, r4 ; vmov.f64 d20, d0 ; vmov.f64 d21, d1 ; vmov.i32 q11, #0x80000000 ; vbit q10, q9, q11 ; vmov.i32 q11, #0x4b000000 ; vacge.f32 q11, q9, q11 ; vbit q10, q9, q11 ; vceq.f32 q11, q9, q9 ; vadd.f32 q8, q9, q9 ; vbit q8, q10, q11",
            &[
                0xeeb00b62, 0xeeb01b63, 0xeef14a10, 0xe3c4c503, 0xe38cc502, 0xeee1ca10, 0xeebd0a40,
                0xeeb80ac0, 0xeefd0a60, 0xeef80ae0, 0xeebd1a41, 0xeeb81ac1, 0xeefd1a61, 0xeef81ae1,
                0xeee14a10, 0xeef04b40, 0xeef05b41, 0xf3c06650, 0xf36241f6, 0xf2c4665b, 0xf3426ef6,
                0xf36241f6, 0xf2426ee2, 0xf2420de2, 0xf36401f6,
            ],
        ),
        (
            Inst::VecRoundF32 {
                mode: FpuRoundMode::Trunc,
                rd: wq(12),
                rn: q(15),
                qa: wq(8),
                qb: wq(9),
                t1: wd(14),
                t2: wd(3),
                gtmp: w(0),
            },
            "vmov.f64 d14, d30 ; vmov.f64 d3, d31 ; vcvt.s32.f32 s28, s28 ; vcvt.f32.s32 s28, s28 ; vcvt.s32.f32 s29, s29 ; vcvt.f32.s32 s29, s29 ; vcvt.s32.f32 s6, s6 ; vcvt.f32.s32 s6, s6 ; vcvt.s32.f32 s7, s7 ; vcvt.f32.s32 s7, s7 ; vmov.f64 d16, d14 ; vmov.f64 d17, d3 ; vmov.i32 q9, #0x80000000 ; vbit q8, q15, q9 ; vmov.i32 q9, #0x4b000000 ; vacge.f32 q9, q15, q9 ; vbit q8, q15, q9 ; vceq.f32 q9, q15, q15 ; vadd.f32 q12, q15, q15 ; vbit q12, q8, q9",
            &[
                0xeeb0eb6e, 0xeeb03b6f, 0xeebdeace, 0xeeb8eace, 0xeefdeaee, 0xeef8eaee, 0xeebd3ac3,
                0xeeb83ac3, 0xeefd3ae3, 0xeef83ae3, 0xeef00b4e, 0xeef01b43, 0xf3c02650, 0xf36e01f2,
                0xf2c4265b, 0xf34e2ef2, 0xf36e01f2, 0xf24e2eee, 0xf24e8dee, 0xf36081f2,
            ],
        ),
        (
            Inst::VecRoundF32 {
                mode: FpuRoundMode::Ceil,
                rd: wq(9),
                rn: q(8),
                qa: wq(14),
                qb: wq(15),
                t1: wd(6),
                t2: wd(7),
                gtmp: w(10),
            },
            "vmov.f64 d6, d16 ; vmov.f64 d7, d17 ; vmrs r10, fpscr ; bic ip, r10, #0xc00000 ; orr ip, ip, #0x400000 ; vmsr fpscr, ip ; vcvtr.s32.f32 s12, s12 ; vcvt.f32.s32 s12, s12 ; vcvtr.s32.f32 s13, s13 ; vcvt.f32.s32 s13, s13 ; vcvtr.s32.f32 s14, s14 ; vcvt.f32.s32 s14, s14 ; vcvtr.s32.f32 s15, s15 ; vcvt.f32.s32 s15, s15 ; vmsr fpscr, r10 ; vmov.f64 d28, d6 ; vmov.f64 d29, d7 ; vmov.i32 q15, #0x80000000 ; vbit q14, q8, q15 ; vmov.i32 q15, #0x4b000000 ; vacge.f32 q15, q8, q15 ; vbit q14, q8, q15 ; vceq.f32 q15, q8, q8 ; vadd.f32 q9, q8, q8 ; vbit q9, q14, q15",
            &[
                0xeeb06b60, 0xeeb07b61, 0xeef1aa10, 0xe3cac503, 0xe38cc501, 0xeee1ca10, 0xeebd6a46,
                0xeeb86ac6, 0xeefd6a66, 0xeef86ae6, 0xeebd7a47, 0xeeb87ac7, 0xeefd7a67, 0xeef87ae7,
                0xeee1aa10, 0xeef0cb46, 0xeef0db47, 0xf3c0e650, 0xf360c1fe, 0xf2c4e65b, 0xf340eefe,
                0xf360c1fe, 0xf240eee0, 0xf2402de0, 0xf36c21fe,
            ],
        ),
        (
            Inst::VecRoundF32 {
                mode: FpuRoundMode::Nearest,
                rd: wq(10),
                rn: q(11),
                qa: wq(12),
                qb: wq(13),
                t1: wd(2),
                t2: wd(5),
                gtmp: w(3),
            },
            "vmov.f64 d2, d22 ; vmov.f64 d5, d23 ; vcvtr.s32.f32 s4, s4 ; vcvt.f32.s32 s4, s4 ; vcvtr.s32.f32 s5, s5 ; vcvt.f32.s32 s5, s5 ; vcvtr.s32.f32 s10, s10 ; vcvt.f32.s32 s10, s10 ; vcvtr.s32.f32 s11, s11 ; vcvt.f32.s32 s11, s11 ; vmov.f64 d24, d2 ; vmov.f64 d25, d5 ; vmov.i32 q13, #0x80000000 ; vbit q12, q11, q13 ; vmov.i32 q13, #0x4b000000 ; vacge.f32 q13, q11, q13 ; vbit q12, q11, q13 ; vceq.f32 q13, q11, q11 ; vadd.f32 q10, q11, q11 ; vbit q10, q12, q13",
            &[
                0xeeb02b66, 0xeeb05b67, 0xeebd2a42, 0xeeb82ac2, 0xeefd2a62, 0xeef82ae2, 0xeebd5a45,
                0xeeb85ac5, 0xeefd5a65, 0xeef85ae5, 0xeef08b42, 0xeef09b45, 0xf3c0a650, 0xf36681fa,
                0xf2c4a65b, 0xf346aefa, 0xf36681fa, 0xf246aee6, 0xf2464de6, 0xf36841fa,
            ],
        ),
        (
            Inst::VecMinMaxF32 {
                is_min: true,
                rd: wq(8),
                rn: q(9),
                rm: q(10),
                ta: wq(11),
                tb: wq(12),
            },
            "vshr.s32 q11, q9, #31 ; vshr.u32 q11, q11, #1 ; veor q11, q11, q9 ; vshr.s32 q12, q10, #31 ; vshr.u32 q12, q12, #1 ; veor q12, q12, q10 ; vcgt.s32 q8, q12, q11 ; vbsl q8, q9, q10 ; vceq.f32 q11, q9, q9 ; vceq.f32 q12, q10, q10 ; vand q11, q11, q12 ; vadd.f32 q12, q9, q10 ; vbif q8, q12, q11",
            &[
                0xf2e16072, 0xf3ff6076, 0xf34661f2, 0xf2e18074, 0xf3ff8078, 0xf34881f4, 0xf26803e6,
                0xf35201f4, 0xf2426ee2, 0xf2448ee4, 0xf24661f8, 0xf2428de4, 0xf37801f6,
            ],
        ),
        (
            Inst::VecMinMaxF32 {
                is_min: false,
                rd: wq(15),
                rn: q(8),
                rm: q(9),
                ta: wq(13),
                tb: wq(14),
            },
            "vshr.s32 q13, q8, #31 ; vshr.u32 q13, q13, #1 ; veor q13, q13, q8 ; vshr.s32 q14, q9, #31 ; vshr.u32 q14, q14, #1 ; veor q14, q14, q9 ; vcgt.s32 q15, q13, q14 ; vbsl q15, q8, q9 ; vceq.f32 q13, q8, q8 ; vceq.f32 q14, q9, q9 ; vand q13, q13, q14 ; vadd.f32 q14, q8, q9 ; vbif q15, q14, q13",
            &[
                0xf2e1a070, 0xf3ffa07a, 0xf34aa1f0, 0xf2e1c072, 0xf3ffc07c, 0xf34cc1f2, 0xf26ae3ec,
                0xf350e1f2, 0xf240aee0, 0xf242cee2, 0xf24aa1fc, 0xf240cde2, 0xf37ce1fa,
            ],
        ),
        (
            Inst::VecCmpF32 {
                cond: FloatCC::Equal,
                rd: wq(8),
                rn: q(9),
                rm: q(10),
                ta: wq(11),
                tb: wq(12),
            },
            "vshr.s32 q11, q9, #31 ; vshr.u32 q12, q11, #1 ; veor q12, q12, q9 ; vsub.i32 q12, q12, q11 ; vshr.s32 q8, q10, #31 ; vshr.u32 q11, q8, #1 ; veor q11, q11, q10 ; vsub.i32 q11, q11, q8 ; vceq.i32 q8, q12, q11 ; vceq.f32 q11, q9, q9 ; vceq.f32 q12, q10, q10 ; vand q11, q11, q12 ; vand q8, q8, q11",
            &[
                0xf2e16072, 0xf3ff8076, 0xf34881f2, 0xf36888e6, 0xf2e10074, 0xf3ff6070, 0xf34661f4,
                0xf36668e0, 0xf36808f6, 0xf2426ee2, 0xf2448ee4, 0xf24661f8, 0xf24001f6,
            ],
        ),
        (
            Inst::VecCmpF32 {
                cond: FloatCC::NotEqual,
                rd: wq(8),
                rn: q(9),
                rm: q(10),
                ta: wq(11),
                tb: wq(12),
            },
            "vshr.s32 q11, q9, #31 ; vshr.u32 q12, q11, #1 ; veor q12, q12, q9 ; vsub.i32 q12, q12, q11 ; vshr.s32 q8, q10, #31 ; vshr.u32 q11, q8, #1 ; veor q11, q11, q10 ; vsub.i32 q11, q11, q8 ; vceq.i32 q8, q12, q11 ; vceq.f32 q11, q9, q9 ; vceq.f32 q12, q10, q10 ; vand q11, q11, q12 ; vand q8, q8, q11 ; vmvn q8, q8",
            &[
                0xf2e16072, 0xf3ff8076, 0xf34881f2, 0xf36888e6, 0xf2e10074, 0xf3ff6070, 0xf34661f4,
                0xf36668e0, 0xf36808f6, 0xf2426ee2, 0xf2448ee4, 0xf24661f8, 0xf24001f6, 0xf3f005e0,
            ],
        ),
        (
            Inst::VecCmpF32 {
                cond: FloatCC::LessThan,
                rd: wq(8),
                rn: q(9),
                rm: q(10),
                ta: wq(11),
                tb: wq(12),
            },
            "vshr.s32 q11, q9, #31 ; vshr.u32 q12, q11, #1 ; veor q12, q12, q9 ; vsub.i32 q12, q12, q11 ; vshr.s32 q8, q10, #31 ; vshr.u32 q11, q8, #1 ; veor q11, q11, q10 ; vsub.i32 q11, q11, q8 ; vcgt.s32 q8, q11, q12 ; vceq.f32 q11, q9, q9 ; vceq.f32 q12, q10, q10 ; vand q11, q11, q12 ; vand q8, q8, q11",
            &[
                0xf2e16072, 0xf3ff8076, 0xf34881f2, 0xf36888e6, 0xf2e10074, 0xf3ff6070, 0xf34661f4,
                0xf36668e0, 0xf26603e8, 0xf2426ee2, 0xf2448ee4, 0xf24661f8, 0xf24001f6,
            ],
        ),
        (
            Inst::VecCmpF32 {
                cond: FloatCC::LessThanOrEqual,
                rd: wq(8),
                rn: q(9),
                rm: q(10),
                ta: wq(11),
                tb: wq(12),
            },
            "vshr.s32 q11, q9, #31 ; vshr.u32 q12, q11, #1 ; veor q12, q12, q9 ; vsub.i32 q12, q12, q11 ; vshr.s32 q8, q10, #31 ; vshr.u32 q11, q8, #1 ; veor q11, q11, q10 ; vsub.i32 q11, q11, q8 ; vcge.s32 q8, q11, q12 ; vceq.f32 q11, q9, q9 ; vceq.f32 q12, q10, q10 ; vand q11, q11, q12 ; vand q8, q8, q11",
            &[
                0xf2e16072, 0xf3ff8076, 0xf34881f2, 0xf36888e6, 0xf2e10074, 0xf3ff6070, 0xf34661f4,
                0xf36668e0, 0xf26603f8, 0xf2426ee2, 0xf2448ee4, 0xf24661f8, 0xf24001f6,
            ],
        ),
        (
            Inst::VecCmpF32 {
                cond: FloatCC::GreaterThan,
                rd: wq(8),
                rn: q(9),
                rm: q(10),
                ta: wq(11),
                tb: wq(12),
            },
            "vshr.s32 q11, q9, #31 ; vshr.u32 q12, q11, #1 ; veor q12, q12, q9 ; vsub.i32 q12, q12, q11 ; vshr.s32 q8, q10, #31 ; vshr.u32 q11, q8, #1 ; veor q11, q11, q10 ; vsub.i32 q11, q11, q8 ; vcgt.s32 q8, q12, q11 ; vceq.f32 q11, q9, q9 ; vceq.f32 q12, q10, q10 ; vand q11, q11, q12 ; vand q8, q8, q11",
            &[
                0xf2e16072, 0xf3ff8076, 0xf34881f2, 0xf36888e6, 0xf2e10074, 0xf3ff6070, 0xf34661f4,
                0xf36668e0, 0xf26803e6, 0xf2426ee2, 0xf2448ee4, 0xf24661f8, 0xf24001f6,
            ],
        ),
        (
            Inst::VecCmpF32 {
                cond: FloatCC::GreaterThanOrEqual,
                rd: wq(8),
                rn: q(9),
                rm: q(10),
                ta: wq(11),
                tb: wq(12),
            },
            "vshr.s32 q11, q9, #31 ; vshr.u32 q12, q11, #1 ; veor q12, q12, q9 ; vsub.i32 q12, q12, q11 ; vshr.s32 q8, q10, #31 ; vshr.u32 q11, q8, #1 ; veor q11, q11, q10 ; vsub.i32 q11, q11, q8 ; vcge.s32 q8, q12, q11 ; vceq.f32 q11, q9, q9 ; vceq.f32 q12, q10, q10 ; vand q11, q11, q12 ; vand q8, q8, q11",
            &[
                0xf2e16072, 0xf3ff8076, 0xf34881f2, 0xf36888e6, 0xf2e10074, 0xf3ff6070, 0xf34661f4,
                0xf36668e0, 0xf26803f6, 0xf2426ee2, 0xf2448ee4, 0xf24661f8, 0xf24001f6,
            ],
        ),
        (
            Inst::VecMulI64 {
                rd: wq(8),
                rn: q(9),
                rm: q(10),
                tb: wq(11),
            },
            "vrev64.32 q8, q10 ; vmul.i32 q8, q8, q9 ; vpaddl.u32 q8, q8 ; vshl.i64 q8, q8, #32 ; vmovn.i64 d22, q9 ; vmovn.i64 d23, q10 ; vmlal.u32 q8, d22, d23",
            &[
                0xf3f80064, 0xf26009f2, 0xf3f802e0, 0xf2e005f0, 0xf3fa6222, 0xf3fa7224, 0xf3e608a7,
            ],
        ),
        (
            Inst::VecMulI64 {
                rd: wq(12),
                rn: q(8),
                rm: q(8),
                tb: wq(15),
            },
            "vrev64.32 q12, q8 ; vmul.i32 q12, q12, q8 ; vpaddl.u32 q12, q12 ; vshl.i64 q12, q12, #32 ; vmovn.i64 d30, q8 ; vmovn.i64 d31, q8 ; vmlal.u32 q12, d30, d31",
            &[
                0xf3f88060, 0xf26889f0, 0xf3f882e8, 0xf2e085f8, 0xf3fae220, 0xf3faf220, 0xf3ee88af,
            ],
        ),
        (
            Inst::PopcntNeon {
                rd: w(0),
                lo: r(1),
                hi: r(1),
                tmp: wd(3),
                is64: false,
            },
            "vmov.32 d3[0], r1 ; vcnt.8 d3, d3 ; vpaddl.u8 d3, d3 ; vpaddl.u16 d3, d3 ; vmov.32 r0, d3[0]",
            &[0xee031b10, 0xf3b03503, 0xf3b03283, 0xf3b43283, 0xee130b10],
        ),
        (
            Inst::PopcntNeon {
                rd: w(4),
                lo: r(2),
                hi: r(3),
                tmp: wd(17),
                is64: true,
            },
            "vmov d17, r2, r3 ; vcnt.8 d17, d17 ; vpaddl.u8 d17, d17 ; vpaddl.u16 d17, d17 ; vpaddl.u32 d17, d17 ; vmov.32 r4, d17[0]",
            &[
                0xec432b31, 0xf3f01521, 0xf3f012a1, 0xf3f412a1, 0xf3f812a1, 0xee114b90,
            ],
        ),
        (
            Inst::PopcntNeon {
                rd: w(10),
                lo: r(8),
                hi: r(9),
                tmp: wd(0),
                is64: true,
            },
            "vmov d0, r8, r9 ; vcnt.8 d0, d0 ; vpaddl.u8 d0, d0 ; vpaddl.u16 d0, d0 ; vpaddl.u32 d0, d0 ; vmov.32 r10, d0[0]",
            &[
                0xec498b10, 0xf3b00500, 0xf3b00280, 0xf3b40280, 0xf3b80280, 0xee10ab10,
            ],
        ),
        (
            Inst::Rot64 {
                left: true,
                rd_lo: w(0),
                rd_hi: w(1),
                rn_lo: r(2),
                rn_hi: r(3),
                amt: r(4),
                t1: w(5),
                t2: w(6),
            },
            "and r5, r4, #63 ; tst r5, #32 ; and r5, r5, #31 ; rsb r6, r5, #32 ; moveq r0, r2, lsl r5 ; orreq r0, r0, r3, lsr r6 ; moveq r1, r3, lsl r5 ; orreq r1, r1, r2, lsr r6 ; movne r0, r3, lsl r5 ; orrne r0, r0, r2, lsr r6 ; movne r1, r2, lsl r5 ; orrne r1, r1, r3, lsr r6",
            &[
                0xe204503f, 0xe3150020, 0xe205501f, 0xe2656020, 0x01a00512, 0x01800633, 0x01a01513,
                0x01811632, 0x11a00513, 0x11800632, 0x11a01512, 0x11811633,
            ],
        ),
        (
            Inst::Rot64 {
                left: false,
                rd_lo: w(7),
                rd_hi: w(8),
                rn_lo: r(0),
                rn_hi: r(1),
                amt: r(10),
                t1: w(2),
                t2: w(3),
            },
            "and r2, r10, #63 ; tst r2, #32 ; and r2, r2, #31 ; rsb r3, r2, #32 ; moveq r7, r0, lsr r2 ; orreq r7, r7, r1, lsl r3 ; moveq r8, r1, lsr r2 ; orreq r8, r8, r0, lsl r3 ; movne r7, r1, lsr r2 ; orrne r7, r7, r0, lsl r3 ; movne r8, r0, lsr r2 ; orrne r8, r8, r1, lsl r3",
            &[
                0xe20a203f, 0xe3120020, 0xe202201f, 0xe2623020, 0x01a07230, 0x01877311, 0x01a08231,
                0x01888310, 0x11a07231, 0x11877310, 0x11a08230, 0x11888311,
            ],
        ),
        (
            Inst::VecRoundF64 {
                mode: FpuRoundMode::Floor,
                rd: wq(8),
                rn: q(9),
                qa: wq(10),
                qb: wq(11),
                qc: wq(12),
                t: wd(3),
                gtmp: w(4),
            },
            "vmov.i32 q10, #0x43000000 ; vorr.i32 q10, #0x00300000 ; vshl.i64 q10, q10, #32 ; vmov.i32 q12, #0x80000000 ; vshl.i64 q12, q12, #32 ; vorr q11, q10, q10 ; vbit q11, q9, q12 ; vmrs r4, fpscr ; bic ip, r4, #0xc00000 ; orr ip, ip, #0x800000 ; vmsr fpscr, ip ; vadd.f64 d16, d18, d22 ; vsub.f64 d16, d16, d22 ; vabs.f64 d3, d18 ; vcmp.f64 d3, d20 ; vmrs APSR_nzcv, fpscr ; vmovge.f64 d16, d18 ; vadd.f64 d17, d19, d23 ; vsub.f64 d17, d17, d23 ; vabs.f64 d3, d19 ; vcmp.f64 d3, d21 ; vmrs APSR_nzcv, fpscr ; vmovge.f64 d17, d19 ; vmsr fpscr, r4 ; vbit q8, q9, q12",
            &[
                0xf2c44653, 0xf2c34550, 0xf2e045f4, 0xf3c08650, 0xf2e085f8, 0xf26461f4, 0xf36261f8,
                0xeef14a10, 0xe3c4c503, 0xe38cc502, 0xeee1ca10, 0xee720ba6, 0xee700be6, 0xeeb03be2,
                0xeeb43b64, 0xeef1fa10, 0xaef00b62, 0xee731ba7, 0xee711be7, 0xeeb03be3, 0xeeb43b65,
                0xeef1fa10, 0xaef01b63, 0xeee14a10, 0xf36201f8,
            ],
        ),
        (
            Inst::VecRoundF64 {
                mode: FpuRoundMode::Nearest,
                rd: wq(13),
                rn: q(8),
                qa: wq(9),
                qb: wq(10),
                qc: wq(11),
                t: wd(20),
                gtmp: w(0),
            },
            "vmov.i32 q9, #0x43000000 ; vorr.i32 q9, #0x00300000 ; vshl.i64 q9, q9, #32 ; vmov.i32 q11, #0x80000000 ; vshl.i64 q11, q11, #32 ; vorr q10, q9, q9 ; vbit q10, q8, q11 ; vadd.f64 d26, d16, d20 ; vsub.f64 d26, d26, d20 ; vabs.f64 d20, d16 ; vcmp.f64 d20, d18 ; vmrs APSR_nzcv, fpscr ; vmovge.f64 d26, d16 ; vadd.f64 d27, d17, d21 ; vsub.f64 d27, d27, d21 ; vabs.f64 d20, d17 ; vcmp.f64 d20, d19 ; vmrs APSR_nzcv, fpscr ; vmovge.f64 d27, d17 ; vbit q13, q8, q11",
            &[
                0xf2c42653, 0xf2c32550, 0xf2e025f2, 0xf3c06650, 0xf2e065f6, 0xf26241f2, 0xf36041f6,
                0xee70aba4, 0xee7aabe4, 0xeef04be0, 0xeef44b62, 0xeef1fa10, 0xaef0ab60, 0xee71bba5,
                0xee7bbbe5, 0xeef04be1, 0xeef44b63, 0xeef1fa10, 0xaef0bb61, 0xf360a1f6,
            ],
        ),
        (
            Inst::VecOrrHalves {
                rd: wd(0),
                rn: q(8),
            },
            "vorr d0, d16, d17",
            &[0xf22001b1],
        ),
        (
            Inst::VecOrrHalves {
                rd: wd(15),
                rn: q(15),
            },
            "vorr d15, d30, d31",
            &[0xf22ef1bf],
        ),
        (
            Inst::VecOrrHalves {
                rd: wd(7),
                rn: q(12),
            },
            "vorr d7, d24, d25",
            &[0xf22871b9],
        ),
        (
            Inst::VecToFpu {
                rd: wd(1),
                rn: q(9),
                half: 1,
            },
            "vmov.f64 d1, d19",
            &[0xeeb01b63],
        ),
        (
            Inst::VecFromFpu {
                rd: wq(9),
                ri: q(9),
                rn: d(1),
                half: 0,
            },
            "vmov.f64 d18, d1",
            &[0xeef02b41],
        ),
        (
            Inst::VecLoadLow {
                size: 3,
                zero_rest: false,
                rd: wq(8),
                mem: mem(1, 0),
                flags: flags(),
            },
            "vld1.8 {d16}, [r1]",
            &[0xf461070f],
        ),
        (
            Inst::VecLoadLow {
                size: 3,
                zero_rest: true,
                rd: wq(9),
                mem: mem(2, 0),
                flags: flags(),
            },
            "vmov.i32 q9, #0 ; vld1.8 {d18}, [r2]",
            &[0xf2c02050, 0xf462270f],
        ),
        (
            Inst::VecLoadLow {
                size: 2,
                zero_rest: false,
                rd: wq(15),
                mem: mem(0, 0),
                flags: flags(),
            },
            "vld1.32 {d30[0]}, [r0]",
            &[0xf4e0e80f],
        ),
        (
            Inst::VecLoadLow {
                size: 2,
                zero_rest: true,
                rd: wq(12),
                mem: mem(7, 0),
                flags: flags(),
            },
            "vmov.i32 q12, #0 ; vld1.32 {d24[0]}, [r7]",
            &[0xf2c08050, 0xf4e7880f],
        ),
        (
            Inst::VecLoadLane {
                size: 0,
                rd: wq(8),
                ri: q(8),
                lane: 0,
                mem: mem(1, 0),
                flags: flags(),
            },
            "vld1.8 {d16[0]}, [r1]",
            &[0xf4e1000f],
        ),
        (
            Inst::VecLoadLane {
                size: 0,
                rd: wq(9),
                ri: q(9),
                lane: 13,
                mem: mem(2, 0),
                flags: flags(),
            },
            "vld1.8 {d19[5]}, [r2]",
            &[0xf4e230af],
        ),
        (
            Inst::VecLoadLane {
                size: 1,
                rd: wq(10),
                ri: q(10),
                lane: 3,
                mem: mem(3, 0),
                flags: flags(),
            },
            "vld1.16 {d20[3]}, [r3]",
            &[0xf4e344cf],
        ),
        (
            Inst::VecLoadLane {
                size: 1,
                rd: wq(15),
                ri: q(15),
                lane: 7,
                mem: mem(0, 0),
                flags: flags(),
            },
            "vld1.16 {d31[3]}, [r0]",
            &[0xf4e0f4cf],
        ),
        (
            Inst::VecLoadLane {
                size: 2,
                rd: wq(8),
                ri: q(8),
                lane: 1,
                mem: mem(4, 0),
                flags: flags(),
            },
            "vld1.32 {d16[1]}, [r4]",
            &[0xf4e4088f],
        ),
        (
            Inst::VecLoadLane {
                size: 2,
                rd: wq(12),
                ri: q(12),
                lane: 2,
                mem: mem(5, 0),
                flags: flags(),
            },
            "vld1.32 {d25[0]}, [r5]",
            &[0xf4e5980f],
        ),
        (
            Inst::VecLoadLane {
                size: 3,
                rd: wq(8),
                ri: q(8),
                lane: 0,
                mem: mem(1, 0),
                flags: flags(),
            },
            "vld1.8 {d16}, [r1]",
            &[0xf461070f],
        ),
        (
            Inst::VecLoadLane {
                size: 3,
                rd: wq(14),
                ri: q(14),
                lane: 1,
                mem: mem(6, 0),
                flags: flags(),
            },
            "vld1.8 {d29}, [r6]",
            &[0xf466d70f],
        ),
        (
            Inst::VecStoreLane {
                size: 0,
                rt: q(8),
                lane: 0,
                mem: mem(1, 0),
                flags: flags(),
            },
            "vst1.8 {d16[0]}, [r1]",
            &[0xf4c1000f],
        ),
        (
            Inst::VecStoreLane {
                size: 0,
                rt: q(9),
                lane: 13,
                mem: mem(2, 0),
                flags: flags(),
            },
            "vst1.8 {d19[5]}, [r2]",
            &[0xf4c230af],
        ),
        (
            Inst::VecStoreLane {
                size: 1,
                rt: q(10),
                lane: 3,
                mem: mem(3, 0),
                flags: flags(),
            },
            "vst1.16 {d20[3]}, [r3]",
            &[0xf4c344cf],
        ),
        (
            Inst::VecStoreLane {
                size: 1,
                rt: q(15),
                lane: 7,
                mem: mem(0, 0),
                flags: flags(),
            },
            "vst1.16 {d31[3]}, [r0]",
            &[0xf4c0f4cf],
        ),
        (
            Inst::VecStoreLane {
                size: 2,
                rt: q(8),
                lane: 1,
                mem: mem(4, 0),
                flags: flags(),
            },
            "vst1.32 {d16[1]}, [r4]",
            &[0xf4c4088f],
        ),
        (
            Inst::VecStoreLane {
                size: 2,
                rt: q(12),
                lane: 2,
                mem: mem(5, 0),
                flags: flags(),
            },
            "vst1.32 {d25[0]}, [r5]",
            &[0xf4c5980f],
        ),
        (
            Inst::VecStoreLane {
                size: 3,
                rt: q(8),
                lane: 0,
                mem: mem(1, 0),
                flags: flags(),
            },
            "vst1.8 {d16}, [r1]",
            &[0xf441070f],
        ),
        (
            Inst::VecStoreLane {
                size: 3,
                rt: q(14),
                lane: 1,
                mem: mem(6, 0),
                flags: flags(),
            },
            "vst1.8 {d29}, [r6]",
            &[0xf446d70f],
        ),
        (
            Inst::VecLoadSplat {
                size: 0,
                rd: wq(8),
                mem: mem(1, 0),
                flags: flags(),
            },
            "vld1.8 {d16[], d17[]}, [r1]",
            &[0xf4e10c2f],
        ),
        (
            Inst::VecLoadSplat {
                size: 1,
                rd: wq(10),
                mem: mem(3, 0),
                flags: flags(),
            },
            "vld1.16 {d20[], d21[]}, [r3]",
            &[0xf4e34c6f],
        ),
        (
            Inst::VecLoadSplat {
                size: 2,
                rd: wq(15),
                mem: mem(9, 0),
                flags: flags(),
            },
            "vld1.32 {d30[], d31[]}, [r9]",
            &[0xf4e9ecaf],
        ),
        (
            Inst::VecLoad {
                rd: wq(8),
                mem: mem(1, 0),
                flags: flags(),
            },
            "vld1.8 {d16, d17}, [r1]",
            &[0xf4610a0f],
        ),
        (
            Inst::VecStore {
                rt: q(15),
                mem: mem(13, 32),
                flags: flags(),
            },
            "add ip, sp, #32\nvst1.8 {d30, d31}, [ip]",
            &[0xe28dc020, 0xf44cea0f],
        ),
        (
            Inst::VecTbl {
                rd: wq(8),
                tbl1: q(9),
                tbl2: None,
                idx: q(10),
            },
            "vtbl.8 d16, {d18, d19}, d20\nvtbl.8 d17, {d18, d19}, d21",
            &[0xf3f209a4, 0xf3f219a5],
        ),
        (
            Inst::VecTbl {
                rd: wq(8),
                tbl1: q(14),
                tbl2: Some(q(15)),
                idx: q(10),
            },
            "vtbl.8 d16, {d28, d29, d30, d31}, d20\nvtbl.8 d17, {d28, d29, d30, d31}, d21",
            &[0xf3fc0ba4, 0xf3fc1ba5],
        ),
        (
            Inst::VecExt {
                rd: wq(8),
                rn: q(9),
                rm: q(10),
                imm: 4,
            },
            "vext.8 q8, q9, q10, #4",
            &[0xf2f204e4],
        ),
        (
            Inst::VecFpuRRR {
                op: FpuOp2::Add,
                size: FpuSize::F64,
                rd: wq(8),
                rn: q(9),
                rm: q(10),
                t1: wd(0),
                t2: wd(1),
            },
            "vadd.f64 d16, d18, d20\nvadd.f64 d17, d19, d21",
            &[0xee720ba4, 0xee731ba5],
        ),
        (
            Inst::VecFpuRRR {
                op: FpuOp2::Mul,
                size: FpuSize::F32,
                rd: wq(8),
                rn: q(9),
                rm: q(10),
                t1: wd(0),
                t2: wd(1),
            },
            "vmov.f64 d0, d18\nvmov.f64 d1, d20\nvmul.f32 s0, s0, s2\nvmul.f32 s1, s1, s3\nvmov.f64 d16, d0\nvmov.f64 d0, d19\nvmov.f64 d1, d21\nvmul.f32 s0, s0, s2\nvmul.f32 s1, s1, s3\nvmov.f64 d17, d0",
            &[
                0xeeb00b62, 0xeeb01b64, 0xee200a01, 0xee600aa1, 0xeef00b40, 0xeeb00b63, 0xeeb01b65,
                0xee200a01, 0xee600aa1, 0xeef01b40,
            ],
        ),
        (
            Inst::VecToFpu32 {
                rd: wd(2),
                rn: q(9),
                lane: 3,
                tmp: wd(3),
            },
            "vmov.f64 d3, d19\nvmov.f32 s4, s7",
            &[0xeeb03b63, 0xeeb02a63],
        ),
        (
            Inst::VecFromFpu32 {
                rd: wq(9),
                ri: q(9),
                rn: d(2),
                lane: 0,
                tmp: wd(3),
            },
            "vmov.f64 d3, d18\nvmov.f32 s6, s4\nvmov.f64 d18, d3",
            &[0xeeb03b62, 0xeeb03a42, 0xeef02b43],
        ),
        (
            Inst::AtomicLoad64 {
                rd_lo: w(0),
                rd_hi: w(1),
                addr: r(7),
                flags: flags(),
                acq_rel: false,
            },
            "ldrexd r0, r1, [r7]\nclrex\ndmb ish",
            &[0xe1b70f9f, 0xf57ff01f, 0xf57ff05b],
        ),
        (
            Inst::AtomicLoad64 {
                rd_lo: w(0),
                rd_hi: w(1),
                addr: r(9),
                flags: flags(),
                acq_rel: true,
            },
            "ldaexd r0, r1, [r9]\nclrex",
            &[0xe1b90e9f, 0xf57ff01f],
        ),
        (
            Inst::LoadAcquire {
                size: 4,
                rd: w(3),
                addr: r(10),
                flags: flags(),
            },
            "lda r3, [r10]",
            &[0xe19a3c9f],
        ),
        (
            Inst::LoadAcquire {
                size: 1,
                rd: w(14),
                addr: r(0),
                flags: flags(),
            },
            "ldab lr, [r0]",
            &[0xe1d0ec9f],
        ),
        (
            Inst::LoadAcquire {
                size: 2,
                rd: w(5),
                addr: r(6),
                flags: flags(),
            },
            "ldah r5, [r6]",
            &[0xe1f65c9f],
        ),
        (
            Inst::StoreRelease {
                size: 4,
                rt: r(8),
                addr: r(2),
                flags: flags(),
            },
            "stl r8, [r2]",
            &[0xe182fc98],
        ),
        (
            Inst::StoreRelease {
                size: 1,
                rt: r(1),
                addr: r(14),
                flags: flags(),
            },
            "stlb r1, [lr]",
            &[0xe1cefc91],
        ),
        (
            Inst::StoreRelease {
                size: 2,
                rt: r(10),
                addr: r(4),
                flags: flags(),
            },
            "stlh r10, [r4]",
            &[0xe1e4fc9a],
        ),
    ]
}

#[test]
fn test_arm32_binemit() {
    let flags = settings::Flags::new(settings::builder());
    let isa_flags = arm32_settings::Flags::new(&flags, &arm32_settings::builder());
    let emit_info = EmitInfo::new(flags, isa_flags);
    for (inst, asm, expected) in cases() {
        let mut buffer = MachBuffer::new();
        inst.emit(&mut buffer, &emit_info, &mut Default::default());
        let buffer = buffer.finish(&Default::default(), &mut Default::default());
        let words: Vec<u32> = buffer
            .data()
            .chunks(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(
            &words[..],
            expected,
            "encoding mismatch for `{asm}` ({inst:?}): got {words:08x?}"
        );
    }
}

#[test]
fn test_arm32_mod_imm() {
    assert_eq!(ModImm::maybe_from_u32(0).map(|i| i.bits()), Some(0));
    assert_eq!(ModImm::maybe_from_u32(0xff).map(|i| i.bits()), Some(0xff));
    assert_eq!(ModImm::maybe_from_u32(0x101), None);
    assert_eq!(
        ModImm::maybe_from_u32(0xf000_000f).map(|i| i.value()),
        Some(0xf000_000f)
    );
    assert_eq!(
        ModImm::maybe_from_u32(0x8000_0000).map(|i| i.value()),
        Some(0x8000_0000)
    );
    for v in [0u32, 1, 0xff, 0x3fc, 0xff00_0000, 0xc000_0003, 0x0001_0000] {
        assert_eq!(ModImm::maybe_from_u32(v).unwrap().value(), v);
    }
}

#[test]
fn test_arm32_cond_invert() {
    for c in [
        Cond::Eq,
        Cond::Ne,
        Cond::Hs,
        Cond::Lo,
        Cond::Mi,
        Cond::Pl,
        Cond::Vs,
        Cond::Vc,
        Cond::Hi,
        Cond::Ls,
        Cond::Ge,
        Cond::Lt,
        Cond::Gt,
        Cond::Le,
    ] {
        assert_eq!(c.invert().invert(), c);
        assert_eq!(c.invert().bits(), c.bits() ^ 1);
    }
}

#[test]
fn test_vfp_imm8_roundtrip() {
    let mut count = 0;
    for imm in 0..=255u8 {
        let v = vfp_expand_imm8(imm);
        assert_eq!(vfp_imm8(v.to_bits(), true), Some(imm), "f64 {v}");
        assert_eq!(
            vfp_imm8(u64::from((v as f32).to_bits()), false),
            Some(imm),
            "f32 {v}"
        );
        count += 1;
    }
    assert_eq!(count, 256);
    for v in [
        0.0f64,
        -0.0,
        0.1,
        32.0,
        0.0625,
        1.0 / 3.0,
        f64::INFINITY,
        f64::NAN,
    ] {
        assert_eq!(vfp_imm8(v.to_bits(), true), None, "f64 {v}");
        assert_eq!(
            vfp_imm8(u64::from((v as f32).to_bits()), false),
            None,
            "f32 {v}"
        );
    }
}
