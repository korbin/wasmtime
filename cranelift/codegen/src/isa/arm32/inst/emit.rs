//! ARMv7 (A32) binary code emission.

use crate::ir;
use crate::ir::TrapCode;
use crate::isa::arm32::abi::{
    Arm32MachineDeps, gen_clobber_restore_insts, merged_clobber_mask, mod_imm_chunks,
    shadow_dreg_to_sreg,
};
use crate::isa::arm32::inst::*;
use crate::isa::arm32::settings as arm32_settings;
use crate::settings;
use cranelift_control::ControlPlane;
use smallvec::smallvec;

/// Constant state used during emissions of a sequence of instructions.
pub struct EmitInfo {
    #[expect(dead_code, reason = "may be used in the future")]
    flags: settings::Flags,
    #[expect(dead_code, reason = "may be used in the future")]
    isa_flags: arm32_settings::Flags,
}

impl EmitInfo {
    /// Emission info for sub-instructions emitted as part of a larger
    /// sequence; none of the flags are consulted by those instructions.
    pub(crate) fn dummy() -> Self {
        let flags = settings::Flags::new(settings::builder());
        let isa_flags = arm32_settings::Flags::new(&flags, &arm32_settings::builder());
        Self { flags, isa_flags }
    }

    /// Create a constant state for emission of instructions.
    pub fn new(flags: settings::Flags, isa_flags: arm32_settings::Flags) -> Self {
        Self { flags, isa_flags }
    }
}

/// State carried between emissions of a sequence of instructions.
#[derive(Default, Clone, Debug)]
pub struct EmitState {
    /// The user stack map for the upcoming instruction, as provided to
    /// `pre_safepoint()`.
    user_stack_map: Option<ir::UserStackMap>,

    /// Only used during fuzz-testing. Otherwise, it is a zero-sized struct and
    /// optimized away at compiletime. See [cranelift_control].
    ctrl_plane: ControlPlane,

    frame_layout: FrameLayout,

    /// The comparison whose result is in the flags, if known. Reset at the
    /// start of each block.
    known_flags: Option<FlagsKey>,

    /// Nesting depth of `emit` calls (instructions emitted as part of
    /// another instruction's expansion).
    emit_depth: u32,
}

impl MachInstEmitState<Inst> for EmitState {
    fn new(abi: &Callee<Arm32MachineDeps>, ctrl_plane: ControlPlane) -> Self {
        EmitState {
            user_stack_map: None,
            ctrl_plane,
            frame_layout: abi.frame_layout().clone(),
            known_flags: None,
            emit_depth: 0,
        }
    }

    fn on_new_block(&mut self) {
        self.known_flags = None;
    }

    fn pre_safepoint(&mut self, user_stack_map: Option<ir::UserStackMap>) {
        self.user_stack_map = user_stack_map;
    }

    fn ctrl_plane_mut(&mut self) -> &mut ControlPlane {
        &mut self.ctrl_plane
    }

    fn take_ctrl_plane(self) -> ControlPlane {
        self.ctrl_plane
    }

    fn frame_layout(&self) -> &FrameLayout {
        &self.frame_layout
    }
}

impl EmitState {
    fn take_stack_map(&mut self) -> Option<ir::UserStackMap> {
        self.user_stack_map.take()
    }
}

//=============================================================================
// Encoding helpers.

const COND_AL: u32 = 0xe;
const IP: u32 = 12;
const SP: u32 = 13;
const LR: u32 = 14;
const PC: u32 = 15;

fn gpr(r: Reg) -> u32 {
    machreg_to_gpr(r)
}

fn gprw(r: Writable<Reg>) -> u32 {
    machreg_to_gpr(r.to_reg())
}

fn dr(r: Reg) -> u32 {
    machreg_to_dreg(r)
}

fn drw(r: Writable<Reg>) -> u32 {
    machreg_to_dreg(r.to_reg())
}

fn alu_opcode(op: AluOp) -> u32 {
    match op {
        AluOp::And => 0b0000,
        AluOp::Eor => 0b0001,
        AluOp::Sub => 0b0010,
        AluOp::Rsb => 0b0011,
        AluOp::Add => 0b0100,
        AluOp::Adc => 0b0101,
        AluOp::Sbc => 0b0110,
        AluOp::Rsc => 0b0111,
        AluOp::Orr => 0b1100,
        AluOp::Bic => 0b1110,
    }
}

fn cmp_opcode(op: CmpOp) -> u32 {
    match op {
        CmpOp::Tst => 0b1000,
        CmpOp::Teq => 0b1001,
        CmpOp::Cmp => 0b1010,
        CmpOp::Cmn => 0b1011,
    }
}

const OPC_MOV: u32 = 0b1101;
const OPC_MVN: u32 = 0b1111;

/// Data-processing (register, immediate shift).
pub(crate) fn enc_dp_reg(
    cond: u32,
    opc: u32,
    s: bool,
    rn: u32,
    rd: u32,
    rm: u32,
    shift: ShiftOpAndAmt,
) -> u32 {
    (cond << 28) | (opc << 21) | (u32::from(s) << 20) | (rn << 16) | (rd << 12) | shift.bits() | rm
}

/// Data-processing (register-shifted register).
pub(crate) fn enc_dp_regshift(
    cond: u32,
    opc: u32,
    s: bool,
    rn: u32,
    rd: u32,
    rm: u32,
    shift_op: ShiftOp,
    rs: u32,
) -> u32 {
    (cond << 28)
        | (opc << 21)
        | (u32::from(s) << 20)
        | (rn << 16)
        | (rd << 12)
        | (rs << 8)
        | (shift_op_bits(shift_op) << 5)
        | (1 << 4)
        | rm
}

/// Data-processing (immediate).
pub(crate) fn enc_dp_imm(cond: u32, opc: u32, s: bool, rn: u32, rd: u32, imm: ModImm) -> u32 {
    (cond << 28)
        | (1 << 25)
        | (opc << 21)
        | (u32::from(s) << 20)
        | (rn << 16)
        | (rd << 12)
        | imm.bits()
}

fn put(sink: &mut MachBuffer<Inst>, word: u32) {
    sink.put4(word);
}

/// The fields of a VFP register operand `d(n)`, or its low half `s(2n)` if
/// `!is_f64`: (4-bit field, 1-bit extension). Only `d0`-`d15` have S halves,
/// which operand collection ensures for registers accessed as S registers.
fn vfp_fields(d: u32, is_f64: bool) -> (u32, u32) {
    if is_f64 {
        debug_assert!(d < 32);
        (d & 0xf, d >> 4)
    } else {
        debug_assert!(d < 16, "d{d} has no single-precision half");
        let s = d * 2;
        (s >> 1, s & 1)
    }
}

/// `vswp dd, dm` (NEON).
fn enc_vswp(d: u32, m: u32) -> u32 {
    0xf3b2_0000 | ((d >> 4) << 22) | ((d & 0xf) << 12) | ((m >> 4) << 5) | (m & 0xf)
}

/// A VFP data-processing instruction with destination `d`, operands `n` and
/// `m`. `d_f64`, `n_f64` and `m_f64` say whether each operand is a D register
/// (as opposed to an S register).
fn enc_vfp(cond: u32, base: u32, sz: bool, d: (u32, bool), n: (u32, bool), m: (u32, bool)) -> u32 {
    let (vd, dd) = vfp_fields(d.0, d.1);
    let (vn, nn) = vfp_fields(n.0, n.1);
    let (vm, mm) = vfp_fields(m.0, m.1);
    (cond << 28)
        | base
        | (u32::from(sz) << 8)
        | (dd << 22)
        | (vn << 16)
        | (vd << 12)
        | (nn << 7)
        | (mm << 5)
        | vm
}

const VADD: u32 = 0x0e30_0a00;
const VSUB: u32 = 0x0e30_0a40;
const VMUL: u32 = 0x0e20_0a00;
const VDIV: u32 = 0x0e80_0a00;
const VFMA: u32 = 0x0ea0_0a00;
const VFMS: u32 = 0x0ea0_0a40;
const VFNMA: u32 = 0x0e90_0a40;
const VFNMS: u32 = 0x0e90_0a00;
const VMOV: u32 = 0x0eb0_0a40;
const VABS: u32 = 0x0eb0_0ac0;
const VNEG: u32 = 0x0eb1_0a40;
const VSQRT: u32 = 0x0eb1_0ac0;
const VCMP: u32 = 0x0eb4_0a40;
const VCVT_FF: u32 = 0x0eb7_0ac0;
const VCVT_TO_S32: u32 = 0x0ebd_0ac0;
const VCVT_TO_U32: u32 = 0x0ebc_0ac0;
const VCVT_FROM_S32: u32 = 0x0eb8_0ac0;
const VCVT_FROM_U32: u32 = 0x0eb8_0a40;

/// `vmrs APSR_nzcv, fpscr`
const VMRS_APSR: u32 = 0xeef1_fa10;

fn enc_vmrs(rt: u32) -> u32 {
    0xeef1_0a10 | (rt << 12)
}

pub(crate) fn enc_vmsr(rt: u32) -> u32 {
    0xeee1_0a10 | (rt << 12)
}

/// `vmov s(2d), rt` / `vmov rt, s(2d)`
fn enc_vmov_core_single(cond: u32, to_core: bool, d: u32, rt: u32) -> u32 {
    let (vn, n) = vfp_fields(d, false);
    (cond << 28) | 0x0e00_0a10 | (u32::from(to_core) << 20) | (vn << 16) | (rt << 12) | (n << 7)
}

/// `vmov dm, rt, rt2` / `vmov rt, rt2, dm`
fn enc_vmov_core_double(to_core: bool, d: u32, rt: u32, rt2: u32) -> u32 {
    let (vm, m) = vfp_fields(d, true);
    0xec40_0b10 | (u32::from(to_core) << 20) | (rt2 << 16) | (rt << 12) | (m << 5) | vm
}

/// `vmov.32 rt, d[1]` / `vmov.32 d[1], rt`
fn enc_vmov_core_scalar_hi(cond: u32, to_core: bool, d: u32, rt: u32) -> u32 {
    let (vn, n) = vfp_fields(d, true);
    (cond << 28) | 0x0e20_0b10 | (u32::from(to_core) << 20) | (vn << 16) | (rt << 12) | (n << 7)
}

/// `vldr` / `vstr` with an immediate offset in `[-1020, 1020]`, multiple of 4.
fn enc_vldr_vstr(load: bool, is_f64: bool, d: u32, rn: u32, offset: i32) -> u32 {
    debug_assert!(offset % 4 == 0 && offset.abs() <= 1020);
    let (vd, dd) = vfp_fields(d, is_f64);
    let u = offset >= 0;
    let imm8 = offset.unsigned_abs() / 4;
    0xed00_0a00
        | (u32::from(u) << 23)
        | (dd << 22)
        | (u32::from(load) << 20)
        | (rn << 16)
        | (vd << 12)
        | (u32::from(is_f64) << 8)
        | imm8
}

fn enc_mul(rd: u32, rn: u32, rm: u32) -> u32 {
    0xe000_0090 | (rd << 16) | (rm << 8) | rn
}

fn enc_branch(cond: u32, link: bool) -> u32 {
    (cond << 28) | 0x0a00_0000 | (u32::from(link) << 24)
}

fn enc_bx(rm: u32) -> u32 {
    0xe12f_ff10 | rm
}

fn enc_blx(rm: u32) -> u32 {
    0xe12f_ff30 | rm
}

fn enc_udf(imm: u16) -> u32 {
    let imm = u32::from(imm);
    0xe7f0_00f0 | ((imm >> 4) << 8) | (imm & 0xf)
}

fn enc_mov_reg(cond: u32, rd: u32, rm: u32) -> u32 {
    enc_dp_reg(cond, OPC_MOV, false, 0, rd, rm, ShiftOpAndAmt::none())
}

fn enc_mov_imm(cond: u32, rd: u32, imm: u32) -> u32 {
    enc_dp_imm(
        cond,
        OPC_MOV,
        false,
        0,
        rd,
        ModImm::maybe_from_u32(imm).unwrap(),
    )
}

/// Emit a backward branch to the absolute buffer offset `target`.
fn emit_branch_to_offset(sink: &mut MachBuffer<Inst>, cond: u32, target: CodeOffset) {
    let here = sink.cur_offset();
    let off = i64::from(target) - (i64::from(here) + 8);
    debug_assert_eq!(off & 3, 0);
    let imm24 = ((off >> 2) as u32) & 0x00ff_ffff;
    put(sink, enc_branch(cond, false) | imm24);
}

/// Emit a forward branch that skips the given number of following
/// instructions.
fn emit_branch_skip(sink: &mut MachBuffer<Inst>, cond: u32, insns_to_skip: u32) {
    // Target is `here + 4 + 4 * insns_to_skip`; the offset is relative to
    // `here + 8`.
    let off = (4 * insns_to_skip as i64) - 4;
    let imm24 = ((off >> 2) as u32) & 0x00ff_ffff;
    put(sink, enc_branch(cond, false) | imm24);
}

/// Emit `rd = rn + value` (or minus) using only modified-immediate chunks.
pub(crate) fn emit_add_imm_chunks(sink: &mut MachBuffer<Inst>, rd: u32, rn: u32, value: i64) {
    let (opc, abs) = if value >= 0 {
        (alu_opcode(AluOp::Add), value as u32)
    } else {
        (alu_opcode(AluOp::Sub), value.unsigned_abs() as u32)
    };
    let mut src = rn;
    let chunks = mod_imm_chunks(abs);
    if chunks.is_empty() {
        if rd != rn {
            put(sink, enc_mov_reg(COND_AL, rd, rn));
        }
        return;
    }
    for chunk in chunks {
        put(sink, enc_dp_imm(COND_AL, opc, false, src, rd, chunk));
        src = rd;
    }
}

/// Load a 32-bit constant into `rd` using `mov`/`mvn` or `movw`/`movt`.
fn emit_load_const32(sink: &mut MachBuffer<Inst>, rd: u32, value: u32) {
    if let Some(imm) = ModImm::maybe_from_u32(value) {
        put(sink, enc_dp_imm(COND_AL, OPC_MOV, false, 0, rd, imm));
    } else if let Some(imm) = ModImm::maybe_from_u32(!value) {
        put(sink, enc_dp_imm(COND_AL, OPC_MVN, false, 0, rd, imm));
    } else {
        put(sink, enc_movw(rd, value & 0xffff));
        if value >> 16 != 0 {
            put(sink, enc_movt(rd, value >> 16));
        }
    }
}

/// The kind of memory access, which determines the immediate offset range
/// and available addressing forms.
#[derive(Clone, Copy, PartialEq, Eq)]
enum MemKind {
    /// `ldr`/`str`/`ldrb`/`strb`: 12-bit offset, register offset with shift.
    Word,
    /// `ldrh`/`strh`/`ldrsb`/`ldrsh`: 8-bit offset, register offset.
    Half,
    /// `vldr`/`vstr`: 8-bit offset scaled by 4.
    Vfp,
}

/// A finalized addressing mode.
enum FinalMem {
    Imm { base: u32, offset: i32 },
    Reg { base: u32, index: u32, shift: u8 },
}

/// Finalize an addressing mode for a memory access of the given kind,
/// emitting address computations into `ip` if needed.
fn finalize_mem(
    sink: &mut MachBuffer<Inst>,
    state: &EmitState,
    mem: &AMode,
    kind: MemKind,
) -> FinalMem {
    match mem {
        AMode::RegReg { base, index, shift } => {
            let (base, index) = (gpr(*base), gpr(*index));
            match kind {
                MemKind::Word => FinalMem::Reg {
                    base,
                    index,
                    shift: *shift,
                },
                MemKind::Half if *shift == 0 => FinalMem::Reg {
                    base,
                    index,
                    shift: 0,
                },
                _ => {
                    // add ip, base, index, lsl #shift
                    let sh = if *shift == 0 {
                        ShiftOpAndAmt::none()
                    } else {
                        ShiftOpAndAmt::new(ShiftOp::Lsl, *shift)
                    };
                    put(
                        sink,
                        enc_dp_reg(COND_AL, alu_opcode(AluOp::Add), false, base, IP, index, sh),
                    );
                    FinalMem::Imm {
                        base: IP,
                        offset: 0,
                    }
                }
            }
        }
        _ => {
            let (base, offset) = mem.resolve(state).unwrap();
            let base = gpr(base);
            let fits = match kind {
                MemKind::Word => offset.abs() <= 4095,
                MemKind::Half => offset.abs() <= 255,
                MemKind::Vfp => offset.abs() <= 1020 && offset % 4 == 0,
            };
            if fits {
                return FinalMem::Imm {
                    base,
                    offset: offset as i32,
                };
            }
            // Split off the low part that fits in the instruction and add
            // the rest into `ip`.
            let (high, low) = match kind {
                MemKind::Word => {
                    let low = (offset.abs() & 0xfff) * offset.signum();
                    (offset - low, low)
                }
                MemKind::Half => {
                    let low = (offset.abs() & 0xff) * offset.signum();
                    (offset - low, low)
                }
                MemKind::Vfp => {
                    let low = (offset.abs() & 0x3fc) * offset.signum();
                    (offset - low, low)
                }
            };
            emit_add_imm_chunks(sink, IP, base, high);
            FinalMem::Imm {
                base: IP,
                offset: low as i32,
            }
        }
    }
}

fn enc_ldst_word(load: bool, byte: bool, rt: u32, mem: FinalMem) -> u32 {
    match mem {
        FinalMem::Imm { base, offset } => {
            let u = offset >= 0;
            0xe500_0000
                | (u32::from(u) << 23)
                | (u32::from(byte) << 22)
                | (u32::from(load) << 20)
                | (base << 16)
                | (rt << 12)
                | offset.unsigned_abs()
        }
        FinalMem::Reg { base, index, shift } => {
            0xe780_0000
                | (u32::from(byte) << 22)
                | (u32::from(load) << 20)
                | (base << 16)
                | (rt << 12)
                | (u32::from(shift) << 7)
                | index
        }
    }
}

/// Halfword and signed-byte loads/stores. `sh` is the 2-bit `SH` field.
fn enc_ldst_half(load: bool, sh: u32, rt: u32, mem: FinalMem) -> u32 {
    match mem {
        FinalMem::Imm { base, offset } => {
            let u = offset >= 0;
            let imm = offset.unsigned_abs();
            0xe140_0090
                | (u32::from(u) << 23)
                | (u32::from(load) << 20)
                | (base << 16)
                | (rt << 12)
                | ((imm >> 4) << 8)
                | (sh << 5)
                | (imm & 0xf)
        }
        FinalMem::Reg { base, index, shift } => {
            debug_assert_eq!(shift, 0);
            0xe180_0090 | (u32::from(load) << 20) | (base << 16) | (rt << 12) | (sh << 5) | index
        }
    }
}

fn emit_load(sink: &mut MachBuffer<Inst>, state: &EmitState, op: LoadOp, rd: u32, mem: &AMode) {
    match op {
        LoadOp::Ldr | LoadOp::Ldrb => {
            let m = finalize_mem(sink, state, mem, MemKind::Word);
            put(sink, enc_ldst_word(true, op == LoadOp::Ldrb, rd, m));
        }
        LoadOp::Ldrh | LoadOp::Ldrsb | LoadOp::Ldrsh => {
            let sh = match op {
                LoadOp::Ldrh => 0b01,
                LoadOp::Ldrsb => 0b10,
                _ => 0b11,
            };
            let m = finalize_mem(sink, state, mem, MemKind::Half);
            put(sink, enc_ldst_half(true, sh, rd, m));
        }
    }
}

fn emit_store(sink: &mut MachBuffer<Inst>, state: &EmitState, op: StoreOp, rt: u32, mem: &AMode) {
    debug_assert!(rt != IP || matches!(mem, AMode::SPOffset(o) if o.abs() <= 255));
    match op {
        StoreOp::Str | StoreOp::Strb => {
            let m = finalize_mem(sink, state, mem, MemKind::Word);
            put(sink, enc_ldst_word(false, op == StoreOp::Strb, rt, m));
        }
        StoreOp::Strh => {
            let m = finalize_mem(sink, state, mem, MemKind::Half);
            put(sink, enc_ldst_half(false, 0b01, rt, m));
        }
    }
}

/// Pick the trap-code immediate for `udf`. The immediate does not affect
/// the trap semantics; we use zero.
fn udf_word() -> u32 {
    enc_udf(0)
}

impl MachInstEmit for Inst {
    type State = EmitState;
    type Info = EmitInfo;

    fn emit(&self, sink: &mut MachBuffer<Inst>, emit_info: &Self::Info, state: &mut EmitState) {
        // Instructions emitted as part of another instruction's expansion
        // are not tracked: the expansion may contain loops.
        if state.emit_depth > 0 {
            state.emit_depth += 1;
            self.emit_body(sink, emit_info, state);
            state.emit_depth -= 1;
            return;
        }

        // Skip a comparison whose result is already in the flags: the same
        // comparison was the last flag-setting instruction in this block,
        // and its operands have not been redefined since.
        let key = self.flags_key();
        if key.is_some() && key == state.known_flags {
            return;
        }
        let keep = match state.known_flags {
            Some(known) if key.is_none() && self.preserves_flags_and(known) => Some(known),
            _ => None,
        };
        state.known_flags = None;
        state.emit_depth = 1;
        self.emit_body(sink, emit_info, state);
        state.emit_depth = 0;
        state.known_flags = key.or(keep);
    }

    fn pretty_print_inst(&self, state: &mut Self::State) -> String {
        self.print_with_state(state)
    }
}

impl Inst {
    fn emit_body(&self, sink: &mut MachBuffer<Inst>, emit_info: &EmitInfo, state: &mut EmitState) {
        // N.B.: we *must* not exceed the "worst-case size" used to compute
        // where to insert islands, except when islands are explicitly
        // triggered (with an `EmitIsland`). We check this in debug builds.
        let mut start_off = sink.cur_offset();

        if super::neon::emit_vec(self, sink, state) {
            let end_off = sink.cur_offset();
            assert!(
                (end_off - start_off) <= Inst::worst_case_size(),
                "Inst:{:?} length:{} worst_case_size:{}",
                self,
                end_off - start_off,
                Inst::worst_case_size()
            );
            return;
        }

        match self {
            Inst::Nop0 => {}
            Inst::Nop4 => put(sink, 0xe320_f000),
            Inst::AluRRR {
                op,
                s,
                rd,
                rn,
                rm,
                shift,
            } => put(
                sink,
                enc_dp_reg(
                    COND_AL,
                    alu_opcode(*op),
                    *s,
                    gpr(*rn),
                    gprw(*rd),
                    gpr(*rm),
                    *shift,
                ),
            ),
            Inst::AluRRRShift {
                op,
                s,
                rd,
                rn,
                rm,
                shift_op,
                rs,
            } => put(
                sink,
                enc_dp_regshift(
                    COND_AL,
                    alu_opcode(*op),
                    *s,
                    gpr(*rn),
                    gprw(*rd),
                    gpr(*rm),
                    *shift_op,
                    gpr(*rs),
                ),
            ),
            Inst::AluRRImm { op, s, rd, rn, imm } => put(
                sink,
                enc_dp_imm(COND_AL, alu_opcode(*op), *s, gpr(*rn), gprw(*rd), *imm),
            ),
            Inst::MovRR {
                op,
                s,
                rd,
                rm,
                shift,
            } => {
                let opc = match op {
                    MovOp::Mov => OPC_MOV,
                    MovOp::Mvn => OPC_MVN,
                };
                put(
                    sink,
                    enc_dp_reg(COND_AL, opc, *s, 0, gprw(*rd), gpr(*rm), *shift),
                );
            }
            Inst::MovRRShift {
                rd,
                rm,
                shift_op,
                rs,
            } => put(
                sink,
                enc_dp_regshift(
                    COND_AL,
                    OPC_MOV,
                    false,
                    0,
                    gprw(*rd),
                    gpr(*rm),
                    *shift_op,
                    gpr(*rs),
                ),
            ),
            Inst::MovImm { op, rd, imm } => {
                let opc = match op {
                    MovOp::Mov => OPC_MOV,
                    MovOp::Mvn => OPC_MVN,
                };
                put(sink, enc_dp_imm(COND_AL, opc, false, 0, gprw(*rd), *imm));
            }
            Inst::MovW { rd, imm } => put(sink, enc_movw(gprw(*rd), u32::from(*imm))),
            Inst::MovT { rd, rn, imm } => {
                debug_assert_eq!(rd.to_reg(), *rn);
                put(sink, enc_movt(gprw(*rd), u32::from(*imm)));
            }
            Inst::Mov { rd, rm } => {
                if rd.to_reg() != *rm {
                    put(sink, enc_mov_reg(COND_AL, gprw(*rd), gpr(*rm)));
                }
            }
            Inst::CmpRR { op, rn, rm, shift } => put(
                sink,
                enc_dp_reg(
                    COND_AL,
                    cmp_opcode(*op),
                    true,
                    gpr(*rn),
                    0,
                    gpr(*rm),
                    *shift,
                ),
            ),
            Inst::CCmpRR { op, cond, rn, rm } => put(
                sink,
                enc_dp_reg(
                    cond.bits(),
                    cmp_opcode(*op),
                    true,
                    gpr(*rn),
                    0,
                    gpr(*rm),
                    ShiftOpAndAmt::none(),
                ),
            ),
            Inst::CmpRImm { op, rn, imm } => put(
                sink,
                enc_dp_imm(COND_AL, cmp_opcode(*op), true, gpr(*rn), 0, *imm),
            ),
            Inst::CCmpRImm { op, cond, rn, imm } => put(
                sink,
                enc_dp_imm(cond.bits(), cmp_opcode(*op), true, gpr(*rn), 0, *imm),
            ),
            Inst::CSel { rd, cond, rn, rm } => {
                let (rd, rn, rm) = (gprw(*rd), gpr(*rn), gpr(*rm));
                if rd == rn {
                    if rd != rm {
                        put(sink, enc_mov_reg(cond.invert().bits(), rd, rm));
                    }
                } else if rd == rm {
                    put(sink, enc_mov_reg(cond.bits(), rd, rn));
                } else {
                    put(sink, enc_mov_reg(COND_AL, rd, rm));
                    put(sink, enc_mov_reg(cond.bits(), rd, rn));
                }
            }
            Inst::CSelImm {
                op,
                rd,
                cond,
                imm,
                rm,
            } => {
                let rd = gprw(*rd);
                debug_assert_eq!(rd, gpr(*rm));
                let opc = match op {
                    MovOp::Mov => OPC_MOV,
                    MovOp::Mvn => OPC_MVN,
                };
                put(sink, enc_dp_imm(cond.bits(), opc, false, 0, rd, *imm));
            }
            Inst::CSet { rd, cond } => {
                let rd = gprw(*rd);
                put(sink, enc_mov_imm(COND_AL, rd, 0));
                put(sink, enc_mov_imm(cond.bits(), rd, 1));
            }
            Inst::CSetm { rd, cond } => {
                let rd = gprw(*rd);
                put(sink, enc_mov_imm(COND_AL, rd, 0));
                put(
                    sink,
                    enc_dp_imm(
                        cond.bits(),
                        OPC_MVN,
                        false,
                        0,
                        rd,
                        ModImm::maybe_from_u32(0).unwrap(),
                    ),
                );
            }
            Inst::Mul { rd, rn, rm } => put(sink, enc_mul(gprw(*rd), gpr(*rn), gpr(*rm))),
            Inst::MulAcc { op, rd, rn, rm, ra } => {
                let base = match op {
                    MulAccOp::Mla => 0xe020_0090,
                    MulAccOp::Mls => 0xe060_0090,
                };
                put(
                    sink,
                    base | (gprw(*rd) << 16) | (gpr(*ra) << 12) | (gpr(*rm) << 8) | gpr(*rn),
                );
            }
            Inst::MulLong {
                op,
                rd_lo,
                rd_hi,
                rn,
                rm,
            } => {
                let base = match op {
                    MulLongOp::Umull => 0xe080_0090,
                    MulLongOp::Smull => 0xe0c0_0090,
                };
                put(
                    sink,
                    base | (gprw(*rd_hi) << 16) | (gprw(*rd_lo) << 12) | (gpr(*rm) << 8) | gpr(*rn),
                );
            }
            Inst::Umaal {
                rd_lo,
                rd_hi,
                ra_lo,
                ra_hi,
                rn,
                rm,
            } => {
                debug_assert_eq!(rd_lo.to_reg(), *ra_lo);
                debug_assert_eq!(rd_hi.to_reg(), *ra_hi);
                put(
                    sink,
                    0xe040_0090
                        | (gprw(*rd_hi) << 16)
                        | (gprw(*rd_lo) << 12)
                        | (gpr(*rm) << 8)
                        | gpr(*rn),
                );
            }
            Inst::Div { signed, rd, rn, rm } => {
                let base = if *signed { 0xe710_f010 } else { 0xe730_f010 };
                put(sink, base | (gprw(*rd) << 16) | (gpr(*rm) << 8) | gpr(*rn));
            }
            Inst::UDivMod32 {
                rd_quot,
                rd_rem,
                rn,
                rm,
            } => {
                let (q, r, n, d) = (gprw(*rd_quot), gprw(*rd_rem), gpr(*rn), gpr(*rm));
                let add = alu_opcode(AluOp::Add);
                let adc = alu_opcode(AluOp::Adc);
                let sub = alu_opcode(AluOp::Sub);
                let orr = alu_opcode(AluOp::Orr);
                let cmp = cmp_opcode(CmpOp::Cmp);
                let none = ShiftOpAndAmt::none();
                // The loop shifts `r:q` left once per iteration, subtracting
                // `d` from `r` (and setting the low bit of `q`) whenever
                // `r >= d`, starting from `r = 0, q = n`. The first `clz(n)`
                // iterations only shift zeros into `r`, so skip them:
                //
                //   clz ip, n ; lsl q, n, ip ; mov r, #0
                //   rsbs ip, ip, #32 ; beq done
                put(sink, 0xe16f_0f10 | (IP << 12) | n);
                put(
                    sink,
                    enc_dp_regshift(COND_AL, OPC_MOV, false, 0, q, n, ShiftOp::Lsl, IP),
                );
                put(sink, enc_mov_imm(COND_AL, r, 0));
                put(
                    sink,
                    enc_dp_imm(
                        COND_AL,
                        alu_opcode(AluOp::Rsb),
                        true,
                        IP,
                        IP,
                        ModImm::maybe_from_u32(32).unwrap(),
                    ),
                );
                // Skip the 7-instruction loop if `n` is zero.
                emit_branch_skip(sink, Cond::Eq.bits(), 7);
                let loop_start = sink.cur_offset();
                // adds q, q, q
                put(sink, enc_dp_reg(COND_AL, add, true, q, q, q, none));
                // adcs r, r, r
                put(sink, enc_dp_reg(COND_AL, adc, true, r, r, r, none));
                // cmpcc r, d
                put(sink, enc_dp_reg(Cond::Lo.bits(), cmp, true, r, 0, d, none));
                // subhs r, r, d
                put(sink, enc_dp_reg(Cond::Hs.bits(), sub, false, r, r, d, none));
                // orrhs q, q, #1
                put(
                    sink,
                    enc_dp_imm(
                        Cond::Hs.bits(),
                        orr,
                        false,
                        q,
                        q,
                        ModImm::maybe_from_u32(1).unwrap(),
                    ),
                );
                // subs ip, ip, #1
                put(
                    sink,
                    enc_dp_imm(
                        COND_AL,
                        sub,
                        true,
                        IP,
                        IP,
                        ModImm::maybe_from_u32(1).unwrap(),
                    ),
                );
                // bne loop
                emit_branch_to_offset(sink, Cond::Ne.bits(), loop_start);
            }
            Inst::UDivMod64 {
                idiv,
                quot_lo,
                quot_hi,
                rem_lo,
                rem_hi,
                n_lo,
                n_hi,
                d_lo,
                d_hi,
            } => {
                let (ql, qh, rl, rh) =
                    (gprw(*quot_lo), gprw(*quot_hi), gprw(*rem_lo), gprw(*rem_hi));
                let (nl, nh, dl, dh) = (gpr(*n_lo), gpr(*n_hi), gpr(*d_lo), gpr(*d_hi));
                let add = alu_opcode(AluOp::Add);
                let adc = alu_opcode(AluOp::Adc);
                let sub = alu_opcode(AluOp::Sub);
                let sbc = alu_opcode(AluOp::Sbc);
                let orr = alu_opcode(AluOp::Orr);
                let cmp = cmp_opcode(CmpOp::Cmp);
                let none = ShiftOpAndAmt::none();
                let imm0 = ModImm::maybe_from_u32(0).unwrap();
                // The bit-by-bit loop below shifts the 128-bit value `r:q`
                // left once per iteration, subtracting `d` from `r` (and
                // setting the low bit of `q`) whenever `r >= d`. It starts
                // from `r = 0, q = n` and runs 64 iterations, but can start
                // later: the state after the first 32 iterations is
                // `r = n_hi % d, q = (n_lo : n_hi / d)` when `d` fits in 32
                // bits, and `r = 0, q = (n_lo : 0)` when `n_hi` is zero.
                //
                // Instructions (after the optional fast path), in order:
                // 9 for the setup, 13 for the loop.
                const SETUP_LEN: u32 = 9;
                const LOOP_LEN: u32 = 13;
                let udiv = |rd: u32, rn: u32, rm: u32| 0xe730_f010 | (rd << 16) | (rm << 8) | rn;
                let mls = |rd: u32, rn: u32, rm: u32, ra: u32| {
                    0xe060_0090 | (rd << 16) | (ra << 12) | (rm << 8) | rn
                };
                if *idiv {
                    // if d_hi != 0, go to the general setup.
                    put(sink, enc_dp_imm(COND_AL, cmp, true, dh, 0, imm0));
                    emit_branch_skip(sink, Cond::Ne.bits(), 13);
                    // if n_hi != 0, go to `hi_nonzero`.
                    put(sink, enc_dp_imm(COND_AL, cmp, true, nh, 0, imm0));
                    emit_branch_skip(sink, Cond::Ne.bits(), 5);
                    // Both operands fit in 32 bits.
                    put(sink, udiv(ql, nl, dl));
                    put(sink, mls(rl, ql, dl, nl));
                    put(sink, enc_mov_imm(COND_AL, qh, 0));
                    put(sink, enc_mov_imm(COND_AL, rh, 0));
                    // b done
                    emit_branch_skip(sink, COND_AL, 6 + SETUP_LEN + LOOP_LEN);
                    // hi_nonzero: divide the high word in hardware, then do
                    // the remaining 32 iterations.
                    put(sink, udiv(ql, nh, dl));
                    put(sink, mls(rl, ql, dl, nh));
                    put(sink, enc_mov_reg(COND_AL, qh, nl));
                    put(sink, enc_mov_imm(COND_AL, rh, 0));
                    put(sink, enc_mov_imm(COND_AL, IP, 32));
                    // b loop
                    emit_branch_skip(sink, COND_AL, SETUP_LEN);
                }
                // General setup, skipping 32 iterations if n_hi is zero.
                let setup_start = sink.cur_offset();
                put(sink, enc_dp_imm(COND_AL, cmp, true, nh, 0, imm0));
                put(sink, enc_mov_reg(Cond::Ne.bits(), qh, nh));
                put(sink, enc_mov_reg(Cond::Ne.bits(), ql, nl));
                put(sink, enc_mov_imm(Cond::Ne.bits(), IP, 64));
                put(sink, enc_mov_reg(Cond::Eq.bits(), qh, nl));
                put(sink, enc_mov_imm(Cond::Eq.bits(), ql, 0));
                put(sink, enc_mov_imm(Cond::Eq.bits(), IP, 32));
                put(sink, enc_mov_imm(COND_AL, rl, 0));
                put(sink, enc_mov_imm(COND_AL, rh, 0));
                debug_assert_eq!(sink.cur_offset() - setup_start, 4 * SETUP_LEN);
                let loop_start = sink.cur_offset();
                // Shift r:q left by one.
                put(sink, enc_dp_reg(COND_AL, add, true, ql, ql, ql, none));
                put(sink, enc_dp_reg(COND_AL, adc, true, qh, qh, qh, none));
                put(sink, enc_dp_reg(COND_AL, adc, true, rl, rl, rl, none));
                put(sink, enc_dp_reg(COND_AL, adc, true, rh, rh, rh, none));
                // If the shift overflowed, r >= d certainly: go subtract.
                // bcs do_sub (skip the next 3 instructions)
                emit_branch_skip(sink, Cond::Hs.bits(), 3);
                // cmp rh, dh ; cmpeq rl, dl ; blo skip
                put(sink, enc_dp_reg(COND_AL, cmp, true, rh, 0, dh, none));
                put(
                    sink,
                    enc_dp_reg(Cond::Eq.bits(), cmp, true, rl, 0, dl, none),
                );
                emit_branch_skip(sink, Cond::Lo.bits(), 3);
                // do_sub: subs rl, rl, dl ; sbc rh, rh, dh ; orr ql, ql, #1
                put(sink, enc_dp_reg(COND_AL, sub, true, rl, rl, dl, none));
                put(sink, enc_dp_reg(COND_AL, sbc, false, rh, rh, dh, none));
                put(
                    sink,
                    enc_dp_imm(
                        COND_AL,
                        orr,
                        false,
                        ql,
                        ql,
                        ModImm::maybe_from_u32(1).unwrap(),
                    ),
                );
                // skip: subs ip, ip, #1 ; bne loop
                put(
                    sink,
                    enc_dp_imm(
                        COND_AL,
                        sub,
                        true,
                        IP,
                        IP,
                        ModImm::maybe_from_u32(1).unwrap(),
                    ),
                );
                emit_branch_to_offset(sink, Cond::Ne.bits(), loop_start);
                debug_assert_eq!(sink.cur_offset() - loop_start, 4 * LOOP_LEN);
            }
            Inst::BitRR { op, rd, rm } => {
                let base = match op {
                    BitOp::Clz => 0xe16f_0f10,
                    BitOp::Rbit => 0xe6ff_0f30,
                    BitOp::Rev => 0xe6bf_0f30,
                    BitOp::Rev16 => 0xe6bf_0fb0,
                    BitOp::Revsh => 0xe6ff_0fb0,
                    BitOp::Sxtb => 0xe6af_0070,
                    BitOp::Sxth => 0xe6bf_0070,
                    BitOp::Uxtb => 0xe6ef_0070,
                    BitOp::Uxth => 0xe6ff_0070,
                };
                put(sink, base | (gprw(*rd) << 12) | gpr(*rm));
            }
            Inst::Rot64 {
                left,
                rd_lo,
                rd_hi,
                rn_lo,
                rn_hi,
                amt,
                t1,
                t2,
            } => {
                let (ol, oh, il, ih) = (gprw(*rd_lo), gprw(*rd_hi), gpr(*rn_lo), gpr(*rn_hi));
                let (n, m) = (gprw(*t1), gprw(*t2));
                let and = alu_opcode(AluOp::And);
                let orr = alu_opcode(AluOp::Orr);
                let imm = |v: u32| ModImm::maybe_from_u32(v).unwrap();
                // n = amt & 63 ; tst n, #32 ; n &= 31 ; m = 32 - n (shifts by
                // 32 give zero, for n = 0)
                put(sink, enc_dp_imm(COND_AL, and, false, gpr(*amt), n, imm(63)));
                put(
                    sink,
                    enc_dp_imm(COND_AL, cmp_opcode(CmpOp::Tst), true, n, 0, imm(32)),
                );
                put(sink, enc_dp_imm(COND_AL, and, false, n, n, imm(31)));
                put(
                    sink,
                    enc_dp_imm(COND_AL, alu_opcode(AluOp::Rsb), false, n, m, imm(32)),
                );
                // Each output half is `a SH1 n | b SH2 m`; by 32 or more
                // (`ne`), the input halves are exchanged.
                let (sh1, sh2) = if *left {
                    (ShiftOp::Lsl, ShiftOp::Lsr)
                } else {
                    (ShiftOp::Lsr, ShiftOp::Lsl)
                };
                for (cond, (a_lo, b_lo)) in [(Cond::Eq, (il, ih)), (Cond::Ne, (ih, il))] {
                    let c = cond.bits();
                    for (out, a, b) in [(ol, a_lo, b_lo), (oh, b_lo, a_lo)] {
                        put(sink, enc_dp_regshift(c, OPC_MOV, false, 0, out, a, sh1, n));
                        put(sink, enc_dp_regshift(c, orr, false, out, out, b, sh2, m));
                    }
                }
            }
            Inst::Shift64 {
                op,
                rd_lo,
                rd_hi,
                rn_lo,
                rn_hi,
                amt,
                tmp,
            } => {
                let (ol, oh, il, ih, n, t) = (
                    gprw(*rd_lo),
                    gprw(*rd_hi),
                    gpr(*rn_lo),
                    gpr(*rn_hi),
                    gpr(*amt),
                    gprw(*tmp),
                );
                let rsb = alu_opcode(AluOp::Rsb);
                let sub = alu_opcode(AluOp::Sub);
                let orr = alu_opcode(AluOp::Orr);
                let imm32 = ModImm::maybe_from_u32(32).unwrap();
                match op {
                    ShiftOp::Lsl => {
                        // rsb t, n, #32
                        put(sink, enc_dp_imm(COND_AL, rsb, false, n, t, imm32));
                        // lsl oh, ih, n
                        put(
                            sink,
                            enc_dp_regshift(COND_AL, OPC_MOV, false, 0, oh, ih, ShiftOp::Lsl, n),
                        );
                        // orr oh, oh, il, lsr t
                        put(
                            sink,
                            enc_dp_regshift(COND_AL, orr, false, oh, oh, il, ShiftOp::Lsr, t),
                        );
                        // sub t, n, #32
                        put(sink, enc_dp_imm(COND_AL, sub, false, n, t, imm32));
                        // orr oh, oh, il, lsl t
                        put(
                            sink,
                            enc_dp_regshift(COND_AL, orr, false, oh, oh, il, ShiftOp::Lsl, t),
                        );
                        // lsl ol, il, n
                        put(
                            sink,
                            enc_dp_regshift(COND_AL, OPC_MOV, false, 0, ol, il, ShiftOp::Lsl, n),
                        );
                    }
                    ShiftOp::Lsr => {
                        put(sink, enc_dp_imm(COND_AL, rsb, false, n, t, imm32));
                        put(
                            sink,
                            enc_dp_regshift(COND_AL, OPC_MOV, false, 0, ol, il, ShiftOp::Lsr, n),
                        );
                        put(
                            sink,
                            enc_dp_regshift(COND_AL, orr, false, ol, ol, ih, ShiftOp::Lsl, t),
                        );
                        put(sink, enc_dp_imm(COND_AL, sub, false, n, t, imm32));
                        put(
                            sink,
                            enc_dp_regshift(COND_AL, orr, false, ol, ol, ih, ShiftOp::Lsr, t),
                        );
                        put(
                            sink,
                            enc_dp_regshift(COND_AL, OPC_MOV, false, 0, oh, ih, ShiftOp::Lsr, n),
                        );
                    }
                    ShiftOp::Asr => {
                        put(sink, enc_dp_imm(COND_AL, rsb, false, n, t, imm32));
                        put(
                            sink,
                            enc_dp_regshift(COND_AL, OPC_MOV, false, 0, ol, il, ShiftOp::Lsr, n),
                        );
                        put(
                            sink,
                            enc_dp_regshift(COND_AL, orr, false, ol, ol, ih, ShiftOp::Lsl, t),
                        );
                        // subs t, n, #32
                        put(sink, enc_dp_imm(COND_AL, sub, true, n, t, imm32));
                        // movpl ol, ih, asr t
                        put(
                            sink,
                            enc_dp_regshift(
                                Cond::Pl.bits(),
                                OPC_MOV,
                                false,
                                0,
                                ol,
                                ih,
                                ShiftOp::Asr,
                                t,
                            ),
                        );
                        put(
                            sink,
                            enc_dp_regshift(COND_AL, OPC_MOV, false, 0, oh, ih, ShiftOp::Asr, n),
                        );
                    }
                    ShiftOp::Ror => unreachable!("64-bit rotates are lowered to shifts"),
                }
            }
            Inst::Load { op, rd, mem, flags } => {
                if let Some(trap_code) = flags.trap_code() {
                    // Any address computation into `ip` is emitted first; the
                    // trap is registered at the load itself below.
                    let rd = gprw(*rd);
                    let m = match op {
                        LoadOp::Ldr | LoadOp::Ldrb => finalize_mem(sink, state, mem, MemKind::Word),
                        _ => finalize_mem(sink, state, mem, MemKind::Half),
                    };
                    sink.add_trap(trap_code);
                    match op {
                        LoadOp::Ldr | LoadOp::Ldrb => {
                            put(sink, enc_ldst_word(true, *op == LoadOp::Ldrb, rd, m))
                        }
                        LoadOp::Ldrh => put(sink, enc_ldst_half(true, 0b01, rd, m)),
                        LoadOp::Ldrsb => put(sink, enc_ldst_half(true, 0b10, rd, m)),
                        LoadOp::Ldrsh => put(sink, enc_ldst_half(true, 0b11, rd, m)),
                    }
                } else {
                    emit_load(sink, state, *op, gprw(*rd), mem);
                }
            }
            Inst::Store { op, rt, mem, flags } => {
                if let Some(trap_code) = flags.trap_code() {
                    let rt = gpr(*rt);
                    let m = match op {
                        StoreOp::Str | StoreOp::Strb => {
                            finalize_mem(sink, state, mem, MemKind::Word)
                        }
                        StoreOp::Strh => finalize_mem(sink, state, mem, MemKind::Half),
                    };
                    sink.add_trap(trap_code);
                    match op {
                        StoreOp::Str | StoreOp::Strb => {
                            put(sink, enc_ldst_word(false, *op == StoreOp::Strb, rt, m))
                        }
                        StoreOp::Strh => put(sink, enc_ldst_half(false, 0b01, rt, m)),
                    }
                } else {
                    emit_store(sink, state, *op, gpr(*rt), mem);
                }
            }
            Inst::LoadAddr { rd, mem } => {
                let rd = gprw(*rd);
                match mem {
                    AMode::RegReg { base, index, shift } => {
                        let sh = if *shift == 0 {
                            ShiftOpAndAmt::none()
                        } else {
                            ShiftOpAndAmt::new(ShiftOp::Lsl, *shift)
                        };
                        put(
                            sink,
                            enc_dp_reg(
                                COND_AL,
                                alu_opcode(AluOp::Add),
                                false,
                                gpr(*base),
                                rd,
                                gpr(*index),
                                sh,
                            ),
                        );
                    }
                    _ => {
                        let (base, offset) = mem.resolve(state).unwrap();
                        emit_add_imm_chunks(sink, rd, gpr(base), offset);
                    }
                }
            }
            Inst::FpuRint { mode, size, rd, rn } => {
                // F32 values live in the low half `s(2n)` of `d(n)`.
                let f64 = size.is_f64();
                let (d, m) = if f64 {
                    (drw(*rd), dr(*rn))
                } else {
                    (2 * drw(*rd), 2 * dr(*rn))
                };
                put(sink, enc_vrint(*mode, f64, d, m));
            }
            Inst::FpuZero { rd } => {
                let d = drw(*rd);
                put(sink, 0xf280_0010 | ((d >> 4) << 22) | ((d & 0xf) << 12));
            }
            Inst::FpuLoadConst { size, rd, constant } => {
                // vldr rd, [pc, #off], with the offset patched in once the
                // constant's location is known.
                let label = sink.get_label_for_constant(*constant);
                sink.use_label_at_offset(sink.cur_offset(), label, LabelUse::Vldr8);
                put(sink, enc_vldr_vstr(true, size.is_f64(), drw(*rd), PC, 0));
            }
            Inst::FpuMovImm { size, rd, imm } => {
                let f64 = size.is_f64();
                let (vd, dd) = vfp_fields(drw(*rd), f64);
                let imm = u32::from(*imm);
                put(
                    sink,
                    (COND_AL << 28)
                        | 0x0eb0_0a00
                        | (u32::from(f64) << 8)
                        | (dd << 22)
                        | ((imm >> 4) << 16)
                        | (vd << 12)
                        | (imm & 0xf),
                );
            }
            Inst::FpuRR { op, size, rd, rn } => {
                let base = match op {
                    FpuOp1::Abs => VABS,
                    FpuOp1::Neg => VNEG,
                    FpuOp1::Sqrt => VSQRT,
                    FpuOp1::Mov => VMOV,
                };
                let f64 = size.is_f64();
                put(
                    sink,
                    enc_vfp(
                        COND_AL,
                        base,
                        f64,
                        (drw(*rd), f64),
                        (0, f64),
                        (dr(*rn), f64),
                    ),
                );
            }
            Inst::FpuRRR {
                op,
                size,
                rd,
                rn,
                rm,
            } => {
                let base = match op {
                    FpuOp2::Add => VADD,
                    FpuOp2::Sub => VSUB,
                    FpuOp2::Mul => VMUL,
                    FpuOp2::Div => VDIV,
                };
                let f64 = size.is_f64();
                put(
                    sink,
                    enc_vfp(
                        COND_AL,
                        base,
                        f64,
                        (drw(*rd), f64),
                        (dr(*rn), f64),
                        (dr(*rm), f64),
                    ),
                );
            }
            Inst::FpuRRRR {
                op,
                size,
                rd,
                ra,
                rn,
                rm,
            } => {
                debug_assert_eq!(rd.to_reg(), *ra);
                let base = match op {
                    FpuOp3::Fma => VFMA,
                    FpuOp3::Fms => VFMS,
                    FpuOp3::Fnma => VFNMA,
                    FpuOp3::Fnms => VFNMS,
                };
                let f64 = size.is_f64();
                let (d, n, m) = (drw(*rd), dr(*rn), dr(*rm));
                if f64 || d < 16 {
                    put(
                        sink,
                        enc_vfp(COND_AL, base, f64, (d, f64), (n, f64), (m, f64)),
                    );
                } else {
                    // The accumulator is a reused input, which the register
                    // allocator may place in a register without an S half
                    // (`n` and `m` are limited to `d0`-`d15`). Swap it with a
                    // low register for the operation; `d16`-`d31` are only
                    // allocated when NEON is available.
                    let t = (0..3).find(|t| *t != n && *t != m).unwrap();
                    put(sink, enc_vswp(t, d));
                    put(
                        sink,
                        enc_vfp(COND_AL, base, false, (t, false), (n, false), (m, false)),
                    );
                    put(sink, enc_vswp(t, d));
                }
            }
            Inst::FpuCmp { size, rn, rm } => {
                let f64 = size.is_f64();
                put(
                    sink,
                    enc_vfp(COND_AL, VCMP, f64, (dr(*rn), f64), (0, f64), (dr(*rm), f64)),
                );
                put(sink, VMRS_APSR);
            }
            Inst::FpuCvt { op, rd, rn } => {
                let (rd, rn) = (drw(*rd), dr(*rn));
                let word = match op {
                    // sz=0: double from single.
                    FpuCvtOp::F32ToF64 => {
                        enc_vfp(COND_AL, VCVT_FF, false, (rd, true), (0, false), (rn, false))
                    }
                    // sz=1: single from double.
                    FpuCvtOp::F64ToF32 => {
                        enc_vfp(COND_AL, VCVT_FF, true, (rd, false), (0, false), (rn, true))
                    }
                    FpuCvtOp::F32ToS32 => enc_vfp(
                        COND_AL,
                        VCVT_TO_S32,
                        false,
                        (rd, false),
                        (0, false),
                        (rn, false),
                    ),
                    FpuCvtOp::F32ToU32 => enc_vfp(
                        COND_AL,
                        VCVT_TO_U32,
                        false,
                        (rd, false),
                        (0, false),
                        (rn, false),
                    ),
                    FpuCvtOp::F64ToS32 => enc_vfp(
                        COND_AL,
                        VCVT_TO_S32,
                        true,
                        (rd, false),
                        (0, false),
                        (rn, true),
                    ),
                    FpuCvtOp::F64ToU32 => enc_vfp(
                        COND_AL,
                        VCVT_TO_U32,
                        true,
                        (rd, false),
                        (0, false),
                        (rn, true),
                    ),
                    FpuCvtOp::S32ToF32 => enc_vfp(
                        COND_AL,
                        VCVT_FROM_S32,
                        false,
                        (rd, false),
                        (0, false),
                        (rn, false),
                    ),
                    FpuCvtOp::U32ToF32 => enc_vfp(
                        COND_AL,
                        VCVT_FROM_U32,
                        false,
                        (rd, false),
                        (0, false),
                        (rn, false),
                    ),
                    FpuCvtOp::S32ToF64 => enc_vfp(
                        COND_AL,
                        VCVT_FROM_S32,
                        true,
                        (rd, true),
                        (0, false),
                        (rn, false),
                    ),
                    FpuCvtOp::U32ToF64 => enc_vfp(
                        COND_AL,
                        VCVT_FROM_U32,
                        true,
                        (rd, true),
                        (0, false),
                        (rn, false),
                    ),
                };
                put(sink, word);
            }
            Inst::FpuCSel {
                rd, cond, rn, rm, ..
            } => {
                // Moving the whole D register works for both precisions.
                let (rd, rn, rm) = (drw(*rd), dr(*rn), dr(*rm));
                let vmov = |cond: u32, d: u32, m: u32| {
                    enc_vfp(cond, VMOV, true, (d, true), (0, true), (m, true))
                };
                if rd == rn {
                    if rd != rm {
                        put(sink, vmov(cond.invert().bits(), rd, rm));
                    }
                } else if rd == rm {
                    put(sink, vmov(cond.bits(), rd, rn));
                } else {
                    put(sink, vmov(COND_AL, rd, rm));
                    put(sink, vmov(cond.bits(), rd, rn));
                }
            }
            Inst::FpuMinMax {
                is_min,
                size,
                rd,
                rn,
                rm,
                tmp,
            } => {
                let f64 = size.is_f64();
                let (rd, rn, rm, tmp) = (drw(*rd), dr(*rn), dr(*rm), gprw(*tmp));
                // vcmp rn, rm ; vmrs
                put(
                    sink,
                    enc_vfp(COND_AL, VCMP, f64, (rn, f64), (0, f64), (rm, f64)),
                );
                put(sink, VMRS_APSR);
                // rd = rn
                put(
                    sink,
                    enc_vfp(COND_AL, VMOV, true, (rd, true), (0, true), (rn, true)),
                );
                // min: if rn > rm, rd = rm. max: if rn < rm, rd = rm.
                let cond = if *is_min { Cond::Gt } else { Cond::Mi };
                put(
                    sink,
                    enc_vfp(cond.bits(), VMOV, true, (rd, true), (0, true), (rm, true)),
                );
                // Unordered: rd = rn + rm (a NaN).
                put(
                    sink,
                    enc_vfp(Cond::Vs.bits(), VADD, f64, (rd, f64), (rn, f64), (rm, f64)),
                );
                // Equal: merge the sign bits (only relevant for zeros).
                let eq = Cond::Eq.bits();
                if f64 {
                    put(sink, enc_vmov_core_scalar_hi(eq, true, rn, tmp));
                    put(sink, enc_vmov_core_scalar_hi(eq, true, rm, IP));
                } else {
                    put(sink, enc_vmov_core_single(eq, true, rn, tmp));
                    put(sink, enc_vmov_core_single(eq, true, rm, IP));
                }
                let op = if *is_min { AluOp::Orr } else { AluOp::And };
                put(
                    sink,
                    enc_dp_reg(
                        eq,
                        alu_opcode(op),
                        false,
                        tmp,
                        tmp,
                        IP,
                        ShiftOpAndAmt::none(),
                    ),
                );
                if f64 {
                    put(sink, enc_vmov_core_scalar_hi(eq, false, rd, tmp));
                } else {
                    put(sink, enc_vmov_core_single(eq, false, rd, tmp));
                }
            }
            Inst::FpuRound {
                mode,
                size,
                rd,
                rn,
                tmp1,
                tmp2,
                ftmp,
            } => {
                emit_fpu_round(
                    sink,
                    *mode,
                    size.is_f64(),
                    drw(*rd),
                    dr(*rn),
                    gprw(*tmp1),
                    gprw(*tmp2),
                    drw(*ftmp),
                );
            }
            Inst::FpuLoad {
                size,
                rd,
                mem,
                flags,
            } => {
                let is_f64 = size.is_f64();
                let m = finalize_mem(sink, state, mem, MemKind::Vfp);
                if let Some(trap_code) = flags.trap_code() {
                    sink.add_trap(trap_code);
                }
                match m {
                    FinalMem::Imm { base, offset } => {
                        put(sink, enc_vldr_vstr(true, is_f64, drw(*rd), base, offset))
                    }
                    FinalMem::Reg { .. } => unreachable!(),
                }
            }
            Inst::FpuLoadUnaligned {
                size,
                rd,
                mem,
                flags,
            } => {
                let base = super::neon::vec_mem_base(sink, state, mem);
                if let Some(trap_code) = flags.trap_code() {
                    sink.add_trap(trap_code);
                }
                let d = drw(*rd);
                let (vd, dd) = (d & 0xf, d >> 4);
                let base_word = match size {
                    // vld1.8 {dd}, [base]
                    FpuSize::F64 => 0xf420_070f,
                    // vld1.32 {dd[0]}, [base]
                    FpuSize::F32 => 0xf4a0_080f,
                };
                put(sink, base_word | (dd << 22) | (base << 16) | (vd << 12));
            }
            Inst::FpuStoreUnaligned32 { rt, mem, flags } => {
                let base = super::neon::vec_mem_base(sink, state, mem);
                if let Some(trap_code) = flags.trap_code() {
                    sink.add_trap(trap_code);
                }
                let d = dr(*rt);
                let (vd, dd) = (d & 0xf, d >> 4);
                // vst1.32 {dd[0]}, [base]
                put(sink, 0xf480_080f | (dd << 22) | (base << 16) | (vd << 12));
            }
            Inst::FpuStore {
                size,
                rt,
                mem,
                flags,
            } => {
                let is_f64 = size.is_f64();
                let m = finalize_mem(sink, state, mem, MemKind::Vfp);
                if let Some(trap_code) = flags.trap_code() {
                    sink.add_trap(trap_code);
                }
                match m {
                    FinalMem::Imm { base, offset } => {
                        put(sink, enc_vldr_vstr(false, is_f64, dr(*rt), base, offset))
                    }
                    FinalMem::Reg { .. } => unreachable!(),
                }
            }
            Inst::MovToFpu32 { rd, rn } => put(
                sink,
                enc_vmov_core_single(COND_AL, false, drw(*rd), gpr(*rn)),
            ),
            Inst::MovFromFpu32 { rd, rn } => put(
                sink,
                enc_vmov_core_single(COND_AL, true, dr(*rn), gprw(*rd)),
            ),
            Inst::MovToFpu64 { rd, lo, hi } => put(
                sink,
                enc_vmov_core_double(false, drw(*rd), gpr(*lo), gpr(*hi)),
            ),
            Inst::MovFromFpu64 { rd_lo, rd_hi, rn } => put(
                sink,
                enc_vmov_core_double(true, dr(*rn), gprw(*rd_lo), gprw(*rd_hi)),
            ),
            Inst::FpuMove { rd, rn } => {
                if rd.to_reg() != *rn {
                    put(
                        sink,
                        enc_vfp(
                            COND_AL,
                            VMOV,
                            true,
                            (drw(*rd), true),
                            (0, true),
                            (dr(*rn), true),
                        ),
                    );
                }
            }
            Inst::Jump { dest } => {
                let off = sink.cur_offset();
                sink.use_label_at_offset(off, *dest, LabelUse::Branch24);
                sink.add_uncond_branch(off, off + 4, *dest);
                put(sink, enc_branch(COND_AL, false));
            }
            Inst::CondBr {
                taken,
                not_taken,
                cond,
            } => {
                // Conditional part first.
                let cond_off = sink.cur_offset();
                sink.use_label_at_offset(cond_off, *taken, LabelUse::Branch24);
                let inverted = enc_branch(cond.invert().bits(), false).to_le_bytes();
                sink.add_cond_branch(cond_off, cond_off + 4, *taken, &inverted[..]);
                put(sink, enc_branch(cond.bits(), false));

                // Unconditional part next.
                let uncond_off = sink.cur_offset();
                sink.use_label_at_offset(uncond_off, *not_taken, LabelUse::Branch24);
                sink.add_uncond_branch(uncond_off, uncond_off + 4, *not_taken);
                put(sink, enc_branch(COND_AL, false));
            }
            Inst::TrapIf { cond, trap_code } => {
                let label = sink.defer_trap(*trap_code);
                let off = sink.cur_offset();
                sink.use_label_at_offset(off, label, LabelUse::Branch24);
                put(sink, enc_branch(cond.bits(), false));
            }
            Inst::Udf { trap_code } => {
                sink.add_trap(*trap_code);
                put(sink, udf_word());
            }
            Inst::Bkpt => put(sink, 0xe120_0070),
            Inst::BrTable { index, targets } => {
                let default = targets[0];
                let table = &targets[1..];
                let n = u32::try_from(table.len()).unwrap();
                // Make sure the whole sequence fits before the next island
                // deadline.
                let size = 4 * (n + 8);
                if sink.island_needed(size) {
                    let jump_around = sink.get_label();
                    Inst::Jump { dest: jump_around }.emit(sink, emit_info, state);
                    sink.emit_island(size + 4, &mut state.ctrl_plane);
                    sink.bind_label(jump_around, &mut state.ctrl_plane);
                }
                let index = gpr(*index);
                // cmp index, #n (or via ip)
                if let Some(imm) = ModImm::maybe_from_u32(n) {
                    put(
                        sink,
                        enc_dp_imm(COND_AL, cmp_opcode(CmpOp::Cmp), true, index, 0, imm),
                    );
                } else {
                    emit_load_const32(sink, IP, n);
                    put(
                        sink,
                        enc_dp_reg(
                            COND_AL,
                            cmp_opcode(CmpOp::Cmp),
                            true,
                            index,
                            0,
                            IP,
                            ShiftOpAndAmt::none(),
                        ),
                    );
                }
                // bhs default
                let off = sink.cur_offset();
                sink.use_label_at_offset(off, default, LabelUse::Branch24);
                put(sink, enc_branch(Cond::Hs.bits(), false));
                // add pc, pc, index, lsl #2
                put(
                    sink,
                    enc_dp_reg(
                        COND_AL,
                        alu_opcode(AluOp::Add),
                        false,
                        PC,
                        PC,
                        index,
                        ShiftOpAndAmt::new(ShiftOp::Lsl, 2),
                    ),
                );
                // Filler: never executed.
                put(sink, udf_word());
                for &target in table {
                    let off = sink.cur_offset();
                    sink.use_label_at_offset(off, target, LabelUse::Branch24);
                    put(sink, enc_branch(COND_AL, false));
                }
                start_off = sink.cur_offset();
            }
            Inst::Call { info } => {
                emit_shadow_arg_copies(sink, &info.uses);
                let start = sink.cur_offset();
                sink.add_reloc(Reloc::Arm32Call, &info.dest, 0);
                put(sink, enc_branch(COND_AL, true));
                emit_post_call(sink, emit_info, state, info, start);
                start_off = sink.cur_offset();
            }
            Inst::CallInd { info } => {
                emit_shadow_arg_copies(sink, &info.uses);
                let start = sink.cur_offset();
                put(sink, enc_blx(gpr(info.dest)));
                emit_post_call(sink, emit_info, state, info, start);
                start_off = sink.cur_offset();
            }
            Inst::ReturnCall { info } => {
                emit_return_call_common_sequence(sink, emit_info, state, info.new_stack_arg_size);
                sink.add_call_site();
                sink.add_reloc(Reloc::Arm32Call, &info.dest, 0);
                put(sink, enc_branch(COND_AL, false));
                start_off = sink.cur_offset();
            }
            Inst::ReturnCallInd { info } => {
                // Move the target into `ip`, which is not touched by the
                // epilogue sequence.
                let dest = gpr(info.dest);
                if dest != IP {
                    put(sink, enc_mov_reg(COND_AL, IP, dest));
                }
                emit_return_call_common_sequence(sink, emit_info, state, info.new_stack_arg_size);
                sink.add_call_site();
                put(sink, enc_bx(IP));
                start_off = sink.cur_offset();
            }
            Inst::Args { args } => {
                // Copy `f32` arguments passed in odd S registers into their
                // shadow registers; see `aapcs_sreg_to_dreg`.
                for arg in args.iter() {
                    let preg = arg.preg.to_real_reg().unwrap();
                    if preg.class() == RegClass::Float {
                        let d = u32::from(preg.hw_enc());
                        if let Some(odd_s) = shadow_dreg_to_sreg(d) {
                            put(sink, enc_vmov_f32_sreg(2 * d, odd_s));
                        }
                    }
                }
            }
            Inst::Rets { .. } => {
                // A pseudo-instruction that only constrains registers.
            }
            Inst::Ret => put(sink, enc_bx(LR)),
            Inst::LoadExtName { rd, name, offset } => {
                // ldr rd, [pc]   ; pc reads as this instruction + 8
                // b   skip
                // .word name + offset
                // skip:
                let rd = gprw(*rd);
                put(sink, 0xe59f_0000 | (rd << 12));
                put(sink, enc_branch(COND_AL, false));
                sink.add_reloc(Reloc::Abs4, &**name, i64::from(*offset));
                put(sink, 0);
            }
            Inst::LoadExtNameNear { rd, name, offset } => {
                // A:    ldr rd, [pc, #4]   ; loads the word at A + 12
                // A+4:  add rd, pc, rd     ; pc reads as A + 12
                // A+8:  b   A+16
                // A+12: .word name + offset - (A + 12)
                let rd = gprw(*rd);
                put(sink, 0xe59f_0004 | (rd << 12));
                put(
                    sink,
                    enc_dp_reg(
                        COND_AL,
                        alu_opcode(AluOp::Add),
                        false,
                        PC,
                        rd,
                        rd,
                        ShiftOpAndAmt::none(),
                    ),
                );
                put(sink, enc_branch(COND_AL, false));
                sink.add_reloc(Reloc::X86PCRel4, &**name, i64::from(*offset));
                put(sink, 0);
            }
            Inst::LabelAddress { rd, label } => {
                // movw rd, #lo ; movt rd, #hi ; add rd, pc, rd
                let rd = gprw(*rd);
                let off = sink.cur_offset();
                sink.use_label_at_offset(off, *label, LabelUse::MovwMovtPcRel);
                put(sink, enc_movw(rd, 0));
                put(sink, enc_movt(rd, 0));
                put(
                    sink,
                    enc_dp_reg(
                        COND_AL,
                        alu_opcode(AluOp::Add),
                        false,
                        PC,
                        rd,
                        rd,
                        ShiftOpAndAmt::none(),
                    ),
                );
            }
            Inst::MovFromPReg { rd, rm } => {
                put(sink, enc_mov_reg(COND_AL, gprw(*rd), gpr(Reg::from(*rm))));
            }
            Inst::Fence => put(sink, 0xf57f_f05b),
            Inst::LoadExclusive { size, rd, addr } => {
                let base = match size {
                    1 => 0xe1d0_0f9f,
                    2 => 0xe1f0_0f9f,
                    _ => 0xe190_0f9f,
                };
                put(sink, base | (gpr(*addr) << 16) | (gprw(*rd) << 12));
            }
            Inst::AtomicRmwLoop {
                op,
                size,
                rd,
                addr,
                operand,
                tmp1,
                tmp2,
                flags,
                acq_rel,
            } => {
                emit_atomic_rmw_loop(
                    sink,
                    *op,
                    *size,
                    gprw(*rd),
                    gpr(*addr),
                    gpr(*operand),
                    gprw(*tmp1),
                    gprw(*tmp2),
                    flags.trap_code(),
                    *acq_rel,
                );
            }
            Inst::AtomicCasLoop {
                size,
                rd,
                addr,
                expected,
                replacement,
                tmp,
                flags,
                acq_rel,
            } => {
                emit_atomic_cas_loop(
                    sink,
                    *size,
                    gprw(*rd),
                    gpr(*addr),
                    gpr(*expected),
                    gpr(*replacement),
                    gprw(*tmp),
                    flags.trap_code(),
                    *acq_rel,
                );
            }
            Inst::AtomicRmwLoop64 {
                op,
                addr,
                flags,
                acq_rel,
                ..
            } => {
                emit_atomic_rmw_loop64(sink, *op, gpr(*addr), flags.trap_code(), *acq_rel);
            }
            Inst::AtomicCasLoop64 {
                addr,
                flags,
                acq_rel,
                ..
            } => {
                emit_atomic_cas_loop64(sink, gpr(*addr), flags.trap_code(), *acq_rel);
            }
            Inst::LoadAcquire {
                size,
                rd,
                addr,
                flags,
            } => {
                let base = match size {
                    1 => 0xe1d0_0c9f,
                    2 => 0xe1f0_0c9f,
                    _ => 0xe190_0c9f,
                };
                add_trap_opt(sink, flags.trap_code());
                put(sink, base | (gpr(*addr) << 16) | (gprw(*rd) << 12));
            }
            Inst::StoreRelease {
                size,
                rt,
                addr,
                flags,
            } => {
                let base = match size {
                    1 => 0xe1c0_fc90,
                    2 => 0xe1e0_fc90,
                    _ => 0xe180_fc90,
                };
                add_trap_opt(sink, flags.trap_code());
                put(sink, base | (gpr(*addr) << 16) | gpr(*rt));
            }
            Inst::AtomicLoad64 {
                rd_lo,
                rd_hi,
                addr,
                flags,
                acq_rel,
            } => {
                debug_assert_eq!((gprw(*rd_lo), gprw(*rd_hi)), (0, 1));
                // ldrexd r0, r1, [addr] ; clrex ; dmb ish
                // (or `ldaexd r0, r1, [addr] ; clrex`)
                add_trap_opt(sink, flags.trap_code());
                put(sink, exclusive(0xe1b0_0f9f, *acq_rel) | (gpr(*addr) << 16));
                put(sink, 0xf57f_f01f);
                if !*acq_rel {
                    put(sink, 0xf57f_f05b);
                }
            }
            Inst::Push { mask } => put(sink, 0xe92d_0000 | u32::from(*mask)),
            Inst::Pop { mask } => put(sink, 0xe8bd_0000 | u32::from(*mask)),
            Inst::VPush { first, count } => {
                let (vd, d) = vfp_fields(u32::from(*first), true);
                put(
                    sink,
                    0xed2d_0b00 | (d << 22) | (vd << 12) | (2 * u32::from(*count)),
                );
            }
            Inst::VPop { first, count } => {
                let (vd, d) = vfp_fields(u32::from(*first), true);
                put(
                    sink,
                    0xecbd_0b00 | (d << 22) | (vd << 12) | (2 * u32::from(*count)),
                );
            }
            Inst::SpAdjust { amount } => {
                emit_add_imm_chunks(sink, SP, SP, i64::from(*amount));
            }
            Inst::StackProbeLoop {
                guard_size,
                probe_count,
            } => {
                // mov ip, #probe_count
                // loop:
                //   sub sp, sp, #guard_size
                //   str ip, [sp]
                //   subs ip, ip, #1
                //   bne loop
                // add sp, sp, #guard_size * probe_count
                emit_load_const32(sink, IP, *probe_count);
                let loop_start = sink.cur_offset();
                emit_add_imm_chunks(sink, SP, SP, -i64::from(*guard_size));
                put(
                    sink,
                    enc_ldst_word(
                        false,
                        false,
                        IP,
                        FinalMem::Imm {
                            base: SP,
                            offset: 0,
                        },
                    ),
                );
                put(
                    sink,
                    enc_dp_imm(
                        COND_AL,
                        alu_opcode(AluOp::Sub),
                        true,
                        IP,
                        IP,
                        ModImm::maybe_from_u32(1).unwrap(),
                    ),
                );
                emit_branch_to_offset(sink, Cond::Ne.bits(), loop_start);
                emit_add_imm_chunks(
                    sink,
                    SP,
                    SP,
                    i64::from(*guard_size) * i64::from(*probe_count),
                );
            }
            Inst::Unwind { inst } => {
                sink.add_unwind(inst.clone());
            }
            Inst::DummyUse { .. } => {}
            Inst::EmitIsland { needed_space } => {
                if sink.island_needed(*needed_space + 4) {
                    let jump_around = sink.get_label();
                    Inst::Jump { dest: jump_around }.emit(sink, emit_info, state);
                    sink.emit_island(needed_space + 4, &mut state.ctrl_plane);
                    sink.bind_label(jump_around, &mut state.ctrl_plane);
                }
            }
            Inst::SequencePoint => {}
            _ => unreachable!("vector instructions are emitted by `emit_vec`: {self:?}"),
        }

        let end_off = sink.cur_offset();
        assert!(
            (end_off - start_off) <= Inst::worst_case_size(),
            "Inst:{:?} length:{} worst_case_size:{}",
            self,
            end_off - start_off,
            Inst::worst_case_size()
        );
    }

    /// If this is an unconditional comparison of allocatable registers, the
    /// key identifying the flags it produces.
    fn flags_key(&self) -> Option<FlagsKey> {
        let key = match *self {
            Inst::CmpRR { op, rn, rm, shift } => FlagsKey::RR { op, rn, rm, shift },
            Inst::CmpRImm { op, rn, imm } => FlagsKey::RImm { op, rn, imm },
            _ => return None,
        };
        // `ip` and `sp` are modified implicitly by many instruction
        // sequences; don't track comparisons involving them.
        let tracked = |r: Reg| {
            r.to_real_reg()
                .is_some_and(|r| r.class() == RegClass::Int && r.hw_enc() != 12 && r.hw_enc() != 13)
        };
        let ok = match key {
            FlagsKey::RR { rn, rm, .. } => tracked(rn) && tracked(rm),
            FlagsKey::RImm { rn, .. } => tracked(rn),
        };
        ok.then_some(key)
    }

    /// Whether this instruction leaves the flags unchanged and does not write
    /// any register that `known` reads.
    fn preserves_flags_and(&self, known: FlagsKey) -> bool {
        let Some(defs) = self.defs_if_flags_preserved() else {
            return false;
        };
        let (a, b) = match known {
            FlagsKey::RR { rn, rm, .. } => (rn, Some(rm)),
            FlagsKey::RImm { rn, .. } => (rn, None),
        };
        !defs.contains(&a) && !b.is_some_and(|b| defs.contains(&b))
    }

    /// If this instruction is known to leave the flags unchanged, the
    /// registers it writes (other than `ip`).
    ///
    /// (The registers are listed explicitly: at emission time all operands
    /// are physical registers, which `get_operands` does not report as
    /// defs.)
    fn defs_if_flags_preserved(&self) -> Option<SmallVec<[Reg; 2]>> {
        let one = |rd: &Writable<Reg>| Some(smallvec![rd.to_reg()]);
        match self {
            Inst::Nop0 | Inst::Nop4 | Inst::Unwind { .. } => Some(smallvec![]),
            Inst::AluRRR { s, rd, .. }
            | Inst::AluRRRShift { s, rd, .. }
            | Inst::AluRRImm { s, rd, .. }
            | Inst::MovRR { s, rd, .. } => {
                if *s {
                    None
                } else {
                    one(rd)
                }
            }
            Inst::MovRRShift { rd, .. }
            | Inst::MovImm { rd, .. }
            | Inst::MovW { rd, .. }
            | Inst::MovT { rd, .. }
            | Inst::Mov { rd, .. }
            | Inst::CSel { rd, .. }
            | Inst::CSelImm { rd, .. }
            | Inst::CSet { rd, .. }
            | Inst::CSetm { rd, .. }
            | Inst::Mul { rd, .. }
            | Inst::MulAcc { rd, .. }
            | Inst::Div { rd, .. }
            | Inst::BitRR { rd, .. }
            | Inst::Load { rd, .. }
            | Inst::LoadAddr { rd, .. }
            | Inst::FpuRR { rd, .. }
            | Inst::FpuRRR { rd, .. }
            | Inst::FpuRRRR { rd, .. }
            | Inst::FpuMovImm { rd, .. }
            | Inst::FpuZero { rd }
            | Inst::FpuRint { rd, .. }
            | Inst::FpuLoadConst { rd, .. }
            | Inst::FpuCvt { rd, .. }
            | Inst::FpuCSel { rd, .. }
            | Inst::FpuLoad { rd, .. }
            | Inst::FpuLoadUnaligned { rd, .. }
            | Inst::MovToFpu32 { rd, .. }
            | Inst::MovFromFpu32 { rd, .. }
            | Inst::MovToFpu64 { rd, .. } => one(rd),
            Inst::MulLong { rd_lo, rd_hi, .. }
            | Inst::Umaal { rd_lo, rd_hi, .. }
            | Inst::MovFromFpu64 { rd_lo, rd_hi, .. } => {
                Some(smallvec![rd_lo.to_reg(), rd_hi.to_reg()])
            }
            Inst::Store { .. } | Inst::FpuStore { .. } | Inst::FpuStoreUnaligned32 { .. } => {
                Some(smallvec![])
            }
            _ => None,
        }
    }
}

/// The comparison whose result is known to be in the flags.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FlagsKey {
    RR {
        op: CmpOp,
        rn: Reg,
        rm: Reg,
        shift: ShiftOpAndAmt,
    },
    RImm {
        op: CmpOp,
        rn: Reg,
        imm: ModImm,
    },
}

/// Set the FPSCR rounding mode for `mode` (not `Nearest`, the default),
/// saving the old FPSCR in the core register `save`; clobbers ip. Restore it
/// with `vmsr fpscr, save` (`enc_vmsr`).
pub(crate) fn emit_set_rounding_mode(sink: &mut MachBuffer<Inst>, mode: FpuRoundMode, save: u32) {
    let rmode = match mode {
        FpuRoundMode::Ceil => 0b01,
        FpuRoundMode::Floor => 0b10,
        FpuRoundMode::Trunc => 0b11,
        FpuRoundMode::Nearest => unreachable!(),
    };
    // vmrs save, fpscr ; bic ip, save, #(3 << 22) ; orr ip, ip, #(rmode << 22) ; vmsr fpscr, ip
    put(sink, enc_vmrs(save));
    put(
        sink,
        enc_dp_imm(
            COND_AL,
            alu_opcode(AluOp::Bic),
            false,
            save,
            IP,
            ModImm::maybe_from_u32(3 << 22).unwrap(),
        ),
    );
    put(
        sink,
        enc_dp_imm(
            COND_AL,
            alu_opcode(AluOp::Orr),
            false,
            IP,
            IP,
            ModImm::maybe_from_u32(rmode << 22).unwrap(),
        ),
    );
    put(sink, enc_vmsr(IP));
}

/// `vrint{p,m,z,n}.f32/f64 rd, rm` (ARMv8), with `d` and `m` the D register
/// numbers for F64 and the S register numbers for F32.
pub(crate) fn enc_vrint(mode: FpuRoundMode, f64: bool, d: u32, m: u32) -> u32 {
    let base = match mode {
        FpuRoundMode::Ceil => 0xfeba_0a40,
        FpuRoundMode::Floor => 0xfebb_0a40,
        FpuRoundMode::Nearest => 0xfeb9_0a40,
        FpuRoundMode::Trunc => 0xeeb6_0ac0,
    };
    let (vd, dd, vm, mm) = if f64 {
        (d & 0xf, d >> 4, m & 0xf, m >> 4)
    } else {
        (d >> 1, d & 1, m >> 1, m & 1)
    };
    base | (u32::from(f64) << 8) | (dd << 22) | (vd << 12) | (mm << 5) | vm
}

/// `vmov.f32 s<sd>, s<sm>` with arbitrary S register numbers.
fn enc_vmov_f32_sreg(sd: u32, sm: u32) -> u32 {
    (COND_AL << 28) | VMOV | ((sd & 1) << 22) | ((sd >> 1) << 12) | ((sm & 1) << 5) | (sm >> 1)
}

/// Before a call, copy `f32` arguments whose AAPCS location is an odd S
/// register from their shadow registers; see `aapcs_sreg_to_dreg`.
fn emit_shadow_arg_copies(sink: &mut MachBuffer<Inst>, uses: &CallArgList) {
    for arg in uses {
        let preg = arg.preg.to_real_reg().unwrap();
        if preg.class() == RegClass::Float {
            let d = u32::from(preg.hw_enc());
            if let Some(odd_s) = shadow_dreg_to_sreg(d) {
                put(sink, enc_vmov_f32_sreg(odd_s, 2 * d));
            }
        }
    }
}

/// Emit the bookkeeping that follows a call instruction.
fn emit_post_call<T>(
    sink: &mut MachBuffer<Inst>,
    emit_info: &EmitInfo,
    state: &mut EmitState,
    info: &CallInfo<T>,
    start: CodeOffset,
) {
    if let Some(s) = state.take_stack_map() {
        let offset = sink.cur_offset();
        sink.push_user_stack_map(state, offset, s);
    }

    if let Some(try_call) = info.try_call_info.as_ref() {
        sink.add_try_call_site(
            Some(state.frame_layout.sp_to_fp()),
            try_call.exception_handlers(&state.frame_layout),
        );
    } else {
        sink.add_call_site();
    }

    let callee_pop_size = i32::try_from(info.callee_pop_size).unwrap();
    if callee_pop_size > 0 {
        for inst in Arm32MachineDeps::gen_sp_reg_adjust(-callee_pop_size) {
            inst.emit(sink, emit_info, state);
        }
    }

    if info.patchable {
        sink.add_patchable_call_site(sink.cur_offset() - start);
    } else {
        // Load any stack-carried return values.
        info.emit_retval_loads::<Arm32MachineDeps, _, _>(
            state.frame_layout().stackslots_size,
            |inst| inst.emit(sink, emit_info, state),
            |needed_space| Some(Inst::EmitIsland { needed_space }),
        );
    }

    // If this is a try-call, jump to the continuation (normal-return) block.
    if let Some(try_call) = info.try_call_info.as_ref() {
        Inst::Jump {
            dest: try_call.continuation,
        }
        .emit(sink, emit_info, state);
    }
}

/// Tear down the current frame before a tail call, leaving SP pointing at
/// the callee's stack arguments. Does not clobber `ip`.
fn emit_return_call_common_sequence(
    sink: &mut MachBuffer<Inst>,
    emit_info: &EmitInfo,
    state: &mut EmitState,
    new_stack_arg_size: u32,
) {
    // The sequence is short (bounded by the number of callee-saved
    // registers) so no island check is needed beyond the usual one.
    let frame_layout = state.frame_layout().clone();
    let mut insts: SmallVec<[Inst; 16]> = SmallVec::new();
    gen_clobber_restore_insts(&frame_layout, &mut |inst| insts.push(inst));
    if frame_layout.setup_area_size > 0 {
        insts.push(Inst::Pop {
            mask: merged_clobber_mask(&frame_layout) | (1 << 11) | (1 << 14),
        });
    }
    let incoming_args_diff = frame_layout.tail_args_size - new_stack_arg_size;
    if incoming_args_diff > 0 {
        insts.push(Inst::SpAdjust {
            amount: i32::try_from(incoming_args_diff).unwrap(),
        });
    }
    for inst in insts {
        inst.emit(sink, emit_info, state);
    }
}

/// Emit a floating-point round-to-integral sequence (VFPv3 has no `vrint`).
fn emit_fpu_round(
    sink: &mut MachBuffer<Inst>,
    mode: FpuRoundMode,
    f64: bool,
    rd: u32,
    rn: u32,
    tmp1: u32,
    tmp2: u32,
    ftmp: u32,
) {
    // ftmp = |x|
    put(
        sink,
        enc_vfp(COND_AL, VABS, f64, (ftmp, f64), (0, f64), (rn, f64)),
    );
    // rd = 2^52 (f64) or 2^23 (f32)
    if f64 {
        emit_load_const32(sink, tmp1, 0);
        emit_load_const32(sink, tmp2, 0x4330_0000);
        put(sink, enc_vmov_core_double(false, rd, tmp1, tmp2));
    } else {
        emit_load_const32(sink, tmp2, 0x4b00_0000);
        put(sink, enc_vmov_core_single(COND_AL, false, rd, tmp2));
    }
    // vcmp ftmp, rd ; vmrs
    put(
        sink,
        enc_vfp(COND_AL, VCMP, f64, (ftmp, f64), (0, f64), (rd, f64)),
    );
    put(sink, VMRS_APSR);

    // If not (|x| < magic), the value is already integral, infinite or NaN.
    let set_rmode = mode != FpuRoundMode::Nearest;
    let rounding_len = 12 + if set_rmode { 5 } else { 0 };
    // bpl not_small (skip the rounding sequence)
    emit_branch_skip(sink, Cond::Pl.bits(), rounding_len);
    let rounding_start = sink.cur_offset();

    // tmp2 = the word holding x's sign bit.
    if f64 {
        put(sink, enc_vmov_core_scalar_hi(COND_AL, true, rn, tmp2));
    } else {
        put(sink, enc_vmov_core_single(COND_AL, true, rn, tmp2));
    }
    // If x is negative, negate the magic constant.
    put(
        sink,
        enc_dp_imm(
            COND_AL,
            cmp_opcode(CmpOp::Tst),
            true,
            tmp2,
            0,
            ModImm::maybe_from_u32(0x8000_0000).unwrap(),
        ),
    );
    put(
        sink,
        enc_vfp(Cond::Ne.bits(), VNEG, f64, (rd, f64), (0, f64), (rd, f64)),
    );
    if set_rmode {
        emit_set_rounding_mode(sink, mode, tmp1);
    }
    // ftmp = (x + magic) - magic
    put(
        sink,
        enc_vfp(COND_AL, VADD, f64, (ftmp, f64), (rn, f64), (rd, f64)),
    );
    put(
        sink,
        enc_vfp(COND_AL, VSUB, f64, (ftmp, f64), (ftmp, f64), (rd, f64)),
    );
    if set_rmode {
        put(sink, enc_vmsr(tmp1));
    }
    // Copy the sign of x onto the result: the rounded magnitude may have
    // lost the sign of a zero result.
    if f64 {
        put(sink, enc_vmov_core_scalar_hi(COND_AL, true, ftmp, tmp1));
    } else {
        put(sink, enc_vmov_core_single(COND_AL, true, ftmp, tmp1));
    }
    put(
        sink,
        enc_dp_imm(
            COND_AL,
            alu_opcode(AluOp::Bic),
            false,
            tmp1,
            tmp1,
            ModImm::maybe_from_u32(0x8000_0000).unwrap(),
        ),
    );
    put(
        sink,
        enc_dp_imm(
            COND_AL,
            alu_opcode(AluOp::And),
            false,
            tmp2,
            IP,
            ModImm::maybe_from_u32(0x8000_0000).unwrap(),
        ),
    );
    put(
        sink,
        enc_dp_reg(
            COND_AL,
            alu_opcode(AluOp::Orr),
            false,
            tmp1,
            tmp1,
            IP,
            ShiftOpAndAmt::none(),
        ),
    );
    // rd = ftmp with the sign word replaced.
    put(
        sink,
        enc_vfp(COND_AL, VMOV, true, (rd, true), (0, true), (ftmp, true)),
    );
    if f64 {
        put(sink, enc_vmov_core_scalar_hi(COND_AL, false, rd, tmp1));
    } else {
        put(sink, enc_vmov_core_single(COND_AL, false, rd, tmp1));
    }
    // b done (skip the two instructions of the not-small path)
    emit_branch_skip(sink, COND_AL, 2);
    debug_assert_eq!(
        sink.cur_offset() - rounding_start,
        4 * rounding_len,
        "rounding sequence length mismatch"
    );

    // not_small: rd = x; if NaN, rd = x + x (quieting it).
    put(
        sink,
        enc_vfp(COND_AL, VMOV, true, (rd, true), (0, true), (rn, true)),
    );
    put(
        sink,
        enc_vfp(Cond::Vs.bits(), VADD, f64, (rd, f64), (rn, f64), (rn, f64)),
    );
    // done:
}

fn atomic_size_ldrex_strex(size: u8) -> (u32, u32) {
    match size {
        1 => (0xe1d0_0f9f, 0xe1c0_0f90),
        2 => (0xe1f0_0f9f, 0xe1e0_0f90),
        4 => (0xe190_0f9f, 0xe180_0f90),
        _ => unreachable!(),
    }
}

/// Emit a load-exclusive/store-exclusive read-modify-write loop.
///
/// ```plain
///   dmb ish
/// loop:
///   ldrex{b,h} rd, [addr]
///   <op> tmp1, rd, operand
///   strex{b,h} tmp2, tmp1, [addr]
///   cmp tmp2, #0
///   bne loop
///   dmb ish
/// ```
fn emit_atomic_rmw_loop(
    sink: &mut MachBuffer<Inst>,
    op: AtomicRmwOp,
    size: u8,
    rd: u32,
    addr: u32,
    operand: u32,
    tmp1: u32,
    tmp2: u32,
    trap_code: Option<TrapCode>,
    acq_rel: bool,
) {
    let (ldrex, strex) = atomic_size_ldrex_strex(size);
    let (ldrex, strex) = (exclusive(ldrex, acq_rel), exclusive(strex, acq_rel));
    let none = ShiftOpAndAmt::none();
    barrier_unless(sink, acq_rel);
    let loop_start = sink.cur_offset();
    add_trap_opt(sink, trap_code);
    put(sink, ldrex | (addr << 16) | (rd << 12));
    // For narrow signed min/max, compare sign-extended values.
    let (cmp_lhs, cmp_rhs) = match (op, size) {
        (AtomicRmwOp::Smin | AtomicRmwOp::Smax, 1 | 2) => {
            let ext = if size == 1 { 0xe6af_0070 } else { 0xe6bf_0070 };
            // sxt{b,h} tmp1, rd ; sxt{b,h} ip, operand
            put(sink, ext | (tmp1 << 12) | rd);
            put(sink, ext | (IP << 12) | operand);
            (tmp1, IP)
        }
        (AtomicRmwOp::Umin | AtomicRmwOp::Umax, 1 | 2) => {
            let ext = if size == 1 { 0xe6ef_0070 } else { 0xe6ff_0070 };
            put(sink, ext | (IP << 12) | operand);
            (rd, IP)
        }
        _ => (rd, operand),
    };
    match op {
        AtomicRmwOp::Add => put(
            sink,
            enc_dp_reg(
                COND_AL,
                alu_opcode(AluOp::Add),
                false,
                rd,
                tmp1,
                operand,
                none,
            ),
        ),
        AtomicRmwOp::Sub => put(
            sink,
            enc_dp_reg(
                COND_AL,
                alu_opcode(AluOp::Sub),
                false,
                rd,
                tmp1,
                operand,
                none,
            ),
        ),
        AtomicRmwOp::And => put(
            sink,
            enc_dp_reg(
                COND_AL,
                alu_opcode(AluOp::And),
                false,
                rd,
                tmp1,
                operand,
                none,
            ),
        ),
        AtomicRmwOp::Or => put(
            sink,
            enc_dp_reg(
                COND_AL,
                alu_opcode(AluOp::Orr),
                false,
                rd,
                tmp1,
                operand,
                none,
            ),
        ),
        AtomicRmwOp::Xor => put(
            sink,
            enc_dp_reg(
                COND_AL,
                alu_opcode(AluOp::Eor),
                false,
                rd,
                tmp1,
                operand,
                none,
            ),
        ),
        AtomicRmwOp::Nand => {
            put(
                sink,
                enc_dp_reg(
                    COND_AL,
                    alu_opcode(AluOp::And),
                    false,
                    rd,
                    tmp1,
                    operand,
                    none,
                ),
            );
            put(
                sink,
                enc_dp_reg(COND_AL, OPC_MVN, false, 0, tmp1, tmp1, none),
            );
        }
        AtomicRmwOp::Xchg => put(sink, enc_mov_reg(COND_AL, tmp1, operand)),
        AtomicRmwOp::Umin | AtomicRmwOp::Umax | AtomicRmwOp::Smin | AtomicRmwOp::Smax => {
            // tmp1 = cond(lhs, rhs) ? rd : operand
            let keep_old = match op {
                AtomicRmwOp::Umin => Cond::Lo,
                AtomicRmwOp::Umax => Cond::Hi,
                AtomicRmwOp::Smin => Cond::Lt,
                _ => Cond::Gt,
            };
            put(
                sink,
                enc_dp_reg(
                    COND_AL,
                    cmp_opcode(CmpOp::Cmp),
                    true,
                    cmp_lhs,
                    0,
                    cmp_rhs,
                    none,
                ),
            );
            put(sink, enc_mov_reg(COND_AL, tmp1, operand));
            put(sink, enc_mov_reg(keep_old.bits(), tmp1, rd));
        }
    }
    add_trap_opt(sink, trap_code);
    put(sink, strex | (addr << 16) | (tmp2 << 12) | tmp1);
    put(
        sink,
        enc_dp_imm(
            COND_AL,
            cmp_opcode(CmpOp::Cmp),
            true,
            tmp2,
            0,
            ModImm::maybe_from_u32(0).unwrap(),
        ),
    );
    emit_branch_to_offset(sink, Cond::Ne.bits(), loop_start);
    barrier_unless(sink, acq_rel);
}

/// The load-acquire/store-release form of a load/store-exclusive encoding
/// (ARMv8), if `acq_rel`.
fn exclusive(enc: u32, acq_rel: bool) -> u32 {
    if acq_rel { enc & !0x100 } else { enc }
}

/// `dmb ish`, unless the sequence uses acquire/release exclusives.
fn barrier_unless(sink: &mut MachBuffer<Inst>, acq_rel: bool) {
    if !acq_rel {
        put(sink, 0xf57f_f05b);
    }
}

/// Record `trap_code`, if any, for the instruction about to be emitted (the
/// exclusive accesses of atomic sequences may fault on out-of-bounds
/// addresses).
fn add_trap_opt(sink: &mut MachBuffer<Inst>, trap_code: Option<TrapCode>) {
    if let Some(trap_code) = trap_code {
        sink.add_trap(trap_code);
    }
}

/// Emit a compare-and-swap loop.
///
/// ```plain
///   dmb ish
/// loop:
///   ldrex{b,h} rd, [addr]
///   uxt{b,h} ip, expected       (narrow sizes)
///   cmp rd, ip
///   bne done
///   strex{b,h} tmp, replacement, [addr]
///   cmp tmp, #0
///   bne loop
/// done:
///   dmb ish
/// ```
fn emit_atomic_cas_loop(
    sink: &mut MachBuffer<Inst>,
    size: u8,
    rd: u32,
    addr: u32,
    expected: u32,
    replacement: u32,
    tmp: u32,
    trap_code: Option<TrapCode>,
    acq_rel: bool,
) {
    let (ldrex, strex) = atomic_size_ldrex_strex(size);
    let (ldrex, strex) = (exclusive(ldrex, acq_rel), exclusive(strex, acq_rel));
    let none = ShiftOpAndAmt::none();
    barrier_unless(sink, acq_rel);
    let loop_start = sink.cur_offset();
    add_trap_opt(sink, trap_code);
    put(sink, ldrex | (addr << 16) | (rd << 12));
    let exp = match size {
        1 => {
            put(sink, 0xe6ef_0070 | (IP << 12) | expected);
            IP
        }
        2 => {
            put(sink, 0xe6ff_0070 | (IP << 12) | expected);
            IP
        }
        _ => expected,
    };
    put(
        sink,
        enc_dp_reg(COND_AL, cmp_opcode(CmpOp::Cmp), true, rd, 0, exp, none),
    );
    // bne done (skip strex, cmp, bne)
    emit_branch_skip(sink, Cond::Ne.bits(), 3);
    add_trap_opt(sink, trap_code);
    put(sink, strex | (addr << 16) | (tmp << 12) | replacement);
    put(
        sink,
        enc_dp_imm(
            COND_AL,
            cmp_opcode(CmpOp::Cmp),
            true,
            tmp,
            0,
            ModImm::maybe_from_u32(0).unwrap(),
        ),
    );
    emit_branch_to_offset(sink, Cond::Ne.bits(), loop_start);
    barrier_unless(sink, acq_rel);
}

/// 64-bit read-modify-write loop with fixed registers:
/// old value in r0:r1, operand in r2:r3, new value in r4:r5, status in r6,
/// address in r7.
fn emit_atomic_rmw_loop64(
    sink: &mut MachBuffer<Inst>,
    op: AtomicRmwOp,
    addr: u32,
    trap_code: Option<TrapCode>,
    acq_rel: bool,
) {
    debug_assert_eq!(addr, 7);
    let none = ShiftOpAndAmt::none();
    let (r0, r1, r2, r3, r4, r5, r6) = (0, 1, 2, 3, 4, 5, 6);
    barrier_unless(sink, acq_rel);
    let loop_start = sink.cur_offset();
    // ldrexd r0, r1, [r7]
    add_trap_opt(sink, trap_code);
    put(
        sink,
        exclusive(0xe1b0_0f9f, acq_rel) | (addr << 16) | (r0 << 12),
    );
    let dp = |sink: &mut MachBuffer<Inst>, op: AluOp, s: bool, rd: u32, rn: u32, rm: u32| {
        put(
            sink,
            enc_dp_reg(COND_AL, alu_opcode(op), s, rn, rd, rm, none),
        );
    };
    match op {
        AtomicRmwOp::Add => {
            dp(sink, AluOp::Add, true, r4, r0, r2);
            dp(sink, AluOp::Adc, false, r5, r1, r3);
        }
        AtomicRmwOp::Sub => {
            dp(sink, AluOp::Sub, true, r4, r0, r2);
            dp(sink, AluOp::Sbc, false, r5, r1, r3);
        }
        AtomicRmwOp::And => {
            dp(sink, AluOp::And, false, r4, r0, r2);
            dp(sink, AluOp::And, false, r5, r1, r3);
        }
        AtomicRmwOp::Or => {
            dp(sink, AluOp::Orr, false, r4, r0, r2);
            dp(sink, AluOp::Orr, false, r5, r1, r3);
        }
        AtomicRmwOp::Xor => {
            dp(sink, AluOp::Eor, false, r4, r0, r2);
            dp(sink, AluOp::Eor, false, r5, r1, r3);
        }
        AtomicRmwOp::Nand => {
            dp(sink, AluOp::And, false, r4, r0, r2);
            dp(sink, AluOp::And, false, r5, r1, r3);
            put(sink, enc_dp_reg(COND_AL, OPC_MVN, false, 0, r4, r4, none));
            put(sink, enc_dp_reg(COND_AL, OPC_MVN, false, 0, r5, r5, none));
        }
        AtomicRmwOp::Xchg => {
            put(sink, enc_mov_reg(COND_AL, r4, r2));
            put(sink, enc_mov_reg(COND_AL, r5, r3));
        }
        AtomicRmwOp::Umin | AtomicRmwOp::Umax | AtomicRmwOp::Smin | AtomicRmwOp::Smax => {
            // Compute old - operand to set flags for the 64-bit comparison.
            // subs r4, r0, r2 ; sbcs r4, r1, r3  (flags: old vs operand)
            dp(sink, AluOp::Sub, true, r4, r0, r2);
            dp(sink, AluOp::Sbc, true, r4, r1, r3);
            // Conditions on (old - operand): keep old when it wins.
            let keep_old = match op {
                AtomicRmwOp::Umin => Cond::Lo,
                AtomicRmwOp::Smin => Cond::Lt,
                AtomicRmwOp::Umax => Cond::Hs,
                _ => Cond::Ge,
            };
            put(sink, enc_mov_reg(COND_AL, r4, r2));
            put(sink, enc_mov_reg(COND_AL, r5, r3));
            put(sink, enc_mov_reg(keep_old.bits(), r4, r0));
            put(sink, enc_mov_reg(keep_old.bits(), r5, r1));
        }
    }
    // strexd r6, r4, r5, [r7]
    add_trap_opt(sink, trap_code);
    put(
        sink,
        exclusive(0xe1a0_0f90, acq_rel) | (addr << 16) | (r6 << 12) | r4,
    );
    put(
        sink,
        enc_dp_imm(
            COND_AL,
            cmp_opcode(CmpOp::Cmp),
            true,
            r6,
            0,
            ModImm::maybe_from_u32(0).unwrap(),
        ),
    );
    emit_branch_to_offset(sink, Cond::Ne.bits(), loop_start);
    barrier_unless(sink, acq_rel);
}

/// 64-bit compare-and-swap loop with fixed registers:
/// old value in r0:r1, expected in r2:r3, replacement in r4:r5, status in
/// r6, address in r7.
fn emit_atomic_cas_loop64(
    sink: &mut MachBuffer<Inst>,
    addr: u32,
    trap_code: Option<TrapCode>,
    acq_rel: bool,
) {
    debug_assert_eq!(addr, 7);
    let none = ShiftOpAndAmt::none();
    barrier_unless(sink, acq_rel);
    let loop_start = sink.cur_offset();
    // ldrexd r0, r1, [r7]
    add_trap_opt(sink, trap_code);
    put(sink, exclusive(0xe1b0_0f9f, acq_rel) | (addr << 16));
    // cmp r1, r3 ; cmpeq r0, r2 ; bne done
    put(
        sink,
        enc_dp_reg(COND_AL, cmp_opcode(CmpOp::Cmp), true, 1, 0, 3, none),
    );
    put(
        sink,
        enc_dp_reg(Cond::Eq.bits(), cmp_opcode(CmpOp::Cmp), true, 0, 0, 2, none),
    );
    emit_branch_skip(sink, Cond::Ne.bits(), 3);
    // strexd r6, r4, r5, [r7]
    add_trap_opt(sink, trap_code);
    put(
        sink,
        exclusive(0xe1a0_0f90, acq_rel) | (addr << 16) | (6 << 12) | 4,
    );
    put(
        sink,
        enc_dp_imm(
            COND_AL,
            cmp_opcode(CmpOp::Cmp),
            true,
            6,
            0,
            ModImm::maybe_from_u32(0).unwrap(),
        ),
    );
    emit_branch_to_offset(sink, Cond::Ne.bits(), loop_start);
    barrier_unless(sink, acq_rel);
}
