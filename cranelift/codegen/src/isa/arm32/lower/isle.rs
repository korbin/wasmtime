//! ISLE integration glue code for ARMv7 lowering.

// Pull in the ISLE generated code.
pub mod generated_code;
use generated_code::{FloatCond, MInst};

// Types that the generated ISLE code uses via `use super::*`.
use crate::ir::condcodes::{FloatCC, IntCC};
use crate::isa::arm32::Arm32Backend;
use crate::isa::arm32::inst::*;
use crate::machinst::{
    ArgPair, CallArgList, CallInfo, CallRetList, InstOutput, MachInst, Reg, isle::*,
};
use crate::{
    ir::{
        AtomicRmwOp, BlockCall, ExternalName, Inst, InstructionData, LibCall, MemFlagsData, Opcode,
        TrapCode, Value, ValueList, immediates::*, types::*,
    },
    machinst::{VCodeConstant, VCodeConstantData, ValueRegs as MachValueRegs},
};
use alloc::boxed::Box;
use alloc::vec::Vec;
use regalloc2::PReg;
use wasmtime_core::math::{f32_cvt_to_int_bounds, f64_cvt_to_int_bounds};

type BoxCallInfo = Box<CallInfo<ExternalName>>;
type BoxCallIndInfo = Box<CallInfo<Reg>>;
type BoxReturnCallInfo = Box<ReturnCallInfo<ExternalName>>;
type BoxReturnCallIndInfo = Box<ReturnCallInfo<Reg>>;
type BoxExternalName = Box<ExternalName>;
type BoxedSliceArgPair = Box<[ArgPair]>;
type OptionReg = Option<Reg>;

pub(crate) struct Arm32IsleContext<'a, 'b, I, B>
where
    I: VCodeInst,
    B: LowerBackend,
{
    pub lower_ctx: &'a mut Lower<'b, I>,
    pub backend: &'a B,
}

impl<'a, 'b> Arm32IsleContext<'a, 'b, MInst, Arm32Backend> {
    fn new(lower_ctx: &'a mut Lower<'b, MInst>, backend: &'a Arm32Backend) -> Self {
        Self { lower_ctx, backend }
    }

    pub(crate) fn dfg(&self) -> &crate::ir::DataFlowGraph {
        &self.lower_ctx.f.dfg
    }

    fn iconst_value(&self, value: Value) -> Option<u64> {
        let inst = self.lower_ctx.dfg().value_def(value).inst()?;
        match self.lower_ctx.data(inst) {
            InstructionData::UnaryImm {
                opcode: Opcode::Iconst,
                imm,
            } => {
                let ty = self.lower_ctx.dfg().value_type(value);
                let bits = ty.bits();
                let v = imm.bits() as u64;
                Some(if bits >= 64 {
                    v
                } else {
                    v & ((1u64 << bits) - 1)
                })
            }
            _ => None,
        }
    }

    /// The value of a 32-bit-or-smaller constant, sign-extended from its
    /// type's width to 32 bits.
    fn iconst_u32(&self, value: Value) -> Option<u32> {
        let ty = self.lower_ctx.dfg().value_type(value);
        if ty.bits() > 32 {
            return None;
        }
        let v = self.iconst_value(value)?;
        let shift = 64 - ty.bits();
        Some((((v << shift) as i64) >> shift) as u32)
    }

    /// A constant operand of an 8- or 16-bit comparison with condition `cc`,
    /// extended to 32 bits like the other operand (see `icmp_operand`).
    fn icmp_narrow_const(&self, cc: &IntCC, ty: Type, value: Value) -> Option<u32> {
        debug_assert!(ty.bits() <= 16);
        // Sign-extended to 32 bits.
        let v = self.iconst_u32(value)?;
        Some(if intcc_is_signed(*cc) {
            v
        } else {
            v & ((1 << ty.bits()) - 1)
        })
    }
}

impl generated_code::Context for Arm32IsleContext<'_, '_, MInst, Arm32Backend> {
    isle_lower_prelude_methods!();

    fn emit(&mut self, arg0: &MInst) -> Unit {
        self.lower_ctx.emit(arg0.clone());
    }

    fn gen_call_info(
        &mut self,
        sig: Sig,
        dest: ExternalName,
        uses: CallArgList,
        defs: CallRetList,
        try_call_info: Option<TryCallInfo>,
        patchable: bool,
    ) -> BoxCallInfo {
        let stack_ret_space = self.lower_ctx.sigs()[sig].sized_stack_ret_space();
        let stack_arg_space = self.lower_ctx.sigs()[sig].sized_stack_arg_space();
        self.lower_ctx
            .abi_mut()
            .accumulate_outgoing_args_size(stack_ret_space + stack_arg_space);

        Box::new(
            self.lower_ctx
                .gen_call_info(sig, dest, uses, defs, try_call_info, patchable),
        )
    }

    fn gen_call_ind_info(
        &mut self,
        sig: Sig,
        dest: Reg,
        uses: CallArgList,
        defs: CallRetList,
        try_call_info: Option<TryCallInfo>,
    ) -> BoxCallIndInfo {
        let stack_ret_space = self.lower_ctx.sigs()[sig].sized_stack_ret_space();
        let stack_arg_space = self.lower_ctx.sigs()[sig].sized_stack_arg_space();
        self.lower_ctx
            .abi_mut()
            .accumulate_outgoing_args_size(stack_ret_space + stack_arg_space);

        Box::new(
            self.lower_ctx
                .gen_call_info(sig, dest, uses, defs, try_call_info, false),
        )
    }

    fn gen_return_call_info(
        &mut self,
        sig: Sig,
        dest: ExternalName,
        uses: CallArgList,
    ) -> BoxReturnCallInfo {
        let new_stack_arg_size = self.lower_ctx.sigs()[sig].sized_stack_arg_space();
        self.lower_ctx
            .abi_mut()
            .accumulate_tail_args_size(new_stack_arg_size);

        Box::new(ReturnCallInfo {
            dest,
            uses,
            new_stack_arg_size,
        })
    }

    fn gen_return_call_ind_info(
        &mut self,
        sig: Sig,
        dest: Reg,
        uses: CallArgList,
    ) -> BoxReturnCallIndInfo {
        let new_stack_arg_size = self.lower_ctx.sigs()[sig].sized_stack_arg_space();
        self.lower_ctx
            .abi_mut()
            .accumulate_tail_args_size(new_stack_arg_size);

        Box::new(ReturnCallInfo {
            dest,
            uses,
            new_stack_arg_size,
        })
    }

    fn mod_imm_from_u32(&mut self, value: u32) -> Option<ModImm> {
        ModImm::maybe_from_u32(value)
    }

    fn mod_imm(&mut self, value: u32) -> ModImm {
        ModImm::maybe_from_u32(value)
            .unwrap_or_else(|| panic!("{value:#x} is not a modified immediate"))
    }

    fn mod_imm_from_value(&mut self, value: Value) -> Option<ModImm> {
        ModImm::maybe_from_u32(self.iconst_u32(value)?)
    }

    fn shuffle_ext(&mut self, mask: u128) -> Option<u8> {
        let b = mask.to_le_bytes();
        let n = b[0];
        ((1..16).contains(&n) && (0..16).all(|i| usize::from(b[i]) == usize::from(n) + i))
            .then_some(n)
    }

    fn shuffle_rotate(&mut self, mask: u128) -> Option<u8> {
        let b = mask.to_le_bytes();
        let n = b[0] & 15;
        (n != 0 && (0..16).all(|i| usize::from(b[i] & 15) == (usize::from(n) + i) % 16))
            .then_some(n)
    }

    fn shuffle_rev(&mut self, mask: u128) -> Option<(VecMiscOp, u8)> {
        let b = mask.to_le_bytes();
        for (group, op) in [
            (2, VecMiscOp::Rev16),
            (4, VecMiscOp::Rev32),
            (8, VecMiscOp::Rev64),
        ] {
            for size in 0..3u8 {
                let elem = 1usize << size;
                if elem >= group {
                    continue;
                }
                let reversed = (0..16).all(|i| {
                    let start = i / group * group;
                    let index = (i % group) / elem;
                    let byte = i % elem;
                    usize::from(b[i] & 15) == start + (group / elem - 1 - index) * elem + byte
                });
                if reversed {
                    return Some((op, size));
                }
            }
        }
        None
    }

    fn shuffle_dup_lane(&mut self, mask: u128) -> Option<(u8, u8)> {
        let b = mask.to_le_bytes();
        let first = usize::from(b[0] & 15);
        (0..3u8).rev().find_map(|size| {
            let elem = 1usize << size;
            (first % elem == 0 && (0..16).all(|i| usize::from(b[i] & 15) == first + i % elem))
                .then(|| (size, u8::try_from(first / elem).unwrap()))
        })
    }

    fn shuffle_first_only(&mut self, mask: u128) -> Option<u128> {
        mask.to_le_bytes().iter().all(|i| *i < 16).then_some(mask)
    }

    fn shuffle_second_only(&mut self, mask: u128) -> Option<u128> {
        let b = mask.to_le_bytes();
        (b.iter().all(|i| (16..32).contains(i)))
            .then_some(mask & 0x0f0f_0f0f_0f0f_0f0f_0f0f_0f0f_0f0f_0f0f)
    }

    fn vec_cmp_f32_supported(&mut self, cc: &FloatCC) -> bool {
        matches!(
            cc,
            FloatCC::Equal
                | FloatCC::NotEqual
                | FloatCC::LessThan
                | FloatCC::LessThanOrEqual
                | FloatCC::GreaterThan
                | FloatCC::GreaterThanOrEqual
        )
    }

    fn vec_size_below_64(&mut self, ty: Type) -> Option<u8> {
        match ty.lane_bits() {
            8 => Some(0),
            16 => Some(1),
            32 => Some(2),
            _ => None,
        }
    }

    fn icmp_narrow_imm(&mut self, cc: &IntCC, ty: Type, value: Value) -> Option<ModImm> {
        ModImm::maybe_from_u32(self.icmp_narrow_const(cc, ty, value)?)
    }

    fn icmp_narrow_neg_imm(&mut self, cc: &IntCC, ty: Type, value: Value) -> Option<ModImm> {
        // The extended constant is never `i32::MIN`, for which `cmn` and
        // `cmp` would set V differently (see `mod_imm_from_neg_value`).
        let v = self.icmp_narrow_const(cc, ty, value)?;
        if v == 0 {
            return None;
        }
        ModImm::maybe_from_u32(v.wrapping_neg())
    }

    fn mod_imm_from_neg_value(&mut self, value: Value) -> Option<ModImm> {
        let v = self.iconst_u32(value)?;
        // Zero and `i32::MIN` are their own negations; excluding the latter
        // also keeps `cmn x, #c` equivalent to `cmp x, #-c` in all flags.
        if v == 0 || v == 0x8000_0000 {
            return None;
        }
        ModImm::maybe_from_u32(v.wrapping_neg())
    }

    fn i64_cmp_imm_lo(&mut self, cc: &IntCC, c: u64) -> Option<ModImm> {
        ModImm::maybe_from_u32(i64_cmp_imm_adjusted(*cc, c)? as u32)
    }

    fn i64_cmp_imm_hi(&mut self, cc: &IntCC, c: u64) -> Option<ModImm> {
        ModImm::maybe_from_u32((i64_cmp_imm_adjusted(*cc, c)? >> 32) as u32)
    }

    fn i64_cmp_imm_cond(&mut self, cc: &IntCC) -> Cond {
        match cc {
            IntCC::SignedLessThan | IntCC::SignedLessThanOrEqual => Cond::Lt,
            IntCC::SignedGreaterThanOrEqual | IntCC::SignedGreaterThan => Cond::Ge,
            IntCC::UnsignedLessThan | IntCC::UnsignedLessThanOrEqual => Cond::Lo,
            IntCC::UnsignedGreaterThanOrEqual | IntCC::UnsignedGreaterThan => Cond::Hs,
            IntCC::Equal | IntCC::NotEqual => unreachable!(),
        }
    }

    fn i64_const_halves(&mut self, value: Value) -> Option<(u32, u32)> {
        let v = self.iconst_value(value)?;
        Some((v as u32, (v >> 32) as u32))
    }

    fn u32_inv(&mut self, v: u32) -> u32 {
        !v
    }

    fn i64_imm_halves(&mut self, value: Value) -> Option<(ModImm, ModImm)> {
        let v = self.iconst_value(value)?;
        Some((
            ModImm::maybe_from_u32(v as u32)?,
            ModImm::maybe_from_u32((v >> 32) as u32)?,
        ))
    }

    fn i64_neg_imm_halves(&mut self, value: Value) -> Option<(ModImm, ModImm)> {
        let v = self.iconst_value(value)?.wrapping_neg();
        Some((
            ModImm::maybe_from_u32(v as u32)?,
            ModImm::maybe_from_u32((v >> 32) as u32)?,
        ))
    }

    fn vfp_imm8_f32(&mut self, bits: u32) -> Option<u8> {
        vfp_imm8(u64::from(bits), false)
    }

    fn vfp_imm8_f64(&mut self, bits: u64) -> Option<u8> {
        vfp_imm8(bits, true)
    }

    fn u64_same_halves(&mut self, v: u64) -> Option<u32> {
        (v as u32 == (v >> 32) as u32).then_some(v as u32)
    }

    fn small_shift64(&mut self, amt: Value) -> Option<u8> {
        let k = self.iconst_value(amt)? & 63;
        (1..32).contains(&k).then_some(k as u8)
    }

    fn large_shift64(&mut self, amt: Value) -> Option<u8> {
        let k = self.iconst_value(amt)? & 63;
        (k >= 32).then(|| (k - 32) as u8)
    }

    fn mod_imm_from_inv_value(&mut self, value: Value) -> Option<ModImm> {
        ModImm::maybe_from_u32(!self.iconst_u32(value)?)
    }

    fn u64_low32(&mut self, value: u64) -> u32 {
        value as u32
    }

    fn u64_high32(&mut self, value: u64) -> u32 {
        (value >> 32) as u32
    }

    fn u32_low16(&mut self, value: u32) -> u16 {
        value as u16
    }

    fn u32_high16(&mut self, value: u32) -> u16 {
        (value >> 16) as u16
    }

    fn shift_none(&mut self) -> ShiftOpAndAmt {
        ShiftOpAndAmt::none()
    }

    fn shift_imm(&mut self, op: &ShiftOp, amt: u8) -> ShiftOpAndAmt {
        ShiftOpAndAmt::new(*op, amt)
    }

    fn shift_amount_masked(&mut self, ty: Type, amt: Value) -> Option<u8> {
        let v = self.iconst_value(amt)?;
        let masked = (v & u64::from(ty.bits() - 1)) as u8;
        if masked == 0 { None } else { Some(masked) }
    }

    fn ty_shift_mask_u32(&mut self, ty: Type) -> u32 {
        ty.bits() - 1
    }

    fn ty_bytes_u8(&mut self, ty: Type) -> u8 {
        u8::try_from(ty.bytes()).unwrap()
    }

    fn ty_bits_u32(&mut self, ty: Type) -> u32 {
        ty.bits()
    }

    fn ty_smin_u32(&mut self, ty: Type) -> u32 {
        match ty {
            I8 => i8::MIN as i32 as u32,
            I16 => i16::MIN as i32 as u32,
            I32 => i32::MIN as u32,
            _ => unreachable!(),
        }
    }

    fn ty_smax_u32(&mut self, ty: Type) -> u32 {
        match ty {
            I8 => i8::MAX as u32,
            I16 => i16::MAX as u32,
            I32 => i32::MAX as u32,
            _ => unreachable!(),
        }
    }

    fn ty_umax_u32(&mut self, ty: Type) -> u32 {
        match ty {
            I8 => u8::MAX.into(),
            I16 => u16::MAX.into(),
            I32 => u32::MAX,
            _ => unreachable!(),
        }
    }

    fn has_idiv(&mut self) -> bool {
        self.backend.isa_flags.has_idiv()
    }

    fn has_vfp4(&mut self) -> bool {
        self.backend.isa_flags.has_vfp4()
    }

    fn has_neon(&mut self) -> bool {
        self.backend.isa_flags.has_neon()
    }

    fn has_armv8(&mut self) -> bool {
        self.backend.isa_flags.has_armv8()
    }

    fn vec_size(&mut self, ty: Type) -> u8 {
        match ty.lane_bits() {
            8 => 0,
            16 => 1,
            32 => 2,
            64 => 3,
            _ => unreachable!("unsupported lane type in {ty}"),
        }
    }

    fn vec_size_le_32(&mut self, ty: Type) -> bool {
        ty.lane_bits() <= 32
    }

    fn u32_to_u64_bits(&mut self, value: u32) -> u64 {
        u64::from(value)
    }

    fn u128_half(&mut self, value: u128, idx: u8) -> u64 {
        (value >> (64 * u32::from(idx))) as u64
    }

    fn u128_word(&mut self, value: u128, idx: u8) -> u32 {
        (value >> (32 * u32::from(idx))) as u32
    }

    fn u128_and_15(&mut self, value: u128) -> u128 {
        value & 0x0f0f_0f0f_0f0f_0f0f_0f0f_0f0f_0f0f_0f0f
    }

    fn libcall_3(&mut self, libcall: &LibCall, a: Reg, b: Reg, c: Reg) -> Reg {
        let outputs = crate::isa::arm32::lower::emit_libcall(
            self.lower_ctx,
            &self.backend.flags,
            &self.backend.triple,
            *libcall,
            &[
                MachValueRegs::one(a),
                MachValueRegs::one(b),
                MachValueRegs::one(c),
            ],
        )
        .expect("failed to emit libcall");
        debug_assert_eq!(outputs.len(), 1);
        outputs[0].only_reg().unwrap()
    }

    fn option_reg_none(&mut self) -> OptionReg {
        None
    }

    fn option_reg_some(&mut self, r: Reg) -> OptionReg {
        Some(r)
    }

    fn cond_from_intcc(&mut self, cc: &IntCC) -> Cond {
        Cond::from_intcc(*cc)
    }

    fn cond_invert(&mut self, cond: &Cond) -> Cond {
        cond.invert()
    }

    fn intcc_signed(&mut self, cc: &IntCC) -> bool {
        intcc_is_signed(*cc)
    }

    fn float_cond(&mut self, cc: &FloatCC) -> FloatCond {
        match Cond::from_floatcc(*cc) {
            (c, None) => FloatCond::One { c },
            (c1, Some(c2)) => FloatCond::Two { c1, c2 },
        }
    }

    fn fp_reg(&mut self) -> PReg {
        preg(11)
    }

    fn sp_reg(&mut self) -> PReg {
        preg(13)
    }

    fn amode_reg_reg(&mut self, base: Reg, index: Reg, shift: u8) -> AMode {
        AMode::RegReg { base, index, shift }
    }

    fn u64_le_3(&mut self, value: u64) -> Option<u8> {
        if (1..=3).contains(&value) {
            Some(value as u8)
        } else {
            None
        }
    }

    fn amode_reg_offset(&mut self, base: Reg, offset: i32) -> AMode {
        AMode::RegOffset { base, offset }
    }

    fn amode_add4(&mut self, amode: AMode) -> AMode {
        match amode {
            AMode::RegOffset { base, offset } => AMode::RegOffset {
                base,
                offset: offset.wrapping_add(4),
            },
            _ => unreachable!("amode_add4 on {amode:?}"),
        }
    }

    fn guard_fold_offset(&mut self, offset: i32, k: u64) -> Option<i32> {
        // `k` must be a 32-bit constant in `[1, 251]` so that `k - 1` is a
        // modified immediate, and the combined offset (plus 4, for split
        // accesses) must fit the 8-bit immediate of `ldrh` and friends.
        if !(1..=251).contains(&k) {
            return None;
        }
        let off = offset.checked_add(k as i32)?;
        (0..=251).contains(&off).then_some(off)
    }

    fn u64_minus1_mod_imm(&mut self, k: u64) -> ModImm {
        ModImm::maybe_from_u32((k - 1) as u32).unwrap()
    }

    fn i32_add4(&mut self, value: i32) -> i32 {
        value.wrapping_add(4)
    }

    fn i64_low32(&mut self, value: i64) -> u32 {
        value as u32
    }

    fn i64_low32_signed(&mut self, value: i64) -> i32 {
        value as i32
    }

    fn memflags_aligned(&mut self, flags: MemFlagsData) -> bool {
        flags.aligned()
    }

    fn lower_br_table(&mut self, index: Reg, targets: &[MachLabel]) -> Unit {
        self.emit(&MInst::BrTable {
            index,
            targets: Box::new(targets.to_vec()),
        });
    }

    fn gen_stack_addr(&mut self, slot: StackSlot, offset: Offset32) -> Reg {
        let result = self.temp_writable_reg(I32);
        let i = self
            .lower_ctx
            .abi()
            .sized_stackslot_addr(slot, i32::from(offset), result);
        self.emit(&i);
        result.to_reg()
    }

    fn value_regs_vec2(&mut self, a: ValueRegs, b: ValueRegs) -> ValueRegsVec {
        let mut v = ValueRegsVec::new();
        v.push(a);
        v.push(b);
        v
    }

    fn value_regs_vec_get(&mut self, regs: &ValueRegsVec, idx: usize) -> ValueRegs {
        regs[idx]
    }

    fn udivmod64(&mut self, n: ValueRegs, d: ValueRegs) -> ValueRegsVec {
        let quot_lo = self.temp_writable_reg(I32);
        let quot_hi = self.temp_writable_reg(I32);
        let rem_lo = self.temp_writable_reg(I32);
        let rem_hi = self.temp_writable_reg(I32);
        self.emit(&MInst::UDivMod64 {
            idiv: self.backend.isa_flags.has_idiv(),
            quot_lo,
            quot_hi,
            rem_lo,
            rem_hi,
            n_lo: n.regs()[0],
            n_hi: n.regs()[1],
            d_lo: d.regs()[0],
            d_hi: d.regs()[1],
        });
        let mut v = ValueRegsVec::new();
        v.push(MachValueRegs::two(quot_lo.to_reg(), quot_hi.to_reg()));
        v.push(MachValueRegs::two(rem_lo.to_reg(), rem_hi.to_reg()));
        v
    }

    fn cvt_lower_bound(&mut self, signed: bool, in_ty: Type, out_ty: Type) -> u64 {
        match in_ty {
            F32 => f32_cvt_to_int_bounds(signed, out_ty.bits())
                .0
                .to_bits()
                .into(),
            F64 => f64_cvt_to_int_bounds(signed, out_ty.bits()).0.to_bits(),
            _ => unreachable!(),
        }
    }

    fn cvt_upper_bound(&mut self, signed: bool, in_ty: Type, out_ty: Type) -> u64 {
        match in_ty {
            F32 => f32_cvt_to_int_bounds(signed, out_ty.bits())
                .1
                .to_bits()
                .into(),
            F64 => f64_cvt_to_int_bounds(signed, out_ty.bits()).1.to_bits(),
            _ => unreachable!(),
        }
    }
}

/// The main entry point for lowering with ISLE.
pub(crate) fn lower(
    lower_ctx: &mut Lower<MInst>,
    backend: &Arm32Backend,
    inst: Inst,
) -> Option<InstOutput> {
    // TODO: reuse the ISLE context across lowerings so we can reuse its
    // internal heap allocations.
    let mut isle_ctx = Arm32IsleContext::new(lower_ctx, backend);
    generated_code::constructor_lower(&mut isle_ctx, inst)
}

/// The main entry point for branch lowering with ISLE.
pub(crate) fn lower_branch(
    lower_ctx: &mut Lower<MInst>,
    backend: &Arm32Backend,
    branch: Inst,
    targets: &[MachLabel],
) -> Option<()> {
    // TODO: reuse the ISLE context across lowerings so we can reuse its
    // internal heap allocations.
    let mut isle_ctx = Arm32IsleContext::new(lower_ctx, backend);
    generated_code::constructor_lower_branch(&mut isle_ctx, branch, targets)
}

/// Whether an integer condition compares signed values.
fn intcc_is_signed(cc: IntCC) -> bool {
    matches!(
        cc,
        IntCC::SignedLessThan
            | IntCC::SignedLessThanOrEqual
            | IntCC::SignedGreaterThan
            | IntCC::SignedGreaterThanOrEqual
    )
}

/// The constant to compare against with `cmp lo ; sbcs hi` for an ordered
/// 64-bit comparison `x <cc> c`: `c` itself for `lt`/`ge`/`lo`/`hs`, and
/// `c + 1` for `gt`/`le`/`hi`/`ls` (which become `ge`/`lt`/`hs`/`lo`), unless
/// that overflows.
fn i64_cmp_imm_adjusted(cc: IntCC, c: u64) -> Option<u64> {
    match cc {
        IntCC::SignedLessThan
        | IntCC::SignedGreaterThanOrEqual
        | IntCC::UnsignedLessThan
        | IntCC::UnsignedGreaterThanOrEqual => Some(c),
        IntCC::SignedGreaterThan | IntCC::SignedLessThanOrEqual => {
            (c as i64 != i64::MAX).then(|| c.wrapping_add(1))
        }
        IntCC::UnsignedGreaterThan | IntCC::UnsignedLessThanOrEqual => c.checked_add(1),
        IntCC::Equal | IntCC::NotEqual => None,
    }
}
