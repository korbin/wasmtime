//! Implementation of the ARMv7 ABI (AAPCS with the VFP hard-float variant).
//!
//! Frame layout, from high to low addresses:
//!
//! ```plain
//!   (high address)
//!                              +---------------------------+
//!                              |          ...              |
//!                              | stack args                |
//!                              | (accessed via FP)         |
//!                              +---------------------------+
//!                              | saved LR                  |
//!            FP after prologue | saved FP                  |
//!                              +---------------------------+
//!                              | callee-saved core regs    |
//!                              | (padding to 8 bytes)      |
//!                              | callee-saved d regs       |
//!                              +---------------------------+
//!                              | stack slots / spill slots |
//!                              +---------------------------+
//!                              | outgoing call args        |
//!            SP after prologue +---------------------------+
//!   (low address)
//! ```
//!
//! Single-precision values live in the low half `s(2n)` of a double register
//! `d(n)`. AAPCS-VFP arguments in odd-numbered S registers are handled with
//! shadow registers; see `aapcs_sreg_to_dreg`.
//!
//! The `tail` calling convention (used for all Wasm functions) differs from
//! the AAPCS: it passes integer arguments and return values in r0-r5 (r4 and
//! r5 are caller-saved in it), assigns each float its own D register, and
//! has the callee pop its stack arguments.

use crate::CodegenResult;
use crate::ir::types::*;
use crate::ir::{self, ExternalName, LibCall, MemFlagsData, Signature, TrapCode};
use crate::isa;
use crate::isa::arm32::inst::*;
use crate::isa::arm32::settings::Flags as Arm32Flags;
use crate::isa::unwind::UnwindInst;
use crate::machinst::*;
use crate::settings;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::format;
use alloc::vec::Vec;
use regalloc2::{MachineEnv, PRegSet};
use smallvec::{SmallVec, smallvec};

/// Support for the ARMv7 ABI from the callee side (within a function body).
pub(crate) type Arm32Callee = Callee<Arm32MachineDeps>;

/// ARMv7-specific ABI behavior. This struct just serves as an implementation
/// point for the trait; it is never actually instantiated.
pub struct Arm32MachineDeps;

impl IsaFlags for Arm32Flags {}

/// Map the first S register of an AAPCS-VFP argument to the D register that
/// the register allocator sees for it.
///
/// Our float register class only contains D registers, with `f32` values in
/// the low half `s(2n)` of `d(n)`. An `f32` argument assigned to an odd S
/// register `s(2k+1)` (the high half of `d(k)`) is instead modeled as living
/// in the "shadow" register `d(8+k)`, which is never an argument register;
/// calls copy the shadow into `s(2k+1)` right before the branch, and the
/// incoming-arguments pseudo-instruction copies `s(2k+1)` into the shadow.
/// Since `s(2k+1)` is only ever used together with `s(2k)`, `d(k)` is always
/// an argument register too, so no live value is clobbered by these copies.
pub(crate) fn aapcs_sreg_to_dreg(s: u32) -> u32 {
    if s % 2 == 0 { s / 2 } else { 8 + s / 2 }
}

/// Whether a D register is the shadow of an odd S argument register; see
/// `aapcs_sreg_to_dreg`. Returns the odd S register number.
pub(crate) fn shadow_dreg_to_sreg(d: u32) -> Option<u32> {
    if (8..16).contains(&d) {
        Some(2 * (d - 8) + 1)
    } else {
        None
    }
}

/// Simulates the AAPCS-VFP allocation of S/D registers.
struct VfpAllocator {
    /// Bitmask of allocated S registers `s0`-`s15`.
    used: u16,
    /// Set once a float argument has been placed on the stack.
    exhausted: bool,
}

impl VfpAllocator {
    fn new() -> Self {
        VfpAllocator {
            used: 0,
            exhausted: false,
        }
    }

    /// Allocate the AAPCS register for a value, returning its first S
    /// register number.
    fn alloc(&mut self, is_f64: bool) -> Option<u32> {
        if self.exhausted {
            return None;
        }
        if is_f64 {
            for d in 0..8 {
                let mask = 0b11 << (2 * d);
                if self.used & mask == 0 {
                    self.used |= mask;
                    return Some(2 * d);
                }
            }
        } else {
            for s in 0..16 {
                if self.used & (1 << s) == 0 {
                    self.used |= 1 << s;
                    return Some(s);
                }
            }
        }
        self.exhausted = true;
        None
    }
}

impl ABIMachineSpec for Arm32MachineDeps {
    type I = Inst;
    type F = Arm32Flags;

    /// This is the limit for the size of argument and return-value areas on
    /// the stack. We place a reasonable limit here to avoid integer overflow
    /// issues with 32-bit arithmetic: for now, 128 MB.
    const STACK_ARG_RET_SIZE_LIMIT: u32 = 128 * 1024 * 1024;

    fn word_bits() -> u32 {
        32
    }

    fn stack_align(_call_conv: isa::CallConv) -> u32 {
        8
    }

    fn compute_arg_locs(
        call_conv: isa::CallConv,
        flags: &settings::Flags,
        params: &[ir::AbiParam],
        args_or_rets: ArgsOrRets,
        add_ret_area_ptr: bool,
        mut args: ArgsAccumulator,
    ) -> CodegenResult<(u32, Option<usize>)> {
        if call_conv == isa::CallConv::Winch {
            return Err(crate::CodegenError::Unsupported(
                "arm32 does not support the 'winch' calling convention".to_owned(),
            ));
        }

        // The `tail` calling convention also passes arguments (and returns
        // values) in r4 and r5, which are caller-saved in that convention.
        let max_int_regs: u8 = if call_conv == isa::CallConv::Tail {
            6
        } else {
            4
        };
        const MAX_FLOAT_REGS: u8 = 8;

        // Arguments of non-`tail` calling conventions follow AAPCS-VFP
        // exactly; see `aapcs_sreg_to_dreg` for how odd S registers are
        // handled. Return values (and `tail` arguments) use one D register
        // per float value.
        let aapcs_args = call_conv != isa::CallConv::Tail && args_or_rets == ArgsOrRets::Args;
        let mut next_int: u8 = 0;
        let mut next_float: u8 = 0;
        let mut next_vec: u8 = 0;
        const MAX_VEC_REGS: u8 = 8;
        let mut next_stack: u32 = 0;
        let mut vfp = VfpAllocator::new();

        let ret_area_ptr = if add_ret_area_ptr {
            debug_assert_eq!(args_or_rets, ArgsOrRets::Args);
            next_int = 1;
            Some(ABIArg::reg(
                rreg(0).to_real_reg().unwrap(),
                I32,
                ir::ArgumentExtension::None,
                ir::ArgumentPurpose::Normal,
            ))
        } else {
            None
        };

        let too_many_rets = || -> CodegenResult<()> {
            if args_or_rets == ArgsOrRets::Rets && !flags.enable_multi_ret_implicit_sret() {
                Err(crate::CodegenError::Unsupported(
                    "Too many return values to fit in registers. \
                     Use a StructReturn argument instead. (#9510)"
                        .to_owned(),
                ))
            } else {
                Ok(())
            }
        };

        for param in params {
            if let ir::ArgumentPurpose::StructArgument(_) = param.purpose {
                return Err(crate::CodegenError::Unsupported(
                    "StructArgument parameters are not supported on arm32".to_owned(),
                ));
            }

            let (rcs, reg_tys) = Inst::rc_for_type(&param.value_type)?;
            match (rcs, reg_tys) {
                // A 64-bit integer in a pair of registers.
                ([RegClass::Int, RegClass::Int], [lo_ty, hi_ty]) => {
                    // AAPCS: double-word aligned arguments start at an even
                    // register.
                    next_int = (next_int + 1) & !1;
                    if next_int + 2 <= max_int_regs {
                        args.push(ABIArg::Slots {
                            slots: smallvec![
                                ABIArgSlot::Reg {
                                    reg: rreg(next_int).to_real_reg().unwrap(),
                                    ty: *lo_ty,
                                    extension: param.extension,
                                },
                                ABIArgSlot::Reg {
                                    reg: rreg(next_int + 1).to_real_reg().unwrap(),
                                    ty: *hi_ty,
                                    extension: param.extension,
                                },
                            ],
                            purpose: param.purpose,
                        });
                        next_int += 2;
                    } else {
                        too_many_rets()?;
                        next_int = max_int_regs;
                        next_stack = align_to(next_stack, 8);
                        args.push(ABIArg::Slots {
                            slots: smallvec![
                                ABIArgSlot::Stack {
                                    offset: i64::from(next_stack),
                                    ty: *lo_ty,
                                    extension: param.extension,
                                },
                                ABIArgSlot::Stack {
                                    offset: i64::from(next_stack + 4),
                                    ty: *hi_ty,
                                    extension: param.extension,
                                },
                            ],
                            purpose: param.purpose,
                        });
                        next_stack += 8;
                    }
                }
                ([RegClass::Int], [ty]) => {
                    if next_int < max_int_regs {
                        args.push(ABIArg::reg(
                            rreg(next_int).to_real_reg().unwrap(),
                            *ty,
                            param.extension,
                            param.purpose,
                        ));
                        next_int += 1;
                    } else {
                        too_many_rets()?;
                        next_stack = align_to(next_stack, 4);
                        args.push(ABIArg::stack(
                            i64::from(next_stack),
                            *ty,
                            param.extension,
                            param.purpose,
                        ));
                        next_stack += 4;
                    }
                }
                ([RegClass::Float], [ty]) => {
                    let is_f64 = *ty == F64;
                    let reg = if aapcs_args {
                        // AAPCS-VFP: f32 values use the lowest free S
                        // register (with back-filling), and f64 values the
                        // lowest free D register.
                        vfp.alloc(is_f64).map(|s| aapcs_sreg_to_dreg(s) as u8)
                    } else if next_float < MAX_FLOAT_REGS {
                        next_float += 1;
                        Some(next_float - 1)
                    } else {
                        None
                    };
                    if let Some(reg) = reg {
                        args.push(ABIArg::reg(
                            dreg(reg).to_real_reg().unwrap(),
                            *ty,
                            param.extension,
                            param.purpose,
                        ));
                    } else {
                        too_many_rets()?;
                        let size = if is_f64 { 8 } else { 4 };
                        next_stack = align_to(next_stack, size);
                        args.push(ABIArg::stack(
                            i64::from(next_stack),
                            *ty,
                            param.extension,
                            param.purpose,
                        ));
                        next_stack += size;
                    }
                }
                ([RegClass::Vector], [ty]) => {
                    // 128-bit vectors are passed in q8-q15 (which don't alias
                    // the scalar float registers) and then on the stack.
                    if next_vec < MAX_VEC_REGS {
                        args.push(ABIArg::reg(
                            qreg(8 + next_vec).to_real_reg().unwrap(),
                            *ty,
                            param.extension,
                            param.purpose,
                        ));
                        next_vec += 1;
                    } else {
                        too_many_rets()?;
                        next_stack = align_to(next_stack, 8);
                        args.push(ABIArg::stack(
                            i64::from(next_stack),
                            *ty,
                            param.extension,
                            param.purpose,
                        ));
                        next_stack += 16;
                    }
                }
                _ => {
                    return Err(crate::CodegenError::Unsupported(format!(
                        "arm32: unsupported argument type {}",
                        param.value_type
                    )));
                }
            }
        }

        let extra_arg = if let Some(ret_area_ptr) = ret_area_ptr {
            args.push_non_formal(ret_area_ptr);
            Some(args.args().len() - 1)
        } else {
            None
        };

        next_stack = align_to(next_stack, Self::stack_align(call_conv));

        Ok((next_stack, extra_arg))
    }

    fn gen_load_stack(mem: StackAMode, into_reg: Writable<Reg>, ty: Type) -> Inst {
        Inst::gen_load(into_reg, mem.into(), ty, MemFlagsData::trusted())
    }

    fn gen_store_stack(mem: StackAMode, from_reg: Reg, ty: Type) -> Inst {
        Inst::gen_store(mem.into(), from_reg, ty, MemFlagsData::trusted())
    }

    fn gen_move(to_reg: Writable<Reg>, from_reg: Reg, ty: Type) -> Inst {
        Inst::gen_move(to_reg, from_reg, ty)
    }

    fn gen_extend(
        to_reg: Writable<Reg>,
        from_reg: Reg,
        signed: bool,
        from_bits: u8,
        to_bits: u8,
    ) -> Inst {
        assert!(from_bits < to_bits && to_bits <= 32);
        let op = match (signed, from_bits) {
            (false, 8) => BitOp::Uxtb,
            (false, 16) => BitOp::Uxth,
            (true, 8) => BitOp::Sxtb,
            (true, 16) => BitOp::Sxth,
            _ => panic!("unsupported extension from {from_bits} bits"),
        };
        Inst::BitRR {
            op,
            rd: to_reg,
            rm: from_reg,
        }
    }

    fn get_ext_mode(
        _call_conv: isa::CallConv,
        specified: ir::ArgumentExtension,
        _location: ABIArgLocation,
    ) -> ir::ArgumentExtension {
        specified
    }

    fn gen_args(args: Box<[ArgPair]>) -> Inst {
        Inst::Args { args }
    }

    fn gen_rets(rets: Box<[RetPair]>) -> Inst {
        Inst::Rets { rets }
    }

    fn get_stacklimit_reg(_call_conv: isa::CallConv) -> Reg {
        ip_reg()
    }

    fn gen_add_imm(
        _call_conv: isa::CallConv,
        into_reg: Writable<Reg>,
        from_reg: Reg,
        imm: u32,
    ) -> SmallInstVec<Inst> {
        let mut insts = SmallInstVec::new();
        let mut src = from_reg;
        for chunk in mod_imm_chunks(imm) {
            insts.push(Inst::AluRRImm {
                op: AluOp::Add,
                s: false,
                rd: into_reg,
                rn: src,
                imm: chunk,
            });
            src = into_reg.to_reg();
        }
        if insts.is_empty() && into_reg.to_reg() != from_reg {
            insts.push(Inst::Mov {
                rd: into_reg,
                rm: from_reg,
            });
        }
        insts
    }

    fn gen_stack_lower_bound_trap(limit_reg: Reg) -> SmallInstVec<Inst> {
        smallvec![
            Inst::CmpRR {
                op: CmpOp::Cmp,
                rn: stack_reg(),
                rm: limit_reg,
                shift: ShiftOpAndAmt::none(),
            },
            Inst::TrapIf {
                cond: Cond::Lo,
                trap_code: TrapCode::STACK_OVERFLOW,
            }
        ]
    }

    fn gen_get_stack_addr(mem: StackAMode, into_reg: Writable<Reg>) -> Inst {
        Inst::LoadAddr {
            rd: into_reg,
            mem: mem.into(),
        }
    }

    fn gen_load_base_offset(into_reg: Writable<Reg>, base: Reg, offset: i32, ty: Type) -> Inst {
        Inst::gen_load(
            into_reg,
            AMode::RegOffset { base, offset },
            ty,
            MemFlagsData::trusted(),
        )
    }

    fn gen_store_base_offset(base: Reg, offset: i32, from_reg: Reg, ty: Type) -> Inst {
        Inst::gen_store(
            AMode::RegOffset { base, offset },
            from_reg,
            ty,
            MemFlagsData::trusted(),
        )
    }

    fn gen_sp_reg_adjust(amount: i32) -> SmallInstVec<Inst> {
        if amount == 0 {
            SmallInstVec::new()
        } else {
            smallvec![Inst::SpAdjust { amount }]
        }
    }

    fn gen_prologue_frame_setup(
        _call_conv: isa::CallConv,
        flags: &settings::Flags,
        _isa_flags: &Arm32Flags,
        frame_layout: &FrameLayout,
    ) -> SmallInstVec<Inst> {
        let mut insts = SmallVec::new();
        if frame_layout.setup_area_size > 0 {
            // push {<core clobbers>, fp, lr}: the core callee-saved registers
            // are pushed along with the frame record when possible.
            let merged = prologue_merged_mask(frame_layout);
            insts.push(Inst::Push {
                mask: merged | (1 << 11) | (1 << 14),
            });
            let below = 4 * merged.count_ones();
            if flags.unwind_info() {
                insts.push(Inst::Unwind {
                    inst: UnwindInst::PushFrameRegs {
                        offset_upward_to_caller_sp: frame_layout.setup_area_size,
                    },
                });
                if below > 0 {
                    insts.push(Inst::Unwind {
                        inst: UnwindInst::StackAlloc { size: below },
                    });
                }
            }
            // fp = the address of the saved fp.
            if below == 0 {
                insts.push(Inst::Mov {
                    rd: writable_fp_reg(),
                    rm: stack_reg(),
                });
            } else {
                insts.push(Inst::AluRRImm {
                    op: AluOp::Add,
                    s: false,
                    rd: writable_fp_reg(),
                    rn: stack_reg(),
                    imm: ModImm::maybe_from_u32(below).unwrap(),
                });
            }
        }
        insts
    }

    fn gen_epilogue_frame_restore(
        call_conv: isa::CallConv,
        _flags: &settings::Flags,
        _isa_flags: &Arm32Flags,
        frame_layout: &FrameLayout,
    ) -> SmallInstVec<Inst> {
        let mut insts = SmallVec::new();
        let callee_pop = callee_pop_size(call_conv, frame_layout);
        if frame_layout.setup_area_size > 0 && callee_pop > 0 {
            // pop {<core clobbers>, fp, lr}
            insts.push(Inst::Pop {
                mask: merged_clobber_mask(frame_layout) | (1 << 11) | (1 << 14),
            });
        }
        // Otherwise, the frame record is popped by the return.
        if callee_pop > 0 {
            insts.push(Inst::SpAdjust {
                amount: i32::try_from(callee_pop).unwrap(),
            });
        }
        insts
    }

    fn gen_return(
        call_conv: isa::CallConv,
        _isa_flags: &Arm32Flags,
        frame_layout: &FrameLayout,
    ) -> SmallInstVec<Inst> {
        if frame_layout.setup_area_size > 0 && callee_pop_size(call_conv, frame_layout) == 0 {
            // Return by popping the saved LR into PC (which interworks like
            // `bx lr`): `pop {<core clobbers>, fp, pc}`.
            smallvec![Inst::Pop {
                mask: merged_clobber_mask(frame_layout) | (1 << 11) | (1 << 15),
            }]
        } else {
            smallvec![Inst::Ret]
        }
    }

    fn gen_probestack(insts: &mut SmallInstVec<Self::I>, frame_size: u32) {
        // There is no out-of-line probestack routine available; probe inline
        // with a conservative guard size.
        Self::gen_inline_probestack(insts, isa::CallConv::SystemV, frame_size, 4096);
    }

    fn gen_inline_probestack(
        insts: &mut SmallInstVec<Self::I>,
        _call_conv: isa::CallConv,
        frame_size: u32,
        guard_size: u32,
    ) {
        let probe_count = frame_size / guard_size;
        if probe_count == 0 {
            return;
        }
        insts.push(Inst::StackProbeLoop {
            guard_size,
            probe_count,
        });
    }

    fn gen_clobber_save(
        _call_conv: isa::CallConv,
        flags: &settings::Flags,
        frame_layout: &FrameLayout,
    ) -> SmallVec<[Inst; 16]> {
        let mut insts = SmallVec::new();
        let setup_frame = frame_layout.setup_area_size > 0;

        if flags.unwind_info() && setup_frame {
            insts.push(Inst::Unwind {
                inst: UnwindInst::DefineNewFrame {
                    offset_downward_to_clobbers: frame_layout.clobber_size,
                    offset_upward_to_caller_sp: frame_layout.setup_area_size,
                },
            });
        }

        let incoming_args_diff = frame_layout.tail_args_size - frame_layout.incoming_args_size;
        if incoming_args_diff > 0 {
            // Grow the incoming argument area, moving the frame record down.
            insts.push(Inst::SpAdjust {
                amount: -i32::try_from(incoming_args_diff).unwrap(),
            });
            if setup_frame {
                // The saved LR is still in LR.
                insts.push(Inst::gen_store(
                    AMode::SPOffset(4),
                    link_reg(),
                    I32,
                    MemFlagsData::trusted(),
                ));
                // Copy the saved FP down.
                insts.push(Inst::gen_load(
                    writable_ip_reg(),
                    AMode::SPOffset(i32::try_from(incoming_args_diff).unwrap()),
                    I32,
                    MemFlagsData::trusted(),
                ));
                insts.push(Inst::gen_store(
                    AMode::SPOffset(0),
                    ip_reg(),
                    I32,
                    MemFlagsData::trusted(),
                ));
                insts.push(Inst::Mov {
                    rd: writable_fp_reg(),
                    rm: stack_reg(),
                });
            }
        }

        let layout = ClobberLayout::of(frame_layout);
        if layout.int_mask != 0 {
            // Unless already pushed along with the frame record.
            if prologue_merged_mask(frame_layout) == 0 {
                insts.push(Inst::Push {
                    mask: layout.int_mask,
                });
            }
            if flags.unwind_info() {
                // The core registers are pushed in ascending order, starting
                // at the bottom of the core register area.
                let int_base = frame_layout.clobber_size - layout.int_size()
                    + if layout.int_padding { 4 } else { 0 };
                for (i, reg) in layout.int_regs.iter().enumerate() {
                    // LR is saved in the frame record.
                    debug_assert_ne!(*reg, link_reg().to_real_reg().unwrap());
                    insts.push(Inst::Unwind {
                        inst: UnwindInst::SaveReg {
                            clobber_offset: int_base + 4 * i as u32,
                            reg: *reg,
                        },
                    });
                }
            }
        }
        // The padding below the core registers is allocated along with the
        // fixed frame storage if there are no float registers in between.
        let fold_padding = layout.int_padding && layout.float_ranges.is_empty();
        if layout.int_padding && !fold_padding {
            insts.push(Inst::SpAdjust { amount: -4 });
        }
        // Float registers: each range is pushed below the previous one.
        let mut float_top = frame_layout.clobber_size - layout.int_size();
        for &(first, count) in &layout.float_ranges {
            insts.push(Inst::VPush { first, count });
            float_top -= 8 * u32::from(count);
            if flags.unwind_info() {
                for i in 0..count {
                    insts.push(Inst::Unwind {
                        inst: UnwindInst::SaveReg {
                            clobber_offset: float_top + 8 * u32::from(i),
                            reg: dreg(first + i).to_real_reg().unwrap(),
                        },
                    });
                }
            }
        }
        debug_assert_eq!(float_top, 0);

        let stack_size = frame_layout.fixed_frame_storage_size
            + frame_layout.outgoing_args_size
            + if fold_padding { 4 } else { 0 };
        if stack_size > 0 {
            insts.push(Inst::SpAdjust {
                amount: -i32::try_from(stack_size).unwrap(),
            });
        }
        insts
    }

    fn gen_clobber_restore(
        _call_conv: isa::CallConv,
        _flags: &settings::Flags,
        frame_layout: &FrameLayout,
    ) -> SmallVec<[Inst; 16]> {
        let mut insts = SmallVec::new();
        gen_clobber_restore_insts(frame_layout, &mut |inst| insts.push(inst));
        insts
    }

    fn gen_memcpy<F: FnMut(Type) -> Writable<Reg>>(
        call_conv: isa::CallConv,
        dst: Reg,
        src: Reg,
        size: usize,
        mut alloc_tmp: F,
    ) -> SmallVec<[Self::I; 8]> {
        let mut insts = SmallVec::new();
        let arg0 = rreg(0);
        let arg1 = rreg(1);
        let arg2 = rreg(2);
        let tmp = alloc_tmp(Self::word_type());
        insts.extend(Inst::load_constant32(tmp, u32::try_from(size).unwrap()));
        insts.push(Inst::Call {
            info: Box::new(CallInfo {
                dest: ExternalName::LibCall(LibCall::Memcpy),
                uses: smallvec![
                    CallArgPair {
                        vreg: dst,
                        preg: arg0
                    },
                    CallArgPair {
                        vreg: src,
                        preg: arg1
                    },
                    CallArgPair {
                        vreg: tmp.to_reg(),
                        preg: arg2
                    }
                ],
                defs: smallvec![],
                clobbers: Self::get_regs_clobbered_by_call(call_conv, false),
                caller_conv: call_conv,
                callee_conv: call_conv,
                callee_pop_size: 0,
                try_call_info: None,
                patchable: false,
            }),
        });
        insts
    }

    fn get_number_of_spillslots_for_value(
        rc: RegClass,
        _target_vector_bytes: u32,
        _isa_flags: &Arm32Flags,
    ) -> u32 {
        // Slots are 4 bytes.
        match rc {
            RegClass::Int => 1,
            RegClass::Float => 2,
            RegClass::Vector => 4,
        }
    }

    fn get_machine_env(_flags: &settings::Flags, call_conv: isa::CallConv) -> &MachineEnv {
        static MACHINE_ENV: MachineEnv = create_reg_env(false, false);
        static TAIL_MACHINE_ENV: MachineEnv = create_reg_env(true, false);
        match call_conv {
            isa::CallConv::Tail => &TAIL_MACHINE_ENV,
            _ => &MACHINE_ENV,
        }
    }

    fn get_machine_env_for_vreg_classes<'a>(
        flags: &'a settings::Flags,
        isa_flags: &'a Arm32Flags,
        call_conv: isa::CallConv,
        vreg_classes: [bool; 3],
    ) -> &'a MachineEnv {
        // The vector registers q8-q15 are d16-d31, which (with NEON, which
        // implies 32 D registers) can hold scalar floats in functions without
        // vector values. This relies on `Limit` operand constraints, which
        // the single-pass allocator doesn't support.
        if isa_flags.has_neon()
            && !vreg_classes[RegClass::Vector as usize]
            && flags.regalloc_algorithm() == settings::RegallocAlgorithm::Backtracking
        {
            static WIDE_MACHINE_ENV: MachineEnv = create_reg_env(false, true);
            static WIDE_TAIL_MACHINE_ENV: MachineEnv = create_reg_env(true, true);
            match call_conv {
                isa::CallConv::Tail => &WIDE_TAIL_MACHINE_ENV,
                _ => &WIDE_MACHINE_ENV,
            }
        } else {
            Self::get_machine_env(flags, call_conv)
        }
    }

    fn get_regs_clobbered_by_call(call_conv: isa::CallConv, is_exception: bool) -> PRegSet {
        match call_conv {
            isa::CallConv::Tail if is_exception => ALL_CLOBBERS,
            isa::CallConv::PreserveAll if is_exception => ALL_CLOBBERS,
            isa::CallConv::PreserveAll => NO_CLOBBERS,
            isa::CallConv::Tail => TAIL_CLOBBERS,
            _ => DEFAULT_CLOBBERS,
        }
    }

    fn compute_frame_layout(
        call_conv: isa::CallConv,
        flags: &settings::Flags,
        sig: &Signature,
        regs: &[Writable<RealReg>],
        function_calls: FunctionCalls,
        incoming_args_size: u32,
        tail_args_size: u32,
        stackslots_size: u32,
        fixed_frame_storage_size: u32,
        outgoing_args_size: u32,
    ) -> FrameLayout {
        // LR is allocatable; if it is clobbered, the frame record saves it.
        let lr_clobbered = regs
            .iter()
            .any(|r| r.to_reg() == link_reg().to_real_reg().unwrap());
        let callee_saves = match call_conv {
            isa::CallConv::Tail => TAIL_CALLEE_SAVES,
            _ => DEFAULT_CALLEE_SAVES,
        };
        let is_callee_saved = |reg: &Writable<RealReg>| match call_conv {
            // LR is restored from the frame record.
            isa::CallConv::PreserveAll => reg.to_reg() != link_reg().to_real_reg().unwrap(),
            _ => callee_saves.contains(reg.to_reg().into()),
        };
        let mut regs: Vec<Writable<RealReg>> =
            regs.iter().cloned().filter(is_callee_saved).collect();

        // Incoming `f32` arguments in odd S registers are copied into their
        // (callee-saved) shadow registers by the `Args` pseudo-instruction,
        // which does not count towards clobbers, so add them here.
        if call_conv != isa::CallConv::Tail {
            let mut vfp = VfpAllocator::new();
            for param in &sig.params {
                if param.value_type == F32 || param.value_type == F64 {
                    if let Some(s) = vfp.alloc(param.value_type == F64) {
                        if s % 2 == 1 {
                            let shadow = Writable::from_reg(
                                dreg(aapcs_sreg_to_dreg(s) as u8).to_real_reg().unwrap(),
                            );
                            if !regs.contains(&shadow) {
                                regs.push(shadow);
                            }
                        }
                    }
                }
            }
        }
        // Keep SP 8-byte aligned. Prefer pushing an unclobbered callee-saved
        // register (restoring it is harmless) over 4 bytes of padding below
        // the core registers, which takes an SP adjustment unless it can be
        // folded into the one for the fixed frame storage. (Caller-saved
        // registers can't be used: they may hold return values, or arguments
        // of a tail call, when the clobbers are popped. Neither can LR, which
        // would keep the core registers from being pushed and popped
        // together with the frame record.)
        let align = needs_sp_alignment(
            function_calls,
            fixed_frame_storage_size + outgoing_args_size,
        );
        let num_int = regs
            .iter()
            .filter(|r| r.to_reg().class() == RegClass::Int)
            .count();
        if align && num_int % 2 == 1 {
            let is_saved = |r: RealReg| regs.iter().any(|s| s.to_reg() == r);
            let pad = (4..=10u8)
                .map(|n| rreg(n).to_real_reg().unwrap())
                .find(|&r| callee_saves.contains(r.into()) && !is_saved(r));
            if let Some(pad) = pad {
                regs.push(Writable::from_reg(pad));
            }
        }
        regs.sort_unstable();

        let clobber_size = ClobberLayout::new(&regs, align).size();

        let setup_area_size = if flags.preserve_frame_pointers()
            || function_calls != FunctionCalls::None
            || lr_clobbered
            // Incoming stack arguments are addressed relative to SP at a
            // known distance; keep the frame record for unwinding.
            || incoming_args_size > 0
            || clobber_size > 0
            || fixed_frame_storage_size > 0
        {
            8 // FP, LR
        } else {
            0
        };

        FrameLayout {
            word_bytes: 4,
            incoming_args_size,
            tail_args_size,
            setup_area_size,
            clobber_size,
            fixed_frame_storage_size,
            stackslots_size,
            outgoing_args_size,
            clobbered_callee_saves: regs,
            function_calls,
        }
    }

    fn retval_temp_reg(_call_conv_of_callee: isa::CallConv) -> Writable<Reg> {
        // LR is clobbered by every call and is never a return register.
        writable_link_reg()
    }

    fn exception_payload_regs(call_conv: isa::CallConv) -> &'static [Reg] {
        const PAYLOAD_REGS: &'static [Reg] = &[rreg(0), rreg(1)];
        match call_conv {
            isa::CallConv::SystemV | isa::CallConv::Tail | isa::CallConv::PreserveAll => {
                PAYLOAD_REGS
            }
            _ => &[],
        }
    }
}

/// How the callee-saved registers are laid out in the clobber save area.
pub(crate) struct ClobberLayout {
    /// Bitmask of core registers to push.
    pub int_mask: u16,
    /// The core registers, in ascending order.
    pub int_regs: SmallVec<[RealReg; 16]>,
    /// Whether 4 bytes of padding follow the core registers.
    pub int_padding: bool,
    /// Contiguous ranges `(first, count)` of D registers (including the
    /// halves of saved Q registers), pushed in order.
    pub float_ranges: SmallVec<[(u8, u8); 8]>,
}

/// Whether SP must be kept 8-byte aligned below the clobber area: for calls,
/// and for stack slots, which may have that alignment. (FP, which points into
/// the frame record above the clobbers, stays aligned regardless.)
fn needs_sp_alignment(function_calls: FunctionCalls, stack_size: u32) -> bool {
    function_calls != FunctionCalls::None || stack_size > 0
}

impl ClobberLayout {
    /// The layout of a frame's clobber area.
    pub(crate) fn of(frame_layout: &FrameLayout) -> Self {
        Self::new(
            &frame_layout.clobbered_callee_saves,
            needs_sp_alignment(
                frame_layout.function_calls,
                frame_layout.fixed_frame_storage_size + frame_layout.outgoing_args_size,
            ),
        )
    }

    /// The layout for the given registers, padding the core registers to
    /// keep SP 8-byte aligned if `align`.
    pub(crate) fn new(regs: &[Writable<RealReg>], align: bool) -> Self {
        let mut int_mask = 0u16;
        let mut int_regs = SmallVec::new();
        let mut floats: SmallVec<[u8; 32]> = SmallVec::new();
        for reg in regs {
            let r = reg.to_reg();
            match r.class() {
                RegClass::Int => {
                    int_mask |= 1 << r.hw_enc();
                    int_regs.push(r);
                }
                RegClass::Float => floats.push(r.hw_enc()),
                // Vector registers are only saved by the `preserve_all`
                // calling convention; save `q<n>` as `d<2n>`-`d<2n+1>`.
                RegClass::Vector => {
                    floats.push(2 * r.hw_enc());
                    floats.push(2 * r.hw_enc() + 1);
                }
            }
        }
        int_regs.sort_unstable_by_key(|r: &RealReg| r.hw_enc());
        floats.sort_unstable();
        // A clobbered `d16`-`d31` may also be half of a clobbered vector
        // register.
        floats.dedup();
        let mut float_ranges: SmallVec<[(u8, u8); 8]> = SmallVec::new();
        for d in floats {
            match float_ranges.last_mut() {
                Some((first, count)) if *first + *count == d && *count < 16 => *count += 1,
                _ => float_ranges.push((d, 1)),
            }
        }
        // `compute_frame_layout` normally pads the core registers to an even
        // number; otherwise, keep SP 8-byte aligned with an adjustment.
        let int_padding = align && int_regs.len() % 2 == 1;
        ClobberLayout {
            int_mask,
            int_regs,
            int_padding,
            float_ranges,
        }
    }

    /// The size of the pushed core registers, including padding.
    pub(crate) fn int_size(&self) -> u32 {
        4 * (self.int_regs.len() as u32 + u32::from(self.int_padding))
    }

    pub(crate) fn size(&self) -> u32 {
        let floats: u32 = self.float_ranges.iter().map(|(_, c)| u32::from(*c)).sum();
        self.int_size() + 8 * floats
    }
}

/// The size of the stack arguments that the callee pops on return.
fn callee_pop_size(call_conv: isa::CallConv, frame_layout: &FrameLayout) -> u32 {
    if call_conv == isa::CallConv::Tail {
        frame_layout.tail_args_size
    } else {
        0
    }
}

/// The core callee-saved registers that are popped together with the frame
/// record (`pop {<regs>, fp, lr/pc}`) rather than by
/// `gen_clobber_restore_insts`. This is possible when they are all numbered
/// below FP (they always sit directly below the frame record, with any
/// padding below them).
pub(crate) fn merged_clobber_mask(frame_layout: &FrameLayout) -> u16 {
    let layout = ClobberLayout::of(frame_layout);
    if frame_layout.setup_area_size > 0 && layout.int_mask >> 11 == 0 {
        layout.int_mask
    } else {
        0
    }
}

/// The core callee-saved registers that the prologue pushes together with the
/// frame record (`push {<regs>, fp, lr}`): those that the epilogue pops
/// together with it, unless the incoming argument area is resized (which
/// moves the frame record after it is pushed).
fn prologue_merged_mask(frame_layout: &FrameLayout) -> u16 {
    if frame_layout.tail_args_size == frame_layout.incoming_args_size {
        merged_clobber_mask(frame_layout)
    } else {
        0
    }
}

/// Generate the instructions that restore callee-saved registers and reset
/// SP to point at the frame record (the reverse of `gen_clobber_save`),
/// except for the core registers in `merged_clobber_mask`, which the caller
/// pops together with the frame record. Does not use any scratch register.
pub(crate) fn gen_clobber_restore_insts(frame_layout: &FrameLayout, emit: &mut dyn FnMut(Inst)) {
    let layout = ClobberLayout::of(frame_layout);
    let stack_size = frame_layout.fixed_frame_storage_size + frame_layout.outgoing_args_size;
    // Padding allocated along with the fixed frame storage (see
    // `gen_clobber_save`) is skipped when resetting SP.
    let fold_padding = layout.int_padding && layout.float_ranges.is_empty();
    // SP is invariant in the function body (callee-popped arguments are
    // re-reserved after each call), so it only needs to be reset when there
    // is fixed frame storage below the clobber area.
    if stack_size > 0 {
        debug_assert!(frame_layout.setup_area_size > 0);
        // sp = fp - clobber_size (+ padding)
        let offset = frame_layout.clobber_size - if fold_padding { 4 } else { 0 };
        if offset == 0 {
            emit(Inst::Mov {
                rd: writable_stack_reg(),
                rm: fp_reg(),
            });
        } else {
            emit(Inst::AluRRImm {
                op: AluOp::Sub,
                s: false,
                rd: writable_stack_reg(),
                rn: fp_reg(),
                imm: ModImm::maybe_from_u32(offset).unwrap(),
            });
        }
    } else if fold_padding {
        emit(Inst::SpAdjust { amount: 4 });
    }
    for &(first, count) in layout.float_ranges.iter().rev() {
        emit(Inst::VPop { first, count });
    }
    if layout.int_padding && !fold_padding {
        emit(Inst::SpAdjust { amount: 4 });
    }
    if layout.int_mask != 0 && merged_clobber_mask(frame_layout) == 0 {
        emit(Inst::Pop {
            mask: layout.int_mask,
        });
    }
}

/// Split a value into modified-immediate chunks whose sum is the value.
pub(crate) fn mod_imm_chunks(mut value: u32) -> SmallVec<[ModImm; 4]> {
    let mut chunks = SmallVec::new();
    while value != 0 {
        // Take the 8 bits starting at the lowest set bit, aligned down to an
        // even bit position.
        let low = value.trailing_zeros() & !1;
        let chunk = value & (0xff << low);
        chunks.push(ModImm::maybe_from_u32(chunk).unwrap());
        value &= !chunk;
    }
    chunks
}

const DEFAULT_CALLEE_SAVES: PRegSet = PRegSet::empty()
    .with(preg(4))
    .with(preg(5))
    .with(preg(6))
    .with(preg(7))
    .with(preg(8))
    .with(preg(9))
    .with(preg(10))
    .with(preg(11))
    .with(pdreg(8))
    .with(pdreg(9))
    .with(pdreg(10))
    .with(pdreg(11))
    .with(pdreg(12))
    .with(pdreg(13))
    .with(pdreg(14))
    .with(pdreg(15));

const DEFAULT_CLOBBERS: PRegSet = PRegSet::empty()
    .with(preg(0))
    .with(preg(1))
    .with(preg(2))
    .with(preg(3))
    .with(preg(12))
    .with(preg(14))
    .with(pdreg(0))
    .with(pdreg(1))
    .with(pdreg(2))
    .with(pdreg(3))
    .with(pdreg(4))
    .with(pdreg(5))
    .with(pdreg(6))
    .with(pdreg(7))
    .with(pdreg(16))
    .with(pdreg(17))
    .with(pdreg(18))
    .with(pdreg(19))
    .with(pdreg(20))
    .with(pdreg(21))
    .with(pdreg(22))
    .with(pdreg(23))
    .with(pdreg(24))
    .with(pdreg(25))
    .with(pdreg(26))
    .with(pdreg(27))
    .with(pdreg(28))
    .with(pdreg(29))
    .with(pdreg(30))
    .with(pdreg(31))
    .with(pqreg(8))
    .with(pqreg(9))
    .with(pqreg(10))
    .with(pqreg(11))
    .with(pqreg(12))
    .with(pqreg(13))
    .with(pqreg(14))
    .with(pqreg(15));

const ALL_CLOBBERS: PRegSet = PRegSet::empty()
    .with(preg(0))
    .with(preg(1))
    .with(preg(2))
    .with(preg(3))
    .with(preg(4))
    .with(preg(5))
    .with(preg(6))
    .with(preg(7))
    .with(preg(8))
    .with(preg(9))
    .with(preg(10))
    .with(preg(12))
    .with(preg(14))
    .with(pdreg(0))
    .with(pdreg(1))
    .with(pdreg(2))
    .with(pdreg(3))
    .with(pdreg(4))
    .with(pdreg(5))
    .with(pdreg(6))
    .with(pdreg(7))
    .with(pdreg(8))
    .with(pdreg(9))
    .with(pdreg(10))
    .with(pdreg(11))
    .with(pdreg(12))
    .with(pdreg(13))
    .with(pdreg(14))
    .with(pdreg(15))
    .with(pdreg(16))
    .with(pdreg(17))
    .with(pdreg(18))
    .with(pdreg(19))
    .with(pdreg(20))
    .with(pdreg(21))
    .with(pdreg(22))
    .with(pdreg(23))
    .with(pdreg(24))
    .with(pdreg(25))
    .with(pdreg(26))
    .with(pdreg(27))
    .with(pdreg(28))
    .with(pdreg(29))
    .with(pdreg(30))
    .with(pdreg(31))
    .with(pqreg(8))
    .with(pqreg(9))
    .with(pqreg(10))
    .with(pqreg(11))
    .with(pqreg(12))
    .with(pqreg(13))
    .with(pqreg(14))
    .with(pqreg(15));

const TAIL_CLOBBERS: PRegSet = DEFAULT_CLOBBERS.with(preg(4)).with(preg(5));

/// Callee-saved registers of the `tail` calling convention, which uses r4 and
/// r5 for arguments.
const TAIL_CALLEE_SAVES: PRegSet = PRegSet::empty()
    .with(preg(6))
    .with(preg(7))
    .with(preg(8))
    .with(preg(9))
    .with(preg(10))
    .with(preg(11))
    .with(pdreg(8))
    .with(pdreg(9))
    .with(pdreg(10))
    .with(pdreg(11))
    .with(pdreg(12))
    .with(pdreg(13))
    .with(pdreg(14))
    .with(pdreg(15));

// Even calls that preserve all registers overwrite LR.
const NO_CLOBBERS: PRegSet = PRegSet::empty().with(preg(14));

/// The register environment. In the `tail` calling convention, r4 and r5 are
/// caller-saved, and preferred like the other caller-saved registers.
///
/// With `wide_float`, for functions without vector values, the float class
/// also contains `d16`-`d31` (the vector registers' halves, caller-saved);
/// operands accessed as S registers are limited to `d0`-`d15`.
const fn create_reg_env(tail: bool, wide_float: bool) -> MachineEnv {
    let (preferred_int, non_preferred_int) = if tail {
        (
            PRegSet::empty()
                .with(preg(0))
                .with(preg(1))
                .with(preg(2))
                .with(preg(3))
                .with(preg(4))
                .with(preg(5)),
            // See below for LR.
            PRegSet::empty()
                .with(preg(14))
                .with(preg(6))
                .with(preg(7))
                .with(preg(8))
                .with(preg(9))
                .with(preg(10)),
        )
    } else {
        (
            PRegSet::empty()
                .with(preg(0))
                .with(preg(1))
                .with(preg(2))
                .with(preg(3)),
            // LR is saved in the frame record by the prologue whenever it is
            // clobbered, so it is allocatable as a caller-saved register. It
            // is not preferred because using it in a leaf function requires
            // setting up a frame.
            PRegSet::empty()
                .with(preg(14))
                .with(preg(4))
                .with(preg(5))
                .with(preg(6))
                .with(preg(7))
                .with(preg(8))
                .with(preg(9))
                .with(preg(10)),
        )
    };
    let mut preferred_float = PRegSet::empty();
    let mut d = 0;
    while d < 32 {
        if d < 8 || (wide_float && d >= 16) {
            preferred_float = preferred_float.with(pdreg(d));
        }
        d += 1;
    }
    let vector = if wide_float {
        PRegSet::empty()
    } else {
        PRegSet::empty()
            .with(pqreg(8))
            .with(pqreg(9))
            .with(pqreg(10))
            .with(pqreg(11))
            .with(pqreg(12))
            .with(pqreg(13))
            .with(pqreg(14))
            .with(pqreg(15))
    };
    MachineEnv {
        preferred_regs_by_class: [preferred_int, preferred_float, PRegSet::empty()],
        non_preferred_regs_by_class: [
            non_preferred_int,
            PRegSet::empty()
                .with(pdreg(8))
                .with(pdreg(9))
                .with(pdreg(10))
                .with(pdreg(11))
                .with(pdreg(12))
                .with(pdreg(13))
                .with(pdreg(14))
                .with(pdreg(15)),
            vector,
        ],
        fixed_stack_slots: vec![],
        scratch_by_class: [None, None, None],
    }
}
