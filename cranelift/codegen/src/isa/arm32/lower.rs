//! Lowering rules for ARMv7.
use crate::CodegenResult;
use crate::ir::{self, Inst as IRInst, LibCall, types};
use crate::isa::CallConv;
use crate::isa::arm32::Arm32Backend;
use crate::isa::arm32::abi::Arm32MachineDeps;
use crate::isa::arm32::inst::*;
use crate::machinst::*;
use crate::settings::Flags;
use alloc::boxed::Box;
use target_lexicon::Triple;
pub mod isle;

/// Emit a call to a libcall, returning its results.
pub(crate) fn emit_libcall(
    ctx: &mut Lower<Inst>,
    flags: &Flags,
    triple: &Triple,
    libcall: LibCall,
    inputs: &[ValueRegs<Reg>],
) -> CodegenResult<InstOutput> {
    let extname = ir::ExternalName::LibCall(libcall);
    let call_conv = CallConv::for_libcall(flags, CallConv::triple_default(triple));
    let sig = libcall.signature(call_conv, types::I32);
    let outputs = ctx.gen_call_output(&sig);

    if !ctx.sigs().have_abi_sig_for_signature(&sig) {
        ctx.sigs_mut()
            .make_abi_sig_from_ir_signature::<Arm32MachineDeps>(sig.clone(), flags)?;
    }
    let sig = ctx.sigs().abi_sig_for_signature(&sig);

    let uses = ctx.gen_call_args(sig, inputs);
    let defs = ctx.gen_call_rets(sig, &outputs);

    let stack_ret_space = ctx.sigs()[sig].sized_stack_ret_space();
    let stack_arg_space = ctx.sigs()[sig].sized_stack_arg_space();
    ctx.abi_mut()
        .accumulate_outgoing_args_size(stack_ret_space + stack_arg_space);

    if flags.use_colocated_libcalls() {
        let call_info = ctx.gen_call_info(sig, extname, uses, defs, None, false);
        ctx.emit(Inst::Call {
            info: Box::new(call_info),
        });
    } else {
        let tmp = ctx.alloc_tmp(types::I32).only_reg().unwrap();
        ctx.emit(Inst::LoadExtName {
            rd: tmp,
            name: Box::new(extname),
            offset: 0,
        });
        let call_info = ctx.gen_call_info(sig, tmp.to_reg(), uses, defs, None, false);
        ctx.emit(Inst::CallInd {
            info: Box::new(call_info),
        });
    }
    Ok(outputs)
}

//=============================================================================
// Lowering-backend trait implementation.

impl LowerBackend for Arm32Backend {
    type MInst = Inst;

    fn lower(&self, ctx: &mut Lower<Inst>, ir_inst: IRInst) -> Option<InstOutput> {
        isle::lower(ctx, self, ir_inst)
    }

    fn lower_branch(
        &self,
        ctx: &mut Lower<Inst>,
        ir_inst: IRInst,
        targets: &[MachLabel],
    ) -> Option<()> {
        isle::lower_branch(ctx, self, ir_inst, targets)
    }
}
