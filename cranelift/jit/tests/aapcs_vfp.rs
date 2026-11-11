//! Checks that Cranelift's 32-bit ARM backend follows the AAPCS-VFP
//! (hard-float) procedure call standard as implemented by rustc, in both
//! directions: JIT code calling Rust and Rust calling JIT code.
//!
//! The interesting cases are `f32` arguments assigned to odd-numbered S
//! registers (the high halves of D registers), including back-filling.

#![cfg(target_arch = "arm")]

use cranelift_codegen::Context;
use cranelift_codegen::ir::types::{F32, F64, I32};
use cranelift_codegen::ir::*;
use cranelift_codegen::isa::OwnedTargetIsa;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::*;
use cranelift_jit::*;
use cranelift_module::*;

fn isa() -> OwnedTargetIsa {
    let mut flag_builder = settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "false").unwrap();
    cranelift_native::builder()
        .unwrap()
        .finish(settings::Flags::new(flag_builder))
        .unwrap()
}

/// Mixes the arguments with distinct weights so that any misplaced argument
/// changes the result.
extern "C" fn rust_mix(a: f32, b: f64, c: f32, d: f32, e: i32, f: f64, g: f32) -> f64 {
    f64::from(a) * 1.0
        + b * 10.0
        + f64::from(c) * 100.0
        + f64::from(d) * 1000.0
        + f64::from(e) * 10000.0
        + f * 100000.0
        + f64::from(g) * 1000000.0
}

fn mix_signature(module: &JITModule) -> Signature {
    let mut sig = module.make_signature();
    for ty in [F32, F64, F32, F32, I32, F64, F32] {
        sig.params.push(AbiParam::new(ty));
    }
    sig.returns.push(AbiParam::new(F64));
    sig
}

const ARGS: (f32, f64, f32, f32, i32, f64, f32) = (1.0, 2.0, 3.0, 4.0, 5, 6.0, 7.0);

fn expected() -> f64 {
    let (a, b, c, d, e, f, g) = ARGS;
    rust_mix(a, b, c, d, e, f, g)
}

#[test]
fn jit_calls_rust() {
    let mut builder = JITBuilder::with_isa(isa(), default_libcall_names());
    builder.symbol("rust_mix", rust_mix as *const u8);
    let mut module = JITModule::new(builder);

    let callee_sig = mix_signature(&module);
    let callee = module
        .declare_function("rust_mix", Linkage::Import, &callee_sig)
        .unwrap();

    // fn() -> f64 { rust_mix(ARGS...) }
    let mut sig = module.make_signature();
    sig.returns.push(AbiParam::new(F64));
    let func_id = module
        .declare_function("caller", Linkage::Local, &sig)
        .unwrap();
    let mut ctx = Context::new();
    ctx.func = Function::with_name_signature(UserFuncName::user(0, 0), sig);
    let mut fn_ctx = FunctionBuilderContext::new();
    {
        let mut bcx = FunctionBuilder::new(&mut ctx.func, &mut fn_ctx);
        let block = bcx.create_block();
        bcx.switch_to_block(block);
        let (a, b, c, d, e, f, g) = ARGS;
        let args = [
            bcx.ins().f32const(a),
            bcx.ins().f64const(b),
            bcx.ins().f32const(c),
            bcx.ins().f32const(d),
            bcx.ins().iconst(I32, i64::from(e)),
            bcx.ins().f64const(f),
            bcx.ins().f32const(g),
        ];
        let callee = module.declare_func_in_func(callee, bcx.func);
        let call = bcx.ins().call(callee, &args);
        let ret = bcx.inst_results(call)[0];
        bcx.ins().return_(&[ret]);
        bcx.seal_all_blocks();
        bcx.finalize(module.target_config());
    }
    module.define_function(func_id, &mut ctx).unwrap();
    module.finalize_definitions().unwrap();

    let code = module.get_finalized_function(func_id);
    let code = unsafe { core::mem::transmute::<*const u8, extern "C" fn() -> f64>(code) };
    assert_eq!(code(), expected());
}

#[test]
fn rust_calls_jit() {
    let mut module = JITModule::new(JITBuilder::with_isa(isa(), default_libcall_names()));
    let sig = mix_signature(&module);
    let func_id = module
        .declare_function("callee", Linkage::Local, &sig)
        .unwrap();

    // A JIT version of `rust_mix`.
    let mut ctx = Context::new();
    ctx.func = Function::with_name_signature(UserFuncName::user(0, 0), sig);
    let mut fn_ctx = FunctionBuilderContext::new();
    {
        let mut bcx = FunctionBuilder::new(&mut ctx.func, &mut fn_ctx);
        let block = bcx.create_block();
        bcx.append_block_params_for_function_params(block);
        bcx.switch_to_block(block);
        let params = bcx.block_params(block).to_vec();
        let mut acc = bcx.ins().f64const(0.0);
        let mut weight = 1.0;
        for (i, p) in params.iter().enumerate() {
            let v = match i {
                4 => bcx.ins().fcvt_from_sint(F64, *p),
                1 | 5 => *p,
                _ => bcx.ins().fpromote(F64, *p),
            };
            let w = bcx.ins().f64const(weight);
            let t = bcx.ins().fmul(v, w);
            acc = bcx.ins().fadd(acc, t);
            weight *= 10.0;
        }
        bcx.ins().return_(&[acc]);
        bcx.seal_all_blocks();
        bcx.finalize(module.target_config());
    }
    module.define_function(func_id, &mut ctx).unwrap();
    module.finalize_definitions().unwrap();

    let code = module.get_finalized_function(func_id);
    let code = unsafe {
        core::mem::transmute::<*const u8, extern "C" fn(f32, f64, f32, f32, i32, f64, f32) -> f64>(
            code,
        )
    };
    let (a, b, c, d, e, f, g) = ARGS;
    assert_eq!(code(a, b, c, d, e, f, g), expected());
}
