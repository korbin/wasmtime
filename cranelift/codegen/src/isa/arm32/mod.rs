//! ARMv7 (32-bit ARM, A32 encoding) Instruction Set Architecture.
//!
//! This backend targets ARMv7-A in ARM state with VFPv3 and the hard-float
//! procedure call standard (AAPCS-VFP), i.e. `armv7-unknown-linux-gnueabihf`.
//! Optional extensions are enabled with ISA settings (and detected on the
//! host by `cranelift-native`): `has_idiv` (hardware division; otherwise
//! software loops are emitted), `has_vfp4` (fused multiply-add; otherwise a
//! libcall is used), `has_neon` (required for 128-bit vector types) and
//! `has_armv8` (an ARMv8 core in AArch32 state: rounding instructions and
//! load-acquire/store-release atomics instead of barriers).
//!
//! Floating-point and vector constants without an immediate encoding are
//! loaded from PC-relative constant islands (`vldr`, +/- 1 KiB).
//!
//! Register usage:
//!
//! * `r0`-`r10` and `lr` are allocatable (`lr` as a caller-saved register,
//!   saved in the frame record whenever it is clobbered); `r11` is the frame
//!   pointer, `r12` (`ip`) is a scratch register used within instruction
//!   sequences and veneers, and `r13` and `r15` are `sp` and `pc`.
//! * Scalar floats live in `d0`-`d15`; an `f32` in `d<n>` occupies its low
//!   half `s<2n>`. (See `abi.rs` for how odd S registers in the AAPCS are
//!   handled.)
//! * Vectors live in `q8`-`q15` (`d16`-`d31`), which don't alias the scalar
//!   float registers. In functions without vector values (and with NEON,
//!   which implies 32 D registers), scalar floats may also use `d16`-`d31`;
//!   operands accessed as S registers are then limited to `d0`-`d15` (see
//!   `create_reg_env` in `abi.rs`).
//!
//! Values of type `i64` occupy a pair of core registers `[lo, hi]`. `i128`
//! (and `f128`) is not supported, since values are limited to two registers;
//! Wasmtime translates the Wasm wide-arithmetic proposal without `i128` for
//! this target.
//!
//! NEON floating-point instructions flush denormals to zero (and ARMv7 NEON
//! has no 64-bit lane arithmetic), so vector float operations that must be
//! exact either go through VFP lane by lane (arithmetic, f64x2 operations,
//! and rounding, which converts lanes to integers and back), or work on the
//! bit patterns with NEON integer instructions: f32x4 `fmin`/`fmax` and
//! comparisons compare order-preserving integer keys of the lanes.

use crate::CodegenError;
use crate::dominator_tree::DominatorTree;
use crate::ir::{self, Function, Type};
use crate::isa::arm32::settings as arm32_settings;
use crate::isa::{
    Builder as IsaBuilder, FunctionAlignment, IsaFlagsHashKey, OwnedTargetIsa, TargetIsa,
};
use crate::loop_analysis::LoopAnalysis;
use crate::machinst::{
    CompiledCode, CompiledCodeStencil, MachInst, MachTextSectionBuilder, Reg, SigSet,
    TextSectionBuilder, VCode, compile,
};
use crate::result::CodegenResult;
use crate::settings::{self as shared_settings, Flags};
use alloc::string::String;
use alloc::{boxed::Box, vec::Vec};
use core::fmt;
use cranelift_control::ControlPlane;
use target_lexicon::{Architecture, Triple};

mod abi;
pub(crate) mod inst;
mod lower;
mod settings;

use self::inst::EmitInfo;

/// An ARMv7 backend.
pub struct Arm32Backend {
    triple: Triple,
    flags: shared_settings::Flags,
    isa_flags: arm32_settings::Flags,
}

impl Arm32Backend {
    /// Create a new ARMv7 backend with the given (shared) flags.
    pub fn new_with_flags(
        triple: Triple,
        flags: shared_settings::Flags,
        isa_flags: arm32_settings::Flags,
    ) -> Self {
        Self {
            triple,
            flags,
            isa_flags,
        }
    }

    /// This performs lowering to VCode, register-allocates the code, computes
    /// block layout and finalizes branches. The result is ready for binary
    /// emission.
    fn compile_vcode(
        &self,
        func: &Function,
        domtree: &DominatorTree,
        loop_analysis: &LoopAnalysis,
        regalloc_ctx: &mut regalloc2::Ctx,
        ctrl_plane: &mut ControlPlane,
    ) -> CodegenResult<VCode<inst::Inst>> {
        // Vector types are lowered to NEON instructions, so reject them up
        // front when NEON isn't available.
        if !self.isa_flags.has_neon() && uses_vector_types(func) {
            return Err(CodegenError::Unsupported(
                "vector types require the `has_neon` setting on arm32".into(),
            ));
        }
        let emit_info = EmitInfo::new(self.flags.clone(), self.isa_flags.clone());
        let sigs = SigSet::new::<abi::Arm32MachineDeps>(func, &self.flags)?;
        let abi = abi::Arm32Callee::new(func, self, &self.isa_flags, &sigs)?;
        compile::compile::<Arm32Backend>(
            func,
            domtree,
            loop_analysis,
            regalloc_ctx,
            self,
            abi,
            emit_info,
            sigs,
            ctrl_plane,
        )
    }
}

/// Whether a function uses vector types in its values or signatures.
fn uses_vector_types(func: &Function) -> bool {
    let sig_has_vectors = |sig: &ir::Signature| {
        sig.params
            .iter()
            .chain(sig.returns.iter())
            .any(|p| p.value_type.is_vector())
    };
    sig_has_vectors(&func.signature)
        || func.dfg.signatures.values().any(sig_has_vectors)
        || func
            .dfg
            .values()
            .any(|v| func.dfg.value_type(v).is_vector())
}

impl TargetIsa for Arm32Backend {
    fn compile_function(
        &self,
        func: &Function,
        domtree: &DominatorTree,
        loop_analysis: &LoopAnalysis,
        regalloc_ctx: &mut regalloc2::Ctx,
        want_disasm: bool,
        ctrl_plane: &mut ControlPlane,
    ) -> CodegenResult<CompiledCodeStencil> {
        let vcode = self.compile_vcode(func, domtree, loop_analysis, regalloc_ctx, ctrl_plane)?;

        let want_disasm = want_disasm || log::log_enabled!(log::Level::Debug);
        let emit_result = vcode.emit(&regalloc_ctx.output, want_disasm, &self.flags, ctrl_plane)?;
        let value_labels_ranges = emit_result.value_labels_ranges;
        let buffer = emit_result.buffer;

        if let Some(disasm) = emit_result.disasm.as_ref() {
            log::debug!("disassembly:\n{disasm}");
        }

        Ok(CompiledCodeStencil(CompiledCode {
            buffer,
            vcode: emit_result.disasm,
            value_labels_ranges,
            bb_starts: emit_result.bb_offsets,
            bb_edges: emit_result.bb_edges,
        }))
    }

    fn name(&self) -> &'static str {
        "arm32"
    }

    fn triple(&self) -> &Triple {
        &self.triple
    }

    fn flags(&self) -> &shared_settings::Flags {
        &self.flags
    }

    fn isa_flags(&self) -> Vec<shared_settings::Value> {
        self.isa_flags.iter().collect()
    }

    fn isa_flags_hash_key(&self) -> IsaFlagsHashKey<'_> {
        IsaFlagsHashKey(self.isa_flags.hash_key())
    }

    fn dynamic_vector_bytes(&self, _dyn_ty: Type) -> u32 {
        16
    }

    #[cfg(feature = "unwind")]
    fn emit_unwind_info(
        &self,
        result: &CompiledCode,
        kind: crate::isa::unwind::UnwindInfoKind,
    ) -> CodegenResult<Option<crate::isa::unwind::UnwindInfo>> {
        use crate::isa::unwind::UnwindInfo;
        use crate::isa::unwind::UnwindInfoKind;
        Ok(match kind {
            UnwindInfoKind::SystemV => {
                let mapper = self::inst::unwind::systemv::RegisterMapper;
                Some(UnwindInfo::SystemV(
                    crate::isa::unwind::systemv::create_unwind_info_from_insts(
                        &result.buffer.unwind_info[..],
                        result.buffer.data().len(),
                        &mapper,
                    )?,
                ))
            }
            _ => None,
        })
    }

    #[cfg(feature = "unwind")]
    fn create_systemv_cie(&self) -> Option<gimli::write::CommonInformationEntry> {
        Some(inst::unwind::systemv::create_cie())
    }

    #[cfg(feature = "unwind")]
    fn map_regalloc_reg_to_dwarf(
        &self,
        reg: Reg,
    ) -> Result<u16, crate::isa::unwind::systemv::RegisterMappingError> {
        inst::unwind::systemv::map_reg(reg).map(|reg| reg.0)
    }

    fn text_section_builder(&self, num_funcs: usize) -> Box<dyn TextSectionBuilder> {
        Box::new(MachTextSectionBuilder::<inst::Inst>::new(num_funcs))
    }

    fn function_alignment(&self) -> FunctionAlignment {
        inst::Inst::function_alignment()
    }

    fn page_size_align_log2(&self) -> u8 {
        12
    }

    #[cfg(feature = "disas")]
    fn to_capstone(&self) -> Result<capstone::Capstone, capstone::Error> {
        use capstone::prelude::*;
        let mut cs = Capstone::new()
            .arm()
            .mode(arch::arm::ArchMode::Arm)
            .build()?;
        cs.set_skipdata(true)?;
        Ok(cs)
    }

    fn pretty_print_reg(&self, reg: Reg, _size: u8) -> String {
        inst::regs::show_reg(reg)
    }

    fn has_native_fma(&self) -> bool {
        self.isa_flags.has_vfp4()
    }

    fn has_round(&self) -> bool {
        true
    }

    fn supports_vector_load_store(&self, ty: Type) -> bool {
        self.isa_flags.has_neon() && ty.is_vector() && ty.bits() == 128
    }

    fn has_blendv_lowering(&self, _: Type) -> bool {
        false
    }

    fn has_x86_pshufb_lowering(&self) -> bool {
        false
    }

    fn has_x86_pmulhrsw_lowering(&self) -> bool {
        false
    }

    fn has_x86_pmaddubsw_lowering(&self) -> bool {
        false
    }

    fn default_argument_extension(&self) -> ir::ArgumentExtension {
        // AAPCS requires the caller to extend arguments smaller than a word
        // according to their type's signedness, which Cranelift expresses
        // explicitly with `uext`/`sext`.
        ir::ArgumentExtension::None
    }
}

impl fmt::Display for Arm32Backend {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("MachBackend")
            .field("name", &self.name())
            .field("triple", &self.triple())
            .field("flags", &format!("{}", self.flags()))
            .finish()
    }
}

/// Create a new `isa::Builder`.
pub fn isa_builder(triple: Triple) -> IsaBuilder {
    assert!(matches!(triple.architecture, Architecture::Arm(_)));
    IsaBuilder {
        triple,
        setup: arm32_settings::builder(),
        constructor: isa_constructor,
    }
}

fn isa_constructor(
    triple: Triple,
    shared_flags: Flags,
    builder: &shared_settings::Builder,
) -> CodegenResult<OwnedTargetIsa> {
    let isa_flags = arm32_settings::Flags::new(&shared_flags, builder);
    let backend = Arm32Backend::new_with_flags(triple, shared_flags, isa_flags);
    Ok(backend.wrapped())
}
