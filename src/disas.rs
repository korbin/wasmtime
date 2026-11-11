//! Shared disassembly utilities for ELF `.text` sections.

use capstone::InsnGroupType::{CS_GRP_JUMP, CS_GRP_RET};
use cranelift_codegen::isa::lookup_by_name;
use cranelift_codegen::settings::Flags;
use object::read::elf::ElfFile64;
use object::{Architecture, Endianness, FileFlags, Object};
use pulley_interpreter::decode::{Decoder, DecodingError, OpVisitor};
use pulley_interpreter::disas::Disassembler;
use wasmtime::{Result, bail, error::Context as _};
use wasmtime_environ::obj;

/// Metadata about a single disassembled instruction.
pub struct Inst {
    /// The virtual address of this instruction.
    pub address: u64,
    /// Whether this instruction is a jump/branch.
    pub is_jump: bool,
    /// Whether this instruction is a return.
    pub is_return: bool,
    /// Human-readable disassembly text.
    pub disassembly: String,
    /// The raw bytes of this instruction.
    pub bytes: Vec<u8>,
}

/// Disassemble `func` bytes at the given `addr` using the architecture from
/// `elf`.
pub fn disas(elf: &ElfFile64<'_, Endianness>, func: &[u8], addr: u64) -> Result<Vec<Inst>> {
    let cranelift_target = match elf.architecture() {
        Architecture::X86_64 => "x86_64",
        Architecture::Aarch64 => {
            let e_flags = match elf.flags() {
                FileFlags::Elf { e_flags, .. } => e_flags,
                _ => bail!("not an ELF file"),
            };
            if e_flags.intersects(obj::EF_WASMTIME_ARM32) {
                "armv7"
            } else {
                "aarch64"
            }
        }
        Architecture::S390x => "s390x",
        Architecture::Riscv64 => {
            let e_flags = match elf.flags() {
                FileFlags::Elf { e_flags, .. } => e_flags,
                _ => bail!("not an ELF file"),
            };
            if e_flags.intersects(obj::EF_WASMTIME_PULLEY32 | obj::EF_WASMTIME_PULLEY64) {
                return disas_pulley(func, addr);
            } else {
                "riscv64"
            }
        }
        other => bail!("unknown architecture {other:?}"),
    };
    let builder =
        lookup_by_name(cranelift_target).context("failed to load cranelift ISA builder")?;
    let flags = cranelift_codegen::settings::builder();
    let isa = builder.finish(Flags::new(flags))?;
    let isa = &*isa;
    let capstone = isa
        .to_capstone()
        .context("failed to create a capstone disassembler")?;

    disas_with_capstone(&capstone, func, addr)
}

/// Whether a 32-bit ARM instruction writes PC through an explicit operand,
/// like `pop {.., pc}`, `ldr pc, [..]` or `add pc, ..`, which capstone doesn't
/// put in the jump or return groups.
fn arm_writes_pc(detail: &capstone::InsnDetail) -> bool {
    use capstone::arch::DetailsArchInsn;
    use capstone::arch::arm::{ArmOperandType, ArmReg};
    let arch = detail.arch_detail();
    let Some(arm) = arch.arm() else {
        return false;
    };
    arm.operands().any(|op| {
        matches!(op.op_type, ArmOperandType::Reg(r) if u32::from(r.0) == ArmReg::ARM_REG_PC)
            && op.access.is_some_and(|a| a.is_writable())
    })
}

/// Disassemble `func` bytes at the given `addr` using an already-configured
/// capstone instance.
pub fn disas_with_capstone(
    capstone: &capstone::Capstone,
    func: &[u8],
    addr: u64,
) -> Result<Vec<Inst>> {
    let insts = capstone
        .disasm_all(func, addr)
        .map_err(|e| wasmtime::format_err!("{e}"))?
        .into_iter()
        .map(|inst| {
            let detail = capstone.insn_detail(&inst).ok();
            let detail = detail.as_ref();
            let is_jump = detail
                .map(|d| d.groups().iter().any(|g| g.0 as u32 == CS_GRP_JUMP) || arm_writes_pc(d))
                .unwrap_or(false);

            let is_return = detail
                .map(|d| d.groups().iter().any(|g| g.0 as u32 == CS_GRP_RET))
                .unwrap_or(false);

            let disassembly = match (inst.mnemonic(), inst.op_str()) {
                (Some(i), Some(o)) => {
                    if o.is_empty() {
                        format!("{i}")
                    } else {
                        format!("{i:7} {o}")
                    }
                }
                (Some(i), None) => format!("{i}"),
                _ => unreachable!(),
            };

            let address = inst.address();
            Inst {
                address,
                is_jump,
                is_return,
                bytes: inst.bytes().to_vec(),
                disassembly,
            }
        })
        .collect::<Vec<_>>();
    Ok(insts)
}

/// Disassemble Pulley bytecode at the given `addr`.
pub fn disas_pulley(func: &[u8], addr: u64) -> Result<Vec<Inst>> {
    let mut result = vec![];

    let mut disas = Disassembler::new(func);
    disas.offsets(false);
    disas.hexdump(false);
    disas.start_offset(usize::try_from(addr).unwrap());
    let mut decoder = Decoder::new();
    let mut last_disas_pos = 0;
    loop {
        let start_addr = disas.bytecode().position();

        match decoder.decode_one(&mut disas) {
            // If we got EOF at the initial position, then we're done disassembling.
            Err(DecodingError::UnexpectedEof { position }) if position == start_addr => break,

            // Otherwise, propagate the error.
            Err(e) => {
                return Err(e).context("failed to disassembly pulley bytecode");
            }

            Ok(()) => {
                let bytes_range = start_addr..disas.bytecode().position();
                let disassembly = disas.disas()[last_disas_pos..].trim();
                last_disas_pos = disas.disas().len();
                let address = u64::try_from(start_addr).unwrap() + addr;
                let is_jump = disassembly.contains("jump") || disassembly.contains("br_");
                let is_return = disassembly == "ret";
                result.push(Inst {
                    bytes: func[bytes_range].to_vec(),
                    address,
                    is_jump,
                    is_return,
                    disassembly: disassembly.to_string(),
                });
            }
        }
    }

    Ok(result)
}
