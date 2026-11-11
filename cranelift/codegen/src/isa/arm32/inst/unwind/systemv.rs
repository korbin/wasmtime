//! Unwind information for System V ABI (ARMv7).

use crate::isa::unwind::systemv::RegisterMappingError;
use crate::machinst::{Reg, RegClass};
use gimli::{Register, write::CommonInformationEntry};

/// Creates a new arm32 common information entry (CIE).
pub fn create_cie() -> CommonInformationEntry {
    use gimli::write::CallFrameInstruction;

    let mut entry = CommonInformationEntry::new(
        gimli::Encoding {
            address_size: 4,
            format: gimli::Format::Dwarf32,
            version: 1,
        },
        4,  // Code alignment factor
        -4, // Data alignment factor
        Register(14),
    );

    // Every frame will start with the call frame address (CFA) at SP.
    let sp = Register(13);
    entry.add_instruction(CallFrameInstruction::Cfa(sp, 0));

    entry
}

/// Map Cranelift registers to their corresponding Gimli registers.
pub fn map_reg(reg: Reg) -> Result<Register, RegisterMappingError> {
    let reg = reg.to_real_reg().unwrap();
    match reg.class() {
        // r0-r15 are DWARF registers 0-15.
        RegClass::Int => Ok(Register(u16::from(reg.hw_enc()))),
        // d0-d31 are DWARF registers 256-287.
        RegClass::Float => Ok(Register(256 + u16::from(reg.hw_enc()))),
        RegClass::Vector => Err(RegisterMappingError::UnsupportedRegisterBank("vector")),
    }
}

pub(crate) struct RegisterMapper;

impl crate::isa::unwind::systemv::RegisterMapper<Reg> for RegisterMapper {
    fn map(&self, reg: Reg) -> Result<u16, RegisterMappingError> {
        Ok(map_reg(reg)?.0)
    }
    fn fp(&self) -> Option<u16> {
        Some(11)
    }
    fn lr(&self) -> Option<u16> {
        Some(14)
    }
    fn lr_offset(&self) -> Option<u32> {
        Some(4)
    }
}
