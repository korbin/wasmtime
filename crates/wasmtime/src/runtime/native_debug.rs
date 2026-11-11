use crate::prelude::*;
use core::mem::size_of;
use object::elf::*;
use object::endian::{BigEndian, Endian, Endianness, LittleEndian};
use object::read::elf::{FileHeader, SectionHeader, Sym};
use object::{
    File, NativeEndian as NE, Object, ObjectSection, ObjectSymbol, RelocationEncoding,
    RelocationKind, RelocationTarget, SymbolKind, U32, U64,
};
use wasmtime_environ::obj;

pub(crate) fn create_gdbjit_image(
    mut bytes: Vec<u8>,
    code_region: (*const u8, usize),
) -> Result<Vec<u8>, Error> {
    let e = ensure_supported_elf_format(&bytes)?;

    // patch relocs
    relocate_dwarf_sections(&mut bytes, code_region)?;

    if is_arm32(&bytes) {
        return arm32_image(&bytes, code_region);
    }

    // elf is still missing details...
    match e {
        Endianness::Little => {
            convert_object_elf_to_loadable_file::<LittleEndian>(&mut bytes, code_region)
        }
        Endianness::Big => {
            convert_object_elf_to_loadable_file::<BigEndian>(&mut bytes, code_region)
        }
    }

    Ok(bytes)
}

fn relocate_dwarf_sections(bytes: &mut [u8], code_region: (*const u8, usize)) -> Result<(), Error> {
    let mut relocations = Vec::new();
    let obj = File::parse(&bytes[..]).map_err(obj::ObjectCrateErrorWrapper)?;
    for section in obj.sections() {
        let section_start = match section.file_range() {
            Some((start, _)) => start,
            None => continue,
        };
        for (off, r) in section.relocations() {
            if r.kind() != RelocationKind::Absolute
                || r.encoding() != RelocationEncoding::Generic
                || (r.size() != 64 && r.size() != 32)
            {
                continue;
            }

            let sym = match r.target() {
                RelocationTarget::Symbol(index) => match obj.symbol_by_index(index) {
                    Ok(sym) => sym,
                    Err(_) => continue,
                },
                _ => continue,
            };
            // Code addresses are 64 bits wide, except for 32-bit targets,
            // whose 32-bit relocations are told apart from the (already
            // correct) offsets into DWARF sections by their target.
            if r.size() == 32 && sym.kind() != SymbolKind::Text {
                continue;
            }
            relocations.push((
                section_start + off,
                r.size(),
                (code_region.0 as u64)
                    .wrapping_add(sym.address())
                    .wrapping_add(r.addend() as u64),
            ));
        }
    }

    for (offset, size, value) in relocations {
        let offset: usize = offset
            .try_into()
            .map_err(|_| format_err!("invalid dwarf relocations"))?;
        if size == 64 {
            let (loc, _) = object::from_bytes_mut::<U64<NE>>(&mut bytes[offset..])
                .map_err(|_| format_err!("invalid dwarf relocations"))?;
            loc.set(NE, value);
        } else {
            let (loc, _) = object::from_bytes_mut::<U32<NE>>(&mut bytes[offset..])
                .map_err(|_| format_err!("invalid dwarf relocations"))?;
            let value =
                u32::try_from(value).map_err(|_| format_err!("invalid dwarf relocations"))?;
            loc.set(NE, value);
        }
    }
    Ok(())
}

/// Whether this is a 32-bit ARM code object, which is an ELF64 file marked
/// with `EF_WASMTIME_ARM32` (ELF64 can't describe 32-bit ARM).
fn is_arm32(bytes: &[u8]) -> bool {
    match FileHeader64::<LittleEndian>::parse(bytes) {
        Ok(header) => {
            let e = LittleEndian;
            header.e_machine.get(e) == EM_AARCH64
                && (header.e_flags.get(e) & obj::EF_WASMTIME_ARM32) == obj::EF_WASMTIME_ARM32
        }
        Err(_) => false,
    }
}

/// Build an ELF32 image for a 32-bit ARM code object, which debuggers can't
/// use as is in a 32-bit process: the text section placed at the code's
/// address, the (relocated) DWARF sections, and the function symbols.
fn arm32_image(bytes: &[u8], code_region: (*const u8, usize)) -> Result<Vec<u8>, Error> {
    let e = LittleEndian;
    let header =
        FileHeader64::<LittleEndian>::parse(bytes).map_err(obj::ObjectCrateErrorWrapper)?;
    let sections = header
        .sections(e, bytes)
        .map_err(obj::ObjectCrateErrorWrapper)?;
    let base = u32::try_from(code_region.0 as usize)
        .map_err(|_| format_err!("32-bit code outside of a 32-bit address space"))?;
    let code_len = u32::try_from(code_region.1)?;

    struct Section<'a> {
        name: &'a [u8],
        sh_type: SectionType,
        flags: SectionFlags,
        addr: u32,
        data: alloc::borrow::Cow<'a, [u8]>,
        link: u32,
        info: u32,
        align: u32,
        entsize: u32,
    }
    let mut out_sections = vec![];

    // The text section, at index 1.
    let (text_index, text) = sections
        .section_by_name(e, b".text")
        .ok_or_else(|| format_err!("no text section"))?;
    out_sections.push(Section {
        name: b".text",
        sh_type: SHT_PROGBITS,
        flags: SHF_ALLOC | SHF_EXECINSTR,
        addr: base,
        data: text
            .data(e, bytes)
            .map_err(obj::ObjectCrateErrorWrapper)?
            .into(),
        link: 0,
        info: 0,
        align: 4,
        entsize: 0,
    });

    // The DWARF sections.
    for section in sections.iter() {
        let name = sections
            .section_name(e, section)
            .map_err(obj::ObjectCrateErrorWrapper)?;
        if name.starts_with(b".debug_") && section.sh_type(e) == SHT_PROGBITS {
            out_sections.push(Section {
                name,
                sh_type: SHT_PROGBITS,
                flags: SectionFlags(0),
                addr: 0,
                data: section
                    .data(e, bytes)
                    .map_err(obj::ObjectCrateErrorWrapper)?
                    .into(),
                link: 0,
                info: 0,
                align: 1,
                entsize: 0,
            });
        }
    }

    // The function symbols, with absolute addresses.
    let symbols = sections
        .symbols(e, bytes, SHT_SYMTAB)
        .map_err(obj::ObjectCrateErrorWrapper)?;
    let mut strtab = vec![0u8];
    let mut symtab = vec![0u8; size_of::<Sym32<LittleEndian>>()];
    for sym in symbols.iter() {
        if sym.st_type() != STT_FUNC || usize::from(sym.st_shndx(e).0) != text_index.0 {
            continue;
        }
        let name = symbols
            .symbol_name(e, sym)
            .map_err(obj::ObjectCrateErrorWrapper)?;
        let out = Sym32::<LittleEndian> {
            st_name: U32::new(e, u32::try_from(strtab.len())?),
            st_value: U32::new(e, base.wrapping_add(u32::try_from(sym.st_value(e))?)),
            st_size: U32::new(e, u32::try_from(sym.st_size(e))?),
            st_info: sym.st_info(),
            st_other: sym.st_other(),
            st_shndx: object::U16::new(e, SymbolSection(1)),
        };
        strtab.extend_from_slice(name);
        strtab.push(0);
        symtab.extend_from_slice(object::bytes_of(&out));
    }
    let symtab_index = out_sections.len() + 1;
    let num_symbols = symtab.len() / size_of::<Sym32<LittleEndian>>();
    out_sections.push(Section {
        name: b".symtab",
        sh_type: SHT_SYMTAB,
        flags: SectionFlags(0),
        addr: 0,
        data: symtab.into(),
        link: u32::try_from(symtab_index + 1)?,
        // All symbols are local.
        info: u32::try_from(num_symbols)?,
        align: 4,
        entsize: u32::try_from(size_of::<Sym32<LittleEndian>>())?,
    });
    out_sections.push(Section {
        name: b".strtab",
        sh_type: SHT_STRTAB,
        flags: SectionFlags(0),
        addr: 0,
        data: strtab.into(),
        link: 0,
        info: 0,
        align: 1,
        entsize: 0,
    });

    // Section names.
    let mut shstrtab = vec![0u8];
    let mut name_offsets = vec![];
    for section in &out_sections {
        name_offsets.push(u32::try_from(shstrtab.len())?);
        shstrtab.extend_from_slice(section.name);
        shstrtab.push(0);
    }
    name_offsets.push(u32::try_from(shstrtab.len())?);
    shstrtab.extend_from_slice(b".shstrtab\0");
    out_sections.push(Section {
        name: b".shstrtab",
        sh_type: SHT_STRTAB,
        flags: SectionFlags(0),
        addr: 0,
        data: shstrtab.into(),
        link: 0,
        info: 0,
        align: 1,
        entsize: 0,
    });

    // Layout: file header, program header, section data, section headers.
    let ehdr_size = size_of::<FileHeader32<LittleEndian>>();
    let phdr_size = size_of::<ProgramHeader32<LittleEndian>>();
    let shdr_size = size_of::<SectionHeader32<LittleEndian>>();
    let mut out = vec![0u8; ehdr_size + phdr_size];
    let mut offsets = vec![];
    for section in &out_sections {
        let align = usize::try_from(section.align.max(1))?;
        out.resize(out.len().next_multiple_of(align), 0);
        offsets.push(u32::try_from(out.len())?);
        out.extend_from_slice(&section.data);
    }
    out.resize(out.len().next_multiple_of(4), 0);
    let shoff = u32::try_from(out.len())?;
    out.extend_from_slice(&[0u8; size_of::<SectionHeader32<LittleEndian>>()]);
    for (i, section) in out_sections.iter().enumerate() {
        let shdr = SectionHeader32::<LittleEndian> {
            sh_name: U32::new(e, name_offsets[i]),
            sh_type: U32::new(e, section.sh_type),
            sh_flags: U32::new_u64(e, section.flags)?,
            sh_addr: U32::new(e, section.addr),
            sh_offset: U32::new(e, offsets[i]),
            sh_size: U32::new(e, u32::try_from(section.data.len())?),
            sh_link: U32::new(e, section.link),
            sh_info: U32::new(e, section.info),
            sh_addralign: U32::new(e, section.align),
            sh_entsize: U32::new(e, section.entsize),
        };
        out.extend_from_slice(object::bytes_of(&shdr));
    }

    let ehdr = FileHeader32::<LittleEndian> {
        e_ident: Ident {
            magic: ELFMAG,
            class: ELFCLASS32,
            data: ELFDATA2LSB,
            version: EV_CURRENT,
            os_abi: ELFOSABI_NONE,
            abi_version: 0,
            padding: [0; 7],
        },
        e_type: object::U16::new(e, ET_DYN),
        e_machine: object::U16::new(e, EM_ARM),
        e_version: U32::new(e, u32::from(EV_CURRENT.0)),
        e_entry: U32::new(e, 0),
        e_phoff: U32::new(e, u32::try_from(ehdr_size)?),
        e_shoff: U32::new(e, shoff),
        e_flags: U32::new(e, EF_ARM_EABI_VER5 | FileFlags(EF_ARM_ABI_FLOAT_HARD)),
        e_ehsize: object::U16::new(e, u16::try_from(ehdr_size)?),
        e_phentsize: object::U16::new(e, u16::try_from(phdr_size)?),
        e_phnum: object::U16::new(e, 1),
        e_shentsize: object::U16::new(e, u16::try_from(shdr_size)?),
        e_shnum: object::U16::new(e, u16::try_from(out_sections.len() + 1)?),
        e_shstrndx: object::U16::new(e, SymbolSection(u16::try_from(out_sections.len())?)),
    };
    out[..ehdr_size].copy_from_slice(object::bytes_of(&ehdr));
    let phdr = ProgramHeader32::<LittleEndian> {
        p_type: U32::new(e, PT_LOAD),
        p_offset: U32::new(e, offsets[0]),
        p_vaddr: U32::new(e, base),
        p_paddr: U32::new(e, base),
        p_filesz: U32::new(e, u32::try_from(out_sections[0].data.len())?),
        p_memsz: U32::new(e, code_len),
        p_flags: U32::new(e, PF_R | PF_X),
        p_align: U32::new(e, 4),
    };
    out[ehdr_size..][..phdr_size].copy_from_slice(object::bytes_of(&phdr));
    Ok(out)
}

fn ensure_supported_elf_format(bytes: &[u8]) -> Result<Endianness, Error> {
    use object::elf::*;
    use object::read::elf::*;

    let kind = match object::FileKind::parse(bytes) {
        Ok(file) => file,
        Err(err) => {
            bail!("Failed to parse file: {err}");
        }
    };
    let header = match kind {
        object::FileKind::Elf64 => match object::elf::FileHeader64::<Endianness>::parse(bytes) {
            Ok(header) => header,
            Err(err) => {
                bail!("Unsupported ELF file: {err}");
            }
        },
        _ => {
            bail!("only 64-bit ELF files currently supported")
        }
    };
    let e = header.endian().unwrap();

    match header.e_machine.get(e) {
        EM_AARCH64 => (),
        EM_X86_64 => (),
        EM_S390 => (),
        EM_RISCV => (),
        machine => {
            bail!("Unsupported ELF target machine: {machine:x}");
        }
    }
    ensure!(
        header.e_phoff.get(e) == 0 && header.e_phnum.get(e) == 0,
        "program header table is empty"
    );
    let e_shentsize = header.e_shentsize.get(e);
    let req_shentsize = match e {
        Endianness::Little => size_of::<SectionHeader64<LittleEndian>>(),
        Endianness::Big => size_of::<SectionHeader64<BigEndian>>(),
    };
    ensure!(e_shentsize as usize == req_shentsize, "size of sh");
    Ok(e)
}

fn convert_object_elf_to_loadable_file<E: Endian>(
    bytes: &mut Vec<u8>,
    code_region: (*const u8, usize),
) {
    let e = E::default();

    let header = FileHeader64::<E>::parse(&bytes[..]).unwrap();
    let sections = header.sections(e, &bytes[..]).unwrap();
    let text_range = match sections.section_by_name(e, b".text") {
        Some((i, text)) => {
            let range = text.file_range(e);
            let e_shoff = usize::try_from(header.e_shoff.get(e)).unwrap();
            let off = e_shoff + i.0 * header.e_shentsize.get(e) as usize;

            let section: &mut SectionHeader64<E> =
                object::from_bytes_mut(&mut bytes[off..]).unwrap().0;
            // Patch vaddr, and save file location and its size.
            section.sh_addr.set(e, code_region.0 as u64);
            range
        }
        None => None,
    };

    // LLDB wants segment with virtual address set, placing them at the end of ELF.
    let ph_off = bytes.len();
    let e_phentsize = size_of::<ProgramHeader64<E>>();
    let e_phnum = 1;
    bytes.resize(ph_off + e_phentsize * e_phnum, 0);
    if let Some((sh_offset, sh_size)) = text_range {
        let (v_offset, size) = code_region;
        let program: &mut ProgramHeader64<E> =
            object::from_bytes_mut(&mut bytes[ph_off..]).unwrap().0;
        program.p_type.set(e, PT_LOAD);
        program.p_offset.set(e, sh_offset);
        program.p_vaddr.set(e, v_offset as u64);
        program.p_paddr.set(e, v_offset as u64);
        program.p_filesz.set(e, sh_size);
        program.p_memsz.set(e, size as u64);
    } else {
        unreachable!();
    }

    // It is somewhat loadable ELF file at this moment.
    let header: &mut FileHeader64<E> = object::from_bytes_mut(bytes).unwrap().0;
    header.e_type.set(e, ET_DYN);
    header.e_phoff.set(e, ph_off as u64);
    header
        .e_phentsize
        .set(e, u16::try_from(e_phentsize).unwrap());
    header.e_phnum.set(e, u16::try_from(e_phnum).unwrap());
}

#[cfg(all(test, feature = "cranelift", feature = "all-arch"))]
mod tests {
    use super::*;
    use crate::{Config, Engine};

    #[test]
    fn arm32_image() -> Result<()> {
        let mut config = Config::new();
        config.target("armv7-unknown-linux-gnueabihf")?;
        config.debug_info(true);
        unsafe {
            config.cranelift_flag_enable("has_neon");
        }
        let engine = Engine::new(&config)?;
        let wat = r#"(module (func (export "f") (param i32) (result i32)
            local.get 0 i32.const 1 i32.add))"#;
        let bytes = engine.precompile_module(wat.as_bytes())?;
        let image = create_gdbjit_image(bytes, (0x1000_0000 as *const u8, 0x1000))?;
        if let Ok(path) = std::env::var("WASMTIME_TEST_DUMP_GDBJIT_IMAGE") {
            std::fs::write(path, &image)?;
        }
        let file = object::File::parse(&image[..]).map_err(obj::ObjectCrateErrorWrapper)?;
        assert_eq!(file.architecture(), object::Architecture::Arm);
        assert!(!file.is_64());
        let text = file.section_by_name(".text").unwrap();
        assert_eq!(text.address(), 0x1000_0000);
        assert!(file.section_by_name(".debug_info").is_some());
        let sym = file
            .symbols()
            .find(|s| s.kind() == SymbolKind::Text)
            .unwrap();
        assert_eq!(sym.address(), 0x1000_0000);
        Ok(())
    }
}
