//===- bolt/Rewrite/PECOFFRewriteInstance.cpp - PE/COFF rewriter ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Rewrite/PECOFFRewriteInstance.h"
#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryEmitter.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Core/JumpTable.h"
#include "bolt/Core/MCPlusBuilder.h"
#include "bolt/Passes/BinaryPasses.h"
#include "bolt/Profile/DataReader.h"
#include "bolt/Profile/ETWDataAggregator.h"
#include "bolt/Rewrite/BinaryPassManager.h"
#include "bolt/Rewrite/PDBRewriter.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "bolt/Utils/Utils.h"
#include "llvm/BinaryFormat/COFF.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
#include <memory>

#define DEBUG_TYPE "bolt"

namespace opts {

using namespace llvm;
extern cl::opt<unsigned> AlignText;
extern cl::opt<bool> AggregateOnly;
extern cl::opt<bool> ForcePatch;
extern cl::opt<bolt::JumpTableSupportLevel> JumpTables;
extern cl::opt<bool> KeepTmp;
extern cl::opt<bool> Lite;
extern cl::opt<bool> NeverPrint;
extern cl::opt<std::string> OutputFilename;
extern cl::opt<bool> PrintAfterBranchFixup;
extern cl::opt<bool> PrintFinalized;
extern cl::opt<bool> PrintNormalized;
extern cl::opt<bool> PrintReordered;
extern cl::opt<bool> PrintSections;
extern cl::opt<bool> PrintDisasm;
extern cl::opt<bool> PrintCFG;
extern cl::opt<unsigned> Verbosity;

} // namespace opts

namespace llvm {
namespace bolt {

extern MCPlusBuilder *createMCPlusBuilder(const Triple::ArchType Arch,
                                          const MCInstrAnalysis *Analysis,
                                          const MCInstrInfo *Info,
                                          const MCRegisterInfo *RegInfo,
                                          const MCSubtargetInfo *STI);

Expected<std::unique_ptr<PECOFFRewriteInstance>>
PECOFFRewriteInstance::create(object::COFFObjectFile *InputFile,
                              StringRef ToolPath) {
  Error Err = Error::success();
  auto Instance =
      std::make_unique<PECOFFRewriteInstance>(InputFile, ToolPath, Err);
  if (Err)
    return std::move(Err);
  return std::move(Instance);
}

PECOFFRewriteInstance::PECOFFRewriteInstance(object::COFFObjectFile *InputFile,
                                             StringRef ToolPath, Error &Err)
    : InputFile(InputFile), ToolPath(ToolPath) {
  ErrorAsOutParameter EAO(&Err);

  // Build a proper triple for the PE/COFF input.
  // ObjectFile::makeTriple() does not set the object format for COFF AMD64,
  // so we must fix it here to ensure MCContext creates COFF-aware structures.
  Triple TheTriple = InputFile->makeTriple();
  TheTriple.setObjectFormat(Triple::COFF);
  if (TheTriple.getOS() == Triple::UnknownOS)
    TheTriple.setOS(Triple::Win32);
  if (TheTriple.getEnvironment() == Triple::UnknownEnvironment)
    TheTriple.setEnvironment(Triple::MSVC);

  Relocation::Arch = TheTriple.getArch();

  auto BCOrErr = BinaryContext::createBinaryContext(
      TheTriple, std::make_shared<orc::SymbolStringPool>(),
      InputFile->getFileName(), nullptr,
      /* IsPIC */ true, DWARFContext::create(*InputFile),
      {llvm::outs(), llvm::errs()});
  if (Error E = BCOrErr.takeError()) {
    Err = std::move(E);
    return;
  }
  BC = std::move(BCOrErr.get());
  BC->initializeTarget(std::unique_ptr<MCPlusBuilder>(
      createMCPlusBuilder(BC->TheTriple->getArch(), BC->MIA.get(),
                          BC->MII.get(), BC->MRI.get(), BC->STI.get())));
}

PECOFFRewriteInstance::~PECOFFRewriteInstance() {}

Error PECOFFRewriteInstance::setProfile(StringRef Filename) {
  if (!sys::fs::exists(Filename))
    return errorCodeToError(make_error_code(errc::no_such_file_or_directory));

  if (ProfileReader) {
    return make_error<StringError>(Twine("multiple profiles specified: ") +
                                       ProfileReader->getFilename() + " and " +
                                       Filename,
                                   inconvertibleErrorCode());
  }

  // Choose the right reader based on the file type, same pattern as
  // RewriteInstance::setProfile() for ELF.
  if (ETWDataAggregator::checkETLMagic(Filename))
    ProfileReader = std::make_unique<ETWDataAggregator>(Filename);
  else
    ProfileReader = std::make_unique<DataReader>(Filename);

  return Error::success();
}

void PECOFFRewriteInstance::preprocessProfileData() {
  if (!ProfileReader)
    return;
  if (Error E = ProfileReader->preprocessProfile(*BC))
    report_error("cannot pre-process profile", std::move(E));
}

void PECOFFRewriteInstance::processProfileDataPreCFG() {
  if (!ProfileReader)
    return;
  if (Error E = ProfileReader->readProfilePreCFG(*BC))
    report_error("cannot read profile pre-CFG", std::move(E));
}

void PECOFFRewriteInstance::processProfileData() {
  if (!ProfileReader)
    return;
  if (Error E = ProfileReader->readProfile(*BC))
    report_error("cannot read profile", std::move(E));
}

void PECOFFRewriteInstance::adjustCommandLineOptions() {
  opts::ForcePatch = true;

  // Move jump tables into the emitted object so they get re-emitted with
  // correct entries after block reordering.  Same as MachO.
  opts::JumpTables = JTS_MOVE;

  // Lite mode skips cold functions which does not work well for PE/COFF
  // in-place patching.  Force full processing.
  if (!opts::Lite.getNumOccurrences())
    opts::Lite = false;

  // PE section alignment is typically 4KB (0x1000).
  BC->PageAlign = 0x1000;

  if (!opts::AlignText.getNumOccurrences())
    opts::AlignText = BC->PageAlign;
}

// PE/COFF uses its own pass pipeline (not BinaryPassManager) to avoid
// passes that change instruction sizes.  ShortenInstructions and RemoveNops
// must NOT be registered here because they alter byte offsets within
// functions, invalidating UNWIND_INFO prolog sizes and unwind code offsets
// in .xdata.  Unlike ELF where DWARF CFI is regenerated, Windows unwind
// data is preserved byte-for-byte.

void PECOFFRewriteInstance::readSpecialSections() {
  for (const object::SectionRef &Section : InputFile->sections()) {
    Expected<StringRef> SectionName = Section.getName();
    check_error(SectionName.takeError(), "cannot get section name");
    if (!SectionName->empty()) {
      BC->registerSection(Section);
      LLVM_DEBUG(
          dbgs() << "BOLT-DEBUG: registering section " << *SectionName
                 << " @ 0x" << Twine::utohexstr(Section.getAddress()) << ":0x"
                 << Twine::utohexstr(Section.getAddress() + Section.getSize())
                 << "\n");
    }
  }

  if (opts::PrintSections) {
    outs() << "BOLT-INFO: Sections from original binary:\n";
    BC->printSections(outs());
  }
}

void PECOFFRewriteInstance::readExceptionHandling() {
  const object::coff_section *PDataSec = nullptr;
  const object::coff_section *XDataSec = nullptr;

  for (const object::SectionRef &Section : InputFile->sections()) {
    Expected<StringRef> NameOrErr = Section.getName();
    if (!NameOrErr)
      continue;
    if (*NameOrErr == ".pdata")
      PDataSec = InputFile->getCOFFSection(Section);
    else if (*NameOrErr == ".xdata")
      XDataSec = InputFile->getCOFFSection(Section);
  }

  if (!PDataSec) {
    outs() << "BOLT-WARNING: no .pdata section found\n";
    return;
  }

  // Get .xdata or .rdata contents for UNWIND_INFO parsing.
  // Unwind data may reside in a dedicated .xdata section or in .rdata.
  ArrayRef<uint8_t> XDataContents;
  uint64_t XDataRVA = 0;
  if (XDataSec) {
    if (Error E = InputFile->getSectionContents(XDataSec, XDataContents))
      consumeError(std::move(E));
    else
      XDataRVA = XDataSec->VirtualAddress;
  }
  // Fall back to any section that contains the unwind RVA from the first
  // .pdata entry.  Many PE binaries store UNWIND_INFO in .rdata.
  if (XDataContents.empty()) {
    ArrayRef<uint8_t> PDataPeek;
    if (Error E = InputFile->getSectionContents(PDataSec, PDataPeek))
      consumeError(std::move(E));
    if (PDataPeek.size() >= 12) {
      uint32_t FirstUnwindRVA = support::endian::read32le(PDataPeek.data() + 8);
      for (const object::SectionRef &Section : InputFile->sections()) {
        const object::coff_section *CS = InputFile->getCOFFSection(Section);
        uint32_t SecStart = CS->VirtualAddress;
        uint32_t SecEnd = SecStart + CS->VirtualSize;
        if (FirstUnwindRVA >= SecStart && FirstUnwindRVA < SecEnd) {
          if (Error E = InputFile->getSectionContents(CS, XDataContents))
            consumeError(std::move(E));
          else
            XDataRVA = CS->VirtualAddress;
          break;
        }
      }
    }
  }

  // Get .pdata contents: array of RUNTIME_FUNCTION entries (12 bytes each)
  ArrayRef<uint8_t> PDataContents;
  if (Error E = InputFile->getSectionContents(PDataSec, PDataContents)) {
    consumeError(std::move(E));
    outs() << "BOLT-WARNING: cannot read .pdata section\n";
    return;
  }

  struct RuntimeFunction {
    support::ulittle32_t BeginAddress;
    support::ulittle32_t EndAddress;
    support::ulittle32_t UnwindInfoAddress;
  };

  size_t NumEntries = PDataContents.size() / sizeof(RuntimeFunction);
  auto *Entries =
      reinterpret_cast<const RuntimeFunction *>(PDataContents.data());

  // First pass: parse UNWIND_INFO and detect chained entries
  std::map<uint32_t, uint32_t>
      ChainToParent; // chained begin RVA -> parent begin RVA

  for (size_t I = 0; I < NumEntries; ++I) {
    uint32_t BeginRVA = Entries[I].BeginAddress;
    uint32_t EndRVA = Entries[I].EndAddress;
    uint32_t UnwindRVA = Entries[I].UnwindInfoAddress;

    if (BeginRVA == 0 && EndRVA == 0)
      continue;

    SEHUnwindInfo Info;

    // Parse UNWIND_INFO from .xdata if available
    if (!XDataContents.empty() && UnwindRVA >= XDataRVA) {
      uint32_t Offset = UnwindRVA - XDataRVA;
      if (Offset + 4 <= XDataContents.size()) {
        const uint8_t *UW = XDataContents.data() + Offset;
        Info.Version = UW[0] & 0x7;
        Info.Flags = (UW[0] >> 3) & 0x1F;
        Info.PrologSize = UW[1];
        uint8_t CountOfCodes = UW[2];
        Info.FrameRegister = UW[3] & 0xF;
        Info.FrameOffset = (UW[3] >> 4) & 0xF;

        // Read unwind codes
        uint32_t CodesOffset = Offset + 4;
        for (uint8_t C = 0;
             C < CountOfCodes && CodesOffset + 2 <= XDataContents.size(); ++C) {
          uint16_t Code =
              support::endian::read16le(XDataContents.data() + CodesOffset);
          Info.UnwindCodes.push_back(Code);
          CodesOffset += 2;
        }

        // Aligned offset after unwind codes (must be 4-byte aligned)
        uint32_t HandlerDataOffset = CodesOffset;
        if (CountOfCodes % 2 != 0)
          HandlerDataOffset += 2;

        // Check for exception handler
        const uint8_t UNW_FLAG_EHANDLER = 0x01;
        const uint8_t UNW_FLAG_UHANDLER = 0x02;
        const uint8_t UNW_FLAG_CHAININFO = 0x04;

        if (Info.Flags & UNW_FLAG_CHAININFO) {
          Info.IsChained = true;
          // Chained RUNTIME_FUNCTION follows the unwind codes
          if (HandlerDataOffset + 12 <= XDataContents.size()) {
            auto *Chain = reinterpret_cast<const RuntimeFunction *>(
                XDataContents.data() + HandlerDataOffset);
            Info.ChainedBeginRVA = Chain->BeginAddress;
            Info.ChainedEndRVA = Chain->EndAddress;
            Info.ChainedUnwindRVA = Chain->UnwindInfoAddress;
            ChainToParent[BeginRVA] = Chain->BeginAddress;
          }
        } else if (Info.Flags & (UNW_FLAG_EHANDLER | UNW_FLAG_UHANDLER)) {
          Info.HasExceptionHandler = true;
          if (HandlerDataOffset + 4 <= XDataContents.size()) {
            Info.ExceptionHandlerRVA = support::endian::read32le(
                XDataContents.data() + HandlerDataOffset);
          }
        }
      }
    }

    FunctionSEHInfo[BeginRVA] = std::move(Info);
  }

  // Second pass: resolve chains to root parents and extend parent ranges
  for (auto &[ChainedRVA, ParentRVA] : ChainToParent) {
    // Walk chain to find root parent
    uint32_t Root = ParentRVA;
    while (ChainToParent.count(Root))
      Root = ChainToParent[Root];
    ChainToParent[ChainedRVA] = Root;
  }

  outs() << "BOLT-INFO: parsed " << FunctionSEHInfo.size()
         << " .pdata entries, " << ChainToParent.size() << " chained\n";
}

void PECOFFRewriteInstance::discoverFileObjects() {
  uint64_t ImageBase = InputFile->getImageBase();

  // Build a symbol name map from the COFF symbol table
  StringMap<uint64_t> SymbolAddresses;
  std::map<uint64_t, StringRef> AddressToName;

  for (const object::SymbolRef &Symbol : InputFile->symbols()) {
    Expected<object::SymbolRef::Type> TypeOrErr = Symbol.getType();
    if (!TypeOrErr) {
      consumeError(TypeOrErr.takeError());
      continue;
    }
    if (*TypeOrErr != object::SymbolRef::ST_Function)
      continue;

    Expected<uint64_t> AddressOrErr = Symbol.getAddress();
    if (!AddressOrErr) {
      consumeError(AddressOrErr.takeError());
      continue;
    }
    Expected<StringRef> NameOrErr = Symbol.getName();
    if (!NameOrErr) {
      consumeError(NameOrErr.takeError());
      continue;
    }
    AddressToName[*AddressOrErr] = *NameOrErr;
  }

  // Find .pdata section to enumerate functions
  const object::coff_section *PDataSec = nullptr;
  for (const object::SectionRef &Section : InputFile->sections()) {
    Expected<StringRef> NameOrErr = Section.getName();
    if (NameOrErr && *NameOrErr == ".pdata")
      PDataSec = InputFile->getCOFFSection(Section);
  }

  if (!PDataSec) {
    outs() << "BOLT-WARNING: no .pdata section, cannot discover functions\n";
    return;
  }

  ArrayRef<uint8_t> PDataContents;
  if (Error E = InputFile->getSectionContents(PDataSec, PDataContents)) {
    consumeError(std::move(E));
    return;
  }

  struct RuntimeFunction {
    support::ulittle32_t BeginAddress;
    support::ulittle32_t EndAddress;
    support::ulittle32_t UnwindInfoAddress;
  };

  size_t NumEntries = PDataContents.size() / sizeof(RuntimeFunction);
  auto *Entries =
      reinterpret_cast<const RuntimeFunction *>(PDataContents.data());

  uint64_t FuncsCreated = 0;
  uint64_t FuncsSkippedHandler = 0;

  for (size_t I = 0; I < NumEntries; ++I) {
    uint32_t BeginRVA = Entries[I].BeginAddress;
    uint32_t EndRVA = Entries[I].EndAddress;

    if (BeginRVA == 0 && EndRVA == 0)
      continue;

    // Skip chained entries (they're part of another function)
    auto SEHIt = FunctionSEHInfo.find(BeginRVA);
    if (SEHIt != FunctionSEHInfo.end() && SEHIt->second.IsChained)
      continue;

    uint64_t Address = ImageBase + BeginRVA;
    uint64_t Size = EndRVA - BeginRVA;

    if (Size == 0)
      continue;

    // Find the section containing this function
    ErrorOr<BinarySection &> Section = BC->getSectionForAddress(Address);
    if (!Section)
      continue;

    // Check if function has an exception handler
    bool HasHandler = false;
    if (SEHIt != FunctionSEHInfo.end())
      HasHandler = SEHIt->second.HasExceptionHandler;

    // Generate function name from symbol table or address
    std::string FuncName;
    auto NameIt = AddressToName.find(Address);
    if (NameIt != AddressToName.end())
      FuncName = NameIt->second.str();
    else
      FuncName = ("func_0x" + Twine::utohexstr(Address)).str();

    BinaryFunction *BF =
        BC->createBinaryFunction(FuncName, *Section, Address, Size);
    if (!BF)
      continue;

    BF->setMaxSize(Size);

    // Mark functions with exception handlers as non-simple to skip them
    if (HasHandler) {
      BF->setSimple(false);
      ++FuncsSkippedHandler;
    }

    ++FuncsCreated;
  }

  NumFuncsWithHandlers = FuncsSkippedHandler;

  outs() << "BOLT-INFO: " << FuncsCreated
         << " functions discovered from .pdata\n";
  outs() << "BOLT-INFO: " << FuncsSkippedHandler
         << " functions with exception handlers (skipped)\n";
}

void PECOFFRewriteInstance::disassembleFunctions() {
  uint64_t DisasmCount = 0;
  uint64_t FailCount = 0;
  uint64_t TotalSimple = 0;

  for (auto &BFI : BC->getBinaryFunctions())
    if (BFI.second.isSimple())
      ++TotalSimple;

  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;
    if (!Function.isSimple())
      continue;

    if (Error E = Function.disassemble()) {
      // Non-fatal for PE: skip functions that fail disassembly
      consumeError(std::move(E));
      Function.setSimple(false);
      ++FailCount;
      continue;
    }

    if (opts::PrintDisasm)
      Function.print(outs(), "after disassembly");
    ++DisasmCount;
  }

  outs() << "BOLT-INFO: disassembled " << DisasmCount << " functions";
  if (FailCount)
    outs() << " (" << FailCount << " failed)";
  outs() << "\n";
}

void PECOFFRewriteInstance::buildFunctionsCFG() {
  uint64_t CFGCount = 0;
  uint64_t FailCount = 0;

  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;
    if (!Function.isSimple())
      continue;

    if (Error E = Function.buildCFG(/*AllocId*/ 0)) {
      consumeError(std::move(E));
      Function.setSimple(false);
      ++FailCount;
      continue;
    }
    ++CFGCount;
  }

  outs() << "BOLT-INFO: built CFG for " << CFGCount << " functions";
  if (FailCount)
    outs() << " (" << FailCount << " failed)";
  outs() << "\n";
}

void PECOFFRewriteInstance::postProcessFunctions() {
  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;
    if (Function.empty())
      continue;
    Function.postProcessCFG();
    if (opts::PrintCFG)
      Function.print(outs(), "after building cfg");
  }
}

void PECOFFRewriteInstance::runOptimizationPasses() {
  BinaryFunctionPassManager Manager(*BC);
  Manager.registerPass(std::make_unique<NormalizeCFG>(opts::PrintNormalized));
  Manager.registerPass(
      std::make_unique<ReorderBasicBlocks>(opts::PrintReordered));
  Manager.registerPass(
      std::make_unique<FixupBranches>(opts::PrintAfterBranchFixup));

  Manager.registerPass(std::make_unique<PopulateOutputFunctions>());

  Manager.registerPass(
      std::make_unique<FinalizeFunctions>(opts::PrintFinalized));

  BC->logBOLTErrorsAndQuitOnFatal(Manager.runPasses());
}

void PECOFFRewriteInstance::emitAndLink() {
  std::error_code EC;
  std::unique_ptr<::llvm::ToolOutputFile> TempOut =
      std::make_unique<::llvm::ToolOutputFile>(opts::OutputFilename + ".bolt.o",
                                               EC, sys::fs::OF_None);
  check_error(EC, "cannot create output object file");

  if (opts::KeepTmp)
    TempOut->keep();

  std::unique_ptr<buffer_ostream> BOS =
      std::make_unique<buffer_ostream>(TempOut->os());
  raw_pwrite_stream *OS = BOS.get();
  auto Streamer = BC->createStreamer(*OS);

  emitBinaryContext(*Streamer, *BC, getOrgSecPrefix());
  Streamer->finish();

  StringRef ObjContents = BOS->str();
  outs() << "BOLT-INFO: emitted object size = " << ObjContents.size()
         << " bytes\n";

  // We resolve relocations ourselves instead of going through JITLink.
  // JITLink was designed for small JIT objects and has O(n^2) algorithms
  // that hang when processing a COFF object with thousands of sections
  // (one per function).  Since PE/COFF rewriting is in-place and all
  // function addresses are known, we just need the assembled bytes with
  // relocations applied.

  std::unique_ptr<MemoryBuffer> ObjBuf =
      MemoryBuffer::getMemBuffer(ObjContents, "bolt-coff-object", false);

  Expected<std::unique_ptr<object::ObjectFile>> ObjOrErr =
      object::ObjectFile::createObjectFile(ObjBuf->getMemBufferRef());
  if (!ObjOrErr) {
    errs() << "BOLT-ERROR: cannot parse emitted object: "
           << toString(ObjOrErr.takeError()) << "\n";
    exit(1);
  }

  auto *Obj = cast<object::COFFObjectFile>(ObjOrErr->get());

  // Map section names to the BinaryFunction they belong to.  The MC emitter
  // creates one section per function, named like ".l.text.<id>".
  StringMap<BinaryFunction *> SectionToFunc;
  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &BF = BFI.second;
    if (!BF.isEmitted())
      continue;
    SmallString<32> SecName = BF.getCodeSectionName();
    SectionToFunc[SecName] = &BF;
  }

  // Map section names to their original virtual addresses.  Defined symbols
  // in the COFF object reference these sections, so we need the VA to compute
  // the final symbol address (section_VA + offset_in_section).
  StringMap<uint64_t> SectionNameToVA;
  for (const auto &Sec : Obj->sections()) {
    Expected<StringRef> NameOrErr = Sec.getName();
    if (!NameOrErr) {
      consumeError(NameOrErr.takeError());
      continue;
    }
    auto It = SectionToFunc.find(*NameOrErr);
    if (It != SectionToFunc.end())
      SectionNameToVA[*NameOrErr] = It->second->getAddress();
  }

  // Also map jump table data sections.  With JTS_MOVE the emitter writes
  // JT entries into .rdata with symbol references that need resolution.
  // Find .rdata sections in the emitted object and map them to the original
  // JT address so relocations resolve correctly.
  for (const auto &Sec : Obj->sections()) {
    Expected<StringRef> NameOrErr = Sec.getName();
    if (!NameOrErr) {
      consumeError(NameOrErr.takeError());
      continue;
    }
    if (*NameOrErr != ".rdata")
      continue;
    // Find the first JT symbol defined in this section to get the VA.
    object::section_iterator SecIt(Sec);
    for (const auto &Sym : Obj->symbols()) {
      Expected<object::section_iterator> SymSec = Sym.getSection();
      if (!SymSec || *SymSec == Obj->section_end() || **SymSec != *SecIt)
        continue;
      Expected<StringRef> SymName = Sym.getName();
      if (!SymName)
        continue;
      if (const BinaryData *BD = BC->getBinaryDataByName(*SymName)) {
        Expected<uint64_t> SymVal = Sym.getValue();
        uint64_t Offset = SymVal ? *SymVal : 0;
        SectionNameToVA[*NameOrErr] = BD->getAddress() - Offset;
        break;
      }
    }
  }

  // Process each function section: copy its bytes and resolve relocations.
  uint64_t ResolvedCount = 0;
  for (const auto &Sec : Obj->sections()) {
    Expected<StringRef> NameOrErr = Sec.getName();
    if (!NameOrErr) {
      consumeError(NameOrErr.takeError());
      continue;
    }

    auto FuncIt = SectionToFunc.find(*NameOrErr);
    if (FuncIt == SectionToFunc.end())
      continue;

    BinaryFunction *BF = FuncIt->second;
    uint64_t SectionVA = BF->getAddress();

    Expected<StringRef> ContentsOrErr = Sec.getContents();
    if (!ContentsOrErr) {
      errs() << "BOLT-WARNING: cannot read section " << *NameOrErr << "\n";
      consumeError(ContentsOrErr.takeError());
      continue;
    }

    // Make a writable copy so we can patch in the resolved relocations.
    ResolvedFunctionBytes.emplace_back(ContentsOrErr->begin(),
                                       ContentsOrErr->end());
    auto &Buffer = ResolvedFunctionBytes.back();
    MutableArrayRef<uint8_t> Data(Buffer);

    for (const auto &Rel : Sec.relocations()) {
      uint64_t SymVA = resolveRelocSymbol(Obj, Rel, SectionNameToVA);
      applyCOFFRelocation(Data, SectionVA, Rel, SymVA);
    }

    BF->setImageAddress(reinterpret_cast<uint64_t>(Data.data()));
    BF->setImageSize(Data.size());
    ++ResolvedCount;
  }

  // Process jump table data sections (.rdata).  With JTS_MOVE the emitter
  // writes JT entries with relocations pointing to BB symbols.  We resolve
  // them the same way as code relocations and store the result so
  // rewriteFile() can pwrite them back.
  for (const auto &Sec : Obj->sections()) {
    Expected<StringRef> NameOrErr = Sec.getName();
    if (!NameOrErr) {
      consumeError(NameOrErr.takeError());
      continue;
    }
    if (*NameOrErr != ".rdata")
      continue;
    auto VAIt = SectionNameToVA.find(*NameOrErr);
    if (VAIt == SectionNameToVA.end())
      continue;

    Expected<StringRef> ContentsOrErr = Sec.getContents();
    if (!ContentsOrErr) {
      consumeError(ContentsOrErr.takeError());
      continue;
    }

    if (Sec.relocations().empty())
      continue;

    uint64_t SectionVA = VAIt->second;
    ResolvedFunctionBytes.emplace_back(ContentsOrErr->begin(),
                                       ContentsOrErr->end());
    auto &Buffer = ResolvedFunctionBytes.back();
    MutableArrayRef<uint8_t> Data(Buffer);

    for (const auto &Rel : Sec.relocations()) {
      uint64_t SymVA = resolveRelocSymbol(Obj, Rel, SectionNameToVA);
      applyCOFFRelocation(Data, SectionVA, Rel, SymVA);
    }

    // Store resolved JT data for rewriteFile() to pwrite back.
    ResolvedJTData.push_back({SectionVA, Data.data(), Data.size()});
    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: resolved " << Sec.relocations().end() - Sec.relocations().begin()
                      << " JT relocations in .rdata at VA 0x"
                      << Twine::utohexstr(SectionVA) << "\n");
  }

  outs() << "BOLT-INFO: resolved relocations for " << ResolvedCount
         << " functions\n";
}

uint64_t PECOFFRewriteInstance::resolveRelocSymbol(
    const object::COFFObjectFile *Obj, const object::RelocationRef &Rel,
    const StringMap<uint64_t> &SectionNameToVA) {
  object::symbol_iterator SI = Rel.getSymbol();
  if (SI == Obj->symbol_end()) {
    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: relocation at offset " << Rel.getOffset()
                      << " has no symbol\n");
    return 0;
  }

  object::SymbolRef Sym = *SI;

  // Defined symbol in the emitted object -- its address is the section's
  // original VA plus the symbol's offset within that section.
  Expected<object::section_iterator> SecOrErr = Sym.getSection();
  if (SecOrErr && *SecOrErr != Obj->section_end()) {
    Expected<StringRef> SecName = (*SecOrErr)->getName();
    if (SecName) {
      auto It = SectionNameToVA.find(*SecName);
      if (It != SectionNameToVA.end()) {
        Expected<uint64_t> ValOrErr = Sym.getValue();
        uint64_t Offset = ValOrErr ? *ValOrErr : 0;
        LLVM_DEBUG({
          dbgs() << "BOLT-DEBUG: resolved defined symbol in section "
                 << *SecName << " at VA 0x"
                 << Twine::utohexstr(It->second + Offset) << "\n";
        });
        return It->second + Offset;
      }
    }
  }

  // External symbol -- look it up by name in BinaryContext.
  Expected<StringRef> NameOrErr = Sym.getName();
  if (!NameOrErr) {
    consumeError(NameOrErr.takeError());
    return 0;
  }
  StringRef Name = *NameOrErr;

  if (const BinaryData *BD = BC->getBinaryDataByName(Name)) {
    uint64_t Addr = BD->isMoved() && !BD->isJumpTable() ? BD->getOutputAddress()
                                                        : BD->getAddress();
    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: resolved " << Name << " via BinaryData"
                      << " at 0x" << Twine::utohexstr(Addr) << "\n");
    return Addr;
  }

  // BOLT creates symbols like FUNCat0x<addr> and DATAat0x<addr> for
  // references into the original binary.  Parse the embedded address.
  size_t HexPos = Name.find("0x");
  if (HexPos != StringRef::npos) {
    uint64_t Addr = 0;
    if (!Name.substr(HexPos + 2).getAsInteger(16, Addr) && Addr != 0) {
      LLVM_DEBUG(dbgs() << "BOLT-DEBUG: parsed address 0x"
                        << Twine::utohexstr(Addr) << " from " << Name << "\n");
      return Addr;
    }
  }

  LLVM_DEBUG(dbgs() << "BOLT-DEBUG: unresolved symbol " << Name << "\n");
  return 0;
}

void PECOFFRewriteInstance::applyCOFFRelocation(
    MutableArrayRef<uint8_t> Data, uint64_t SectionVA,
    const object::RelocationRef &Rel, uint64_t SymVA) {
  uint64_t Offset = Rel.getOffset();

  switch (Rel.getType()) {
  case COFF::IMAGE_REL_AMD64_REL32: {
    if (Offset + 4 > Data.size()) {
      LLVM_DEBUG(dbgs() << "BOLT-DEBUG: REL32 at offset " << Offset
                        << " overflows section of size " << Data.size()
                        << "\n");
      return;
    }
    int32_t Existing = support::endian::read32le(&Data[Offset]);
    uint64_t RelocVA = SectionVA + Offset;
    int32_t Value = static_cast<int32_t>(SymVA - RelocVA - 4) + Existing;
    support::endian::write32le(&Data[Offset], Value);
    break;
  }
  case COFF::IMAGE_REL_AMD64_REL32_1:
  case COFF::IMAGE_REL_AMD64_REL32_2:
  case COFF::IMAGE_REL_AMD64_REL32_3:
  case COFF::IMAGE_REL_AMD64_REL32_4:
  case COFF::IMAGE_REL_AMD64_REL32_5: {
    if (Offset + 4 > Data.size())
      return;
    // REL32_N subtracts an extra N bytes from the displacement.
    unsigned Extra = Rel.getType() - COFF::IMAGE_REL_AMD64_REL32;
    int32_t Existing = support::endian::read32le(&Data[Offset]);
    uint64_t RelocVA = SectionVA + Offset;
    int32_t Value =
        static_cast<int32_t>(SymVA - RelocVA - 4 - Extra) + Existing;
    support::endian::write32le(&Data[Offset], Value);
    break;
  }
  case COFF::IMAGE_REL_AMD64_ADDR64: {
    if (Offset + 8 > Data.size())
      return;
    int64_t Existing = support::endian::read64le(&Data[Offset]);
    support::endian::write64le(&Data[Offset], SymVA + Existing);
    break;
  }
  case COFF::IMAGE_REL_AMD64_ADDR32NB: {
    // Image-base-relative 32-bit address.
    if (Offset + 4 > Data.size())
      return;
    int32_t Existing = support::endian::read32le(&Data[Offset]);
    uint64_t ImageBase = InputFile->getImageBase();
    int32_t Value = static_cast<int32_t>(SymVA - ImageBase) + Existing;
    support::endian::write32le(&Data[Offset], Value);
    break;
  }
  case COFF::IMAGE_REL_AMD64_ADDR32: {
    if (Offset + 4 > Data.size())
      return;
    int32_t Existing = support::endian::read32le(&Data[Offset]);
    int32_t Value = static_cast<int32_t>(SymVA) + Existing;
    support::endian::write32le(&Data[Offset], Value);
    break;
  }
  case COFF::IMAGE_REL_AMD64_SECTION:
  case COFF::IMAGE_REL_AMD64_SECREL:
    // Debug info relocations -- not relevant for code patching.
    break;
  default:
    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: unhandled COFF relocation type "
                      << Rel.getType() << " at offset " << Offset << "\n");
    break;
  }
}

void PECOFFRewriteInstance::rewriteFile() {
  std::error_code EC;
  Out = std::make_unique<ToolOutputFile>(opts::OutputFilename, EC,
                                         sys::fs::OF_None);
  check_error(EC, "cannot create output executable file");
  raw_fd_ostream &OS = Out->os();

  // Start with a full copy of the original PE.  We patch individual function
  // bodies below, leaving headers, imports, relocations etc. untouched.
  OS << InputFile->getData();

  // We need to translate virtual addresses to file offsets.  PE sections have
  // a VirtualAddress and a PointerToRawData that together define the mapping.
  uint64_t ImageBase = InputFile->getImageBase();
  struct SectionLayout {
    uint32_t VA;
    uint32_t Size;
    uint32_t FileOffset;
  };
  SmallVector<SectionLayout, 8> SectionMap;

  for (const object::SectionRef &Section : InputFile->sections()) {
    const object::coff_section *CS = InputFile->getCOFFSection(Section);
    SectionMap.push_back(
        {CS->VirtualAddress, CS->VirtualSize, CS->PointerToRawData});
  }

  auto VAToFileOffset = [&](uint64_t VA) -> std::optional<uint64_t> {
    uint32_t RVA = VA - ImageBase;
    for (const auto &S : SectionMap) {
      if (RVA >= S.VA && RVA < S.VA + S.Size)
        return S.FileOffset + (RVA - S.VA);
    }
    return std::nullopt;
  };

  uint64_t RewrittenCount = 0;
  uint64_t OverflowCount = 0;

  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;
    if (!Function.isSimple())
      continue;
    if (!Function.isEmitted())
      continue;

    uint64_t EmittedSize = Function.getImageSize();
    uint64_t OriginalSize = Function.getMaxSize();

    if (EmittedSize > OriginalSize) {
      ++OverflowCount;
      continue;
    }

    auto FileOff = VAToFileOffset(Function.getAddress());
    if (!FileOff) {
      if (opts::Verbosity >= 1)
        outs() << "BOLT-WARNING: cannot map address 0x"
               << Twine::utohexstr(Function.getAddress())
               << " to file offset for \"" << Function << "\", skipping\n";
      continue;
    }

    // Compare emitted bytes with the original.  The MC assembler often
    // produces different encodings (e.g. dropping redundant REX prefixes)
    // even when the optimization passes did not change the layout.  Writing
    // those re-encoded bytes back would corrupt the UNWIND_INFO byte offsets
    // in .xdata and the base relocation entries in .reloc, so we only patch
    // functions whose layout was actually modified by the passes.
    if (!ModifiedFunctions.count(Function.getAddress())) {
      continue;
    }

    if (opts::Verbosity >= 2)
      outs() << "BOLT: rewriting \"" << Function << "\""
             << " size=" << EmittedSize << "/" << OriginalSize
             << " at file offset 0x" << Twine::utohexstr(*FileOff) << "\n";

    OS.pwrite(reinterpret_cast<char *>(Function.getImageAddress()), EmittedSize,
              *FileOff);

    // Fill leftover space with int3 so stale code traps cleanly.
    if (EmittedSize < OriginalSize) {
      std::vector<uint8_t> Padding(OriginalSize - EmittedSize, 0xCC);
      OS.pwrite(reinterpret_cast<char *>(Padding.data()), Padding.size(),
                *FileOff + EmittedSize);
    }

    ++RewrittenCount;
  }

  // Write resolved jump table data back to their original .rdata offsets.
  for (const auto &JTD : ResolvedJTData) {
    auto FileOff = VAToFileOffset(JTD.VA);
    if (!FileOff)
      continue;
    OS.pwrite(reinterpret_cast<const char *>(JTD.Data), JTD.Size, *FileOff);
    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: wrote " << JTD.Size
                      << " bytes of JT data at file offset 0x"
                      << Twine::utohexstr(*FileOff) << "\n");
  }

  NumFuncsOverflow = OverflowCount;
  Out->keep();

  outs() << "BOLT-INFO: " << RewrittenCount << " functions rewritten\n";
  if (OverflowCount)
    outs() << "BOLT-INFO: " << OverflowCount
           << " functions skipped (size overflow)\n";
  outs() << "BOLT-INFO: output binary: " << opts::OutputFilename << "\n";
}

void PECOFFRewriteInstance::identityRewriteFile() {
  std::error_code EC;
  Out = std::make_unique<ToolOutputFile>(opts::OutputFilename, EC,
                                         sys::fs::OF_None);
  check_error(EC, "cannot create output executable file");

  Out->os() << InputFile->getData();
  Out->keep();

  outs() << "BOLT-INFO: identity copy written to " << opts::OutputFilename
         << "\n";
}

void PECOFFRewriteInstance::run() {
  outs() << "BOLT-INFO: processing PE/COFF binary\n";

  adjustCommandLineOptions();

  // Detect binary characteristics that affect rewriting correctness.
  {
    // Incrementally linked binaries contain ILT padding and fixup data
    // that BOLT cannot handle. Check for IMAGE_DEBUG_TYPE_FIXUP entries
    // and the .textbss section (MSVC incremental link marker).
    bool IsIncremental = false;
    for (const auto &Entry : InputFile->debug_directories()) {
      if (Entry.Type == COFF::IMAGE_DEBUG_TYPE_FIXUP) {
        IsIncremental = true;
        break;
      }
    }
    for (const auto &Section : InputFile->sections()) {
      Expected<StringRef> NameOrErr = Section.getName();
      if (NameOrErr && *NameOrErr == ".textbss") {
        IsIncremental = true;
        break;
      }
    }
    if (IsIncremental)
      errs() << "BOLT-WARNING: binary appears to be incrementally linked "
                "(/INCREMENTAL). Results may be incorrect. Re-link with "
                "/INCREMENTAL:NO for best results.\n";

    // Control Flow Guard maintains a bitmap of valid indirect call targets
    // at specific RVAs. After block reordering those RVAs are wrong and the
    // OS will terminate the process on any indirect call.
    // TODO: rewrite the CFG bitmap after reordering so that /GUARD:CF
    // binaries can be optimized safely.
    const object::pe32plus_header *PE = InputFile->getPE32PlusHeader();
    if (PE &&
        (PE->DLLCharacteristics & COFF::IMAGE_DLL_CHARACTERISTICS_GUARD_CF))
      errs() << "BOLT-WARNING: binary has Control Flow Guard enabled "
                "(/GUARD:CF). CFG tables will not be updated and the "
                "rewritten binary may crash on indirect calls.\n";
  }

  readSpecialSections();
  readExceptionHandling();
  discoverFileObjects();

  preprocessProfileData();

  disassembleFunctions();
  processProfileDataPreCFG();
  buildFunctionsCFG();
  processProfileData();
  postProcessFunctions();

  // In aggregate-only mode (etw2bolt), just write the profile and exit.
  // This mirrors how perf2bolt uses AggregateOnly to skip optimization.
  if (opts::AggregateOnly)
    return;

  if (!ProfileReader) {
    outs() << "BOLT-INFO: no profile data, producing identity copy\n";
    identityRewriteFile();
    PDBRewriter::rewritePDB(InputFile->getFileName(), opts::OutputFilename,
                            *BC, ModifiedFunctions, FunctionOffsetMaps);
    return;
  }

  // Save the basic block layout of every function before optimization.
  // After the passes we compare against this snapshot to find which
  // functions actually had their layout modified.  Only those get their
  // bytes replaced in the output -- writing re-encoded bytes for unmodified
  // functions would break the UNWIND_INFO byte offsets and base relocations.
  DenseMap<uint64_t, std::vector<const BinaryBasicBlock *>> OrigLayouts;
  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &BF = BFI.second;
    if (!BF.hasCFG())
      continue;
    auto &Layout = BF.getLayout();
    std::vector<const BinaryBasicBlock *> Order;
    for (const BinaryBasicBlock *BB : Layout.blocks())
      Order.push_back(BB);
    OrigLayouts[BF.getAddress()] = std::move(Order);
  }

  runOptimizationPasses();

  // Find functions whose layout was actually changed by the passes.
  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &BF = BFI.second;
    auto It = OrigLayouts.find(BF.getAddress());
    if (It == OrigLayouts.end())
      continue;

    const auto &OldOrder = It->second;
    auto &Layout = BF.getLayout();
    auto NewBlocks = Layout.blocks();

    auto OldIt = OldOrder.begin();
    bool Changed = false;
    for (const BinaryBasicBlock *BB : NewBlocks) {
      if (OldIt == OldOrder.end() || BB != *OldIt) {
        Changed = true;
        break;
      }
      ++OldIt;
    }
    if (!Changed && OldIt != OldOrder.end())
      Changed = true;

    if (Changed)
      ModifiedFunctions.insert(BF.getAddress());
  }

  outs() << "BOLT-INFO: " << ModifiedFunctions.size()
         << " functions had layout modified\n";

  // Capture BB address translation before emit releases the CFG.
  // For each rewritten function, record how each BB moved so the PDB
  // rewriter can remap line tables.  We store pairs of
  // {original_BB_offset, new_position_in_layout} since absolute output
  // addresses are not available until after emission.
  for (uint64_t FuncVA : ModifiedFunctions) {
    auto It = BC->getBinaryFunctions().find(FuncVA);
    if (It == BC->getBinaryFunctions().end())
      continue;
    const BinaryFunction &BF = It->second;
    if (!BF.hasCFG())
      continue;

    OffsetMap &Map = FunctionOffsetMaps[FuncVA];
    // Walk the new layout order.  Accumulate byte offsets based on BB sizes.
    uint32_t NewOffset = 0;
    for (const BinaryBasicBlock *BB : BF.getLayout().blocks()) {
      uint32_t OldOffset = BB->getOffset();
      Map.push_back({OldOffset, NewOffset});
      NewOffset += BB->getOutputSize() ? BB->getOutputSize()
                                       : BB->estimateSize();
    }
  }

  emitAndLink();
  rewriteFile();

  // Update PDB debug info to match the new binary layout.
  PDBRewriter::rewritePDB(InputFile->getFileName(), opts::OutputFilename,
                          *BC, ModifiedFunctions, FunctionOffsetMaps);
}

} // namespace bolt
} // namespace llvm
