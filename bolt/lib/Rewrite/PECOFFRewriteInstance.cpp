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
#include "bolt/Rewrite/ExecutableFileMemoryManager.h"
#include "bolt/Rewrite/JITLinkLinker.h"
#include "bolt/Rewrite/PDBRewriter.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "bolt/Utils/Utils.h"
#include "llvm/BinaryFormat/COFF.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
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

// x86-64 RUNTIME_FUNCTION entry from .pdata.
struct RuntimeFunction {
  support::ulittle32_t BeginAddress;
  support::ulittle32_t EndAddress;
  support::ulittle32_t UnwindInfoAddress;
};

// x86-64 unwind info flags from UNWIND_INFO.Flags.
constexpr uint8_t UNW_FLAG_EHANDLER = 0x01;
constexpr uint8_t UNW_FLAG_UHANDLER = 0x02;
constexpr uint8_t UNW_FLAG_CHAININFO = 0x04;

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

  ImageBase = InputFile->getImageBase();
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

  // Choose the right reader based on the file type.
  if (ETWDataAggregator::checkETLMagic(Filename) ||
      Filename.ends_with_insensitive(".csv"))
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

/// Check whether the input PE has a CodeView debug directory entry,
/// which indicates a PDB is associated with this binary.
static bool hasCodeViewDebugInfo(const object::COFFObjectFile *File) {
  return llvm::any_of(File->debug_directories(),
                      [](const object::debug_directory &D) {
                        return D.Type == COFF::IMAGE_DEBUG_TYPE_CODEVIEW;
                      });
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

// PE/COFF uses a restricted pass pipeline.  ShortenInstructions and
// RemoveNops are excluded because they alter byte offsets within functions,
// which would corrupt UNWIND_INFO prolog sizes and unwind code offsets
// in .xdata.  Unlike ELF where DWARF CFI can be regenerated, Windows
// unwind data is preserved byte-for-byte in the original .xdata section.

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
    if (!NameOrErr) {
      consumeError(NameOrErr.takeError());
      continue;
    }
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
    Info.EndRVA = EndRVA;

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

        // Check for exception handler or chained info.
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
    // Walk chain to find root parent.  Use a visited set to break cycles
    // in case the unwind data is corrupted.
    uint32_t Root = ParentRVA;
    DenseSet<uint32_t> Visited;
    Visited.insert(ChainedRVA);
    while (ChainToParent.count(Root) && Visited.insert(Root).second)
      Root = ChainToParent[Root];
    ChainToParent[ChainedRVA] = Root;
  }

  outs() << "BOLT-INFO: parsed " << FunctionSEHInfo.size()
         << " .pdata entries, " << ChainToParent.size() << " chained\n";
}

void PECOFFRewriteInstance::discoverFileObjects() {

  // Build address-to-name map from the COFF symbol table.
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

  // Enumerate functions from pre-parsed SEH data (populated by
  // readExceptionHandling) instead of re-scanning .pdata raw bytes.
  uint64_t FuncsCreated = 0;
  uint64_t FuncsSkippedHandler = 0;

  for (const auto &[BeginRVA, Info] : FunctionSEHInfo) {
    if (Info.IsChained)
      continue;

    uint64_t Address = ImageBase + BeginRVA;
    uint64_t Size = Info.EndRVA - BeginRVA;

    if (Size == 0)
      continue;

    ErrorOr<BinarySection &> Section = BC->getSectionForAddress(Address);
    if (!Section)
      continue;

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
    BF->setOutputAddress(BF->getAddress());

    if (Info.HasExceptionHandler) {
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
  // PE/COFF fix: detect basic blocks that fall through to the next function
  // (cross-function fall-through).  MSVC splits logical functions across
  // multiple RUNTIME_FUNCTION entries; the last block of one entry falls
  // through to the first block of the next.  Without an explicit branch,
  // FixupBranches would insert a RET, corrupting the control flow.
  // Add an explicit tail-call JMP to the fall-through target so that block
  // reordering and OOP emission preserve the original semantics.
  uint64_t FTFixups = 0;
  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;
    if (!Function.hasCFG() || !Function.isSimple())
      continue;

    uint64_t FuncEnd = Function.getAddress() + Function.getMaxSize();
    uint32_t FuncEndRVA = static_cast<uint32_t>(FuncEnd - ImageBase);

    // Check if the fall-through target is a known function in .pdata
    // (including chained entries that aren't registered as BinaryFunctions).
    if (FunctionSEHInfo.find(FuncEndRVA) == FunctionSEHInfo.end())
      continue;

    // Find the block that originally ended at the function boundary.
    // After disassembly (before reordering), blocks are in address order.
    // Scan all blocks for one that lacks a terminator and has no
    // successor — this is the block whose fall-through crosses into
    // the next function.
    for (BinaryBasicBlock &BB : Function) {
      if (BB.empty())
        continue;
      // Skip blocks that already have a terminator.
      const MCInst &Last = *BB.rbegin();
      if (BC->MIB->isTerminator(Last) || BC->MIB->isReturn(Last))
        continue;
      // Skip blocks that have successors (fall-through within the function).
      if (BB.succ_size() > 0)
        continue;

      // This block has no terminator and no successors — it was a
      // cross-function fall-through.  Add a tail-call JMP to the
      // next function.
      MCSymbol *FTSym =
          BC->getOrCreateGlobalSymbol(FuncEnd, "FUNCat0x");
      MCInst JmpInst;
      BC->MIB->createTailCall(JmpInst, FTSym, BC->Ctx.get());
      BB.addInstruction(JmpInst);
      ++FTFixups;
    }
  }
  if (FTFixups)
    outs() << "BOLT-INFO: added " << FTFixups
           << " cross-function fall-through fixups\n";

  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;
    if (Function.empty())
      continue;
    Function.postProcessCFG();
    if (opts::PrintCFG)
      Function.print(outs(), "after building cfg");
  }
}

/// Mark prolog instructions as immutable so that size-changing passes
/// (ShortenInstructions, RemoveNops) leave them alone.  SEH unwind data
/// references byte offsets within the prolog, so every instruction there
/// must keep its original encoding and size.
void PECOFFRewriteInstance::freezePrologInstructions() {
  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &BF = BFI.second;
    if (!BF.hasCFG() || !BF.isSimple())
      continue;

    uint64_t FuncRVA = BF.getAddress() - ImageBase;
    auto SEHIt = FunctionSEHInfo.find(FuncRVA);
    if (SEHIt == FunctionSEHInfo.end())
      continue;
    uint8_t PrologSize = SEHIt->second.PrologSize;
    if (PrologSize == 0)
      continue;

    // The prolog occupies the first PrologSize bytes of the entry block.
    BinaryBasicBlock &EntryBB = BF.front();
    uint32_t Offset = 0;
    for (MCInst &Inst : EntryBB) {
      if (Offset >= PrologSize)
        break;
      unsigned InstSize = BC->computeInstructionSize(Inst);
      // Size annotation prevents ShortenInstructions from re-encoding.
      if (InstSize > 0)
        BC->MIB->setSize(Inst, InstSize);
      // Clearing the NOP annotation prevents RemoveNops from deleting
      // alignment padding that is part of the prolog.
      if (BC->MIB->isNoop(Inst))
        BC->MIB->removeAnnotation(Inst, "NOP");
      Offset += InstSize;
    }
  }
}

void PECOFFRewriteInstance::runOptimizationPasses() {
  freezePrologInstructions();

  BinaryFunctionPassManager Manager(*BC);
  Manager.registerPass(std::make_unique<NormalizeCFG>(opts::PrintNormalized));

  Manager.registerPass(
      std::make_unique<ShortenInstructions>(opts::NeverPrint));

  Manager.registerPass(std::make_unique<RemoveNops>(opts::NeverPrint));

  Manager.registerPass(
      std::make_unique<ReorderBasicBlocks>(opts::PrintReordered));
  Manager.registerPass(
      std::make_unique<FixupBranches>(opts::PrintAfterBranchFixup));

  Manager.registerPass(std::make_unique<PopulateOutputFunctions>());

  Manager.registerPass(
      std::make_unique<FinalizeFunctions>(opts::PrintFinalized));

  BC->logBOLTErrorsAndQuitOnFatal(Manager.runPasses());
}

void PECOFFRewriteInstance::mapCodeSections(
    BOLTLinker::SectionMapper MapSection) {
  for (BinaryFunction *Function : BC->getAllBinaryFunctions()) {
    if (!Function->isEmitted())
      continue;
    if (Function->getOutputAddress() == 0)
      continue;
    ErrorOr<BinarySection &> FuncSection = Function->getCodeSection();
    if (!FuncSection) {
      LLVM_DEBUG(dbgs() << "BOLT-DEBUG: no code section for "
                        << Function->getOneName() << "\n");
      continue;
    }

    FuncSection->setOutputAddress(Function->getOutputAddress());
    MapSection(*FuncSection, Function->getOutputAddress());
    Function->setImageAddress(FuncSection->getAllocAddress());
    Function->setImageSize(FuncSection->getOutputSize());
  }

  // Map jump table data sections to their original addresses so JITLink
  // resolves the JT entry relocations correctly.  Track mapped sections
  // to avoid redundant calls when multiple functions share a JT section.
  DenseSet<uint64_t> MappedJTSections;
  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &BF = BFI.second;
    if (!BF.isEmitted())
      continue;
    for (const auto &JTKV : BF.jumpTables()) {
      uint64_t JTVA = JTKV.second->getAddress();
      ErrorOr<BinarySection &> JTSection = BC->getSectionForAddress(JTVA);
      if (JTSection && MappedJTSections.insert(JTSection->getAddress()).second) {
        JTSection->setOutputAddress(JTSection->getAddress());
        MapSection(*JTSection, JTSection->getAddress());
      }
    }
  }
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

  std::unique_ptr<MemoryBuffer> ObjectMemBuffer =
      MemoryBuffer::getMemBuffer(ObjContents, "bolt-coff-object", false);

  auto EFMM = std::make_unique<ExecutableFileMemoryManager>(*BC);
  EFMM->setNewSecPrefix(getNewSecPrefix());
  EFMM->setOrgSecPrefix(getOrgSecPrefix());

  Linker = std::make_unique<JITLinkLinker>(*BC, std::move(EFMM));
  Linker->loadObject(ObjectMemBuffer->getMemBufferRef(),
                     [this](auto MapSection) {
                       mapCodeSections(MapSection);
                     });
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

  const auto *PE = InputFile->getPE32PlusHeader();

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

  uint64_t InPlaceCount = 0;
  uint64_t OOPWritten = 0;
  uint64_t OverflowCount = 0;

  uint32_t NewSecRawPtr = 0;
  uint32_t NewSecVA = 0;
  if (PE) {
    uint32_t LastSecRawEnd = 0;
    uint32_t LastEndVA = 0;
    for (const object::SectionRef &Sec : InputFile->sections()) {
      const auto *CS = InputFile->getCOFFSection(Sec);
      uint32_t EndRaw = CS->PointerToRawData + CS->SizeOfRawData;
      uint32_t EndVA = CS->VirtualAddress + CS->VirtualSize;
      if (EndRaw > LastSecRawEnd)
        LastSecRawEnd = EndRaw;
      if (EndVA > LastEndVA)
        LastEndVA = EndVA;
    }
    NewSecRawPtr = alignTo(LastSecRawEnd, PE->FileAlignment);
    NewSecVA = alignTo(LastEndVA, PE->SectionAlignment);
  }

  // Write all emitted functions.  Regular functions in the main map
  // plus injected functions (patches created by createInstructionPatch).
  auto writeFunction = [&](BinaryFunction &Function) {
    if (!Function.isEmitted() || Function.getImageSize() == 0)
      return;

    uint64_t OutputAddr = Function.getOutputAddress();
    uint64_t OrigAddr = Function.getAddress();
    uint64_t EmittedSize = Function.getImageSize();
    uint32_t OutputRVA = static_cast<uint32_t>(OutputAddr - ImageBase);

    std::optional<uint64_t> FileOff;
    if (OutputRVA >= NewSecVA && NewSecRawPtr > 0)
      FileOff = NewSecRawPtr + (OutputRVA - NewSecVA);
    else
      FileOff = VAToFileOffset(OutputAddr);

    if (!FileOff)
      return;

    // For non-patch regular functions, check ModifiedFunctions.
    if (!Function.isPatch() && OutputAddr == OrigAddr) {
      if (!ModifiedFunctions.count(OrigAddr))
        return;
      if (EmittedSize > Function.getMaxSize()) {
        ++OverflowCount;
        return;
      }
    }

    OS.pwrite(reinterpret_cast<char *>(Function.getImageAddress()),
              EmittedSize, *FileOff);

    // Pad in-place functions.
    if (!Function.isPatch() && OutputAddr == OrigAddr &&
        EmittedSize < Function.getMaxSize()) {
      std::vector<uint8_t> Padding(Function.getMaxSize() - EmittedSize,
                                   0xCC);
      OS.pwrite(reinterpret_cast<char *>(Padding.data()), Padding.size(),
                *FileOff + EmittedSize);
    }

    if (Function.isPatch())
      return;
    if (OutputAddr == OrigAddr)
      ++InPlaceCount;
    else
      ++OOPWritten;
  };

  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &BF = BFI.second;
    if (BF.isSimple())
      writeFunction(BF);
  }
  for (BinaryFunction *BF : BC->getInjectedBinaryFunctions())
    writeFunction(*BF);

  // Write .bolt section header and update PE headers if any OOP functions.
  if (OOPWritten > 0 && PE) {
    uint32_t NumSections = InputFile->getNumberOfSections();
    const auto *COFFHdr = InputFile->getCOFFHeader();
    StringRef FileData = InputFile->getData();

    constexpr uint32_t SecHdrSize = sizeof(object::coff_section);

    // PE structure already validated by COFFObjectFile parser.
    uint32_t PEOff = support::endian::read32le(FileData.data() + 0x3C);
    uint32_t CoffHdrOff = PEOff + 4;
    uint32_t OptHdrOff = CoffHdrOff + sizeof(object::coff_file_header);
    uint32_t SecTableOff = OptHdrOff + COFFHdr->SizeOfOptionalHeader;
    uint32_t SecTableEnd = SecTableOff + NumSections * SecHdrSize;

    if (SecTableEnd + SecHdrSize <= PE->SizeOfHeaders) {
      // Compute .bolt section extent from what was written.
      uint32_t MaxBoltEnd = 0;
      for (BinaryFunction *Function : BC->getAllBinaryFunctions()) {
        if (!Function->isEmitted() || Function->getImageSize() == 0)
          continue;
        uint32_t ORVA =
            static_cast<uint32_t>(Function->getOutputAddress() - ImageBase);
        if (ORVA >= NewSecVA) {
          uint32_t End = (ORVA - NewSecVA) + Function->getImageSize();
          if (End > MaxBoltEnd)
            MaxBoltEnd = End;
        }
      }

      uint32_t NewSecVSize = MaxBoltEnd;
      uint32_t NewSecRawSize = alignTo(NewSecVSize, PE->FileAlignment);

      // Pad to FileAlignment.
      if (NewSecRawSize > NewSecVSize) {
        std::vector<uint8_t> Pad(NewSecRawSize - NewSecVSize, 0xCC);
        OS.seek(NewSecRawPtr + NewSecVSize);
        OS.write(reinterpret_cast<char *>(Pad.data()), Pad.size());
      }

      // PE header updates using offsetof to avoid magic numbers.
      using PEHdr = object::pe32plus_header;
      uint16_t NewNumSections = NumSections + 1;
      OS.pwrite(reinterpret_cast<char *>(&NewNumSections), 2,
                CoffHdrOff + 2);
      uint32_t NewSizeOfImage =
          alignTo(NewSecVA + NewSecVSize, PE->SectionAlignment);
      uint32_t NewSizeOfCode = PE->SizeOfCode + NewSecRawSize;
      OS.pwrite(reinterpret_cast<char *>(&NewSizeOfCode), 4,
                OptHdrOff + offsetof(PEHdr, SizeOfCode));
      OS.pwrite(reinterpret_cast<char *>(&NewSizeOfImage), 4,
                OptHdrOff + offsetof(PEHdr, SizeOfImage));

      // Section header.
      object::coff_section NewSec = {};
      std::memcpy(NewSec.Name, ".bolt\0\0\0", 8);
      NewSec.VirtualSize = NewSecVSize;
      NewSec.VirtualAddress = NewSecVA;
      NewSec.SizeOfRawData = NewSecRawSize;
      NewSec.PointerToRawData = NewSecRawPtr;
      NewSec.Characteristics =
          COFF::IMAGE_SCN_CNT_CODE |
          COFF::IMAGE_SCN_MEM_EXECUTE |
          COFF::IMAGE_SCN_MEM_READ;
      OS.pwrite(reinterpret_cast<char *>(&NewSec), sizeof(NewSec),
                SecTableEnd);

      outs() << "BOLT-INFO: added .bolt section at VA 0x"
             << Twine::utohexstr(NewSecVA) << " (" << OOPWritten
             << " functions)\n";

      // Add .pdata entries for OOP functions in the .bolt section.
      // Without .pdata, the SEH unwinder cannot walk through .bolt
      // functions, causing stack corruption on C++ exceptions.
      //
      // New entries go at the end of the existing .pdata section
      // (they have the highest RVAs, maintaining sort order).  The
      // original .pdata entries are shrunk to cover only the trampoline.
      // The exception directory (data_directory[EXCEPTION_TABLE]) follows
      // the PE32+ optional header.  Use sizeof + LLVM enum for the index.
      uint32_t ExcDirOff = OptHdrOff + sizeof(PEHdr) +
                           COFF::EXCEPTION_TABLE * sizeof(object::data_directory);
      if (ExcDirOff + 8 > FileData.size()) {
        outs() << "BOLT-WARNING: exception directory offset out of bounds\n";
      } else {
      uint32_t ExcRVA =
          support::endian::read32le(FileData.data() + ExcDirOff);
      uint32_t ExcSize =
          support::endian::read32le(FileData.data() + ExcDirOff + 4);
      uint32_t NumPdataEntries = ExcSize / sizeof(RuntimeFunction);

      // Find .pdata section file offset.
      uint32_t PDataFileOff = 0;
      uint32_t PDataSecRawSize = 0;
      for (const object::SectionRef &Sec : InputFile->sections()) {
        const auto *CS = InputFile->getCOFFSection(Sec);
        if (ExcRVA >= CS->VirtualAddress &&
            ExcRVA < CS->VirtualAddress + CS->VirtualSize) {
          PDataFileOff =
              CS->PointerToRawData + (ExcRVA - CS->VirtualAddress);
          PDataSecRawSize = CS->SizeOfRawData;
          break;
        }
      }

      if (PDataFileOff > 0 && ExcSize <= PDataSecRawSize &&
          PDataFileOff + NumPdataEntries * sizeof(RuntimeFunction) <=
              FileData.size()) {
        uint32_t PDataEnd = PDataFileOff + ExcSize;
        uint32_t SlackBytes = PDataSecRawSize - ExcSize;
        const uint32_t PatchSize = 5; // JMP rel32

        SmallVector<RuntimeFunction, 16> NewEntries;
        auto *Entries = reinterpret_cast<const RuntimeFunction *>(
            FileData.data() + PDataFileOff);

        for (auto &BFI : BC->getBinaryFunctions()) {
          BinaryFunction &BF = BFI.second;
          if (!BF.isEmitted() || BF.getImageSize() == 0)
            continue;
          uint64_t OutputAddr = BF.getOutputAddress();
          uint64_t OrigAddr = BF.getAddress();
          if (OutputAddr == OrigAddr)
            continue;
          uint32_t OrigRVA =
              static_cast<uint32_t>(OrigAddr - ImageBase);
          uint32_t BoltRVA =
              static_cast<uint32_t>(OutputAddr - ImageBase);

          auto SEHIt = FunctionSEHInfo.find(OrigRVA);
          if (SEHIt == FunctionSEHInfo.end())
            continue;

          // Find the original .pdata entry and shrink it to the trampoline.
          uint32_t UnwindInfoRVA = 0;
          for (uint32_t J = 0; J < NumPdataEntries; ++J) {
            if (Entries[J].BeginAddress == OrigRVA) {
              UnwindInfoRVA = Entries[J].UnwindInfoAddress;
              RuntimeFunction Patched;
              Patched.BeginAddress = OrigRVA;
              Patched.EndAddress = OrigRVA + PatchSize;
              Patched.UnwindInfoAddress = UnwindInfoRVA;
              OS.pwrite(reinterpret_cast<char *>(&Patched),
                        sizeof(RuntimeFunction),
                        PDataFileOff + J * sizeof(RuntimeFunction));
              break;
            }
          }

          if (UnwindInfoRVA == 0)
            continue;

          RuntimeFunction NewEntry;
          NewEntry.BeginAddress = BoltRVA;
          NewEntry.EndAddress = BoltRVA + BF.getImageSize();
          NewEntry.UnwindInfoAddress = UnwindInfoRVA;
          NewEntries.push_back(NewEntry);
        }

        llvm::sort(NewEntries, [](const RuntimeFunction &A,
                                  const RuntimeFunction &B) {
          return A.BeginAddress < B.BeginAddress;
        });

        uint32_t NewBytes = NewEntries.size() * sizeof(RuntimeFunction);
        if (NewBytes <= SlackBytes) {
          OS.pwrite(reinterpret_cast<char *>(NewEntries.data()), NewBytes,
                    PDataEnd);

          uint32_t NewExcSize =
              ExcSize + NewEntries.size() * sizeof(RuntimeFunction);
          OS.pwrite(reinterpret_cast<char *>(&NewExcSize), 4,
                    ExcDirOff + 4);

          outs() << "BOLT-INFO: added " << NewEntries.size()
                 << " .pdata entries for .bolt functions\n";
        } else {
          outs() << "BOLT-WARNING: not enough .pdata slack for "
                 << NewEntries.size() << " entries (need " << NewBytes
                 << ", have " << SlackBytes << " bytes)\n";
        }
      }
      } // ExcDirOff bounds check
    }
  }

  NumFuncsOverflow = OverflowCount;
  Out->keep();

  outs() << "BOLT-INFO: " << InPlaceCount
         << " functions rewritten in-place\n";
  if (OOPWritten)
    outs() << "BOLT-INFO: " << OOPWritten
           << " functions moved to .bolt section\n";
  if (OverflowCount)
    outs() << "BOLT-INFO: " << OverflowCount
           << " functions could not be optimized\n";
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
  // BOLT's PE/COFF mode uses strict in-place patching: function bodies are
  // overwritten at their original file offsets, no addresses change, no
  // sections are added, and PE headers are untouched.  This means base
  // relocations (.reloc) and ASLR (DYNAMIC_BASE) are safe — all RVAs in
  // the relocation table remain valid.  However, several other features
  // are incompatible with code rewriting.
  {
    const object::pe32plus_header *PE = InputFile->getPE32PlusHeader();

    // --- Hard errors: binary MUST NOT be processed ---

    // Incrementally linked binaries contain ILT padding (5-byte jmp
    // thunks), fixup data, and .textbss BSS sections.  BOLT cannot
    // distinguish thunks from real code and would corrupt the padding.
    bool IsIncremental = false;
    for (const auto &Entry : InputFile->debug_directories()) {
      if (Entry.Type == COFF::IMAGE_DEBUG_TYPE_FIXUP) {
        IsIncremental = true;
        break;
      }
    }
    for (const auto &Section : InputFile->sections()) {
      Expected<StringRef> NameOrErr = Section.getName();
      if (!NameOrErr) {
        consumeError(NameOrErr.takeError());
        continue;
      }
      if (*NameOrErr == ".textbss") {
        IsIncremental = true;
        break;
      }
    }
    if (IsIncremental) {
      errs() << "BOLT-ERROR: binary appears to be incrementally linked "
                "(/INCREMENTAL). Incremental link tables contain padding "
                "and fixup data that would be corrupted by rewriting. "
                "Re-link with /INCREMENTAL:NO.\n";
      exit(1);
    }

    // Control Flow Guard: the GFids table contains function entry RVAs.
    // In-place patching preserves all entry points at their original
    // addresses, so the table remains valid.  Intra-function indirect
    // branches (jump tables) are not covered by CFG and are handled
    // separately via JTS_MOVE.
    if (PE &&
        (PE->DLLCharacteristics & COFF::IMAGE_DLL_CHARACTERISTICS_GUARD_CF)) {
      outs() << "BOLT-INFO: binary has Control Flow Guard (/GUARD:CF). "
                "Entry point RVAs are unchanged; GFids table is valid.\n";
    }

    // Code integrity enforcement requires a valid Authenticode signature.
    // Any byte change invalidates it and the loader rejects the binary.
    if (PE && (PE->DLLCharacteristics &
               COFF::IMAGE_DLL_CHARACTERISTICS_FORCE_INTEGRITY)) {
      errs() << "BOLT-ERROR: binary has /INTEGRITYCHECK. Rewriting "
                "invalidates the Authenticode signature. Remove "
                "/INTEGRITYCHECK from the linker flags.\n";
      exit(1);
    }

    // --- Warnings: binary CAN be processed but results need care ---

    // Authenticode signature without FORCE_INTEGRITY.  The binary is
    // signed but the OS does not enforce the signature at load time
    // (typical for user-mode EXEs).  Rewriting will invalidate the
    // signature, which may cause warnings from antivirus or SmartScreen
    // but will not prevent execution.
    if (PE) {
      const object::data_directory *SecDir =
          InputFile->getDataDirectory(COFF::CERTIFICATE_TABLE);
      if (SecDir && SecDir->RelativeVirtualAddress != 0 && SecDir->Size != 0) {
        errs() << "BOLT-WARNING: binary has an Authenticode signature. "
                  "Rewriting will invalidate it. The binary will still run "
                  "but may trigger antivirus or SmartScreen warnings. "
                  "Re-sign after optimization if needed.\n";
      }
    }

    // LTCG (Link-Time Code Generation) uses COMDAT folding to merge
    // identical functions.  With in-place patching this is usually safe
    // since function addresses do not change, but COMDAT-folded aliases
    // may share code that BOLT optimizes differently based on profile
    // data for one alias.
    {
      bool HasCOMDAT = false;
      for (const auto &Section : InputFile->sections()) {
        const object::coff_section *COFFSec =
            InputFile->getCOFFSection(Section);
        if (COFFSec &&
            (COFFSec->Characteristics & COFF::IMAGE_SCN_LNK_COMDAT)) {
          HasCOMDAT = true;
          break;
        }
      }
      if (HasCOMDAT && opts::Verbosity >= 1)
        outs() << "BOLT-INFO: binary has COMDAT sections (likely /LTCG). "
                  "COMDAT-folded functions share code; profile-guided "
                  "optimization will use the profile of whichever alias "
                  "was profiled.\n";
    }
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
    if (hasCodeViewDebugInfo(InputFile))
      PDBRewriter::rewritePDB(InputFile->getFileName(), opts::OutputFilename,
                              *BC, ImageBase,
                              ModifiedFunctions, FunctionOffsetMaps);
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
      NewOffset +=
          BB->getOutputSize() ? BB->getOutputSize() : BB->estimateSize();
    }
  }

  // Skip emission for functions whose layout did not change.
  uint64_t SkippedEmit = 0;
  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &BF = BFI.second;
    if (!BF.isSimple() || !BF.hasCFG())
      continue;
    if (!ModifiedFunctions.count(BF.getAddress())) {
      BF.setIgnored();
      ++SkippedEmit;
    }
  }

  // Handle oversized functions using the same approach as ELF's
  // PatchEntries pass.  For each function that exceeds its original
  // allocation:
  //   1. Assign a new output address in a .bolt section
  //   2. Create a patch function at the original address containing a
  //      JMP to the function's symbol
  //   3. Both go through emitAndLink() — JITLink resolves the JMP
  //      target via symbol reference, not raw arithmetic
  //
  // This is robust because createInstructionPatch() and
  // createLongTailCall() are the same BOLT APIs used on ELF, where
  // they handle all edge cases (secondary entry points, symbol
  // resolution, proper code section assignment).
  const auto *PE = InputFile->getPE32PlusHeader();
  uint32_t BoltSecVA = 0;
  if (PE) {
    uint32_t SecAlign = PE->SectionAlignment;
    uint32_t LastEndVA = 0;
    for (const object::SectionRef &Sec : InputFile->sections()) {
      const auto *CS = InputFile->getCOFFSection(Sec);
      uint32_t End = CS->VirtualAddress + CS->VirtualSize;
      if (End > LastEndVA)
        LastEndVA = End;
    }
    BoltSecVA = alignTo(LastEndVA, SecAlign);
  }

  // Compute the JMP patch size once.
  size_t PatchSize = 0;
  {
    InstructionListType Seq;
    BC->MIB->createLongTailCall(Seq, BC->Ctx->createTempSymbol(),
                                BC->Ctx.get());
    PatchSize = BC->computeCodeSize(Seq.begin(), Seq.end());
  }

  uint32_t BoltCurOff = 0;
  uint64_t OOPCount = 0;
  for (uint64_t FuncVA : ModifiedFunctions) {
    auto It = BC->getBinaryFunctions().find(FuncVA);
    if (It == BC->getBinaryFunctions().end())
      continue;
    BinaryFunction &BF = It->second;
    if (!BF.isSimple())
      continue;
    uint64_t HotSize, ColdSize;
    std::tie(HotSize, ColdSize) =
        BC->calculateEmittedSize(BF, /*FixBranches=*/false);
    if (HotSize <= BF.getMaxSize())
      continue;

    // Check that the original function is large enough for the patch.
    if (BF.getMaxSize() < PatchSize) {
      if (opts::Verbosity >= 1)
        outs() << "BOLT-INFO: " << BF << " too small for patch ("
               << BF.getMaxSize() << " < " << PatchSize << ")\n";
      BF.setSimple(false);
      continue;
    }

    // Assign new address in .bolt section.
    BoltCurOff = alignTo(BoltCurOff, 16);
    uint64_t NewVA = ImageBase + BoltSecVA + BoltCurOff;
    BF.setOutputAddress(NewVA);
    BoltCurOff += HotSize;

    // Create a patch function at the original address, exactly like
    // ELF's PatchEntries pass.  The patch contains a JMP to the
    // function's symbol; JITLink resolves it during linking.
    bool PatchOK = true;
    BF.forEachEntryPoint([&](uint64_t Offset, const MCSymbol *Symbol) {
      if (Offset + PatchSize > BF.getMaxSize()) {
        PatchOK = false;
        return false;
      }
      InstructionListType JmpSeq;
      BC->MIB->createLongTailCall(JmpSeq, Symbol, BC->Ctx.get());
      BC->createInstructionPatch(
          BF.getAddress() + Offset, JmpSeq,
          NameResolver::append(Symbol->getName(), ".org.0"));
      return true;
    });

    if (!PatchOK) {
      BF.setSimple(false);
      continue;
    }

    ++OOPCount;
    if (opts::Verbosity >= 1)
      outs() << "BOLT-INFO: " << BF << " (" << HotSize << "B) moved to"
             << " .bolt+0x" << Twine::utohexstr(BoltCurOff - HotSize)
             << " with entry patch\n";
  }
  if (OOPCount)
    outs() << "BOLT-INFO: " << OOPCount
           << " functions moved to .bolt section\n";
  LLVM_DEBUG(dbgs() << "BOLT-DEBUG: skipped emission for " << SkippedEmit
                    << " unmodified functions\n");

  emitAndLink();
  rewriteFile();

  if (hasCodeViewDebugInfo(InputFile))
    PDBRewriter::rewritePDB(InputFile->getFileName(), opts::OutputFilename, *BC,
                            ImageBase, ModifiedFunctions,
                            FunctionOffsetMaps);
}

} // namespace bolt
} // namespace llvm

