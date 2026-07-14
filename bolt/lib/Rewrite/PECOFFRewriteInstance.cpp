//===- bolt/Rewrite/PECOFFRewriteInstance.cpp - PE/COFF rewriter ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Rewrite/PECOFFRewriteInstance.h"
#include "bolt/Core/AddressMap.h"
#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryEmitter.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Core/JumpTable.h"
#include "bolt/Core/MCPlusBuilder.h"
#include "bolt/Passes/BinaryPasses.h"
#include "bolt/Passes/InferEdgeCounts.h"
#include "bolt/Profile/DataReader.h"
#include "bolt/Profile/ETWDataAggregator.h"
#include "bolt/Rewrite/BinaryPassManager.h"
#include "bolt/Rewrite/ExecutableFileMemoryManager.h"
#include "bolt/Rewrite/JITLinkLinker.h"
#include "bolt/Rewrite/PDBRewriter.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "bolt/Utils/Utils.h"
#include "llvm/BinaryFormat/COFF.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
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

cl::opt<bool> PECOFFInplaceOnly(
    "pecoff-inplace-only",
    cl::desc("Skip out-of-place emission; only rewrite functions that fit"),
    cl::init(false));

cl::opt<bool> PECOFFRelocateEH(
    "pecoff-relocate-eh",
    cl::desc("Reorder functions with C++ exception handlers, regenerating "
             "their MSVC EH metadata (experimental)"),
    cl::init(false));

cl::opt<bool> PECOFFRelocateEHDryRun(
    "pecoff-relocate-eh-dryrun",
    cl::desc("With --pecoff-relocate-eh, verify and report the regenerated EH "
             "metadata but do not modify the output binary"),
    cl::init(true));

cl::opt<bool> PECOFFRelocateEHOOP(
    "pecoff-relocate-eh-oop",
    cl::desc("With --pecoff-relocate-eh, allow C++ EH functions that no longer "
             "fit their original slot to be relocated out-of-place into .bolt "
             "(experimental)"),
    cl::init(false));

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

// Populate \p Reader so it can resolve any RVA in the image; the C++ EH
// FuncInfo and its sub-tables are typically spread across .rdata/.xdata.
static void populateImageReader(object::COFFObjectFile &InputFile,
                                WinEHImageReader &Reader) {
  for (const object::SectionRef &Section : InputFile.sections()) {
    const object::coff_section *CS = InputFile.getCOFFSection(Section);
    if (!CS || CS->VirtualSize == 0)
      continue;
    ArrayRef<uint8_t> Contents;
    if (Error E = InputFile.getSectionContents(CS, Contents)) {
      consumeError(std::move(E));
      continue;
    }
    if (!Contents.empty())
      Reader.addSection(CS->VirtualAddress, Contents);
  }
}

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

  // ObjectFile::makeTriple() omits the object format for COFF AMD64.
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

  // PE/COFF profiles come in two flavors.
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

  // Relocating C++ EH functions requires a precise per-instruction
  // input->output offset map.  BOLT already produces one via its address
  // translation machinery; enable it so location symbols are emitted for the
  // functions we intend to rewrite.  PE/COFF never writes a BAT section, so
  // this only turns on the per-instruction tracking we consume internally.
  if (opts::PECOFFRelocateEH && !opts::EnableBAT)
    opts::EnableBAT = true;
  // EnableBAT alone only tracks calls/branches; ip2state boundaries may land on
  // any instruction, so retain offsets on all of them.
  if (opts::PECOFFRelocateEH)
    BC->KeepAllOffsets = true;
}

// ShortenInstructions and RemoveNops are excluded from the PE/COFF pipeline
// because they change instruction sizes, corrupting UNWIND_INFO byte offsets.

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
    BC->outs() << "BOLT-INFO: Sections from original binary:\n";
    BC->printSections(BC->outs());
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
    BC->outs() << "BOLT-WARNING: no .pdata section found\n";
    return;
  }

  // Get .xdata contents for UNWIND_INFO parsing.  Unwind data may live in
  // .xdata or .rdata.
  ArrayRef<uint8_t> XDataContents;
  uint64_t XDataRVA = 0;
  if (XDataSec) {
    if (Error E = InputFile->getSectionContents(XDataSec, XDataContents)) {
      BC->outs() << "BOLT-WARNING: cannot read .xdata section contents\n";
      consumeError(std::move(E));
    } else {
      XDataRVA = XDataSec->VirtualAddress;
    }
  }
  // Fall back to whichever section contains the unwind RVA from the first
  // .pdata entry.  Many PE binaries put UNWIND_INFO in .rdata.
  if (XDataContents.empty()) {
    ArrayRef<uint8_t> PDataPeek;
    if (Error E = InputFile->getSectionContents(PDataSec, PDataPeek)) {
      BC->outs()
          << "BOLT-WARNING: cannot peek .pdata for unwind section lookup\n";
      consumeError(std::move(E));
    }
    if (PDataPeek.size() >= 12) {
      uint32_t FirstUnwindRVA = support::endian::read32le(PDataPeek.data() + 8);
      for (const object::SectionRef &Section : InputFile->sections()) {
        const object::coff_section *CS = InputFile->getCOFFSection(Section);
        uint32_t SecStart = CS->VirtualAddress;
        uint32_t SecEnd = SecStart + CS->VirtualSize;
        if (FirstUnwindRVA >= SecStart && FirstUnwindRVA < SecEnd) {
          if (Error E = InputFile->getSectionContents(CS, XDataContents)) {
            BC->outs() << "BOLT-WARNING: cannot read unwind data section\n";
            consumeError(std::move(E));
          } else {
            XDataRVA = CS->VirtualAddress;
          }
          break;
        }
      }
    }
  }

  // Get .pdata contents: array of RUNTIME_FUNCTION entries (12 bytes each)
  ArrayRef<uint8_t> PDataContents;
  if (Error E = InputFile->getSectionContents(PDataSec, PDataContents)) {
    consumeError(std::move(E));
    BC->outs() << "BOLT-WARNING: cannot read .pdata section\n";
    return;
  }

  size_t NumEntries = PDataContents.size() / sizeof(RuntimeFunction);
  auto *Entries =
      reinterpret_cast<const RuntimeFunction *>(PDataContents.data());

  // First pass: parse UNWIND_INFO and detect chained entries
  DenseMap<uint32_t, uint32_t>
      ChainToParent; // chained begin RVA -> parent begin RVA

  // Reader over all sections, used to decode C++ EH FuncInfo referenced from
  // the handler data.
  WinEHImageReader ImageReader;
  populateImageReader(*InputFile, ImageReader);

  for (size_t I = 0; I < NumEntries; ++I) {
    uint32_t BeginRVA = Entries[I].BeginAddress;
    uint32_t EndRVA = Entries[I].EndAddress;
    uint32_t UnwindRVA = Entries[I].UnwindInfoAddress;

    if (BeginRVA == 0 && EndRVA == 0)
      continue;

    SEHUnwindInfo Info;
    Info.EndRVA = EndRVA;

    // Parse UNWIND_INFO from .xdata
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
            Info.ChainedEntryRVA = XDataRVA + HandlerDataOffset;
            ChainToParent[BeginRVA] = Chain->BeginAddress;
          }
        } else if (Info.Flags & (UNW_FLAG_EHANDLER | UNW_FLAG_UHANDLER)) {
          Info.HasExceptionHandler = true;
          if (HandlerDataOffset + 4 <= XDataContents.size()) {
            Info.ExceptionHandlerRVA = support::endian::read32le(
                XDataContents.data() + HandlerDataOffset);
          }
          // For __CxxFrameHandler3 the personality RVA is followed by a single
          // RVA pointing at the FuncInfo structure.  A successful parse both
          // recovers the metadata and confirms the personality (the compressed
          // __CxxFrameHandler4 format does not match the FuncInfo magic).
          if (HandlerDataOffset + 8 <= XDataContents.size()) {
            uint32_t FuncInfoRVA = support::endian::read32le(
                XDataContents.data() + HandlerDataOffset + 4);
            if (Expected<WinEHFuncInfo> FI =
                    parseWinEHFuncInfo(ImageReader, FuncInfoRVA)) {
              Info.IsCxxEH = true;
              Info.CxxFuncInfoRVA = FuncInfoRVA;
              // Record the cleanup/catch funclet entry points so they can be
              // pinned in place; their RVAs are referenced by the EH metadata.
              for (const WinEHFuncInfo::UnwindMapEntry &UM : FI->UnwindMap)
                if (UM.Action)
                  CxxEHFuncletRVAs.insert(UM.Action);
              for (const WinEHFuncInfo::TryBlock &TB : FI->TryBlocks)
                for (const WinEHFuncInfo::HandlerType &HT : TB.Handlers)
                  if (HT.Handler)
                    CxxEHFuncletRVAs.insert(HT.Handler);
              FunctionCxxEHInfo[BeginRVA] = std::move(*FI);
              ++NumCxxEHFuncs;
              // A first-cut reordering candidate: classic personality, no try
              // blocks (simplest shape), and a non-empty state map to relocate.
              if (FunctionCxxEHInfo[BeginRVA].TryBlocks.empty() &&
                  !FunctionCxxEHInfo[BeginRVA].IPToStateMap.empty())
                CxxEHCandidateRVAs.insert(static_cast<uint32_t>(BeginRVA));
            } else {
              consumeError(FI.takeError());
            }
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

  BC->outs() << "BOLT-INFO: parsed " << FunctionSEHInfo.size()
             << " .pdata entries, " << ChainToParent.size() << " chained\n";
  BC->outs() << "BOLT-INFO: parsed " << NumCxxEHFuncs
             << " C++ (__CxxFrameHandler3) EH tables\n";
  BC->outs() << "BOLT-INFO: " << CxxEHFuncletRVAs.size()
             << " C++ EH funclets to pin\n";
}

void PECOFFRewriteInstance::discoverFileObjects() {

  // Build address-to-name map from the COFF symbol table.
  DenseMap<uint64_t, StringRef> AddressToName;

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

  // Enumerate functions from .pdata SEH data.
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
      // With EH relocation enabled, leave reordering candidates simple so they
      // can be optimized; their EH metadata is regenerated after emission.
      bool IsCandidate =
          opts::PECOFFRelocateEH &&
          CxxEHCandidateRVAs.count(static_cast<uint32_t>(BeginRVA)) &&
          !CxxEHFuncletRVAs.count(static_cast<uint32_t>(BeginRVA));
      if (!IsCandidate) {
        BF->setSimple(false);
        ++FuncsSkippedHandler;
      }
    }

    // Pin catch/cleanup funclets: their entry RVAs are referenced by EH
    // metadata and must not move when a parent function is reordered.
    if (CxxEHFuncletRVAs.count(static_cast<uint32_t>(BeginRVA)))
      BF->setSimple(false);

    ++FuncsCreated;
  }

  NumFuncsWithHandlers = FuncsSkippedHandler;

  BC->outs() << "BOLT-INFO: " << FuncsCreated
             << " functions discovered from .pdata\n";
  BC->outs() << "BOLT-INFO: " << FuncsSkippedHandler
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
      Function.print(BC->outs(), "after disassembly");
    ++DisasmCount;
  }

  BC->outs() << "BOLT-INFO: disassembled " << DisasmCount << " functions";
  if (FailCount)
    BC->outs() << " (" << FailCount << " failed)";
  BC->outs() << "\n";
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

  BC->outs() << "BOLT-INFO: built CFG for " << CFGCount << " functions";
  if (FailCount)
    BC->outs() << " (" << FailCount << " failed)";
  BC->outs() << "\n";
}

void PECOFFRewriteInstance::postProcessFunctions() {
  // MSVC splits logical functions across multiple RUNTIME_FUNCTION entries.
  // Blocks at the end of one entry may fall through into the next.  Insert
  // explicit tail-call JMPs to prevent FixupBranches from inserting a RET.
  uint64_t FTFixups = 0;
  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;
    if (!Function.hasCFG() || !Function.isSimple())
      continue;

    for (BinaryBasicBlock &BB : Function) {
      if (BB.empty())
        continue;
      const MCInst &Last = *BB.rbegin();
      if (BC->MIB->isTerminator(Last) || BC->MIB->isReturn(Last))
        continue;
      if (BB.succ_size() > 0)
        continue;

      // Fall-through address: byte after this block's last instruction.
      uint64_t BlockEnd = 0;

      // Try offset annotation on last instruction.
      auto LastOff = BC->MIB->tryGetAnnotationAs<uint32_t>(Last, "Offset");
      if (LastOff) {
        uint32_t LastSize = BC->computeInstructionSize(Last);
        BlockEnd = Function.getAddress() + *LastOff + LastSize;
      }

      // Fallback: function end (common for boundary fall-throughs).
      if (BlockEnd == 0)
        BlockEnd = Function.getAddress() + Function.getMaxSize();

      // Only fixup if the target is a known code address.
      if (!BC->getBinaryFunctionContainingAddress(BlockEnd, false, true) &&
          FunctionSEHInfo.find(static_cast<uint32_t>(BlockEnd - ImageBase)) ==
              FunctionSEHInfo.end())
        continue;

      MCSymbol *FTSym = BC->getOrCreateGlobalSymbol(BlockEnd, "FUNCat0x");
      MCInst JmpInst;
      BC->MIB->createTailCall(JmpInst, FTSym, BC->Ctx.get());
      BB.addInstruction(JmpInst);
      ++FTFixups;
    }
  }
  if (FTFixups)
    BC->outs() << "BOLT-INFO: added " << FTFixups
               << " cross-function fall-through fixups\n";

  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;
    if (Function.empty())
      continue;
    Function.postProcessCFG();
    if (opts::PrintCFG)
      Function.print(BC->outs(), "after building cfg");
  }
}

/// Returns true if \p Byte is an x86 legacy prefix (operand-size, address-
/// size, segment, LOCK, REP).  These may appear before a REX prefix.
static bool isLegacyPrefix(uint8_t Byte) {
  switch (Byte) {
  case 0x66:
  case 0x67:
  case 0xF0:
  case 0xF2:
  case 0xF3:
  case 0x26:
  case 0x2E:
  case 0x36:
  case 0x3E:
  case 0x64:
  case 0x65:
    return true;
  default:
    return false;
  }
}

/// Returns true if \p Byte is an x86-64 REX prefix (0x40–0x4F).
static bool isREXPrefix(uint8_t Byte) { return Byte >= 0x40 && Byte <= 0x4F; }

/// Scan the first \p Len bytes starting at \p Data for a REX prefix,
/// skipping any leading legacy prefixes.  Returns true if a REX byte
/// is found.
static bool originalHasREXPrefix(const uint8_t *Data, uint64_t Len) {
  for (uint64_t I = 0; I < Len; ++I) {
    if (isLegacyPrefix(Data[I]))
      continue;
    return isREXPrefix(Data[I]);
  }
  return false;
}

/// Force REX prefixes on prolog instructions to match the original encoding.
/// MSVC emits redundant REX bytes (e.g. `40 55` for `push rbp`) that LLVM MC
/// would normally drop.  SEH UNWIND_INFO CodeOffset fields depend on exact
/// byte positions, so every prolog instruction must keep its original size.
void PECOFFRewriteInstance::freezePrologInstructions() {
  unsigned FixedCount = 0;

  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &BF = BFI.second;
    if (!BF.hasCFG() || !BF.isSimple())
      continue;

    uint32_t FuncRVA = static_cast<uint32_t>(BF.getAddress() - ImageBase);
    auto SEHIt = FunctionSEHInfo.find(FuncRVA);
    if (SEHIt == FunctionSEHInfo.end())
      continue;
    uint8_t PrologSize = SEHIt->second.PrologSize;
    if (PrologSize == 0)
      continue;

    // Locate the function bytes in the input binary.
    uint32_t FuncRVA32 = static_cast<uint32_t>(FuncRVA);
    const uint8_t *OrigData = nullptr;
    for (const object::SectionRef &Sec : InputFile->sections()) {
      const auto *CS = InputFile->getCOFFSection(Sec);
      if (FuncRVA32 >= CS->VirtualAddress &&
          FuncRVA32 < CS->VirtualAddress + CS->VirtualSize) {
        uint32_t SecOffset = FuncRVA32 - CS->VirtualAddress;
        // Clamp to SizeOfRawData — VirtualSize can exceed it (zero-fill).
        if (SecOffset + PrologSize <= CS->SizeOfRawData) {
          uint64_t FileOff = CS->PointerToRawData + SecOffset;
          if (FileOff + PrologSize <= InputFile->getData().size())
            OrigData = reinterpret_cast<const uint8_t *>(
                InputFile->getData().data() + FileOff);
        }
        break;
      }
    }

    BinaryBasicBlock &EntryBB = BF.front();
    uint32_t OrigOffset = 0;
    for (MCInst &Inst : EntryBB) {
      if (OrigOffset >= PrologSize)
        break;

      // Decode original instruction to get its size.
      uint64_t OrigInstSize = 0;
      if (OrigData) {
        MCInst TmpInst;
        ArrayRef<uint8_t> Bytes(OrigData + OrigOffset, PrologSize - OrigOffset);
        if (!BC->DisAsm->getInstruction(TmpInst, OrigInstSize, Bytes, 0,
                                        nulls()))
          break; // Cannot decode — stop processing this prolog.
      }

      // Force REX prefix if the original had one and MC would drop it.
      if (OrigData && OrigInstSize > 0) {
        unsigned CurSize = BC->computeInstructionSize(Inst);
        if (CurSize < OrigInstSize &&
            originalHasREXPrefix(OrigData + OrigOffset, OrigInstSize)) {
          BC->MIB->forceREXPrefix(Inst);
          unsigned NewSize = BC->computeInstructionSize(Inst);
          if (NewSize == OrigInstSize) {
            ++FixedCount;
          } else {
            // Unexpected: IP_USE_REX didn't fully close the size gap.
            // Log and leave the flag set — it's still closer to correct.
            LLVM_DEBUG(dbgs() << "BOLT-DEBUG: REX fix size mismatch for "
                              << BF.getPrintName() << " prolog inst at +"
                              << OrigOffset << ": orig=" << OrigInstSize
                              << " new=" << NewSize << "\n");
          }
        }
        OrigOffset += OrigInstSize;
      } else {
        // No original data available — use the re-encoded size.
        OrigOffset += BC->computeInstructionSize(Inst);
      }

      // Size annotation prevents ShortenInstructions from re-encoding.
      unsigned FinalSize = BC->computeInstructionSize(Inst);
      if (FinalSize > 0)
        BC->MIB->setSize(Inst, FinalSize);
      // Clearing the NOP annotation prevents RemoveNops from deleting
      // alignment padding that is part of the prolog.
      if (BC->MIB->isNoop(Inst))
        BC->MIB->removeAnnotation(Inst, "NOP");
    }
  }

  if (FixedCount)
    BC->outs() << "BOLT-INFO: preserved REX prefix on " << FixedCount
               << " prolog instructions for SEH correctness\n";
}

void PECOFFRewriteInstance::runOptimizationPasses() {
  freezePrologInstructions();

  BinaryFunctionPassManager Manager(*BC);
  Manager.registerPass(std::make_unique<NormalizeCFG>(opts::PrintNormalized));

  // ShortenInstructions and RemoveNops would change instruction sizes,
  // corrupting UNWIND_INFO byte offsets.  Do not register them.

  Manager.registerPass(std::make_unique<InferEdgeCounts>(opts::NeverPrint));

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

  // Map JT data sections so JITLink resolves JT entry relocations.
  DenseSet<uint64_t> MappedJTSections;
  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &BF = BFI.second;
    if (!BF.isEmitted())
      continue;
    for (const auto &JTKV : BF.jumpTables()) {
      uint64_t JTVA = JTKV.second->getAddress();
      ErrorOr<BinarySection &> JTSection = BC->getSectionForAddress(JTVA);
      if (JTSection &&
          MappedJTSections.insert(JTSection->getAddress()).second) {
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
  // Emit the input->output address map so we can recover precise
  // per-instruction offsets after linking (used to regenerate C++ EH tables).
  // BinaryEmitter only emits this automatically for ELF, so do it explicitly.
  if (opts::PECOFFRelocateEH)
    AddressMap::emit(*Streamer, *BC);
  Streamer->finish();

  StringRef ObjContents = BOS->str();
  BC->outs() << "BOLT-INFO: emitted object size = " << ObjContents.size()
             << " bytes\n";

  std::unique_ptr<MemoryBuffer> ObjectMemBuffer =
      MemoryBuffer::getMemBuffer(ObjContents, "bolt-coff-object", false);

  auto EFMM = std::make_unique<ExecutableFileMemoryManager>(*BC);
  EFMM->setNewSecPrefix(getNewSecPrefix());
  EFMM->setOrgSecPrefix(getOrgSecPrefix());

  Linker = std::make_unique<JITLinkLinker>(*BC, std::move(EFMM));
  Linker->loadObject(ObjectMemBuffer->getMemBufferRef(),
                     [this](auto MapSection) { mapCodeSections(MapSection); });

  // Recover the linked per-instruction address map for EH table regeneration.
  if (opts::PECOFFRelocateEH) {
    if (std::optional<AddressMap> Map = AddressMap::parse(*BC)) {
      BC->setIOAddressMap(std::move(*Map));
      BC->outs() << "BOLT-INFO: address map available for EH relocation\n";
    } else {
      BC->outs() << "BOLT-WARNING: no address map parsed; EH relocation "
                    "disabled for this run\n";
    }
  }
}

void PECOFFRewriteInstance::relocateCxxEHTables(bool DryRun) {
  if (!opts::PECOFFRelocateEH || CxxEHCandidateRVAs.empty())
    return;
  if (!BC->hasIOAddressMap()) {
    BC->outs() << "BOLT-WARNING: no IO address map; skipping EH relocation\n";
    return;
  }
  const AddressMap &IOMap = BC->getIOAddressMap();

  // Revert a candidate to its original layout so the untouched EH metadata
  // stays valid (used on dry-run and on any verification failure).
  auto revert = [&](uint64_t FuncVA) {
    ModifiedFunctions.erase(FuncVA);
    FunctionOffsetMaps.erase(FuncVA);
  };

  // Reordered candidates, in ascending input-address order.  The CFG has been
  // released by the time this runs, so per-instruction offsets are recovered
  // from the input->output address map rather than from basic blocks.
  struct Cand {
    uint32_t BeginRVA;
    uint64_t InStart, InEnd, OutputRVA, FuncVA;
    uint64_t EmittedBase = ~0ULL;
    SmallVector<std::pair<uint64_t, int>, 64> Raw; // (raw output addr, state)
    SmallVector<OutputInsnState, 64> BodyInsns;
    DenseSet<uint32_t> TrackedOffsets;
  };
  SmallVector<Cand, 16> Cands;
  for (uint32_t BeginRVA : CxxEHCandidateRVAs) {
    uint64_t FuncVA = ImageBase + BeginRVA;
    auto BFI = BC->getBinaryFunctions().find(FuncVA);
    if (BFI == BC->getBinaryFunctions().end())
      continue;
    BinaryFunction &BF = BFI->second;
    if (!BF.isEmitted() || !ModifiedFunctions.count(FuncVA))
      continue;
    Cand C;
    C.BeginRVA = BeginRVA;
    C.InStart = BF.getAddress();
    C.InEnd = BF.getAddress() + BF.getMaxSize();
    C.OutputRVA = BF.getOutputAddress() - ImageBase;
    C.FuncVA = FuncVA;
    Cands.push_back(std::move(C));
  }
  llvm::sort(Cands, [](const Cand &A, const Cand &B) {
    return A.InStart < B.InStart;
  });

  // Single pass over the address map: assign each mapped instruction to the
  // candidate whose input range contains it, recording its raw linked output
  // address and the EH state at the corresponding input instruction.  The
  // emitted base (smallest output address, i.e. the entry) is tracked so the
  // final IP RVA can be computed relative to the function's final placement,
  // which works whether it is rewritten in-place or moved to .bolt.
  for (const auto &[InAddr, OutAddr] : IOMap.entries()) {
    uint64_t In = InAddr, Out = OutAddr;
    auto It = llvm::partition_point(
        Cands, [&](const Cand &C) { return C.InEnd <= In; });
    if (It == Cands.end() || In < It->InStart)
      continue;
    uint32_t Offset = static_cast<uint32_t>(In - It->InStart);
    int State = lookupEHState(FunctionCxxEHInfo[It->BeginRVA],
                              static_cast<uint32_t>(It->BeginRVA + Offset));
    It->Raw.push_back({Out, State});
    It->EmittedBase = std::min(It->EmittedBase, Out);
    It->TrackedOffsets.insert(Offset);
  }

  // Resolve each instruction's final IP RVA now that the emitted base and the
  // final output RVA are known.
  for (Cand &C : Cands)
    for (const auto &[Raw, State] : C.Raw)
      C.BodyInsns.push_back(
          {static_cast<uint32_t>(C.OutputRVA + (Raw - C.EmittedBase)), State});

  uint64_t Verified = 0, Failed = 0;
  for (Cand &C : Cands) {
    const WinEHFuncInfo &FI = FunctionCxxEHInfo[C.BeginRVA];
    const uint32_t EndRVA = FunctionSEHInfo[C.BeginRVA].EndRVA;

    // Fail closed: every in-body ip2state boundary must map to a tracked
    // instruction, otherwise the regenerated table cannot be guaranteed.
    bool CoverageOK = !C.BodyInsns.empty();
    if (CoverageOK)
      for (const WinEHFuncInfo::IPToStateEntry &E : FI.IPToStateMap) {
        if (isInBodyIP(E.IP, C.BeginRVA, EndRVA) &&
            !C.TrackedOffsets.count(E.IP - C.BeginRVA)) {
          CoverageOK = false;
          break;
        }
      }

    if (!CoverageOK) {
      ++Failed;
      if (opts::Verbosity >= 1) {
        uint32_t Missing = 0, InBody = 0;
        for (const WinEHFuncInfo::IPToStateEntry &E : FI.IPToStateMap)
          if (isInBodyIP(E.IP, C.BeginRVA, EndRVA)) {
            ++InBody;
            if (!C.TrackedOffsets.count(E.IP - C.BeginRVA) && !Missing)
              Missing = E.IP - C.BeginRVA;
          }
        BC->outs() << "BOLT-INFO: EH relocation: incomplete offset coverage "
                      "for RVA 0x"
                   << Twine::utohexstr(C.BeginRVA) << " (" << InBody
                   << " in-body entries, first uncovered offset 0x"
                   << Twine::utohexstr(Missing)
                   << "); leaving unmodified\n";
      }
      revert(C.FuncVA);
      continue;
    }

    llvm::sort(C.BodyInsns,
               [](const OutputInsnState &A, const OutputInsnState &B) {
                 return A.OutputIP < B.OutputIP;
               });

    // Regenerate the full table: body entries in output-layout order plus the
    // funclet-region entries carried over unchanged (funclets are pinned in
    // place, so their image-relative IP RVAs remain valid).
    SmallVector<WinEHFuncInfo::IPToStateEntry, 16> FuncletEntries;
    for (const WinEHFuncInfo::IPToStateEntry &E : FI.IPToStateMap)
      if (!isInBodyIP(E.IP, C.BeginRVA, EndRVA))
        FuncletEntries.push_back(E);
    SmallVector<WinEHFuncInfo::IPToStateEntry, 16> NewTable =
        regenerateIPToState(C.BodyInsns, FuncletEntries);

    size_t OrigBody = 0;
    for (const WinEHFuncInfo::IPToStateEntry &E : FI.IPToStateMap)
      if (isInBodyIP(E.IP, C.BeginRVA, EndRVA))
        ++OrigBody;

    // Verify the regenerated table reproduces the original per-instruction
    // state for every tracked body instruction (lossless compression).  The
    // appended funclet entries have higher RVAs than the body and do not affect
    // body-IP lookups.
    bool VerifyOK = true;
    for (const OutputInsnState &Insn : C.BodyInsns)
      if (stateAtIP(NewTable, Insn.OutputIP) != Insn.State) {
        VerifyOK = false;
        break;
      }

    auto BFI = BC->getBinaryFunctions().find(C.FuncVA);
    BinaryFunction &BF = BFI->second;
    if (VerifyOK) {
      ++Verified;
      if (opts::Verbosity >= 1)
        BC->outs() << "BOLT-INFO: EH relocation "
                   << (DryRun ? "(dry-run) " : "") << BF << ": body ip2state "
                   << OrigBody << " -> " << (NewTable.size() - FuncletEntries.size())
                   << " entries (" << NewTable.size()
                   << " total) [verified]\n";
      if (!DryRun)
        RegeneratedEHTables[C.BeginRVA] = std::move(NewTable);
    } else {
      ++Failed;
      BC->outs() << "BOLT-WARNING: EH relocation verification failed for " << BF
                 << "; leaving unmodified\n";
    }

    // Revert (keep the original layout and metadata) on dry-run or on any
    // verification failure.  Otherwise the function stays reordered and
    // rewriteFile() emits the regenerated table.
    if (DryRun || !VerifyOK)
      revert(C.FuncVA);
  }

  BC->outs() << "BOLT-INFO: EH relocation candidates: "
             << CxxEHCandidateRVAs.size() << " total, " << Cands.size()
             << " reordered\n";
  BC->outs() << "BOLT-INFO: EH relocation summary: " << Verified
             << " verified, " << Failed << " failed\n";
}

void PECOFFRewriteInstance::rewriteFile() {
  std::error_code EC;
  Out = std::make_unique<ToolOutputFile>(opts::OutputFilename, EC,
                                         sys::fs::OF_None);
  check_error(EC, "cannot create output executable file");
  raw_fd_ostream &OS = Out->os();

  // Start with a copy of the original PE.  We patch individual function
  // bodies below, leaving headers, imports, relocations etc. untouched.
  //
  // If the binary has an Authenticode certificate table, it sits after the
  // last section's raw data but inside the file.  The .bolt section is
  // placed at alignTo(LastSecRawEnd, FileAlignment), which would overlap
  // the cert table.  Since BOLT already warns that rewriting invalidates
  // the signature, we strip the trailing cert blob by not copying it, and
  // zero out the Security data directory entry so no tool tries to parse
  // an absent cert table.
  const auto *PE = InputFile->getPE32PlusHeader();
  StringRef FileData = InputFile->getData();

  // Compute PE header layout offsets once (used by cert stripping and OOP).
  uint32_t PEOff = 0, CoffHdrOff = 0, OptHdrOff = 0;
  if (PE) {
    PEOff = support::endian::read32le(FileData.data() + 0x3C);
    CoffHdrOff = PEOff + 4;
    OptHdrOff = CoffHdrOff + sizeof(object::coff_file_header);
  }

  uint64_t CopySize = FileData.size();
  uint32_t CertDirOff = 0;
  if (PE) {
    const object::data_directory *SecDir =
        InputFile->getDataDirectory(COFF::CERTIFICATE_TABLE);
    if (SecDir && SecDir->RelativeVirtualAddress != 0 && SecDir->Size != 0) {
      // Note: For the certificate table, RelativeVirtualAddress is actually
      // a file offset (not an RVA), per the PE spec.
      uint32_t CertFileOff = SecDir->RelativeVirtualAddress;
      uint32_t CertSize = SecDir->Size;
      if (CertFileOff + CertSize == FileData.size()) {
        CopySize = CertFileOff;
        CertDirOff = OptHdrOff + sizeof(object::pe32plus_header) +
                     COFF::CERTIFICATE_TABLE * sizeof(object::data_directory);

        BC->outs() << "BOLT-INFO: stripping " << CertSize
                   << "-byte Authenticode certificate table at file offset 0x"
                   << Twine::utohexstr(CertFileOff) << "\n";
      }
    }
  }
  OS.write(FileData.data(), CopySize);

  struct SectionLayout {
    uint32_t VA;
    uint32_t VirtualSize;
    uint32_t RawSize;
    uint32_t FileOffset;
  };
  SmallVector<SectionLayout, 8> SectionMap;

  for (const object::SectionRef &Section : InputFile->sections()) {
    const object::coff_section *CS = InputFile->getCOFFSection(Section);
    SectionMap.push_back({CS->VirtualAddress, CS->VirtualSize,
                          CS->SizeOfRawData, CS->PointerToRawData});
  }

  // Map VA to file offset, clamping to SizeOfRawData.
  auto VAToFileOffset = [&](uint64_t VA) -> std::optional<uint64_t> {
    if (VA < ImageBase)
      return std::nullopt;
    uint64_t RVA64 = VA - ImageBase;
    if (RVA64 > UINT32_MAX)
      return std::nullopt;
    uint32_t RVA = static_cast<uint32_t>(RVA64);
    for (const auto &S : SectionMap) {
      if (RVA >= S.VA && RVA < S.VA + S.VirtualSize) {
        uint32_t Offset = RVA - S.VA;
        if (Offset < S.RawSize)
          return S.FileOffset + Offset;
        return std::nullopt;
      }
    }
    return std::nullopt;
  };

  uint64_t InPlaceCount = 0;
  uint64_t OOPWritten = 0;
  uint64_t OverflowCount = 0;

  // Begin RVAs of C++ EH candidates actually written in-place, in ascending
  // order.  This is the authoritative gate for repointing FuncInfo: functions
  // that overflow or whose prolog re-encodes differently are skipped here and
  // keep their original layout and metadata.
  SmallVector<uint32_t, 16> EHInPlaceWritten;

  // Begin RVAs of C++ EH candidates written out-of-place to .bolt, mapped to
  // their final .bolt RVA.  Their regenerated ip2state table uses .bolt body
  // IPs and their FuncInfo is repointed just like the in-place set, but their
  // funclet chained-unwind records must additionally be repointed to the new
  // .bolt parent range.
  SmallVector<std::pair<uint32_t, uint32_t>, 16> EHOOPWritten;

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
  // Pre-allocate a padding buffer to avoid per-function heap allocation.
  SmallVector<uint8_t, 4096> PadBuf;
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

      // Verify prolog bytes match the original — a mismatch means
      // MC re-encoded differently and UNWIND_INFO would be wrong.
      auto SEHIt =
          FunctionSEHInfo.find(static_cast<uint32_t>(OrigAddr - ImageBase));
      if (SEHIt != FunctionSEHInfo.end() && SEHIt->second.PrologSize > 0) {
        uint8_t PrologSize = SEHIt->second.PrologSize;
        auto OrigFileOff = VAToFileOffset(OrigAddr);
        if (OrigFileOff && PrologSize <= EmittedSize &&
            *OrigFileOff + PrologSize <= FileData.size()) {
          const uint8_t *EmittedBytes =
              reinterpret_cast<const uint8_t *>(Function.getImageAddress());
          const uint8_t *OrigBytes =
              reinterpret_cast<const uint8_t *>(FileData.data() + *OrigFileOff);
          if (memcmp(EmittedBytes, OrigBytes, PrologSize) != 0) {
            LLVM_DEBUG(dbgs() << "BOLT-DEBUG: skipping " << Function
                              << " - prolog bytes changed by re-encoding\n");
            if (opts::Verbosity >= 1)
              BC->outs() << "BOLT-INFO: skipping " << Function.getPrintName()
                         << " - prolog re-encoded differently\n";
            return;
          }
        }
      }
    }

    // pwrite for in-place; seek+write for OOP (pwrite asserts on extend).
    bool IsOOP = (OutputRVA >= NewSecVA && NewSecRawPtr > 0);
    if (IsOOP) {
      OS.seek(*FileOff);
      OS.write(reinterpret_cast<char *>(Function.getImageAddress()),
               EmittedSize);
    } else {
      OS.pwrite(reinterpret_cast<char *>(Function.getImageAddress()),
                EmittedSize, *FileOff);
    }

    // Pad in-place functions.
    if (!Function.isPatch() && OutputAddr == OrigAddr &&
        EmittedSize < Function.getMaxSize()) {
      size_t PadSize = Function.getMaxSize() - EmittedSize;
      PadBuf.assign(PadSize, 0xCC);
      OS.pwrite(reinterpret_cast<char *>(PadBuf.data()), PadBuf.size(),
                *FileOff + EmittedSize);
    }

    if (Function.isPatch())
      return;
    if (OutputAddr == OrigAddr) {
      ++InPlaceCount;
      uint32_t RVA = static_cast<uint32_t>(OrigAddr - ImageBase);
      if (RegeneratedEHTables.count(RVA))
        EHInPlaceWritten.push_back(RVA);
    } else {
      ++OOPWritten;
      uint32_t RVA = static_cast<uint32_t>(OrigAddr - ImageBase);
      if (RegeneratedEHTables.count(RVA))
        EHOOPWritten.push_back({RVA, static_cast<uint32_t>(OutputRVA)});
    }
  };

  // The regenerated C++ EH tables are written into the .bolt section by the
  // out-of-place path below.  If that section will not be created (no
  // out-of-place functions, or no room for another section header), the
  // reordered EH bodies would be emitted without their new metadata, so revert
  // them here to keep their original tables valid.
  if (!RegeneratedEHTables.empty() && PE) {
    bool WillCreateBolt = false;
    for (BinaryFunction *F : BC->getAllBinaryFunctions())
      if (F->isEmitted() && F->getImageSize() != 0 &&
          F->getOutputAddress() != F->getAddress()) {
        WillCreateBolt = true;
        break;
      }
    const auto *COFFHdr = InputFile->getCOFFHeader();
    uint32_t SecTableEnd = OptHdrOff + COFFHdr->SizeOfOptionalHeader +
                           InputFile->getNumberOfSections() *
                               sizeof(object::coff_section);
    bool HeaderRoom =
        SecTableEnd + sizeof(object::coff_section) <= PE->SizeOfHeaders;
    if (!WillCreateBolt || !HeaderRoom) {
      BC->outs() << "BOLT-WARNING: no .bolt section available for EH tables; "
                    "reverting "
                 << RegeneratedEHTables.size()
                 << " reordered C++ EH functions\n";
      for (const auto &KV : RegeneratedEHTables)
        ModifiedFunctions.erase(ImageBase + KV.first);
      RegeneratedEHTables.clear();
    }
  }

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

    constexpr uint32_t SecHdrSize = sizeof(object::coff_section);

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

      // PE header updates using offsetof to avoid magic numbers.
      using PEHdr = object::pe32plus_header;
      using CoffHdr = object::coff_file_header;
      uint16_t NewNumSections = NumSections + 1;
      OS.pwrite(reinterpret_cast<char *>(&NewNumSections), 2,
                CoffHdrOff + offsetof(CoffHdr, NumberOfSections));
      uint32_t NewSizeOfImage =
          alignTo(NewSecVA + NewSecVSize, PE->SectionAlignment);
      uint32_t NewSizeOfCode = PE->SizeOfCode + NewSecRawSize;
      OS.pwrite(reinterpret_cast<char *>(&NewSizeOfCode), 4,
                OptHdrOff + offsetof(PEHdr, SizeOfCode));
      OS.pwrite(reinterpret_cast<char *>(&NewSizeOfImage), 4,
                OptHdrOff + offsetof(PEHdr, SizeOfImage));

      // Section header (will be rewritten after .pdata is appended).
      object::coff_section NewSec = {};
      std::memcpy(NewSec.Name, ".bolt\0\0\0", 8);
      NewSec.VirtualSize = NewSecVSize;
      NewSec.VirtualAddress = NewSecVA;
      NewSec.SizeOfRawData = NewSecRawSize;
      NewSec.PointerToRawData = NewSecRawPtr;
      NewSec.Characteristics = COFF::IMAGE_SCN_CNT_CODE |
                               COFF::IMAGE_SCN_MEM_EXECUTE |
                               COFF::IMAGE_SCN_MEM_READ;
      OS.pwrite(reinterpret_cast<char *>(&NewSec), sizeof(NewSec), SecTableEnd);

      BC->outs() << "BOLT-INFO: added .bolt section at VA 0x"
                 << Twine::utohexstr(NewSecVA) << " (" << OOPWritten
                 << " functions)\n";

      // Pad the .bolt section to FileAlignment and rewrite its section-table
      // header plus the image's SizeOfCode/SizeOfImage.  Invoked after each
      // block that appends to the section (.pdata, then the EH tables), since
      // every append can grow NewSecVSize.
      auto finalizeBoltSection = [&]() {
        NewSecRawSize = alignTo(NewSecVSize, PE->FileAlignment);
        if (NewSecRawSize > NewSecVSize) {
          PadBuf.assign(NewSecRawSize - NewSecVSize, 0x00);
          OS.seek(NewSecRawPtr + NewSecVSize);
          OS.write(reinterpret_cast<const char *>(PadBuf.data()),
                   PadBuf.size());
        }
        NewSec.VirtualSize = NewSecVSize;
        NewSec.SizeOfRawData = NewSecRawSize;
        NewSec.Characteristics |= COFF::IMAGE_SCN_CNT_INITIALIZED_DATA;
        OS.pwrite(reinterpret_cast<char *>(&NewSec), sizeof(NewSec),
                  SecTableEnd);
        uint32_t SizeOfImage =
            alignTo(NewSecVA + NewSecVSize, PE->SectionAlignment);
        uint32_t SizeOfCode = PE->SizeOfCode + NewSecRawSize;
        OS.pwrite(reinterpret_cast<char *>(&SizeOfCode), 4,
                  OptHdrOff + offsetof(PEHdr, SizeOfCode));
        OS.pwrite(reinterpret_cast<char *>(&SizeOfImage), 4,
                  OptHdrOff + offsetof(PEHdr, SizeOfImage));
      };

      // Append .pdata for OOP functions.  Copy original entries, remove
      // OOP originals (now leaf JMP trampolines), add .bolt entries.
      uint32_t ExcDirOff =
          OptHdrOff + sizeof(PEHdr) +
          COFF::EXCEPTION_TABLE * sizeof(object::data_directory);
      if (ExcDirOff + 8 > FileData.size()) {
        BC->outs()
            << "BOLT-WARNING: exception directory offset out of bounds\n";
      } else {
        uint32_t ExcRVA =
            support::endian::read32le(FileData.data() + ExcDirOff);
        uint32_t ExcSize =
            support::endian::read32le(FileData.data() + ExcDirOff + 4);
        uint32_t NumPdataEntries = ExcSize / sizeof(RuntimeFunction);

        uint32_t PDataFileOff = 0;
        for (const object::SectionRef &Sec : InputFile->sections()) {
          const auto *CS = InputFile->getCOFFSection(Sec);
          if (ExcRVA >= CS->VirtualAddress &&
              ExcRVA < CS->VirtualAddress + CS->VirtualSize) {
            PDataFileOff = CS->PointerToRawData + (ExcRVA - CS->VirtualAddress);
            break;
          }
        }

        if (PDataFileOff > 0 && PDataFileOff + ExcSize <= FileData.size()) {
          struct OOPInfo {
            uint32_t OrigRVA, BoltRVA, Size;
          };
          SmallVector<OOPInfo, 16> OOPFuncs;
          for (auto &BFI : BC->getBinaryFunctions()) {
            BinaryFunction &BF = BFI.second;
            if (!BF.isEmitted() || BF.getImageSize() == 0)
              continue;
            if (BF.getOutputAddress() == BF.getAddress())
              continue;
            uint32_t OrigRVA =
                static_cast<uint32_t>(BF.getAddress() - ImageBase);
            if (FunctionSEHInfo.find(OrigRVA) == FunctionSEHInfo.end())
              continue;
            OOPFuncs.push_back(
                {OrigRVA,
                 static_cast<uint32_t>(BF.getOutputAddress() - ImageBase),
                 static_cast<uint32_t>(BF.getImageSize())});
          }

          DenseSet<uint32_t> OOPOrigRVAs;
          for (const auto &O : OOPFuncs)
            OOPOrigRVAs.insert(O.OrigRVA);

          // Remove .pdata entries for OOP functions.  The original address
          // now has a leaf JMP trampoline with no valid UNWIND_INFO.
          auto *OrigEntries = reinterpret_cast<const RuntimeFunction *>(
              FileData.data() + PDataFileOff);
          SmallVector<RuntimeFunction> Combined;
          DenseMap<uint32_t, uint32_t> UnwindMap;
          Combined.reserve(NumPdataEntries + OOPFuncs.size());
          for (size_t Idx = 0; Idx < NumPdataEntries; ++Idx) {
            if (OOPOrigRVAs.count(OrigEntries[Idx].BeginAddress))
              UnwindMap[OrigEntries[Idx].BeginAddress] =
                  OrigEntries[Idx].UnwindInfoAddress;
            else
              Combined.push_back(OrigEntries[Idx]);
          }

          // Append entries for .bolt copies with the original UNWIND_INFO.
          for (const auto &O : OOPFuncs) {
            auto It = UnwindMap.find(O.OrigRVA);
            if (It == UnwindMap.end())
              continue;
            RuntimeFunction New;
            New.BeginAddress = O.BoltRVA;
            New.EndAddress = O.BoltRVA + O.Size;
            New.UnwindInfoAddress = It->second;
            Combined.push_back(New);
          }

          // Windows SEH requires .pdata sorted by BeginAddress.
          llvm::sort(Combined,
                     [](const RuntimeFunction &A, const RuntimeFunction &B) {
                       return A.BeginAddress < B.BeginAddress;
                     });

          // Write combined .pdata after code in .bolt (4-byte aligned).
          uint32_t PDataOffset = alignTo(NewSecVSize, 4);
          uint32_t CombinedBytes = Combined.size() * sizeof(RuntimeFunction);
          uint32_t BoltPDataRVA = NewSecVA + PDataOffset;

          OS.seek(NewSecRawPtr + PDataOffset);
          OS.write(reinterpret_cast<const char *>(Combined.data()),
                   CombinedBytes);

          // Recompute section sizes now that .pdata is included.
          NewSecVSize = PDataOffset + CombinedBytes;
          finalizeBoltSection();

          // Redirect exception directory to the .bolt copy.
          OS.pwrite(reinterpret_cast<char *>(&BoltPDataRVA), 4, ExcDirOff);
          OS.pwrite(reinterpret_cast<char *>(&CombinedBytes), 4, ExcDirOff + 4);

          BC->outs() << "BOLT-INFO: " << Combined.size() << " .pdata entries ("
                     << OOPFuncs.size() << " new) written to .bolt section\n";
        }
      } // ExcDirOff bounds check

      // Append the regenerated C++ EH ip2state tables after the code/.pdata in
      // the .bolt section and repoint each affected FuncInfo to its fresh
      // table.  The entry count grows when reordering fragments state regions,
      // so the table cannot be patched in place.  Only functions actually
      // written in-place are repointed; any that overflowed or re-encoded keep
      // their original layout and metadata.
      if (!EHInPlaceWritten.empty() || !EHOOPWritten.empty()) {
        constexpr uint32_t EHEntrySize = 8; // {uint32 Ip RVA, int32 State}

        // Repoint every relocated candidate's FuncInfo, whether it was written
        // in place or moved out-of-place; both share the same regenerated-table
        // machinery.  Out-of-place candidates additionally need their funclet
        // chained-unwind records repointed to the new .bolt parent range.
        SmallVector<uint32_t, 16> EHWritten(EHInPlaceWritten.begin(),
                                            EHInPlaceWritten.end());
        for (const auto &[OrigRVA, BoltRVA] : EHOOPWritten)
          EHWritten.push_back(OrigRVA);
        llvm::sort(EHWritten);

        uint32_t EHTablesWritten = 0, EHEntriesWritten = 0;
        for (uint32_t BeginRVA : EHWritten) {
          auto TI = RegeneratedEHTables.find(BeginRVA);
          if (TI == RegeneratedEHTables.end())
            continue;
          const auto &Table = TI->second;
          uint32_t TableOffset = alignTo(NewSecVSize, 4);
          uint32_t TableRVA = NewSecVA + TableOffset;

          SmallVector<uint8_t, 256> TableBytes(Table.size() * EHEntrySize);
          for (size_t Idx = 0; Idx < Table.size(); ++Idx) {
            support::endian::write32le(TableBytes.data() + Idx * EHEntrySize,
                                       Table[Idx].IP);
            support::endian::write32le(
                TableBytes.data() + Idx * EHEntrySize + 4,
                static_cast<uint32_t>(Table[Idx].State));
          }
          OS.seek(NewSecRawPtr + TableOffset);
          OS.write(reinterpret_cast<const char *>(TableBytes.data()),
                   TableBytes.size());
          NewSecVSize = TableOffset + TableBytes.size();

          // Repoint FuncInfo: NumIPMapEntries (+20) and IPToStateMap RVA (+24).
          uint32_t FIRVA = FunctionSEHInfo[BeginRVA].CxxFuncInfoRVA;
          if (std::optional<uint64_t> FO = VAToFileOffset(ImageBase + FIRVA)) {
            uint32_t Count = static_cast<uint32_t>(Table.size());
            OS.pwrite(reinterpret_cast<char *>(&Count), 4, *FO + 20);
            OS.pwrite(reinterpret_cast<char *>(&TableRVA), 4, *FO + 24);
            ++EHTablesWritten;
            EHEntriesWritten += Table.size();
          }
        }

        // Repoint funclet chained-unwind records for out-of-place parents.  A
        // catch/cleanup funclet with UNW_FLAG_CHAININFO embeds a
        // RUNTIME_FUNCTION pointing at its primary function's original range;
        // once the primary moves to .bolt that BeginAddress/EndAddress is stale
        // and must track the new .bolt .pdata entry.  UnwindInfoAddress is left
        // untouched (the parent's UNWIND_INFO stays in .xdata, unchanged).
        uint32_t ChainsPatched = 0;
        for (const auto &[OrigRVA, BoltRVA] : EHOOPWritten) {
          auto BFI = BC->getBinaryFunctions().find(ImageBase + OrigRVA);
          if (BFI == BC->getBinaryFunctions().end())
            continue;
          uint32_t BoltEnd =
              BoltRVA + static_cast<uint32_t>(BFI->second.getImageSize());
          for (const auto &[FuncletRVA, Info] : FunctionSEHInfo) {
            if (!Info.IsChained || Info.ChainedBeginRVA != OrigRVA ||
                !Info.ChainedEntryRVA)
              continue;
            if (std::optional<uint64_t> FO =
                    VAToFileOffset(ImageBase + Info.ChainedEntryRVA)) {
              OS.pwrite(reinterpret_cast<const char *>(&BoltRVA), 4, *FO);
              OS.pwrite(reinterpret_cast<const char *>(&BoltEnd), 4, *FO + 4);
              ++ChainsPatched;
            }
          }
        }
        if (ChainsPatched)
          BC->outs() << "BOLT-INFO: " << ChainsPatched
                     << " funclet chained-unwind records repointed to .bolt\n";
        finalizeBoltSection();
        if (EHTablesWritten)
          BC->outs() << "BOLT-INFO: " << EHTablesWritten
                     << " C++ EH ip2state tables (" << EHEntriesWritten
                     << " entries) written to .bolt section\n";
      }
    }
  }

  NumFuncsOverflow = OverflowCount;

  // Zero out the Security data directory if we truncated the cert table.
  if (CertDirOff > 0) {
    uint64_t Zero = 0;
    OS.pwrite(reinterpret_cast<char *>(&Zero), sizeof(object::data_directory),
              CertDirOff);
  }

  // Recompute the PE image checksum.  BOLT patches the file in place after
  // copying it, so the inherited checksum is stale.  The loader ignores it for
  // ordinary user-mode modules, but drivers, protected processes and integrity
  // tools verify it, so keep it correct.
  if (PE) {
    OS.flush();
    if (ErrorOr<std::unique_ptr<MemoryBuffer>> MB = MemoryBuffer::getFile(
            opts::OutputFilename, /*IsText=*/false,
            /*RequiresNullTerminator=*/false, /*IsVolatile=*/true)) {
      StringRef Buf = (*MB)->getBuffer();
      const uint64_t CksOff =
          OptHdrOff + offsetof(object::pe32plus_header, CheckSum);
      const auto *D = reinterpret_cast<const uint8_t *>(Buf.data());
      const size_t N = Buf.size();
      uint64_t Sum = 0;
      for (size_t I = 0; I + 1 < N; I += 2) {
        uint16_t W = (I >= CksOff && I < CksOff + 4)
                         ? 0
                         : static_cast<uint16_t>(D[I] | (D[I + 1] << 8));
        Sum += W;
        Sum = (Sum & 0xffff) + (Sum >> 16);
      }
      if (N & 1) {
        Sum += D[N - 1];
        Sum = (Sum & 0xffff) + (Sum >> 16);
      }
      Sum = (Sum & 0xffff) + (Sum >> 16);
      uint32_t Checksum =
          static_cast<uint32_t>(Sum & 0xffff) + static_cast<uint32_t>(N);
      OS.pwrite(reinterpret_cast<char *>(&Checksum), sizeof(Checksum), CksOff);
    }
  }

  Out->keep();

  BC->outs() << "BOLT-INFO: " << InPlaceCount
             << " functions rewritten in-place\n";
  if (OOPWritten)
    BC->outs() << "BOLT-INFO: " << OOPWritten
               << " functions rewritten out-of-place (.bolt section)\n";
  if (OverflowCount)
    BC->outs() << "BOLT-INFO: " << OverflowCount
               << " functions could not be optimized\n";
  BC->outs() << "BOLT-INFO: output binary: " << opts::OutputFilename << "\n";
}

void PECOFFRewriteInstance::identityRewriteFile() {
  std::error_code EC;
  Out = std::make_unique<ToolOutputFile>(opts::OutputFilename, EC,
                                         sys::fs::OF_None);
  check_error(EC, "cannot create output executable file");

  Out->os() << InputFile->getData();
  Out->keep();

  BC->outs() << "BOLT-INFO: identity copy written to " << opts::OutputFilename
             << "\n";
}

void PECOFFRewriteInstance::run() {
  BC->outs() << "BOLT-INFO: processing PE/COFF binary\n";

  adjustCommandLineOptions();

  // Detect binary features that affect rewriting.
  // PE/COFF mode uses in-place patching: function bodies are overwritten at
  // their original offsets, so base relocations and ASLR remain valid.
  {
    const object::pe32plus_header *PE = InputFile->getPE32PlusHeader();

    // --- Hard errors: binary MUST NOT be processed ---

    // Incrementally linked binaries contain ILT padding and fixup data.
    // BOLT cannot distinguish thunks from real code.
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
      BC->errs()
          << "BOLT-ERROR: binary is incrementally linked (/INCREMENTAL). "
             "Re-link with /INCREMENTAL:NO.\n";
      exit(1);
    }

    // Control Flow Guard: entry RVAs unchanged, GFids table valid.
    if (PE &&
        (PE->DLLCharacteristics & COFF::IMAGE_DLL_CHARACTERISTICS_GUARD_CF)) {
      BC->outs() << "BOLT-INFO: binary has Control Flow Guard (/GUARD:CF)\n";
    }

    // Code integrity requires a valid Authenticode signature.
    if (PE && (PE->DLLCharacteristics &
               COFF::IMAGE_DLL_CHARACTERISTICS_FORCE_INTEGRITY)) {
      BC->errs() << "BOLT-ERROR: binary has /INTEGRITYCHECK — rewriting "
                    "invalidates the signature.\n";
      exit(1);
    }

    // --- Warnings: binary CAN be processed but results need care ---

    // Authenticode without FORCE_INTEGRITY: rewriting invalidates the
    // signature but the OS does not enforce it at load time.
    if (PE) {
      const object::data_directory *SecDir =
          InputFile->getDataDirectory(COFF::CERTIFICATE_TABLE);
      if (SecDir && SecDir->RelativeVirtualAddress != 0 && SecDir->Size != 0) {
        BC->errs() << "BOLT-WARNING: binary has an Authenticode signature — "
                      "rewriting will invalidate it.\n";
      }
    }

    // COMDAT: LTCG may have folded identical functions.  Profile-guided
    // optimization uses whichever alias was profiled.
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
        BC->outs() << "BOLT-INFO: binary has COMDAT sections (likely /LTCG)\n";
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
    BC->outs() << "BOLT-INFO: no profile data, producing identity copy\n";
    identityRewriteFile();
    if (hasCodeViewDebugInfo(InputFile))
      PDBRewriter::rewritePDB(InputFile->getFileName(), opts::OutputFilename,
                              *BC, ImageBase, ModifiedFunctions,
                              FunctionOffsetMaps);
    return;
  }

  // Snapshot block layout before optimization.  Only functions whose
  // layout actually changes will have their bytes replaced in the output.
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

  BC->outs() << "BOLT-INFO: " << ModifiedFunctions.size()
             << " functions had layout modified\n";

  // Record BB offset translation for PDB line-table remapping.
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
    // Sort by original offset so the PDB rewriter can do range lookups.
    llvm::sort(Map,
               [](const auto &A, const auto &B) { return A.first < B.first; });
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

  // OOP functions: assign new addresses in a .bolt section and create
  // JMP patches at original locations, same approach as ELF PatchEntries.
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

  // Build set of end-RVAs to detect contiguous .pdata groups (MSVC split
  // functions).  Functions in a group must not be moved OOP.
  DenseSet<uint32_t> SplitTargetRVAs;
  for (const auto &[RVA, Info] : FunctionSEHInfo)
    SplitTargetRVAs.insert(Info.EndRVA);

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
  DenseSet<uint64_t> OOPFunctions;
  for (uint64_t FuncVA : ModifiedFunctions) {
    auto It = BC->getBinaryFunctions().find(FuncVA);
    if (It == BC->getBinaryFunctions().end())
      continue;
    BinaryFunction &BF = It->second;
    if (!BF.isSimple())
      continue;

    // C++ EH reordering (experimental) defaults to in-place rewriting: an
    // out-of-place copy would leave the function's .pdata/.xdata (and the
    // regenerated ip2state table) describing the original address range.  With
    // --pecoff-relocate-eh-oop the out-of-place path also repoints those, so
    // EH candidates are allowed to move; otherwise keep them in place so a
    // non-fitting one simply overflows and is left untouched.
    if (opts::PECOFFRelocateEH && !opts::PECOFFRelocateEHOOP &&
        CxxEHCandidateRVAs.count(static_cast<uint32_t>(FuncVA - ImageBase)))
      continue;

    uint64_t HotSize, ColdSize;
    std::tie(HotSize, ColdSize) =
        BC->calculateEmittedSize(BF, /*FixBranches=*/false);
    if (HotSize <= BF.getMaxSize())
      continue;

    // In inplace-only mode, skip functions that do not fit.
    if (opts::PECOFFInplaceOnly) {
      BF.setSimple(false);
      continue;
    }

    if (BF.getMaxSize() < PatchSize) {
      if (opts::Verbosity >= 1)
        BC->outs() << "BOLT-INFO: " << BF << " too small for patch ("
                   << BF.getMaxSize() << " < " << PatchSize << ")\n";
      BF.setSimple(false);
      continue;
    }

    // Skip OOP for functions with jump tables — JT data in .rdata is
    // outside the emitted LinkGraph.
    if (BF.hasJumpTables()) {
      BF.setSimple(false);
      continue;
    }

    // Skip OOP for contiguous .pdata groups (MSVC split functions).
    uint32_t FuncRVA = static_cast<uint32_t>(BF.getAddress() - ImageBase);
    uint32_t FuncEndRVA = FuncRVA + BF.getMaxSize();
    if (FunctionSEHInfo.count(FuncEndRVA) || SplitTargetRVAs.count(FuncRVA)) {
      BF.setSimple(false);
      continue;
    }

    // Assign new address in .bolt section.
    BoltCurOff = alignTo(BoltCurOff, 16);
    uint64_t NewVA = ImageBase + BoltSecVA + BoltCurOff;
    BF.setOutputAddress(NewVA);
    BoltCurOff += HotSize;

    // Create a JMP patch at the original address.
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
    OOPFunctions.insert(FuncVA);
    if (opts::Verbosity >= 1)
      BC->outs() << "BOLT-INFO: " << BF << " (" << HotSize << "B) moved to"
                 << " .bolt+0x" << Twine::utohexstr(BoltCurOff - HotSize)
                 << " with entry patch\n";
  }
  if (OOPCount)
    BC->outs() << "BOLT-INFO: " << OOPCount
               << " functions rewritten out-of-place (.bolt section)\n";

  // Drop PDB offset maps for OOP and skipped functions — their line
  // offsets should not be remapped.
  {
    SmallVector<uint64_t, 64> ToRemove;
    for (auto &Entry : FunctionOffsetMaps) {
      uint64_t VA = Entry.first;
      if (OOPFunctions.count(VA)) {
        ToRemove.push_back(VA);
        continue;
      }
      auto It = BC->getBinaryFunctions().find(VA);
      if (It == BC->getBinaryFunctions().end() || !It->second.isSimple()) {
        ToRemove.push_back(VA);
        continue;
      }
    }
    for (uint64_t VA : ToRemove)
      FunctionOffsetMaps.erase(VA);
  }
  LLVM_DEBUG(dbgs() << "BOLT-DEBUG: skipped emission for " << SkippedEmit
                    << " unmodified functions\n");

  emitAndLink();
  relocateCxxEHTables(opts::PECOFFRelocateEHDryRun);
  rewriteFile();

  if (hasCodeViewDebugInfo(InputFile))
    PDBRewriter::rewritePDB(InputFile->getFileName(), opts::OutputFilename, *BC,
                            ImageBase, ModifiedFunctions, FunctionOffsetMaps);
}

} // namespace bolt
} // namespace llvm
