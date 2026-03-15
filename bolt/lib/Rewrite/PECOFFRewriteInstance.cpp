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
#include "bolt/Core/MCPlusBuilder.h"
#include "bolt/Passes/BinaryPasses.h"
#include "bolt/Profile/DataReader.h"
#include "bolt/Rewrite/BinaryPassManager.h"
#include "bolt/Rewrite/ExecutableFileMemoryManager.h"
#include "bolt/Rewrite/JITLinkLinker.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "bolt/Utils/Utils.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
#include <memory>

#define DEBUG_TYPE "bolt"

namespace opts {

using namespace llvm;
extern cl::opt<unsigned> AlignText;
extern cl::opt<bool> ForcePatch;
extern cl::opt<bool> KeepTmp;
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

extern MCPlusBuilder *
createMCPlusBuilder(const Triple::ArchType Arch, const MCInstrAnalysis *Analysis,
                    const MCInstrInfo *Info, const MCRegisterInfo *RegInfo,
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
    return make_error<StringError>(
        Twine("multiple profiles specified: ") + ProfileReader->getFilename() +
            " and " + Filename,
        inconvertibleErrorCode());
  }
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

  // PE section alignment is typically 4KB (0x1000).
  BC->PageAlign = 0x1000;

  if (!opts::AlignText.getNumOccurrences())
    opts::AlignText = BC->PageAlign;
}

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

  // Get .xdata contents for UNWIND_INFO parsing
  ArrayRef<uint8_t> XDataContents;
  uint64_t XDataRVA = 0;
  if (XDataSec) {
    if (Error E = InputFile->getSectionContents(XDataSec, XDataContents))
      consumeError(std::move(E));
    else
      XDataRVA = XDataSec->VirtualAddress;
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
  std::map<uint32_t, uint32_t> ChainToParent; // chained begin RVA -> parent begin RVA

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
        for (uint8_t C = 0; C < CountOfCodes && CodesOffset + 2 <= XDataContents.size(); ++C) {
          uint16_t Code = support::endian::read16le(XDataContents.data() + CodesOffset);
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
            Info.ExceptionHandlerRVA =
                support::endian::read32le(XDataContents.data() + HandlerDataOffset);
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

  // NormalizeCFG must run before reordering
  Manager.registerPass(std::make_unique<NormalizeCFG>(opts::PrintNormalized));

  // Block reordering is the primary optimization
  Manager.registerPass(
      std::make_unique<ReorderBasicBlocks>(opts::PrintReordered));

  // Fix up branches after reordering
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
    if (!FuncSection)
      continue;

    FuncSection->setOutputAddress(Function->getOutputAddress());
    MapSection(*FuncSection, Function->getOutputAddress());
    Function->setImageAddress(FuncSection->getAllocAddress());
    Function->setImageSize(FuncSection->getOutputSize());
  }
}

void PECOFFRewriteInstance::emitAndLink() {
  std::error_code EC;
  std::unique_ptr<::llvm::ToolOutputFile> TempOut =
      std::make_unique<::llvm::ToolOutputFile>(
          opts::OutputFilename + ".bolt.o", EC, sys::fs::OF_None);
  check_error(EC, "cannot create output object file");

  if (opts::KeepTmp)
    TempOut->keep();

  std::unique_ptr<buffer_ostream> BOS =
      std::make_unique<buffer_ostream>(TempOut->os());
  raw_pwrite_stream *OS = BOS.get();
  auto Streamer = BC->createStreamer(*OS);

  emitBinaryContext(*Streamer, *BC, getOrgSecPrefix());
  Streamer->finish();

  LLVM_DEBUG(dbgs() << "BOLT-DEBUG: emitted object size = "
                    << BOS->str().size() << "\n");

  std::unique_ptr<MemoryBuffer> ObjectMemBuffer =
      MemoryBuffer::getMemBuffer(BOS->str(), "in-memory object file", false);

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

  // Copy the entire original binary
  OS << InputFile->getData();

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

    // Skip functions that grew beyond their original allocation
    if (EmittedSize > OriginalSize) {
      ++OverflowCount;
      continue;
    }

    if (opts::Verbosity >= 2)
      outs() << "BOLT: rewriting function \"" << Function << "\""
             << " (size: " << EmittedSize << "/" << OriginalSize << ")\n";

    // Write optimized function code at original file offset
    OS.pwrite(reinterpret_cast<char *>(Function.getImageAddress()),
              EmittedSize, Function.getFileOffset());

    // Pad remaining space with int3 (0xCC) breakpoint instructions
    if (EmittedSize < OriginalSize) {
      std::vector<uint8_t> Padding(OriginalSize - EmittedSize, 0xCC);
      OS.pwrite(reinterpret_cast<char *>(Padding.data()), Padding.size(),
                Function.getFileOffset() + EmittedSize);
    }

    ++RewrittenCount;
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

  readSpecialSections();

  readExceptionHandling();

  discoverFileObjects();

  preprocessProfileData();

  disassembleFunctions();

  processProfileDataPreCFG();

  buildFunctionsCFG();

  processProfileData();

  postProcessFunctions();

  if (!ProfileReader) {
    outs() << "BOLT-INFO: no profile data, producing identity copy\n";
    identityRewriteFile();
    return;
  }

  runOptimizationPasses();

  emitAndLink();

  rewriteFile();
}

} // namespace bolt
} // namespace llvm
