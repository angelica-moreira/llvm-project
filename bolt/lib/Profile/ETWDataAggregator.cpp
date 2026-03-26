//===- bolt/Profile/ETWDataAggregator.cpp - ETW data aggregator -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Profile/ETWDataAggregator.h"
#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "bolt-etw"

using namespace llvm;
using namespace bolt;

namespace opts {
extern cl::opt<std::string> OutputFilename;
extern cl::opt<unsigned> Verbosity;

static cl::opt<std::string>
    XperfPathOpt("xperf-path",
                 cl::desc("Path to xperf.exe (auto-detected if not set)"),
                 cl::Optional, cl::cat(AggregatorCategory));

static cl::opt<std::string>
    ETWDumpFile("etw-dump",
                cl::desc("Path to pre-generated xperf dump text file"),
                cl::Optional, cl::cat(AggregatorCategory));
} // namespace opts

ETWDataAggregator::~ETWDataAggregator() {
  if (!DumpFilePath.empty() && opts::ETWDumpFile.empty())
    sys::fs::remove(DumpFilePath);
}

bool ETWDataAggregator::checkETLMagic(StringRef FileName) {
  // ETL files start with a header that varies by version.  Rather than
  // parsing the full header, just check the file extension.  The actual
  // validation happens when xperf tries to open it.
  return FileName.ends_with_insensitive(".etl");
}

std::string ETWDataAggregator::findXperf() const {
  if (auto P = sys::findProgramByName("xperf"))
    return *P;

  const char *Candidates[] = {
      "C:\\Program Files (x86)\\Windows Kits\\10\\Windows Performance "
      "Toolkit\\xperf.exe",
      "C:\\Program Files\\Windows Kits\\10\\Windows Performance "
      "Toolkit\\xperf.exe",
  };
  for (const char *C : Candidates)
    if (sys::fs::exists(C))
      return C;

  return {};
}

Error ETWDataAggregator::launchXperf() {
  XperfPath = opts::XperfPathOpt.empty() ? findXperf() : opts::XperfPathOpt;
  if (XperfPath.empty())
    return createStringError(
        errc::no_such_file_or_directory,
        "cannot find xperf.exe; install Windows Performance Toolkit or use "
        "-xperf-path=<path>, or run xperf manually and use -etw-dump=<file>");

  // Place the dump file next to the ETL file rather than in %TEMP%.
  // Antivirus software may block writes to temp directories.
  DumpFilePath = ETLFilename + ".dump.txt";

  // Shell out to xperf, just like DataAggregator shells to perf script.
  // Use stdout redirect instead of -o flag -- some xperf versions have
  // issues writing to certain paths with -o.
  SmallString<512> CmdLine;
  raw_svector_ostream CmdStream(CmdLine);
  CmdStream << "\"" << XperfPath << "\" -i \"" << ETLFilename
            << "\" -a dumper > \"" << DumpFilePath << "\" 2>&1";

  StringRef CmdArgs[] = {"cmd.exe", "/c", CmdLine};
  SmallVector<StringRef, 4> Argv(std::begin(CmdArgs), std::end(CmdArgs));

  outs() << "ETW2BOLT: running xperf to dump trace data...\n";
  LLVM_DEBUG(dbgs() << "ETW2BOLT: " << CmdLine << "\n");

  std::string ErrMsg;
  int RC = sys::ExecuteAndWait("cmd.exe", Argv, /*Env=*/std::nullopt,
                               /*Redirects=*/{}, /*SecondsToWait=*/600,
                               /*MemoryLimit=*/0, &ErrMsg);

  // xperf returns non-zero when events were lost during tracing, which is
  // common and harmless.  Only fail if the output file is missing or empty.
  uint64_t FileSize = 0;
  sys::fs::file_size(DumpFilePath, FileSize);
  if (FileSize == 0) {
    return createStringError(errc::executable_format_error,
                             "xperf produced no output (exit %d). "
                             "Try running as Administrator, or dump manually:\n"
                             "  xperf -i %s -a dumper > dump.txt\n"
                             "  etw2bolt ... -etw-dump=dump.txt",
                             RC, ETLFilename.c_str());
  }

  if (RC != 0)
    outs() << "ETW2BOLT: xperf reported warnings (exit " << RC
           << "), proceeding with " << (FileSize / 1024) << " KB of data\n";

  return Error::success();
}

bool ETWDataAggregator::recordBranchEvent(uint64_t From, uint64_t To,
                                          uint64_t Count, uint64_t Mispreds) {
  // Resolve absolute addresses to BinaryFunctions, same logic as
  // DataAggregator::doBranch().
  BinaryFunction *FromFunc =
      BC->getBinaryFunctionContainingAddress(From, false, true);
  BinaryFunction *ToFunc =
      BC->getBinaryFunctionContainingAddress(To, false, true);

  if (!FromFunc && !ToFunc)
    return false;
  uint64_t FromOffset = FromFunc ? From - FromFunc->getAddress() : 0;
  uint64_t ToOffset = ToFunc ? To - ToFunc->getAddress() : 0;
  if (FromFunc && ToFunc && FromFunc == ToFunc) {
    StringRef Name = FromFunc->getOneName();
    FuncBranchData &FBD = NamesToBranches[Name];
    FBD.Name = Name;
    FBD.bumpBranchCount(FromOffset, ToOffset, Count, Mispreds);
    return true;
  }
  if (FromFunc) {
    StringRef FromName = FromFunc->getOneName();
    FuncBranchData &FromFBD = NamesToBranches[FromName];
    FromFBD.Name = FromName;
    Location ToLoc(ToFunc != nullptr, ToFunc ? ToFunc->getOneName() : "",
                   ToOffset);
    FromFBD.bumpCallCount(FromOffset, ToLoc, Count, Mispreds);
  }

  if (ToFunc) {
    StringRef ToName = ToFunc->getOneName();
    FuncBranchData &ToFBD = NamesToBranches[ToName];
    ToFBD.Name = ToName;
    Location FromLoc(FromFunc != nullptr,
                     FromFunc ? FromFunc->getOneName() : "", FromOffset);
    ToFBD.bumpEntryCount(FromLoc, ToOffset, Count, Mispreds);
  }

  return true;
}

void ETWDataAggregator::parseImageLoadEvents(StringRef Dump) {
  // Scan for I-Start events (image load) to find the actual load address.
  // Real xperf format:
  //   I-Start, <ts>, z3.exe (PID), 0x00007ff697580000, 0x00007ff6984c0000, ...
  // The 4th field is BaseAddr.

  // Get the preferred ImageBase from the PE header.
  uint64_t PreferredBase = 0;
  {
    ErrorOr<std::unique_ptr<MemoryBuffer>> FileBuf =
        MemoryBuffer::getFile(BC->getFilename());
    if (FileBuf) {
      Expected<std::unique_ptr<object::ObjectFile>> ObjOrErr =
          object::ObjectFile::createObjectFile((*FileBuf)->getMemBufferRef());
      if (ObjOrErr) {
        if (auto *COFF = dyn_cast<object::COFFObjectFile>(ObjOrErr->get()))
          PreferredBase = COFF->getImageBase();
      } else {
        consumeError(ObjOrErr.takeError());
      }
    }
  }
  if (PreferredBase == 0 && !BC->getBinaryFunctions().empty()) {
    // Fallback: derive from first function address.  PE ImageBase is
    // always 64KB-aligned, but .text can start at RVA >= 0x10000 for
    // binaries with large headers or /MERGE sections.  Use a wider mask
    // (1MB) to be safe, then warn that ASLR may be inaccurate.
    PreferredBase =
        BC->getBinaryFunctions().begin()->second.getAddress() & ~0xFFFFFULL;
    errs() << "ETW2BOLT: warning: could not read PE ImageBase from file; "
              "derived 0x"
           << Twine::utohexstr(PreferredBase)
           << " from function addresses. ASLR adjustment may be "
              "inaccurate.\n";
  }

  StringRef BinaryPath = BC->getFilename();
  StringRef BinaryName = llvm::sys::path::filename(BinaryPath);

  while (!Dump.empty()) {
    auto [Line, Rest] = Dump.split('\n');
    Dump = Rest;

    StringRef Trimmed = Line.ltrim();
    if (!Trimmed.starts_with("I-Start"))
      continue;

    // Parse: I-Start, <ts>, z3.exe (PID), 0x<BaseAddr>, ...
    SmallVector<StringRef, 16> Parts;
    Line.split(Parts, ',');
    if (Parts.size() < 4)
      continue;

    // Field 2 is "processname.exe (PID)".  Match the binary name as a
    // delimited token, not as a substring of the entire line.  This
    // prevents "z3.exe" from matching "libz3.exe" or "z3.exe_helper".
    StringRef ProcField = Parts[2].trim();
    if (!ProcField.starts_with_insensitive(BinaryName))
      continue;
    // Ensure the match ends at a delimiter (space, '(' for PID, etc.)
    if (ProcField.size() > BinaryName.size()) {
      char After = ProcField[BinaryName.size()];
      if (After != ' ' && After != '(' && After != '\t')
        continue;
    }

    StringRef BaseField = Parts[3].trim();
    uint64_t ActualBase = 0;
    if (BaseField.consume_front("0x") || BaseField.consume_front("0X"))
      BaseField.getAsInteger(16, ActualBase);

    if (ActualBase == 0)
      continue;

    ASLROffset =
        static_cast<int64_t>(ActualBase) - static_cast<int64_t>(PreferredBase);

    outs() << "ETW2BOLT: " << BinaryName << " loaded at 0x"
           << Twine::utohexstr(ActualBase) << " (preferred 0x"
           << Twine::utohexstr(PreferredBase) << ")\n";
    if (ASLROffset != 0)
      outs() << "ETW2BOLT: ASLR offset: " << (ASLROffset > 0 ? "+" : "")
             << ASLROffset << "\n";
    return;
  }

  outs() << "ETW2BOLT: no I-Start event found for " << BinaryName
         << ", assuming no ASLR\n";
}

Error ETWDataAggregator::parseXperfOutput() {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
      MemoryBuffer::getFile(DumpFilePath);
  if (!BufOrErr)
    return createStringError(BufOrErr.getError(), "cannot read xperf dump %s",
                             DumpFilePath.c_str());

  uint64_t BufSize = (*BufOrErr)->getBufferSize();
  outs() << "ETW2BOLT: parsing xperf dump (" << (BufSize / 1024) << " KB)...\n";

  StringRef Dump = (*BufOrErr)->getBuffer();

  // First pass: find the actual load address from I-Start events.
  parseImageLoadEvents(Dump);

  // Second pass: parse SampledProfile events.  Real xperf format:
  //   SampledProfile, <ts>, z3.exe (PID), <tid>, <PrgrmCtr>, <cpu>, ...
  // PrgrmCtr (field 4) is the actual sampled instruction pointer.

  StringRef Remaining = Dump;
  uint64_t LinesProcessed = 0;

  while (!Remaining.empty()) {
    auto [Line, Rest] = Remaining.split('\n');
    Remaining = Rest;
    ++LinesProcessed;

    if ((LinesProcessed & 0xFFFFF) == 0)
      outs() << "ETW2BOLT: processed " << (LinesProcessed / 1000000)
             << "M lines, " << MatchedSamples << " samples matched\r";

    StringRef Trimmed = Line.ltrim();
    if (!Trimmed.starts_with("SampledProfile"))
      continue;

    ++TotalEvents;

    // Extract the sampled instruction pointer from the PrgrmCtr field
    // (field index 4, 0-based) and ThreadID (field index 3).
    // Format: SampledProfile, <ts>, z3.exe (PID), <tid>, <PrgrmCtr>, <cpu>, ...
    SmallVector<StringRef, 12> Parts;
    Line.split(Parts, ',');
    if (Parts.size() < 6)
      continue;

    uint64_t ThreadID = 0;
    Parts[3].trim().getAsInteger(0, ThreadID);
    uint64_t IP = 0;
    StringRef IPField = Parts[4].trim();
    if (IPField.consume_front("0x") || IPField.consume_front("0X"))
      IPField.getAsInteger(16, IP);

    if (IP == 0)
      continue;

    // Apply ASLR adjustment: convert runtime address to preferred address.
    if (ASLROffset != 0)
      IP = static_cast<uint64_t>(static_cast<int64_t>(IP) - ASLROffset);

    // Check if this IP belongs to a known function.  We use the function
    // lookup directly because BinaryContext::containsAddress() depends on
    // FirstAllocAddress which is not set for PE/COFF.
    if (!BC->getBinaryFunctionContainingAddress(IP, false, true))
      continue;

    ++MatchedSamples;

    // Infer edges from consecutive samples in the same thread.
    if (ThreadID != 0) {
      auto &LastIP = LastIPPerThread[ThreadID];
      if (LastIP != 0 && LastIP != IP) {
        recordBranchEvent(LastIP, IP, 1, 0);
      }
      LastIP = IP;
    }
  }

  outs() << "\n";
  return Error::success();
}

std::error_code
ETWDataAggregator::writeAggregatedFile(StringRef OutputFilename) const {
  return writeBranchProfile(OutputFilename);
}

Error ETWDataAggregator::preprocessProfile(BinaryContext &BC) {
  this->BC = &BC;

  // Get the dump text -- either from a user-provided file or by running xperf.
  if (!opts::ETWDumpFile.empty()) {
    DumpFilePath = opts::ETWDumpFile;
  } else {
    if (Error E = launchXperf())
      return E;
  }

  return Error::success();
}

Error ETWDataAggregator::readProfile(BinaryContext &BC) {
  this->BC = &BC;

  if (Error E = parseXperfOutput())
    return E;

  outs() << "ETW2BOLT: " << TotalEvents << " events, " << MatchedSamples
         << " matched to binary\n";

  if (NamesToBranches.empty()) {
    errs() << "ETW2BOLT: no profile data matched the binary\n";
    return Error::success();
  }

  // In aggregate-only mode, write fdata and stop.
  if (opts::AggregateOnly) {
    if (std::error_code EC = writeAggregatedFile(opts::OutputFilename))
      return errorCodeToError(EC);
  }

  return Error::success();
}

bool ETWDataAggregator::mayHaveProfileData(const BinaryFunction &BF) {
  return BF.hasProfileAvailable();
}
