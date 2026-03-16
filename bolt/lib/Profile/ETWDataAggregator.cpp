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
  // Clean up the temp dump file if we created it.
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
  // Use stdout redirect instead of -o flag — some xperf versions have
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

  // Convert to function-relative offsets.
  uint64_t FromOffset = FromFunc ? From - FromFunc->getAddress() : 0;
  uint64_t ToOffset = ToFunc ? To - ToFunc->getAddress() : 0;

  // Intra-function branch.
  if (FromFunc && ToFunc && FromFunc == ToFunc) {
    StringRef Name = FromFunc->getOneName();
    FuncBranchData &FBD = NamesToBranches[Name];
    FBD.Name = Name;
    FBD.bumpBranchCount(FromOffset, ToOffset, Count, Mispreds);
    return true;
  }

  // Inter-function branch (call or tail call).
  if (FromFunc) {
    StringRef FromName = FromFunc->getOneName();
    FuncBranchData &FromFBD = NamesToBranches[FromName];
    FromFBD.Name = FromName;
    Location ToLoc(ToFunc != nullptr, ToFunc ? ToFunc->getOneName() : "", ToOffset);
    FromFBD.bumpCallCount(FromOffset, ToLoc, Count, Mispreds);
  }

  if (ToFunc) {
    StringRef ToName = ToFunc->getOneName();
    FuncBranchData &ToFBD = NamesToBranches[ToName];
    ToFBD.Name = ToName;
    Location FromLoc(FromFunc != nullptr, FromFunc ? FromFunc->getOneName() : "",
                     FromOffset);
    ToFBD.bumpEntryCount(FromLoc, ToOffset, Count, Mispreds);
  }

  return true;
}

void ETWDataAggregator::parseImageLoadEvents(StringRef Dump) {
  // First pass: scan for ImageLoad events to find the actual load address
  // of the target binary.  xperf ImageLoad lines look like:
  //   ImageLoad, <timestamp>, <process>(PID), 0x7ff6a2f90000, 0x100000, C:\path\to\binary.exe
  //
  // MSVC does the same thing: queries IProcessInfoSource::QueryImages() to
  // get images[j]->Base, then computes RVA = SampleIP - Base.
  //
  // We compute ASLROffset = ActualBase - PreferredImageBase, then subtract
  // it from all sample IPs so they match BinaryContext's address space.

  // PE ImageBase is always at least 64KB aligned.  FirstAllocAddress is
  // the lowest loaded address (ImageBase + first section RVA, typically
  // 0x1000).  Rounding down to 64KB gives us the ImageBase.
  uint64_t PreferredBase = BC->FirstAllocAddress & ~0xFFFFULL;

  // Get just the filename of the target binary for matching.
  StringRef ExeName = llvm::sys::path::filename(ETLFilename);
  // The ETL filename is the trace, not the exe. Get the exe name from BC.
  StringRef BinaryPath = BC->getFilename();
  StringRef BinaryName = llvm::sys::path::filename(BinaryPath);

  SmallVector<StringRef, 0> Lines;
  Dump.split(Lines, '\n');

  for (const StringRef &RawLine : Lines) {
    StringRef Line = RawLine.trim();
    if (!Line.contains_insensitive("ImageLoad"))
      continue;

    // Check if this line references our target binary.
    if (!Line.contains_insensitive(BinaryName))
      continue;

    // Extract the base address (first large hex value after "ImageLoad").
    SmallVector<StringRef, 16> Parts;
    Line.split(Parts, ',');

    for (const StringRef &Part : Parts) {
      StringRef P = Part.trim();
      uint64_t Val = 0;
      StringRef Hex = P;
      if (Hex.consume_front("0x") || Hex.consume_front("0X")) {
        if (!Hex.getAsInteger(16, Val) && Val > 0x10000) {
          // This is the actual load address.
          ASLROffset = static_cast<int64_t>(Val) -
                       static_cast<int64_t>(PreferredBase);
          LLVM_DEBUG(dbgs() << "ETW2BOLT: detected ASLR load at 0x"
                            << Twine::utohexstr(Val)
                            << ", preferred base 0x"
                            << Twine::utohexstr(PreferredBase)
                            << ", offset " << ASLROffset << "\n");
          if (ASLROffset != 0)
            outs() << "ETW2BOLT: ASLR detected, load offset "
                   << (ASLROffset > 0 ? "+" : "") << ASLROffset << "\n";
          return;
        }
      }
    }
  }

  LLVM_DEBUG(dbgs() << "ETW2BOLT: no ImageLoad event found for "
                    << BinaryName << ", assuming no ASLR\n");
}

Error ETWDataAggregator::parseXperfOutput() {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
      MemoryBuffer::getFile(DumpFilePath);
  if (!BufOrErr)
    return createStringError(BufOrErr.getError(), "cannot read xperf dump %s",
                             DumpFilePath.c_str());

  outs() << "ETW2BOLT: parsing xperf dump ("
         << ((*BufOrErr)->getBufferSize() / 1024) << " KB)...\n";

  StringRef Dump = (*BufOrErr)->getBuffer();

  // First pass: find the actual load address from ImageLoad events.
  parseImageLoadEvents(Dump);

  // Second pass: parse SampledProfile events.
  SmallVector<StringRef, 0> Lines;
  Dump.split(Lines, '\n');

  for (const StringRef &RawLine : Lines) {
    StringRef Line = RawLine.trim();
    if (Line.empty() || Line.starts_with("#"))
      continue;

    // Parse xperf SampledProfile events.
    // Format: SampledProfile, TimeStamp, ..., <hex IP>, ...
    if (!Line.contains_insensitive("SampledProfile") &&
        !Line.contains_insensitive("PerfInfo"))
      continue;

    ++TotalEvents;

    // Extract instruction pointer and thread ID from the comma-separated
    // fields.  The IP is typically the largest hex value on the line.
    uint64_t IP = 0;
    uint64_t ThreadID = 0;
    SmallVector<StringRef, 16> Parts;
    Line.split(Parts, ',');

    for (const StringRef &Part : Parts) {
      StringRef P = Part.trim();

      uint64_t Val = 0;
      StringRef Hex = P;
      if (Hex.consume_front("0x") || Hex.consume_front("0X")) {
        if (!Hex.getAsInteger(16, Val) && Val > 0x10000)
          IP = Val;
      }

      if (P.contains_insensitive("ThreadId") ||
          P.contains_insensitive("TID")) {
        StringRef After = P.substr(P.find_last_of("=: ") + 1).trim();
        After.getAsInteger(0, ThreadID);
      }
    }

    if (IP == 0)
      continue;

    // Apply ASLR adjustment: convert runtime address to preferred address.
    // Same concept as DataAggregator::adjustAddress() for Linux, and
    // MSVC's (SampleIP - ActualBase) RVA conversion.
    if (ASLROffset != 0)
      IP = static_cast<uint64_t>(static_cast<int64_t>(IP) - ASLROffset);

    if (!BC->containsAddress(IP))
      continue;

    ++MatchedSamples;

    // Infer edges from consecutive samples in the same thread.  This is
    // the same technique DataAggregator uses in basic (non-LBR) mode.
    if (ThreadID != 0) {
      auto &LastIP = LastIPPerThread[ThreadID];
      if (LastIP != 0 && LastIP != IP) {
        recordBranchEvent(LastIP, IP, 1, 0);
      }
      LastIP = IP;
    }
  }

  return Error::success();
}

std::error_code
ETWDataAggregator::writeAggregatedFile(StringRef OutputFilename) const {
  // Same output format as DataAggregator::writeAggregatedFile().
  std::error_code EC;
  raw_fd_ostream OutFile(OutputFilename, EC, sys::fs::OF_None);
  if (EC)
    return EC;

  uint64_t BranchValues = 0;

  for (const auto &KV : NamesToBranches) {
    const FuncBranchData &FBD = KV.second;
    for (const BranchInfo &BI : FBD.Data) {
      OutFile << (BI.From.IsSymbol ? "1 " : "0 ")
              << (BI.From.Name.empty() ? "[unknown]" : BI.From.Name) << " "
              << Twine::utohexstr(BI.From.Offset) << " "
              << (BI.To.IsSymbol ? "1 " : "0 ")
              << (BI.To.Name.empty() ? "[unknown]" : BI.To.Name) << " "
              << Twine::utohexstr(BI.To.Offset) << " " << BI.Mispreds << " "
              << BI.Branches << "\n";
      ++BranchValues;
    }
    for (const BranchInfo &BI : FBD.EntryData) {
      if (BI.From.IsSymbol)
        continue;
      OutFile << (BI.From.IsSymbol ? "1 " : "0 ")
              << (BI.From.Name.empty() ? "[unknown]" : BI.From.Name) << " "
              << Twine::utohexstr(BI.From.Offset) << " "
              << (BI.To.IsSymbol ? "1 " : "0 ")
              << (BI.To.Name.empty() ? "[unknown]" : BI.To.Name) << " "
              << Twine::utohexstr(BI.To.Offset) << " " << BI.Mispreds << " "
              << BI.Branches << "\n";
      ++BranchValues;
    }
  }

  outs() << "ETW2BOLT: wrote " << BranchValues << " objects to "
         << OutputFilename << "\n";
  return std::error_code();
}

Error ETWDataAggregator::preprocessProfile(BinaryContext &BC) {
  this->BC = &BC;

  // Get the dump text — either from a user-provided file or by running xperf.
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
