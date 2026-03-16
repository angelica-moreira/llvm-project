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

  SmallString<256> TempFile;
  if (std::error_code EC =
          sys::fs::createTemporaryFile("etw2bolt", "txt", TempFile))
    return createStringError(EC, "cannot create temp file for xperf output");

  DumpFilePath = std::string(TempFile);

  // Shell out to xperf, just like DataAggregator shells to perf script.
  StringRef Args[] = {XperfPath,   "-i", ETLFilename,
                      "-o",        DumpFilePath, "-a",
                      "dumper"};
  SmallVector<StringRef, 8> Argv(std::begin(Args), std::end(Args));

  outs() << "ETW2BOLT: running xperf to dump trace data...\n";
  LLVM_DEBUG(dbgs() << "ETW2BOLT: " << XperfPath << " -i " << ETLFilename
                    << " -a dumper\n");

  std::string ErrMsg;
  int RC = sys::ExecuteAndWait(XperfPath, Argv, /*Env=*/std::nullopt,
                               /*Redirects=*/{}, /*SecondsToWait=*/600,
                               /*MemoryLimit=*/0, &ErrMsg);
  if (RC != 0)
    return createStringError(errc::executable_format_error,
                             "xperf failed (exit %d): %s", RC, ErrMsg.c_str());

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

Error ETWDataAggregator::parseXperfOutput() {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
      MemoryBuffer::getFile(DumpFilePath);
  if (!BufOrErr)
    return createStringError(BufOrErr.getError(), "cannot read xperf dump %s",
                             DumpFilePath.c_str());

  outs() << "ETW2BOLT: parsing xperf dump ("
         << ((*BufOrErr)->getBufferSize() / 1024) << " KB)...\n";

  StringRef Dump = (*BufOrErr)->getBuffer();
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
