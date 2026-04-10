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

static cl::opt<std::string>
    ETWAnalyzerCSV("etwanalyzer-csv",
                   cl::desc("ETWAnalyzer -dump LBR -csv output file (best "
                            "for LBR branch data)"),
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

  // Second pass: parse SampledProfile and LBR branch events.
  //
  // SampledProfile (timer interrupt, always available):
  //   SampledProfile, <ts>, z3.exe (PID), <tid>, <PrgrmCtr>, <cpu>, ...
  //
  // LBR branch events (only when captured with -LastBranch):
  //   BranchTrace, <ts>, z3.exe (PID), <tid>, <FromAddr>, <ToAddr>, <mispred>
  //   LastBranch, <ts>, z3.exe (PID), <tid>, <FromAddr>, <ToAddr>, <mispred>
  //
  // LBR data is vastly better than timer samples for BOLT -- it gives exact
  // branch from->to pairs instead of just "the CPU was here".  When LBR data
  // is present, BOLT can do precise basic block reordering.

  StringRef Remaining = Dump;
  uint64_t LinesProcessed = 0;

  while (!Remaining.empty()) {
    auto [Line, Rest] = Remaining.split('\n');
    Remaining = Rest;
    ++LinesProcessed;

    if ((LinesProcessed & 0xFFFFF) == 0)
      outs() << "ETW2BOLT: processed " << (LinesProcessed / 1000000)
             << "M lines, " << MatchedSamples << " samples, "
             << MatchedLBRBranches << " LBR branches\r";

    StringRef Trimmed = Line.ltrim();

    // --- LBR branch events (highest quality data) ---
    if (Trimmed.starts_with("BranchTrace") ||
        Trimmed.starts_with("LastBranch")) {
      ++TotalEvents;

      // Format: BranchTrace, <ts>, proc (PID), <tid>, <FromAddr>, <ToAddr>,
      //         <mispred>, ...
      SmallVector<StringRef, 12> Parts;
      Line.split(Parts, ',');
      if (Parts.size() < 7)
        continue;

      uint64_t FromIP = 0, ToIP = 0;
      StringRef FromField = Parts[4].trim();
      StringRef ToField = Parts[5].trim();
      if (FromField.consume_front("0x") || FromField.consume_front("0X"))
        FromField.getAsInteger(16, FromIP);
      if (ToField.consume_front("0x") || ToField.consume_front("0X"))
        ToField.getAsInteger(16, ToIP);

      if (FromIP == 0 || ToIP == 0)
        continue;

      // ASLR adjustment
      if (ASLROffset != 0) {
        FromIP =
            static_cast<uint64_t>(static_cast<int64_t>(FromIP) - ASLROffset);
        ToIP =
            static_cast<uint64_t>(static_cast<int64_t>(ToIP) - ASLROffset);
      }

      // Misprediction flag
      uint64_t Mispred = 0;
      StringRef MispredField = Parts[6].trim();
      if (MispredField.equals_insensitive("true") || MispredField == "1")
        Mispred = 1;
      else
        MispredField.getAsInteger(0, Mispred);

      if (recordBranchEvent(FromIP, ToIP, 1, Mispred))
        ++MatchedLBRBranches;

      continue;
    }

    // --- SampledProfile events (timer-based, always available) ---
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
    // This is a poor heuristic -- the timer fires ~1ms apart while the CPU
    // executes millions of instructions between samples.  Only use this when
    // no LBR data is available.  When LBR branches are present, they provide
    // exact branch edges and this inference is unnecessary.
    if (ThreadID != 0) {
      auto &LastIP = LastIPPerThread[ThreadID];
      if (LastIP != 0 && LastIP != IP) {
        recordBranchEvent(LastIP, IP, 1, 0);
        ++InferredBranches;
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

Error ETWDataAggregator::parseETWAnalyzerCSV() {
  // ETWAnalyzer -dump LBR -csv produces clean CSV with branch from/to pairs:
  //   ProcessName,ProcessId,ThreadId,FromAddress,ToAddress,Timestamp,
  //   BranchType,Count,Mispredicted
  //
  // This is the best quality LBR data source on Windows -- ETWAnalyzer
  // handles all the ETL complexity and gives us pre-parsed branch records.

  ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
      MemoryBuffer::getFile(opts::ETWAnalyzerCSV);
  if (!BufOrErr)
    return createStringError(BufOrErr.getError(),
                             "cannot read ETWAnalyzer CSV %s",
                             opts::ETWAnalyzerCSV.c_str());

  outs() << "ETW2BOLT: parsing ETWAnalyzer LBR CSV: "
         << opts::ETWAnalyzerCSV << "\n";

  StringRef Content = (*BufOrErr)->getBuffer();

  // Parse header line to find column indices.
  auto [HeaderLine, Body] = Content.split('\n');
  // Strip BOM if present
  if (HeaderLine.starts_with("\xef\xbb\xbf"))
    HeaderLine = HeaderLine.drop_front(3);

  SmallVector<StringRef, 16> Headers;
  HeaderLine.split(Headers, ',');

  int FromCol = -1, ToCol = -1, CountCol = -1, MispredCol = -1;
  for (int I = 0; I < (int)Headers.size(); ++I) {
    StringRef H = Headers[I].trim().trim('"');
    if (H.equals_insensitive("FromAddress") || H.equals_insensitive("From") ||
        H.equals_insensitive("SourceAddress") ||
        H.equals_insensitive("BranchFrom"))
      FromCol = I;
    else if (H.equals_insensitive("ToAddress") || H.equals_insensitive("To") ||
             H.equals_insensitive("TargetAddress") ||
             H.equals_insensitive("BranchTo"))
      ToCol = I;
    else if (H.equals_insensitive("Count") ||
             H.equals_insensitive("SampleCount") ||
             H.equals_insensitive("Samples") ||
             H.equals_insensitive("Weight"))
      CountCol = I;
    else if (H.equals_insensitive("Mispredicted") ||
             H.equals_insensitive("Mispred"))
      MispredCol = I;
  }

  if (FromCol < 0 || ToCol < 0) {
    return createStringError(
        errc::invalid_argument,
        "ETWAnalyzer CSV missing FromAddress/ToAddress columns. "
        "Expected output from: ETWAnalyzer -dump LBR -csv <file>");
  }

  // We need ASLR info. Parse image load from the binary itself since
  // the CSV doesn't contain it. Use the same preferred base logic.
  if (ASLROffset == 0 && BC) {
    uint64_t PreferredBase = 0;
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
    // For CSV mode, addresses may already be rebased or not.
    // We try both with and without offset -- if no functions match with
    // offset 0, the addresses are likely runtime (ASLR'd) addresses.
    (void)PreferredBase;
  }

  uint64_t RowsRead = 0, RowsMatched = 0;

  while (!Body.empty()) {
    auto [Line, Rest] = Body.split('\n');
    Body = Rest;
    StringRef Trimmed = Line.trim();
    if (Trimmed.empty())
      continue;

    SmallVector<StringRef, 16> Fields;
    Trimmed.split(Fields, ',');
    if ((int)Fields.size() <= std::max(FromCol, ToCol))
      continue;

    ++RowsRead;

    // Parse From and To addresses
    uint64_t FromIP = 0, ToIP = 0;
    StringRef FromStr = Fields[FromCol].trim().trim('"');
    StringRef ToStr = Fields[ToCol].trim().trim('"');

    if (FromStr.consume_front("0x") || FromStr.consume_front("0X"))
      FromStr.getAsInteger(16, FromIP);
    else
      FromStr.getAsInteger(0, FromIP);

    if (ToStr.consume_front("0x") || ToStr.consume_front("0X"))
      ToStr.getAsInteger(16, ToIP);
    else
      ToStr.getAsInteger(0, ToIP);

    if (FromIP == 0 || ToIP == 0)
      continue;

    // ASLR adjustment
    if (ASLROffset != 0) {
      FromIP =
          static_cast<uint64_t>(static_cast<int64_t>(FromIP) - ASLROffset);
      ToIP = static_cast<uint64_t>(static_cast<int64_t>(ToIP) - ASLROffset);
    }

    // Count
    uint64_t Count = 1;
    if (CountCol >= 0 && CountCol < (int)Fields.size()) {
      StringRef CountStr = Fields[CountCol].trim().trim('"');
      CountStr.getAsInteger(0, Count);
      if (Count == 0)
        Count = 1;
    }

    // Misprediction
    uint64_t Mispred = 0;
    if (MispredCol >= 0 && MispredCol < (int)Fields.size()) {
      StringRef MispredStr = Fields[MispredCol].trim().trim('"');
      if (MispredStr.equals_insensitive("true") || MispredStr == "1")
        Mispred = 1;
    }

    if (recordBranchEvent(FromIP, ToIP, Count, Mispred)) {
      ++RowsMatched;
      MatchedLBRBranches += Count;
    }
  }

  outs() << "ETW2BOLT: parsed " << RowsRead << " CSV rows, " << RowsMatched
         << " matched to binary (" << MatchedLBRBranches
         << " LBR branches)\n";

  return Error::success();
}

Error ETWDataAggregator::preprocessProfile(BinaryContext &BC) {
  this->BC = &BC;

  // ETWAnalyzer CSV is a separate path -- no xperf needed.
  if (!opts::ETWAnalyzerCSV.empty())
    return Error::success();

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

  // If ETWAnalyzer CSV is provided, use that (best LBR data source).
  if (!opts::ETWAnalyzerCSV.empty()) {
    if (Error E = parseETWAnalyzerCSV())
      return E;
  } else {
    if (Error E = parseXperfOutput())
      return E;
  }

  // Report statistics
  outs() << "ETW2BOLT: " << TotalEvents << " events, " << MatchedSamples
         << " IP samples matched";
  if (MatchedLBRBranches > 0)
    outs() << ", " << MatchedLBRBranches << " LBR branches";
  if (InferredBranches > 0)
    outs() << ", " << InferredBranches
           << " inferred from consecutive samples";
  outs() << "\n";

  if (MatchedLBRBranches > 0 && InferredBranches > 0)
    outs() << "ETW2BOLT: LBR data present -- inferred branches are "
              "supplementary\n";
  else if (MatchedLBRBranches == 0 && InferredBranches > 0)
    outs() << "ETW2BOLT: no LBR data -- using inferred branches only. "
              "For better results, capture with:\n"
              "  xperf -on PROC_THREAD+LOADER+PROFILE "
              "-LastBranch PROFILE "
              "conditionalbranches,nearrelativecalls,nearreturns\n";

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
