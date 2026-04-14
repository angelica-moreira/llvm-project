//===- bolt/Profile/ETWDataAggregator.cpp - ETW data aggregator -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Windows equivalent of DataAggregator.  Reads ETW (Event Tracing for Windows)
// traces and aggregates them into BOLT's branch profile format.
//
// Supports three input modes:
//   1. Raw ETL file  -- shells out to xperf to convert to text
//   2. Pre-dumped xperf text  -- from `xperf -i trace.etl -a dumper`
//   3. ETWAnalyzer LBR CSV  -- from `ETWAnalyzer -dump LBR -csv`
//
// Within the xperf text dump, two event types are parsed:
//   - SampledProfile:  timer-interrupt IP samples (always present)
//   - BranchTrace/LastBranch:  LBR from/to pairs (present only when the
//     trace was captured with -LastBranch)
//
// LBR data is strongly preferred: it gives exact branch edges that map
// directly to BOLT's profile model.  Timer samples are used as fallback
// when LBR is unavailable, with branch edges inferred from consecutive
// samples on the same thread (noisy but better than nothing).
//
//===----------------------------------------------------------------------===//

#include "bolt/Profile/ETWDataAggregator.h"
#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/Debug.h"
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
                   cl::desc("ETWAnalyzer LBR CSV "
                            "(from: ETWAnalyzer -dump LBR -csv <file>)"),
                   cl::Optional, cl::cat(AggregatorCategory));
} // namespace opts

ETWDataAggregator::~ETWDataAggregator() {
  if (!DumpFilePath.empty() && opts::ETWDumpFile.empty())
    sys::fs::remove(DumpFilePath);
}

bool ETWDataAggregator::checkETLMagic(StringRef FileName) {
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
        "-xperf-path=<path>, or dump manually with -etw-dump=<file>");

  // Write next to the ETL file, not %TEMP% (AV may block temp writes).
  DumpFilePath = ETLFilename + ".dump.txt";

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

  // xperf returns non-zero when events were lost, which is common and
  // harmless.  Only fail if the output file is empty or missing.
  uint64_t FileSize = 0;
  sys::fs::file_size(DumpFilePath, FileSize);
  if (FileSize == 0)
    return createStringError(errc::executable_format_error,
                             "xperf produced no output (exit %d). "
                             "Try running as Administrator, or dump manually:\n"
                             "  xperf -i %s -a dumper > dump.txt\n"
                             "  etw2bolt ... -etw-dump=dump.txt",
                             RC, ETLFilename.c_str());

  if (RC != 0)
    outs() << "ETW2BOLT: xperf warnings (exit " << RC << "), proceeding with "
           << (FileSize / 1024) << " KB of data\n";

  return Error::success();
}

/// Parse a hex field from a comma-separated string, stripping the 0x prefix.
/// Returns 0 on failure.
static uint64_t parseHex(StringRef Field) {
  uint64_t Val = 0;
  StringRef S = Field.trim();
  if (S.consume_front("0x") || S.consume_front("0X"))
    S.getAsInteger(16, Val);
  else
    S.getAsInteger(0, Val);
  return Val;
}

/// Apply ASLR correction: subtract the offset between the runtime base
/// address and the preferred ImageBase so that all addresses match what
/// BinaryContext expects.
static uint64_t adjustForASLR(uint64_t Addr, int64_t Offset) {
  if (Offset == 0)
    return Addr;
  return static_cast<uint64_t>(static_cast<int64_t>(Addr) - Offset);
}

bool ETWDataAggregator::recordBranchEvent(uint64_t From, uint64_t To,
                                          uint64_t Count, uint64_t Mispreds) {
  // Mirror DataAggregator::doBranch() -- resolve addresses to functions,
  // compute function-relative offsets, record into NamesToBranches.
  BinaryFunction *FromFunc =
      BC->getBinaryFunctionContainingAddress(From, false, true);
  BinaryFunction *ToFunc =
      BC->getBinaryFunctionContainingAddress(To, false, true);

  if (!FromFunc && !ToFunc)
    return false;

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

  // Inter-function: record as call from source, entry into target.
  if (FromFunc) {
    StringRef FromName = FromFunc->getOneName();
    FuncBranchData &FBD = NamesToBranches[FromName];
    FBD.Name = FromName;
    FBD.bumpCallCount(
        FromOffset,
        Location(ToFunc != nullptr, ToFunc ? ToFunc->getOneName() : "",
                 ToOffset),
        Count, Mispreds);
  }
  if (ToFunc) {
    StringRef ToName = ToFunc->getOneName();
    FuncBranchData &FBD = NamesToBranches[ToName];
    FBD.Name = ToName;
    FBD.bumpEntryCount(
        Location(FromFunc != nullptr,
                 FromFunc ? FromFunc->getOneName() : "", FromOffset),
        ToOffset, Count, Mispreds);
  }

  return true;
}

uint64_t ETWDataAggregator::readPreferredBase() const {
  ErrorOr<std::unique_ptr<MemoryBuffer>> FileBuf =
      MemoryBuffer::getFile(BC->getFilename());
  if (!FileBuf)
    return 0;
  Expected<std::unique_ptr<object::ObjectFile>> ObjOrErr =
      object::ObjectFile::createObjectFile((*FileBuf)->getMemBufferRef());
  if (!ObjOrErr) {
    consumeError(ObjOrErr.takeError());
    return 0;
  }
  if (auto *COFF = dyn_cast<object::COFFObjectFile>(ObjOrErr->get()))
    return COFF->getImageBase();
  return 0;
}

void ETWDataAggregator::parseImageLoadEvents(StringRef Dump) {
  uint64_t PreferredBase = readPreferredBase();
  if (PreferredBase == 0 && !BC->getBinaryFunctions().empty()) {
    PreferredBase =
        BC->getBinaryFunctions().begin()->second.getAddress() & ~0xFFFFFULL;
    errs() << "ETW2BOLT: warning: cannot read PE ImageBase; derived 0x"
           << Twine::utohexstr(PreferredBase)
           << " -- ASLR adjustment may be wrong\n";
  }

  StringRef BinaryName = llvm::sys::path::filename(BC->getFilename());

  while (!Dump.empty()) {
    auto [Line, Rest] = Dump.split('\n');
    Dump = Rest;

    // Handle \r\n line endings (common in Windows tool output).
    StringRef Trimmed = Line.ltrim();
    if (Trimmed.ends_with("\r"))
      Trimmed = Trimmed.drop_back(1);

    if (!Trimmed.starts_with("I-Start"))
      continue;

    SmallVector<StringRef, 16> Parts;
    Trimmed.split(Parts, ',');
    if (Parts.size() < 4)
      continue;

    StringRef ProcField = Parts[2].trim();
    if (!ProcField.starts_with_insensitive(BinaryName))
      continue;
    if (ProcField.size() > BinaryName.size()) {
      char After = ProcField[BinaryName.size()];
      if (After != ' ' && After != '(' && After != '\t')
        continue;
    }

    uint64_t ActualBase = parseHex(Parts[3]);
    if (ActualBase == 0)
      continue;

    ASLROffset =
        static_cast<int64_t>(ActualBase) - static_cast<int64_t>(PreferredBase);
    outs() << "ETW2BOLT: " << BinaryName << " loaded at 0x"
           << Twine::utohexstr(ActualBase) << " (preferred 0x"
           << Twine::utohexstr(PreferredBase) << ")\n";
    if (ASLROffset != 0)
      outs() << "ETW2BOLT: ASLR delta: " << (ASLROffset > 0 ? "+" : "")
             << ASLROffset << "\n";
    return;
  }

  outs() << "ETW2BOLT: no I-Start event for " << BinaryName
         << ", assuming no ASLR\n";
}

/// Infer the ASLR offset by probing stack walk frames in SampledProfile
/// events.  Each frame has the form "binary.exe!0xRuntimeAddr".  We
/// subtract candidate offsets (aligned to 64KB, the ASLR granularity)
/// until the adjusted address resolves to a known function.
void ETWDataAggregator::detectASLRFromSamples(StringRef Dump) {
  uint64_t PreferredBase = readPreferredBase();
  if (PreferredBase == 0)
    PreferredBase =
        BC->getBinaryFunctions().begin()->second.getAddress() & ~0xFFFFFULL;

  StringRef BinaryName = llvm::sys::path::filename(BC->getFilename());

  while (!Dump.empty()) {
    auto [Line, Rest] = Dump.split('\n');
    Dump = Rest;

    StringRef Trimmed = Line.ltrim();
    if (!Trimmed.starts_with("SampledProfile"))
      continue;

    SmallVector<StringRef, 12> Cols;
    Trimmed.split(Cols, ',');

    for (size_t I = 6; I < Cols.size(); ++I) {
      StringRef Token = Cols[I].trim();
      if (!Token.starts_with_insensitive(BinaryName))
        continue;
      size_t Bang = Token.find('!');
      if (Bang == StringRef::npos)
        continue;
      uint64_t RuntimeAddr = parseHex(Token.substr(Bang + 1));
      if (RuntimeAddr == 0)
        continue;

      int64_t Guess = static_cast<int64_t>(RuntimeAddr & ~0xFFFFULL) -
                      static_cast<int64_t>(PreferredBase);

      for (int64_t Delta = -0x10000; Delta <= 0x10000; Delta += 0x10000) {
        int64_t Candidate = Guess + Delta;
        uint64_t Adjusted = RuntimeAddr - Candidate;
        if (BC->getBinaryFunctionContainingAddress(Adjusted, false, true)) {
          ASLROffset = Candidate;
          outs() << "ETW2BOLT: detected ASLR offset from stack sample: "
                 << (ASLROffset > 0 ? "+" : "") << ASLROffset
                 << " (runtime 0x" << Twine::utohexstr(RuntimeAddr)
                 << " -> binary 0x" << Twine::utohexstr(Adjusted) << ")\n";
          return;
        }
      }
    }
  }
}

Error ETWDataAggregator::parseXperfOutput() {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
      MemoryBuffer::getFile(DumpFilePath);
  if (!BufOrErr)
    return createStringError(BufOrErr.getError(), "cannot read xperf dump %s",
                             DumpFilePath.c_str());

  uint64_t BufSize = (*BufOrErr)->getBufferSize();
  outs() << "ETW2BOLT: parsing " << (BufSize / 1024) << " KB xperf dump...\n";

  StringRef RawBuf = (*BufOrErr)->getBuffer();

  // xperf on Windows produces UTF-16 LE output.  Detect the BOM and convert
  // to UTF-8 so the line parser works with normal ASCII comparisons.
  std::string UTF8Storage;
  StringRef Dump;
  if (RawBuf.size() >= 2 &&
      static_cast<uint8_t>(RawBuf[0]) == 0xFF &&
      static_cast<uint8_t>(RawBuf[1]) == 0xFE) {
    outs() << "ETW2BOLT: converting UTF-16 LE dump to UTF-8...\n";
    ArrayRef<char> Src(RawBuf.data() + 2, RawBuf.size() - 2);
    if (!convertUTF16ToUTF8String(Src, UTF8Storage))
      return createStringError(std::errc::illegal_byte_sequence,
                               "failed to convert UTF-16 xperf dump to UTF-8");
    outs() << "ETW2BOLT: converted to " << (UTF8Storage.size() / 1024)
           << " KB UTF-8\n";
    Dump = StringRef(UTF8Storage);
  } else {
    Dump = RawBuf;
  }

  parseImageLoadEvents(Dump);

  // If the target process was already running when the trace started, no
  // I-Start event exists.  Infer the ASLR offset from the first stack
  // walk frame that names our binary.
  if (ASLROffset == 0 && BC && !BC->getBinaryFunctions().empty())
    detectASLRFromSamples(Dump);

  // Two event types in the dump:
  //
  //   SampledProfile, <ts>, proc (PID), <tid>, <IP>, <cpu>, ...
  //     Timer-interrupt sample.  Always present.
  //
  //   BranchTrace, <ts>, proc (PID), <tid>, <From>, <To>, <mispred>, ...
  //   LastBranch, <ts>, proc (PID), <tid>, <From>, <To>, <mispred>, ...
  //     LBR record.  Only present when captured with -LastBranch.
  //     Contains the exact branch source and target addresses.

  StringRef Remaining = Dump;
  uint64_t LinesProcessed = 0;

  while (!Remaining.empty()) {
    auto [Line, Rest] = Remaining.split('\n');
    Remaining = Rest;
    ++LinesProcessed;

    if ((LinesProcessed & 0xFFFFF) == 0)
      outs() << "ETW2BOLT: " << (LinesProcessed / 1000000) << "M lines, "
             << MatchedSamples << " samples, " << MatchedLBRBranches
             << " LBR\r";

    StringRef Trimmed = Line.ltrim();
    if (Trimmed.ends_with("\r"))
      Trimmed = Trimmed.drop_back(1);

    // LBR branch records.
    if (Trimmed.starts_with("BranchTrace") ||
        Trimmed.starts_with("LastBranch")) {
      ++TotalEvents;

      SmallVector<StringRef, 12> Parts;
      Trimmed.split(Parts, ',');
      if (Parts.size() < 7)
        continue;

      uint64_t FromIP = adjustForASLR(parseHex(Parts[4]), ASLROffset);
      uint64_t ToIP = adjustForASLR(parseHex(Parts[5]), ASLROffset);
      if (FromIP == 0 || ToIP == 0)
        continue;

      uint64_t Mispred = 0;
      StringRef MF = Parts[6].trim();
      if (MF.equals_insensitive("true") || MF == "1")
        Mispred = 1;

      if (recordBranchEvent(FromIP, ToIP, 1, Mispred))
        ++MatchedLBRBranches;

      continue;
    }

    // Timer-interrupt IP samples.
    if (!Trimmed.starts_with("SampledProfile"))
      continue;

    ++TotalEvents;

    SmallVector<StringRef, 12> Parts;
    Trimmed.split(Parts, ',');
    if (Parts.size() < 6)
      continue;

    uint64_t ThreadID = 0;
    Parts[3].trim().getAsInteger(0, ThreadID);

    // Column 4 is the raw interrupted IP.  When the sample hits in the
    // kernel (e.g. during a syscall), this is a kernel address that won't
    // match any function in the binary.  The -stackwalk columns (6+)
    // contain "module!0xaddr" tokens; scan them for a user-mode IP that
    // belongs to our binary.
    uint64_t IP = adjustForASLR(parseHex(Parts[4]), ASLROffset);
    if (!BC->getBinaryFunctionContainingAddress(IP, false, true)) {
      // Try stack walk columns for a matching user-mode address.
      IP = 0;
      StringRef BinaryName = llvm::sys::path::filename(BC->getFilename());
      for (size_t I = 6; I < Parts.size(); ++I) {
        StringRef Token = Parts[I].trim();
        if (!Token.starts_with_insensitive(BinaryName))
          continue;
        size_t Bang = Token.find('!');
        if (Bang == StringRef::npos)
          continue;
        IP = adjustForASLR(parseHex(Token.substr(Bang + 1)), ASLROffset);
        if (IP && BC->getBinaryFunctionContainingAddress(IP, false, true))
          break;
        IP = 0;
      }
    }
    if (IP == 0)
      continue;

    ++MatchedSamples;

    // Record the IP sample for no_lbr output.
    const BinaryFunction *Func =
        BC->getBinaryFunctionContainingAddress(IP, false, true);
    if (Func) {
      uint64_t Offset = IP - Func->getAddress();
      std::string Name = Func->getOneName().str();
      BasicSamples[{Name, Offset}] += 1;
    }

    // When LBR data is available, edges come from BranchTrace/LastBranch
    // events above.  As a fallback, infer edges from consecutive timer
    // samples on the same thread.  These are noisy (the timer fires ~1ms
    // apart) but provide some signal when LBR is not available.
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
  // When real LBR branch data was collected, write the standard edge format.
  if (MatchedLBRBranches > 0)
    return writeBranchProfile(OutputFilename);

  // Timer-only profile: write no_lbr format with per-IP sample counts.
  // BOLT uses these to estimate basic block execution counts.
  std::error_code EC;
  raw_fd_ostream OutFile(OutputFilename, EC, sys::fs::OpenFlags::OF_None);
  if (EC)
    return EC;

  OutFile << "no_lbr\n";
  uint64_t Written = 0;
  for (const auto &[Key, Count] : BasicSamples) {
    const auto &[FuncName, Offset] = Key;
    OutFile << "1 " << FuncName << " "
            << Twine::utohexstr(Offset) << " " << Count << "\n";
    ++Written;
  }

  outs() << "BOLT-INFO: wrote " << Written
         << " basic samples (no_lbr) to " << OutputFilename << "\n";
  return std::error_code();
}

Error ETWDataAggregator::parseETWAnalyzerCSV() {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
      MemoryBuffer::getFile(opts::ETWAnalyzerCSV);
  if (!BufOrErr)
    return createStringError(BufOrErr.getError(),
                             "cannot read ETWAnalyzer CSV %s",
                             opts::ETWAnalyzerCSV.c_str());

  outs() << "ETW2BOLT: parsing ETWAnalyzer LBR CSV...\n";

  StringRef Content = (*BufOrErr)->getBuffer();

  auto [HeaderLine, Body] = Content.split('\n');
  if (HeaderLine.starts_with("\xef\xbb\xbf"))
    HeaderLine = HeaderLine.drop_front(3); // strip UTF-8 BOM

  SmallVector<StringRef, 16> Headers;
  HeaderLine.split(Headers, ',');

  // Map column names to indices.  ETWAnalyzer output varies between
  // versions, so accept several synonyms for each field.
  int FromCol = -1, ToCol = -1, CountCol = -1, MispredCol = -1;
  for (int I = 0, E = (int)Headers.size(); I < E; ++I) {
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

  if (FromCol < 0 || ToCol < 0)
    return createStringError(
        errc::invalid_argument,
        "CSV missing From/To address columns.  Expected output from:\n"
        "  ETWAnalyzer -dump LBR -fd <file> -csv lbr.csv");

  int MaxCol = std::max(FromCol, ToCol);

  // ETWAnalyzer CSV contains runtime (ASLR'd) addresses, not file RVAs.
  // Auto-detect the ASLR offset by probing the first address against
  // BinaryContext's function map at various base offsets.
  if (ASLROffset == 0 && BC) {
    uint64_t PreferredBase = readPreferredBase();
    if (PreferredBase != 0) {
      // Peek at the first data row to get a sample address.
      StringRef Peek = Body;
      while (!Peek.empty()) {
        auto [PeekLine, PeekRest] = Peek.split('\n');
        Peek = PeekRest;
        if (PeekLine.trim().empty())
          continue;
        SmallVector<StringRef, 16> PeekFields;
        PeekLine.split(PeekFields, ',');
        if ((int)PeekFields.size() <= MaxCol)
          continue;
        uint64_t SampleAddr = parseHex(PeekFields[FromCol].trim('"'));
        if (SampleAddr == 0)
          continue;
        // If the raw address already resolves, no ASLR adjustment needed.
        if (BC->getBinaryFunctionContainingAddress(SampleAddr, false, true))
          break;
        // Try with a guessed offset: sample is likely near ImageBase.
        uint64_t GuessBase = SampleAddr & ~0xFFFFFULL;
        int64_t GuessOffset =
            static_cast<int64_t>(GuessBase) -
            static_cast<int64_t>(PreferredBase);
        uint64_t Adjusted = static_cast<uint64_t>(
            static_cast<int64_t>(SampleAddr) - GuessOffset);
        if (BC->getBinaryFunctionContainingAddress(Adjusted, false, true)) {
          ASLROffset = GuessOffset;
          outs() << "ETW2BOLT: detected ASLR offset "
                 << (ASLROffset > 0 ? "+" : "") << ASLROffset
                 << " from CSV addresses\n";
        }
        break;
      }
    }
  }

  uint64_t RowsRead = 0, RowsMatched = 0;

  while (!Body.empty()) {
    auto [Line, Rest] = Body.split('\n');
    Body = Rest;
    if (Line.trim().empty())
      continue;

    SmallVector<StringRef, 16> Fields;
    Line.split(Fields, ',');
    if ((int)Fields.size() <= MaxCol)
      continue;

    ++RowsRead;

    uint64_t FromIP =
        adjustForASLR(parseHex(Fields[FromCol].trim('"')), ASLROffset);
    uint64_t ToIP =
        adjustForASLR(parseHex(Fields[ToCol].trim('"')), ASLROffset);
    if (FromIP == 0 || ToIP == 0)
      continue;

    uint64_t Count = 1;
    if (CountCol >= 0 && CountCol < (int)Fields.size()) {
      StringRef S = Fields[CountCol].trim().trim('"');
      S.getAsInteger(0, Count);
      Count = std::max(Count, uint64_t(1));
    }

    uint64_t Mispred = 0;
    if (MispredCol >= 0 && MispredCol < (int)Fields.size()) {
      StringRef S = Fields[MispredCol].trim().trim('"');
      if (S.equals_insensitive("true") || S == "1")
        Mispred = 1;
    }

    if (recordBranchEvent(FromIP, ToIP, Count, Mispred)) {
      ++RowsMatched;
      MatchedLBRBranches += Count;
    }
  }

  outs() << "ETW2BOLT: " << RowsRead << " CSV rows, " << RowsMatched
         << " matched (" << MatchedLBRBranches << " LBR branches)\n";

  return Error::success();
}

Error ETWDataAggregator::preprocessProfile(BinaryContext &BC) {
  this->BC = &BC;

  if (!opts::ETWAnalyzerCSV.empty())
    return Error::success();

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

  if (!opts::ETWAnalyzerCSV.empty()) {
    if (Error E = parseETWAnalyzerCSV())
      return E;
  } else {
    if (Error E = parseXperfOutput())
      return E;
  }

  outs() << "ETW2BOLT: " << TotalEvents << " events, " << MatchedSamples
         << " samples";
  if (MatchedLBRBranches > 0)
    outs() << ", " << MatchedLBRBranches << " LBR branches";
  if (InferredBranches > 0)
    outs() << ", " << InferredBranches << " inferred edges";
  outs() << "\n";

  if (MatchedLBRBranches == 0 && InferredBranches > 0)
    outs() << "ETW2BOLT: no LBR data found.  For better results, capture "
              "with:\n"
              "  xperf -on PROC_THREAD+LOADER+PROFILE "
              "-LastBranch PROFILE "
              "conditionalbranches,nearrelativecalls,nearreturns\n";

  if (NamesToBranches.empty()) {
    errs() << "ETW2BOLT: no profile data matched the binary\n";
    return Error::success();
  }

  if (opts::AggregateOnly) {
    if (std::error_code EC = writeAggregatedFile(opts::OutputFilename))
      return errorCodeToError(EC);
  }

  return Error::success();
}

bool ETWDataAggregator::mayHaveProfileData(const BinaryFunction &BF) {
  return BF.hasProfileAvailable();
}
