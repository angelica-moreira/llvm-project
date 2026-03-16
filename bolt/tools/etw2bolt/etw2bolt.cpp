//===- bolt/tools/etw2bolt/etw2bolt.cpp - ETW to BOLT fdata converter -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Convert Windows ETW (Event Tracing for Windows) LBR branch trace data to
// BOLT's fdata profile format.
//
// Typical workflow:
//   1. Capture ETW LBR traces:
//        xperf -on PROC_THREAD+LOADER+PMC_PROFILE -PmcProfile BranchMispredictions
//        <run workload>
//        xperf -d trace.etl
//   2. Export branch events to CSV:
//        xperf -i trace.etl -o branches.csv -a dumper -provider Microsoft-Windows-...
//      Or use a simpler format (see below).
//   3. Convert to BOLT fdata:
//        etw2bolt -exe=binary.exe -csv=branches.csv -o=profile.fdata
//   4. Optimize:
//        llvm-bolt binary.exe -o opt.exe -data=profile.fdata -reorder-blocks=ext-tsp
//
// Input CSV format (one branch record per line):
//   <from_address>,<to_address>[,<mispredicted>]
//
// Addresses are hex (with optional 0x prefix) virtual addresses that fall
// within the target binary's code sections.  Records targeting other modules
// are silently dropped.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/COFF.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <map>
#include <string>

using namespace llvm;

namespace opts {
cl::opt<std::string> ExecutablePath("exe",
                                    cl::desc("Path to the target PE executable"),
                                    cl::Required);

cl::opt<std::string> CSVPath("csv",
                             cl::desc("Path to ETW branch CSV file"),
                             cl::Required);

cl::opt<std::string> OutputPath("o", cl::desc("Output fdata file path"),
                                cl::Required);

cl::opt<bool> Verbose("v", cl::desc("Verbose output"), cl::init(false));
} // namespace opts

namespace {

// A function range discovered from .pdata.
struct FuncRange {
  uint64_t VA;   // Virtual address (ImageBase + RVA).
  uint32_t Size; // Function size in bytes.
};

// A counted branch edge between two functions.
struct BranchEdge {
  uint64_t FromFunc; // Source function VA.
  uint64_t FromOff;  // Offset within source function.
  uint64_t ToFunc;   // Target function VA.
  uint64_t ToOff;    // Offset within target function.
  uint64_t Count;    // Number of times this branch was taken.
  uint64_t Mispreds; // Number of mispredictions.
};

// Parse a hex address from a string, accepting optional 0x prefix.
bool parseHexAddress(StringRef S, uint64_t &Addr) {
  S = S.trim();
  if (S.consume_front("0x") || S.consume_front("0X"))
    return !S.getAsInteger(16, Addr);
  return !S.getAsInteger(16, Addr);
}

} // namespace

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv,
                               "etw2bolt - convert ETW LBR traces to BOLT "
                               "fdata\n");

  // Load the PE binary and build a function map from .pdata.
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
      MemoryBuffer::getFile(opts::ExecutablePath);
  if (!BufOrErr) {
    errs() << "etw2bolt: cannot open " << opts::ExecutablePath << ": "
           << BufOrErr.getError().message() << "\n";
    return 1;
  }

  Expected<std::unique_ptr<object::ObjectFile>> ObjOrErr =
      object::ObjectFile::createObjectFile((*BufOrErr)->getMemBufferRef());
  if (!ObjOrErr) {
    errs() << "etw2bolt: cannot parse " << opts::ExecutablePath << ": "
           << toString(ObjOrErr.takeError()) << "\n";
    return 1;
  }

  auto *COFF = dyn_cast<object::COFFObjectFile>(ObjOrErr->get());
  if (!COFF) {
    errs() << "etw2bolt: " << opts::ExecutablePath
           << " is not a PE/COFF binary\n";
    return 1;
  }

  uint64_t ImageBase = COFF->getImageBase();

  // Find .text section VA range so we can filter branches that belong to
  // this binary.
  uint64_t TextVA = 0;
  uint64_t TextSize = 0;
  for (const auto &Sec : COFF->sections()) {
    Expected<StringRef> Name = Sec.getName();
    if (Name && *Name == ".text") {
      TextVA = Sec.getAddress();
      TextSize = Sec.getSize();
      break;
    }
  }

  if (!TextVA) {
    errs() << "etw2bolt: no .text section found in " << opts::ExecutablePath
           << "\n";
    return 1;
  }

  // Build sorted function table from .pdata (exception directory).
  // Each RUNTIME_FUNCTION is {BeginAddress, EndAddress, UnwindInfoAddress},
  // all as 32-bit RVAs.
  std::vector<FuncRange> Functions;

  const object::data_directory *ExcDir = COFF->getDataDirectory(
      COFF::EXCEPTION_TABLE);
  if (!ExcDir || ExcDir->Size == 0) {
    errs() << "etw2bolt: no .pdata (exception table) in "
           << opts::ExecutablePath << "\n";
    return 1;
  }

  uintptr_t ExcAddr = 0;
  if (COFF->getRvaPtr(ExcDir->RelativeVirtualAddress, ExcAddr)) {
    errs() << "etw2bolt: cannot read .pdata\n";
    return 1;
  }

  unsigned NumEntries = ExcDir->Size / 12; // Each RUNTIME_FUNCTION is 12 bytes.
  const auto *Entries =
      reinterpret_cast<const support::ulittle32_t *>(ExcAddr);

  for (unsigned I = 0; I < NumEntries; ++I) {
    uint32_t BeginRVA = Entries[I * 3];
    uint32_t EndRVA = Entries[I * 3 + 1];
    if (EndRVA > BeginRVA) {
      Functions.push_back({ImageBase + BeginRVA, EndRVA - BeginRVA});
    }
  }

  llvm::sort(Functions,
             [](const FuncRange &A, const FuncRange &B) { return A.VA < B.VA; });

  if (opts::Verbose)
    outs() << "etw2bolt: loaded " << Functions.size()
           << " functions from .pdata\n"
           << "etw2bolt: .text VA range: 0x" << Twine::utohexstr(TextVA)
           << " - 0x" << Twine::utohexstr(TextVA + TextSize) << "\n"
           << "etw2bolt: first function VA: 0x"
           << Twine::utohexstr(Functions.front().VA) << ", last: 0x"
           << Twine::utohexstr(Functions.back().VA) << "\n";

  // Helper: given a VA, find the containing function using binary search.
  auto findFunction = [&](uint64_t VA) -> const FuncRange * {
    if (VA < TextVA || VA >= TextVA + TextSize)
      return nullptr;

    // Upper bound gives us the first function with VA > target, so we
    // go back one to find the function containing the address.
    auto It = std::upper_bound(
        Functions.begin(), Functions.end(), VA,
        [](uint64_t Addr, const FuncRange &F) { return Addr < F.VA; });
    if (It == Functions.begin())
      return nullptr;
    --It;
    if (VA >= It->VA && VA < It->VA + It->Size)
      return &*It;
    return nullptr;
  };

  // Read the CSV file with branch records.
  ErrorOr<std::unique_ptr<MemoryBuffer>> CSVBufOrErr =
      MemoryBuffer::getFile(opts::CSVPath);
  if (!CSVBufOrErr) {
    errs() << "etw2bolt: cannot open " << opts::CSVPath << ": "
           << CSVBufOrErr.getError().message() << "\n";
    return 1;
  }

  // Parse branch records and aggregate into edge counts.
  // Key: (from_func_va, from_offset, to_func_va, to_offset)
  struct EdgeKey {
    uint64_t FromFunc;
    uint64_t FromOff;
    uint64_t ToFunc;
    uint64_t ToOff;
    bool operator<(const EdgeKey &O) const {
      return std::tie(FromFunc, FromOff, ToFunc, ToOff) <
             std::tie(O.FromFunc, O.FromOff, O.ToFunc, O.ToOff);
    }
  };

  struct EdgeCount {
    uint64_t Count = 0;
    uint64_t Mispreds = 0;
  };

  std::map<EdgeKey, EdgeCount> Edges;
  uint64_t TotalRecords = 0;
  uint64_t MatchedRecords = 0;

  StringRef CSV = (*CSVBufOrErr)->getBuffer();
  SmallVector<StringRef, 0> Lines;
  CSV.split(Lines, '\n');

  for (const StringRef &Line : Lines) {
    StringRef Trimmed = Line.trim();
    if (Trimmed.empty() || Trimmed.starts_with("#") ||
        Trimmed.starts_with("//"))
      continue;

    // Parse: from_addr,to_addr[,mispredicted]
    SmallVector<StringRef, 4> Fields;
    Trimmed.split(Fields, ',');
    if (Fields.size() < 2)
      continue;

    uint64_t FromAddr = 0, ToAddr = 0;
    if (!parseHexAddress(Fields[0], FromAddr) ||
        !parseHexAddress(Fields[1], ToAddr))
      continue;

    bool Mispred = false;
    if (Fields.size() >= 3) {
      StringRef M = Fields[2].trim();
      Mispred = (M == "1" || M == "true" || M == "Y");
    }

    ++TotalRecords;

    const FuncRange *FromFunc = findFunction(FromAddr);
    const FuncRange *ToFunc = findFunction(ToAddr);
    if (!FromFunc || !ToFunc)
      continue;

    ++MatchedRecords;

    EdgeKey Key{FromFunc->VA, FromAddr - FromFunc->VA, ToFunc->VA,
                ToAddr - ToFunc->VA};
    auto &EC = Edges[Key];
    EC.Count++;
    if (Mispred)
      EC.Mispreds++;
  }

  if (opts::Verbose)
    outs() << "etw2bolt: " << TotalRecords << " branch records, "
           << MatchedRecords << " matched to functions, " << Edges.size()
           << " unique edges\n";

  if (Edges.empty()) {
    errs() << "etw2bolt: no branch records matched the target binary\n";
    return 1;
  }

  // Write fdata output.
  // Format: <is_sym> <func_name> <offset> <is_sym> <func_name> <offset>
  //         <mispreds> <count>
  std::error_code EC;
  raw_fd_ostream OS(opts::OutputPath, EC, sys::fs::OF_Text);
  if (EC) {
    errs() << "etw2bolt: cannot open output " << opts::OutputPath << ": "
           << EC.message() << "\n";
    return 1;
  }

  for (const auto &[Key, Count] : Edges) {
    // Use "1" (global symbol) and the BOLT naming convention func_0x<VA>.
    OS << "1 func_0x" << Twine::utohexstr(Key.FromFunc) << " "
       << format_hex_no_prefix(Key.FromOff, 1) << " "
       << "1 func_0x" << Twine::utohexstr(Key.ToFunc) << " "
       << format_hex_no_prefix(Key.ToOff, 1) << " " << Count.Mispreds << " "
       << Count.Count << "\n";
  }

  outs() << "etw2bolt: wrote " << Edges.size() << " branch records to "
         << opts::OutputPath << "\n";

  return 0;
}
