//===- bolt/Profile/ETWDataAggregator.h - ETW data aggregator ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Reads Windows ETW (Event Tracing for Windows) trace data and aggregates it
// into BOLT's profile format.  This is the Windows equivalent of
// DataAggregator which handles Linux perf data.
//
// Architecture mirrors perf2bolt:
//   perf2bolt:  perf.data  -> `perf script` subprocess -> parse text -> fdata
//   etw2bolt:   trace.etl  -> `xperf -a dumper` subprocess -> parse text ->
//   fdata
//
// Inherits DataReader to reuse the NamesToBranches data structures,
// FuncBranchData, BranchInfo, Location types, and the profile matching
// infrastructure.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PROFILE_ETW_DATA_AGGREGATOR_H
#define BOLT_PROFILE_ETW_DATA_AGGREGATOR_H

#include "bolt/Profile/DataReader.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <map>
#include <string>

namespace llvm {
namespace bolt {

class BinaryContext;

class ETWDataAggregator : public DataReader {
public:
  explicit ETWDataAggregator(StringRef Filename)
      : DataReader(Filename), ETLFilename(Filename.str()) {}

  ~ETWDataAggregator() override;

  StringRef getReaderName() const override { return "ETW data aggregator"; }

  bool isTrustedSource() const override { return true; }

  Error preprocessProfile(BinaryContext &BC) override;

  Error readProfilePreCFG(BinaryContext &BC) override {
    return Error::success();
  }

  Error readProfile(BinaryContext &BC) override;

  bool mayHaveProfileData(const BinaryFunction &BF) override;

  /// Check whether a file looks like an ETL trace.
  static bool checkETLMagic(StringRef FileName);

private:
  std::string ETLFilename;
  BinaryContext *BC{nullptr};

  /// Path to xperf.exe, auto-detected or from command line.
  std::string XperfPath;

  /// Path to the temp file holding xperf dumper output.
  std::string DumpFilePath;

  /// Find xperf.exe in standard locations.
  std::string findXperf() const;

  /// Shell out to `xperf -a dumper` to convert ETL to text, just like
  /// DataAggregator shells out to `perf script`.
  Error launchXperf();

  /// First pass: scan for ImageLoad events to detect the actual load
  /// address of the target binary.  With ASLR the runtime address differs
  /// from the preferred ImageBase in the PE header.  This mirrors how
  /// DataAggregator uses mmap events for ASLR on Linux.
  void parseImageLoadEvents(StringRef Dump);

  /// Second pass: parse SampledProfile events, adjust IPs for ASLR, and
  /// aggregate into NamesToBranches.
  Error parseXperfOutput();

  /// Record a branch from absolute address From to To with the given counts.
  /// Resolves addresses to BinaryFunctions via BinaryContext, converts to
  /// function-relative offsets, and updates NamesToBranches -- same logic as
  /// DataAggregator::doBranch().
  bool recordBranchEvent(uint64_t From, uint64_t To, uint64_t Count,
                         uint64_t Mispreds);

  /// Write the aggregated profile to the output fdata file.  Uses the same
  /// format as DataAggregator::writeAggregatedFile().
  std::error_code writeAggregatedFile(StringRef OutputFilename) const;

  /// Per-thread last instruction pointer, used to infer branches from
  /// consecutive samples (basic mode, same as perf2bolt without LBR).
  std::map<uint64_t, uint64_t> LastIPPerThread;

  /// ASLR offset: ActualLoadAddress - PreferredImageBase.
  /// Added to zero when ASLR is not detected, subtracted from sample IPs
  /// to convert runtime addresses to preferred addresses that match
  /// BinaryContext's function map.
  int64_t ASLROffset{0};

  uint64_t TotalEvents{0};
  uint64_t MatchedSamples{0};
};

} // namespace bolt
} // namespace llvm

#endif
