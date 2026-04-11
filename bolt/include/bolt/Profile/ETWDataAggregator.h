//===- bolt/Profile/ETWDataAggregator.h - ETW data aggregator ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Windows equivalent of DataAggregator.  Reads ETW trace data and converts
// it into BOLT's profile format.
//
// Architecture mirrors perf2bolt:
//   perf2bolt:  perf.data  -> `perf script` -> parse text -> profile
//   etw2bolt:   trace.etl  -> `xperf -a dumper` -> parse text -> profile
//               -or-       ETWAnalyzer CSV (LBR) -> parse CSV -> profile
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

  static bool checkETLMagic(StringRef FileName);

private:
  std::string ETLFilename;
  BinaryContext *BC{nullptr};

  std::string XperfPath;
  std::string DumpFilePath;

  std::string findXperf() const;
  Error launchXperf();

  /// Read the preferred ImageBase from the PE header.
  uint64_t readPreferredBase() const;

  /// Scan I-Start events to find the runtime load address (ASLR).
  void parseImageLoadEvents(StringRef Dump);

  /// Parse xperf dump text: SampledProfile events and LBR branch records.
  Error parseXperfOutput();

  /// Parse ETWAnalyzer -dump LBR -csv output.
  Error parseETWAnalyzerCSV();

  /// Record a branch from absolute address From to To.  Resolves to
  /// function-relative offsets and updates NamesToBranches.
  bool recordBranchEvent(uint64_t From, uint64_t To, uint64_t Count,
                         uint64_t Mispreds);

  std::error_code writeAggregatedFile(StringRef OutputFilename) const;

  /// Per-thread last IP, used to infer edges from consecutive samples.
  std::map<uint64_t, uint64_t> LastIPPerThread;

  /// Runtime load address minus preferred ImageBase.
  int64_t ASLROffset{0};

  uint64_t TotalEvents{0};
  uint64_t MatchedSamples{0};
  uint64_t MatchedLBRBranches{0};
  uint64_t InferredBranches{0};
};

} // namespace bolt
} // namespace llvm

#endif
