//===- bolt/Passes/InferEdgeCounts.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Infer CFG edge counts from basic block execution counts.
//
// When profiling without branch records (no LBR), BOLT has per-block sample
// counts but no edge weights.  This pass fills them in using the Wu-Larus
// local frequency propagation algorithm [1], adapted for sample-based block
// counts as described in VESPA [2].
//
//   [1] Wu & Larus, "Static branch frequency and program profile analysis",
//       MICRO-27, 1994.
//   [2] Moreira et al., "VESPA: Static Profiling for Binary Optimization",
//       OOPSLA 2021.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_INFEREDGECOUNTS_H
#define BOLT_PASSES_INFEREDGECOUNTS_H

#include "bolt/Passes/BinaryPasses.h"

namespace llvm {
namespace bolt {

class InferEdgeCounts : public BinaryFunctionPass {
  uint64_t FuncsUpdated{0};

  bool verify(const BinaryFunction &BF) const;

public:
  explicit InferEdgeCounts(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "infer-edge-counts"; }

  Error runOnFunctions(BinaryContext &BC) override;
};

} // namespace bolt
} // namespace llvm

#endif
