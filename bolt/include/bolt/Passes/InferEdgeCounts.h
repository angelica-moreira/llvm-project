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
// When profiling without branch records (no LBR/no brstack), BOLT has
// per-block sample counts but no edge (branch) counts.  Block reordering
// passes such as ext-tsp require edge weights.  This pass fills them in
// by distributing each block's count across its outgoing edges in
// proportion to successor block counts.
//
// References:
//
//   [1] Y. Wu and J. R. Larus, "Static branch frequency and program
//       profile analysis," in Proc. MICRO-27, 1994, pp. 1-11.
//       https://doi.org/10.1145/192724.192725
//
//   [2] A. Moreira et al., "VESPA: Static Profiling for Binary
//       Optimization," in Proc. ACM Program. Lang. (OOPSLA), 2021.
//       https://doi.org/10.1145/3485521
//
// The local frequency propagation follows Section 4 of [1]: process
// blocks in reverse postorder, scale loop headers by the cyclic
// probability, and distribute to outgoing edges.  The adaptation for
// sample-based block counts (rather than heuristic branch probabilities)
// follows the approach used in VESPA [2].
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
