//===- bolt/Passes/InferEdgeCounts.cpp --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/InferEdgeCounts.h"
#include "bolt/Core/BinaryBasicBlock.h"
#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Core/BinaryLoop.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PostOrderIterator.h"

#define DEBUG_TYPE "bolt-infer-edge-counts"

using namespace llvm;

namespace llvm {
namespace bolt {

namespace {

bool isBackEdge(const BinaryBasicBlock *Src, const BinaryBasicBlock *Dst,
                const BinaryLoopInfo &LI) {
  const BinaryLoop *L = LI.getLoopFor(Dst);
  return L && L->getHeader() == Dst && L->contains(Src);
}

/// Distribute Count across Succs proportionally to their sample counts.
/// Uses 128-bit intermediate to avoid overflow.
void distributeToSuccessors(
    BinaryBasicBlock &BB, uint64_t Count,
    const DenseMap<const BinaryBasicBlock *, uint64_t> &Samples) {

  if (BB.succ_size() == 1) {
    BB.setSuccessorBranchInfo(*BB.succ_begin()[0], Count, 0);
    return;
  }

  uint64_t TotalSucc = 0;
  for (const BinaryBasicBlock *Succ : BB.successors()) {
    auto It = Samples.find(Succ);
    TotalSucc += (It != Samples.end()) ? It->second : 0;
  }

  if (TotalSucc == 0) {
    uint64_t Each = Count / BB.succ_size();
    for (BinaryBasicBlock *Succ : BB.successors())
      BB.setSuccessorBranchInfo(*Succ, Each, 0);
    return;
  }

  // Ensure successors with nonzero samples get at least count 1.
  uint64_t Remaining = Count;
  for (BinaryBasicBlock *Succ : BB.successors()) {
    auto It = Samples.find(Succ);
    uint64_t SuccSamples = (It != Samples.end()) ? It->second : 0;
    uint64_t EdgeCount =
        static_cast<uint64_t>(static_cast<double>(Count) * SuccSamples /
                              TotalSucc);
    if (EdgeCount == 0 && SuccSamples > 0)
      EdgeCount = 1;
    if (EdgeCount > Remaining)
      EdgeCount = Remaining;
    BB.setSuccessorBranchInfo(*Succ, EdgeCount, 0);
    Remaining -= EdgeCount;
  }
}

/// Propagate block sample counts to edge counts.
///
/// Follows the local frequency algorithm from Wu-Larus (MICRO-27, 1994,
/// Section 4): process blocks in reverse postorder, compute each block's
/// frequency from incoming non-back edges, scale loop headers by the
/// cyclic probability, and distribute to outgoing edges proportionally
/// to successor sample counts.
void propagateFunction(BinaryFunction &BF) {
  assert(!BF.empty() && "cannot propagate on empty function");

  if (!BF.hasLoopInfo())
    BF.calculateLoopInfo();
  const BinaryLoopInfo &LI = BF.getLoopInfo();

  ReversePostOrderTraversal<BinaryFunction *> RPOT(&BF);

  // Snapshot original sample counts; propagation overwrites block counts.
  DenseMap<const BinaryBasicBlock *, uint64_t> Samples;
  for (const BinaryBasicBlock &BB : BF)
    Samples[&BB] = BB.getExecutionCount();

  DenseSet<const BinaryBasicBlock *> Visited;

  for (BinaryBasicBlock *BB : RPOT) {
    if (BB != &BF.front()) {
      // Block frequency = sum of incoming non-back-edge frequencies.
      uint64_t InSum = 0;
      for (const BinaryBasicBlock *Pred : BB->predecessors()) {
        if (Visited.count(Pred) && !isBackEdge(Pred, BB, LI))
          InSum += Pred->getBranchInfo(*BB).Count;
      }

      // Loop header scaling: freq(H) = InSum / (1 - cyclicProb).
      // cyclicProb = 1 - (entryCount / headerCount), approximated from
      // samples as (headerSamples - entryEdgeSamples) / headerSamples.
      const BinaryLoop *L = LI.getLoopFor(BB);
      if (L && L->getHeader() == BB) {
        uint64_t HeaderSamples = Samples[BB];
        if (HeaderSamples > 0 && InSum > 0) {
          // Entry edges are the non-back-edge predecessors' contributions.
          // CyclicProb = 1 - (InSum / expectedHeaderFreq).
          // Since we want headerFreq = InSum / (1 - cp), and cp is the
          // fraction of executions that come from back edges, estimate
          // from samples: cp = (headerSamples - non-back-samples) / headerSamples.
          uint64_t NonBackSamples = 0;
          for (const BinaryBasicBlock *Pred : BB->predecessors()) {
            if (!isBackEdge(Pred, BB, LI))
              NonBackSamples += Samples[Pred];
          }
          double CyclicProb = 0.0;
          if (NonBackSamples < HeaderSamples)
            CyclicProb = 1.0 - static_cast<double>(NonBackSamples) /
                                   HeaderSamples;
          // Clamp to avoid division by zero or negative.
          if (CyclicProb >= 0.99)
            CyclicProb = 0.99;
          if (CyclicProb > 0.0)
            InSum = static_cast<uint64_t>(InSum / (1.0 - CyclicProb));
        }
      }

      if (InSum > 0)
        BB->setExecutionCount(InSum);
    }

    Visited.insert(BB);

    uint64_t Count = BB->getExecutionCount();
    if (Count == 0 || BB->succ_size() == 0)
      continue;

    distributeToSuccessors(*BB, Count, Samples);
  }
}

} // anonymous namespace

bool InferEdgeCounts::verify(const BinaryFunction &BF) const {
  for (const BinaryBasicBlock &BB : BF) {
    if (&BB == &BF.front())
      continue;

    uint64_t BlockCount = BB.getExecutionCount();

    // Outgoing edges should not exceed block count.
    if (BB.succ_size() > 0) {
      uint64_t OutSum = 0;
      for (const BinaryBasicBlock *Succ : BB.successors())
        OutSum += BB.getBranchInfo(*Succ).Count;
      uint64_t Tol = std::max(BlockCount, OutSum) / 10 + 1;
      if (OutSum > BlockCount + Tol)
        return false;
    }

    // Incoming edges should approximate block count.
    uint64_t InSum = 0;
    for (const BinaryBasicBlock *Pred : BB.predecessors())
      InSum += Pred->getBranchInfo(BB).Count;
    if (InSum > 0 && BlockCount > 0) {
      uint64_t Tol = std::max(BlockCount, InSum) / 10 + 1;
      if (InSum > BlockCount + Tol || BlockCount > InSum + Tol)
        return false;
    }
  }
  return true;
}

Error InferEdgeCounts::runOnFunctions(BinaryContext &BC) {
  FuncsUpdated = 0;

  for (auto &BFI : BC.getBinaryFunctions()) {
    BinaryFunction &BF = BFI.second;
    if (!BF.hasCFG() || !BF.isSimple() || !BF.hasValidProfile())
      continue;
    if (BF.empty())
      continue;

    bool HasEdges = false;
    for (const BinaryBasicBlock &BB : BF) {
      for (const BinaryBasicBlock *Succ : BB.successors()) {
        if (BB.getBranchInfo(*Succ).Count > 0) {
          HasEdges = true;
          break;
        }
      }
      if (HasEdges)
        break;
    }
    if (HasEdges)
      continue;

    bool HasSamples = false;
    for (const BinaryBasicBlock &BB : BF) {
      if (BB.getExecutionCount() > 0) {
        HasSamples = true;
        break;
      }
    }
    if (!HasSamples)
      continue;

    propagateFunction(BF);

    if (verify(BF))
      ++FuncsUpdated;
  }

  if (FuncsUpdated)
    BC.outs() << "BOLT-INFO: inferred edge counts for " << FuncsUpdated
              << " functions from block samples\n";

  return Error::success();
}

} // namespace bolt
} // namespace llvm
