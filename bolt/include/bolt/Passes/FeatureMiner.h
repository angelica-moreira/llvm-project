//===--- Passes/FeatureMiner.h ---------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// A very simple feature extractor based on Calder's paper
// Evidence-based static branch prediction using machine learning
// https://dl.acm.org/doi/10.1145/239912.239923
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_PASSES_FEATUREMINER_H_
#define LLVM_TOOLS_LLVM_BOLT_PASSES_FEATUREMINER_H_

#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Core/BinaryLoop.h"
#include "bolt/Passes/DominatorAnalysis.h"
#include "bolt/Passes/BinaryPasses.h"
#include "bolt/Passes/StaticBranchInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace llvm {
namespace bolt {

class FeatureMiner : public BinaryFunctionPass {
private:
  std::unique_ptr<StaticBranchInfo> SBI;

  /// BasicBlockInfo - This structure holds feature information about the target
  /// BasicBlock of either the taken or the fallthrough paths of a given branch.
  struct BasicBlockInfo {
    std::optional<bool> BranchDominates;     // 1 - dominates, 0 - does not dominate
    std::optional<bool> BranchPostdominates; // 1 - postdominates, 0 - does not PD
    std::optional<bool> LoopHeader; // 1 - loop header, 0 - not a loop header
    std::optional<bool> Backedge;   // 1 - loop back, 0 - not a loop back
    std::optional<bool> Exit;       // 1 - loop exit, 0 - not a loop exit
    std::optional<bool> Call;       // 1 - program call, 0 - not a program call
    std::optional<unsigned> NumCalls;
    std::optional<int32_t> EndOpcode; // 0 = NOTHING
    StringRef EndOpcodeStr = "UNDEF";
    std::optional<int32_t> BasicBlockSize;
    std::string FromFunName = "UNDEF";
    uint32_t FromBb;
    std::string ToFunName = "UNDEF";
    uint32_t ToBb;


    std::optional<unsigned> NumCallsExit;
    std::optional<unsigned> NumCallsInvoke;
    std::optional<unsigned> NumIndirectCalls;
    std::optional<unsigned> NumTailCalls;
  };

  typedef std::unique_ptr<struct BasicBlockInfo> BBIPtr;

  /// BranchFeaturesInfo - This structure holds feature information about each
  /// two-way branch from the program.
  struct BranchFeaturesInfo {
    StringRef OpcodeStr = "UNDEF";
    StringRef CmpOpcodeStr = "UNDEF";
    bool Simple = 0;

    std::optional<int32_t> Opcode;
    std::optional<int32_t> CmpOpcode;
    std::optional<int64_t> Count;
    std::optional<int64_t> MissPredicted;
    std::optional<int64_t> FallthroughCount;
    std::optional<int64_t> FallthroughMissPredicted;
    BBIPtr TrueSuccessor = std::make_unique<struct BasicBlockInfo>();
    BBIPtr FalseSuccessor = std::make_unique<struct BasicBlockInfo>();
    std::optional<int8_t> ProcedureType; // 1 - Leaf, 0 - NonLeaf, 2 - CallSelf
    std::optional<bool> LoopHeader;      // 1 — loop header, 0 - not a loop header
    std::optional<bool> Direction;       // 1 - Forward Branch, 0 - Backward Branch

    std::optional<unsigned> NumOuterLoops;
    std::optional<unsigned> TotalLoops;
    std::optional<unsigned> MaximumLoopDepth;
    std::optional<unsigned> LoopDepth;
    std::optional<unsigned> LoopNumExitEdges;
    std::optional<unsigned> LoopNumLatches;
    std::optional<unsigned> LoopNumBlocks;
    std::optional<unsigned> LoopNumBackEdges;

    std::optional<bool> LocalExitingBlock;
    std::optional<bool> LocalLatchBlock;
    std::optional<bool> LocalLoopHeader;
    std::optional<bool> Call;

    std::optional<unsigned> NumCalls;
    std::optional<unsigned> NumCallsExit;
    std::optional<unsigned> NumCallsInvoke;
    std::optional<unsigned> NumIndirectCalls;
    std::optional<unsigned> NumTailCalls;
    std::optional<unsigned> NumSelfCalls;

    std::optional<unsigned> NumBasicBlocks;

    std::optional<int64_t> DeltaTaken;

    std::optional<int32_t> OperandRAType;
    std::optional<int32_t> OperandRBType;

    std::optional<int32_t> BasicBlockSize;

    std::optional<int64_t> BranchOffset;
  };

  typedef std::unique_ptr<struct BranchFeaturesInfo> BFIPtr;
  std::vector<BFIPtr> BranchesInfoSet;

  /// getProcedureType - Determines which category the function falls into:
  /// Leaf, Non-leaf or Calls-self.
  int8_t getProcedureType(BinaryFunction &Function, BinaryContext &BC);

  /// addSuccessorInfo - Discovers feature information for the target successor
  /// basic block, and inserts it into the static branch info container.
  void addSuccessorInfo(DominatorAnalysis<false> &DA,
                        DominatorAnalysis<true> &PDA, BFIPtr const &BFI,
                        BinaryFunction &Function, BinaryContext &BC,
                        MCInst &Inst, BinaryBasicBlock &BB, bool Succ);

  /// extractFeatures - Extracts the feature information for each two-way branch
  /// from the program.
  void extractFeatures(BinaryFunction &Function, 
                       BinaryContext &BC,
                       raw_ostream &Printer);

  /// dumpSuccessorFeatures - Dumps the feature information about the target
  /// BasicBlock of either the taken or the fallthrough paths of a given branch.
  void dumpSuccessorFeatures(raw_ostream &Printer, BBIPtr &Successor);

  /// dumpFeatures - Dumps the feature information about each two-way branch
  /// from the program.
  void dumpFeatures(raw_ostream &Printer, uint64_t FunctionAddress,
		    uint64_t FunctionFrequency);

  /// dumpProfileData - Dumps a limited version of the inout profile data
  /// that contains only profile for conditional branches, unconditional
  /// branches and terminators that aren't branches.
  void dumpProfileData(BinaryFunction &Function, raw_ostream &Printer);

public:
  explicit FeatureMiner(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "feature-miner"; }

  Error runOnFunctions(BinaryContext &BC) override;
};

} // namespace bolt
} // namespace llvm

#endif /* LLVM_TOOLS_LLVM_BOLT_PASSES_FEATUREMINER_H_ */
