//===- Passes/StaticBranchProbabilities.cpp - Infered Branch Probabilities -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/StaticBranchProbabilities.h"
#include "bolt/Passes/DataflowInfoManager.h"
#include "bolt/Passes/RegAnalysis.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/Error.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"

#include <optional>

#undef DEBUG_TYPE
#define DEBUG_TYPE "bolt-branch-prob"

using namespace llvm;
using namespace bolt;

namespace opts {

extern cl::OptionCategory BoltInferenceCategory;

cl::opt<bolt::StaticBranchProbabilities::HeuristicType> HeuristicBased(
    "heuristic-based",
    cl::desc("generates probabilities statically based on heuristics."),
    cl::values(clEnumValN(bolt::StaticBranchProbabilities::H_ALWAYS_TAKEN,
                          "always", "set as 1 the weight of taken BB edges"),
               clEnumValN(bolt::StaticBranchProbabilities::H_NEVER_TAKEN,
                          "never",
                          "set as 1 the weight of fallthrough BB edges"),
               clEnumValN(bolt::StaticBranchProbabilities::H_WEAKLY_TAKEN,
                          "weakly-taken",
                          "set as 0.2 the weight of taken BB edges "
                          "and set as 0.8 the weight of "
                          "fallthrough BB edges"),
               clEnumValN(bolt::StaticBranchProbabilities::H_WEAKLY_NOT_TAKEN,
                          "weakly-not-taken",
                          "set as 0.8 the weight of taken BB edges "
                          "and set as 0.2 the weight of "
                          "fallthrough BB edges"),
               clEnumValN(bolt::StaticBranchProbabilities::H_UNBIASED,
                          "unbiased", "set as 0.5 the weight of all BB edges"),
               clEnumValN(bolt::StaticBranchProbabilities::H_WU_LARUS,
                          "wularus",
                          "use as edge weights the combined the outcome "
                          "of the some of the heuritics described "
                          "in Wu Larus paper")),
    cl::ZeroOrMore, cl::cat(BoltInferenceCategory));

cl::opt<bool> MLBased("ml-based",
                      cl::desc("reads probabilities based on ML model."),
                      cl::ZeroOrMore, cl::Hidden, cl::cat(BoltInferenceCategory));
} // namespace opts

namespace llvm {
namespace bolt {

double
StaticBranchProbabilities::getCFGBackEdgeProbability(BinaryBasicBlock &SrcBB,
                                                     BinaryBasicBlock &DstBB) {
  Edge CFGEdge = std::make_pair(SrcBB.getLabel(), DstBB.getLabel());
  auto It = CFGBackEdgeProbabilities.find(CFGEdge);
  if (It != CFGBackEdgeProbabilities.end()) {
    if (static_cast<int64_t>(It->second) < 0 ||
        static_cast<int64_t>(It->second) == INT64_MAX)
      return 0.0;
    return It->second;
  }

  auto Function = SrcBB.getFunction();
  return getCFGEdgeProbability(CFGEdge, Function);
}

void StaticBranchProbabilities::setCFGBackEdgeProbability(Edge &CFGEdge,
                                                          double Prob) {
  if (static_cast<int64_t>(Prob) < 0 || static_cast<int64_t>(Prob) == INT64_MAX)
    CFGBackEdgeProbabilities[CFGEdge] = 0.0;
  CFGBackEdgeProbabilities[CFGEdge] = Prob;
}

double
StaticBranchProbabilities::getCFGEdgeProbability(Edge &CFGEdge,
                                                 BinaryFunction *Function) {
 
  auto *BB = Function->getBasicBlockForLabel(CFGEdge.first);
  auto &BC = Function->getBinaryContext();
  auto LastInst = BB->getLastNonPseudoInstr();
 
  auto It = CFGEdgeProbabilities.find(CFGEdge);
  if (It != CFGEdgeProbabilities.end()) {
    if (static_cast<int64_t>(It->second) < 0 ||
        static_cast<int64_t>(It->second) == INT64_MAX){
          return 0;
        }
    return It->second;
  }

  if (LastInst && BC.MIB->isConditionalBranch(*LastInst)){	  
      return 0.5;
  }

  return 1.0;
}

double
StaticBranchProbabilities::getCFGEdgeProbability(BinaryBasicBlock &SrcBB,
                                                 BinaryBasicBlock &DstBB) {
  Edge CFGEdge = std::make_pair(SrcBB.getLabel(), DstBB.getLabel());

  auto Function = SrcBB.getFunction();

  return getCFGEdgeProbability(CFGEdge, Function);
}

int64_t 
StaticBranchProbabilities::getFunctionFrequency(uint64_t FunAddress){
  auto It = OriginalFunctionsFrequency.find(FunAddress);
  if (It != OriginalFunctionsFrequency.end()) {
    return It->second;
  }
  return 1;
}

void StaticBranchProbabilities::clear() {
  BSI->clear();
  CFGBackEdgeProbabilities.clear();
  CFGEdgeProbabilities.clear();
}

void StaticBranchProbabilities::parseProbabilitiesFile(
    std::unique_ptr<MemoryBuffer> MemBuf, BinaryContext &BC) {
  errs() << "BOLT-INFO: Starting passing.\n";
  std::vector<BasicBlockOffset> BasicBlockOffsets;
  auto populateBasicBlockOffsets =
      [&](BinaryFunction &Function,
          std::vector<BasicBlockOffset> &BasicBlockOffsets) {
        for (auto &BB : Function) {
          BasicBlockOffsets.emplace_back(
              std::make_pair(BB.getInputOffset(), &BB));
        }
      };

  auto getBasicBlockAtOffset = [&](uint64_t Offset) -> BinaryBasicBlock * {
    if (BasicBlockOffsets.empty())
      return nullptr;

    auto It = std::upper_bound(
        BasicBlockOffsets.begin(), BasicBlockOffsets.end(),
        BasicBlockOffset(Offset, nullptr), CompareBasicBlockOffsets());
    assert(It != BasicBlockOffsets.begin() &&
           "first basic block not at offset 0");
    --It;
    auto *BB = It->second;
    return (Offset == BB->getInputOffset()) ? BB : nullptr;
  };

  auto ParsingBuf = MemBuf.get()->getBuffer();
  BinaryFunction *Function = nullptr;
  while (ParsingBuf.size() > 0) {
    auto LineEnd = ParsingBuf.find_first_of("\n");
    if (LineEnd == StringRef::npos) {
      errs() << "BOLT-ERROR: File not in the correct format.\n";
      exit(EXIT_FAILURE);
    }

    StringRef Line = ParsingBuf.substr(0, LineEnd);
    auto Type = Line.split(" ");
    if (!Type.first.equals("EDGE") && !Type.first.equals("FUNCTION") &&
        !Line.equals("END")) {
      errs() << "BOLT-ERROR: File not in the correct format, found: " << Line
             << "\n";
      exit(EXIT_FAILURE);
    }

    if (Type.first.equals("FUNCTION")) {
      clear();
      BasicBlockOffsets.clear();
      auto FunLine = Type.second.split(" ");
      //the first substring is the function's name
      StringRef NumStr = FunLine.second.split(" ").first;
      uint64_t FunctionAddress;
      if (NumStr.getAsInteger(16, FunctionAddress)) {
        errs() << "BOLT-ERROR: File not in the correct format.\n";
        exit(EXIT_FAILURE);
      }

      Function = BC.getBinaryFunctionAtAddress(FunctionAddress);
      if (Function){
        populateBasicBlockOffsets(*Function, BasicBlockOffsets);
      }
    }

    if (!Function){
      ParsingBuf = ParsingBuf.drop_front(LineEnd + 1);
      continue;
    }
    
    if (Type.first.equals("EDGE")) {
      auto EdgeLine = Type.second.split(" ");

      StringRef SrcBBAddressStr = EdgeLine.first;
      uint64_t SrcBBAddress;
      if (SrcBBAddressStr.getAsInteger(16, SrcBBAddress)) {
        errs() << "BOLT-ERROR: File not in the correct format.\n";
        exit(EXIT_FAILURE);
      }

      auto EdgeInfo = EdgeLine.second.split(" ");
      StringRef DstBBAddressStr = EdgeInfo.first;
      uint64_t DstBBAddress;
      if (DstBBAddressStr.getAsInteger(16, DstBBAddress)) {
        errs() << "BOLT-ERROR: File not in the correct format.\n";
        exit(EXIT_FAILURE);
      }

      auto SrcBB = getBasicBlockAtOffset(SrcBBAddress);
      auto DstBB = getBasicBlockAtOffset(DstBBAddress);
      if (SrcBB && DstBB) {
        uint64_t Prob;
        StringRef ProbStr = EdgeInfo.second;
        if (ProbStr.getAsInteger(10, Prob)) {
          errs() << "BOLT-ERROR: File not in the correct format.\n";
          exit(EXIT_FAILURE);
        }
        SrcBB->setSuccessorBranchInfo(*DstBB, Prob, 0);
      }
    } else if (Line.equals("END")) {
      BasicBlockOffsets.clear();
      Function->setExecutionCount(1);
    }
    
    ParsingBuf = ParsingBuf.drop_front(LineEnd + 1);
  }
}

void StaticBranchProbabilities::parseProbabilitiesFile(
    std::unique_ptr<MemoryBuffer> MemBuf, BinaryContext &BC,
    std::set<uint64_t> &StaleFunctionsAddresses,
    std::set<uint64_t> &MatchedStaleFuncAddresses) {

  std::vector<BasicBlockOffset> BasicBlockOffsets;
  auto populateBasicBlockOffsets =
      [&](BinaryFunction &Function,
          std::vector<BasicBlockOffset> &BasicBlockOffsets) {
        for (auto &BB : Function) {
          BasicBlockOffsets.emplace_back(
              std::make_pair(BB.getInputOffset(), &BB));
        }
      };

  auto getBasicBlockAtOffset = [&](uint64_t Offset) -> BinaryBasicBlock * {
    if (BasicBlockOffsets.empty())
      return nullptr;

    auto It = std::upper_bound(
        BasicBlockOffsets.begin(), BasicBlockOffsets.end(),
        BasicBlockOffset(Offset, nullptr), CompareBasicBlockOffsets());
    assert(It != BasicBlockOffsets.begin() &&
           "first basic block not at offset 0");
    --It;
    auto *BB = It->second;
    return (Offset == BB->getInputOffset()) ? BB : nullptr;
  };

  auto ParsingBuf = MemBuf.get()->getBuffer();
  BinaryFunction *Function = nullptr;
  while (ParsingBuf.size() > 0) {
    auto LineEnd = ParsingBuf.find_first_of("\n");
    if (LineEnd == StringRef::npos) {
      errs() << "BOLT-ERROR: File not in the correct format.\n";
      exit(EXIT_FAILURE);
    }

    StringRef Line = ParsingBuf.substr(0, LineEnd);
    auto Type = Line.split(" ");
    if (!Type.first.equals("EDGE") && !Type.first.equals("FUNCTION") &&
        !Line.equals("END")) {
      errs() << "BOLT-ERROR: File not in the correct format, found: " << Line
             << "\n";
      exit(EXIT_FAILURE);
    }

    if (Type.first.equals("FUNCTION")) {
      clear();
      BasicBlockOffsets.clear();
      auto FunLine = Type.second.split(" ");
      // the first substring is the function's name
      StringRef NumStr = FunLine.second.split(" ").first;
      uint64_t FunctionAddress = 0;
      if (NumStr.getAsInteger(16, FunctionAddress)) {
        errs() << "BOLT-ERROR: File not in the correct format.\n";
        exit(EXIT_FAILURE);
      }

      auto it = StaleFunctionsAddresses.find(FunctionAddress);
      if (it != StaleFunctionsAddresses.end()) {
        Function = BC.getBinaryFunctionAtAddress(FunctionAddress);
        if(Function){
          Function->clearProfile();
          Function->setExecutionCount(1);
          Function->markProfiled(BinaryFunction::PF_STATIC);
          MatchedStaleFuncAddresses.insert(FunctionAddress);
          populateBasicBlockOffsets(*Function, BasicBlockOffsets);
        }
      } 
    }

    if (!Function){
      ParsingBuf = ParsingBuf.drop_front(LineEnd + 1);
      continue;
    }
    
    if (Type.first.equals("EDGE")) {
      auto EdgeLine = Type.second.split(" ");

      StringRef SrcBBAddressStr = EdgeLine.first;
      uint64_t SrcBBAddress;
      if (SrcBBAddressStr.getAsInteger(16, SrcBBAddress)) {
        errs() << "BOLT-ERROR: File not in the correct format.\n";
        exit(EXIT_FAILURE);
      }

      auto EdgeInfo = EdgeLine.second.split(" ");
      StringRef DstBBAddressStr = EdgeInfo.first;
      uint64_t DstBBAddress;
      if (DstBBAddressStr.getAsInteger(16, DstBBAddress)) {
        errs() << "BOLT-ERROR: File not in the correct format.\n";
        exit(EXIT_FAILURE);
      }

      auto SrcBB = getBasicBlockAtOffset(SrcBBAddress);
      auto DstBB = getBasicBlockAtOffset(DstBBAddress);
      if (SrcBB && DstBB && SrcBB->isSuccessor(DstBB)) {
        uint64_t Prob;
        StringRef ProbStr = EdgeInfo.second;
        if (ProbStr.getAsInteger(10, Prob)) {
          errs() << "BOLT-ERROR: File not in the correct format.\n";
          exit(EXIT_FAILURE);
        }

        SrcBB->setSuccessorBranchInfo(*DstBB, Prob, 0);
      }
    } else if (Line.equals("END")) {
      BasicBlockOffsets.clear();
    }
    
    ParsingBuf = ParsingBuf.drop_front(LineEnd + 1);
  }
}

void StaticBranchProbabilities::parseBeetleProfileFile(
    std::unique_ptr<MemoryBuffer> MemBuf, BinaryContext &BC,
    std::set<uint64_t> &StaleFunctionsAddresses) {

  std::vector<BasicBlockOffset> BasicBlockOffsets;
  auto populateBasicBlockOffsets =
      [&](BinaryFunction &Function,
          std::vector<BasicBlockOffset> &BasicBlockOffsets) {
        for (auto &BB : Function) {
          BasicBlockOffsets.emplace_back(
              std::make_pair(BB.getInputOffset(), &BB));
        }
      };

  auto getBasicBlockAtOffset = [&](uint64_t Offset) -> BinaryBasicBlock * {
    if (BasicBlockOffsets.empty())
      return nullptr;

    auto It = std::upper_bound(
        BasicBlockOffsets.begin(), BasicBlockOffsets.end(),
        BasicBlockOffset(Offset, nullptr), CompareBasicBlockOffsets());
    assert(It != BasicBlockOffsets.begin() &&
           "first basic block not at offset 0");
    --It;
    auto *BB = It->second;
    return (Offset == BB->getInputOffset()) ? BB : nullptr;
  };

  auto ParsingBuf = MemBuf.get()->getBuffer();
  BinaryFunction *Function = nullptr;
  while (ParsingBuf.size() > 0) {
    auto LineEnd = ParsingBuf.find_first_of("\n");
    if (LineEnd == StringRef::npos) {
      errs() << "BOLT-ERROR: File not in the correct format.\n";
      exit(EXIT_FAILURE);
    }

    StringRef Line = ParsingBuf.substr(0, LineEnd);
    auto Type = Line.split(" ");
    if (!Type.first.equals("EDGE") && !Type.first.equals("FUNCTION") &&
        !Line.equals("END")) {
      errs() << "BOLT-ERROR: File not in the correct format, found: " << Line
             << "\n";
      exit(EXIT_FAILURE);
    }

    if (Type.first.equals("FUNCTION")) {
      clear();
      BasicBlockOffsets.clear();
      auto FunLine = Type.second.split(" ");
      // the first substring is the function's name
      StringRef NumStr = FunLine.second.split(" ").first;
      uint64_t FunctionAddress = 0;
      if (NumStr.getAsInteger(16, FunctionAddress)) {
        errs() << "BOLT-ERROR: File not in the correct format.\n";
        exit(EXIT_FAILURE);
      }

      StringRef FunFreqStr = FunLine.second.split(" ").second;
      uint64_t FunFreq = 0;
      if (FunFreqStr.getAsInteger(10, FunFreq)) {
        errs() << "BOLT-ERROR: File not in the correct format.\n";
        exit(EXIT_FAILURE);
      }
      auto it = StaleFunctionsAddresses.find(FunctionAddress);
      if (it != StaleFunctionsAddresses.end()) {
        Function = BC.getBinaryFunctionAtAddress(FunctionAddress);
        if(Function){
          Function->setExecutionCount(FunFreq);
          Function->markProfiled(BinaryFunction::PF_STATIC);
          StaleFunctionsAddresses.erase(it);
          populateBasicBlockOffsets(*Function, BasicBlockOffsets);
          OriginalFunctionsFrequency[FunctionAddress] = FunFreq;
        }
      } 
    }

    if (!Function){
      ParsingBuf = ParsingBuf.drop_front(LineEnd + 1);
      continue;
    }
    
    if (Type.first.equals("EDGE")) {
      auto EdgeLine = Type.second.split(" ");

      StringRef SrcBBAddressStr = EdgeLine.first;
      uint64_t SrcBBAddress;
      if (SrcBBAddressStr.getAsInteger(16, SrcBBAddress)) {
        errs() << "BOLT-ERROR: File not in the correct format.\n";
        exit(EXIT_FAILURE);
      }

      auto EdgeInfo = EdgeLine.second.split(" ");
      StringRef DstBBAddressStr = EdgeInfo.first;
      uint64_t DstBBAddress;
      if (DstBBAddressStr.getAsInteger(16, DstBBAddress)) {
        errs() << "BOLT-ERROR: File not in the correct format.\n";
        exit(EXIT_FAILURE);
      }

      auto SrcBB = getBasicBlockAtOffset(SrcBBAddress);
      auto DstBB = getBasicBlockAtOffset(DstBBAddress);
      if (SrcBB && DstBB && SrcBB->isSuccessor(DstBB)) {
        uint64_t Freq;
        StringRef FreqStr = EdgeInfo.second;

        if (FreqStr.getAsInteger(10, Freq)) {
          errs() << "BOLT-ERROR: File not in the correct format.\n";
          exit(EXIT_FAILURE);
        }

        SrcBB->setSuccessorBranchInfo(*DstBB, Freq, 0);
        uint64_t newBBFreq = DstBB->getKnownExecutionCount() + Freq;
        DstBB->setExecutionCount(newBBFreq);
      }
    } else if (Line.equals("END")) {
      BasicBlockOffsets.clear();
    }
    
    ParsingBuf = ParsingBuf.drop_front(LineEnd + 1);
  }
}


void StaticBranchProbabilities::computeProbabilities(BinaryFunction &Function) {
  Function.setExecutionCount(1);

  double EdgeProbTaken = 0.5;
  switch (opts::HeuristicBased) {
  case bolt::StaticBranchProbabilities::H_ALWAYS_TAKEN:
    EdgeProbTaken = 1.0;
    break;
  case bolt::StaticBranchProbabilities::H_NEVER_TAKEN:
    EdgeProbTaken = 0.0;
    break;
  case bolt::StaticBranchProbabilities::H_WEAKLY_TAKEN:
    EdgeProbTaken = 0.2;
    break;
  case bolt::StaticBranchProbabilities::H_WEAKLY_NOT_TAKEN:
    EdgeProbTaken = 0.8;
    break;
  default:
    EdgeProbTaken = 0.5;
    break;
  }

  double EdgeProbNotTaken = 1 - EdgeProbTaken;

  for (auto &BB : Function) {
    BB.setExecutionCount(0);

    unsigned NumSucc = BB.succ_size();
    if (NumSucc == 0)
      continue;

    if (NumSucc == 1) {
      BinaryBasicBlock *SuccBB = *BB.succ_begin();

      BB.setSuccessorBranchInfo(*SuccBB, 0.0, 0.0);
      Edge CFGEdge = std::make_pair(BB.getLabel(), SuccBB->getLabel());

      // Since it is an unconditional branch, when this branch is reached
      // it has a chance of 100% of being taken (1.0).
      CFGEdgeProbabilities[CFGEdge] = 1.0;
    } 
    else if(NumSucc > 2){
      for (BinaryBasicBlock *SuccBB : BB.successors()) {
        Edge CFGEdge = std::make_pair(BB.getLabel(), SuccBB->getLabel());
        CFGEdgeProbabilities[CFGEdge] = 1.0 / NumSucc;
        BB.setSuccessorBranchInfo(*SuccBB, 0.0, 0.0);
      }
    }
    else if (opts::MLBased) {
      double total_prob = 0.0;
      for (BinaryBasicBlock *SuccBB : BB.successors()) {
        uint64_t Frequency = BB.getBranchInfo(*SuccBB).Count;
        double EdgeProb = (Frequency == UINT64_MAX)
                              ? 0
                              : Frequency / DIVISOR;
        Edge CFGEdge = std::make_pair(BB.getLabel(), SuccBB->getLabel());
        CFGEdgeProbabilities[CFGEdge] = EdgeProb;
        total_prob += EdgeProb;
        BB.setSuccessorBranchInfo(*SuccBB, 0.0, 0.0);
      }

      if(total_prob == 0.0){
        BinaryBasicBlock *TakenSuccBB = BB.getConditionalSuccessor(true);
        BinaryBasicBlock *NotTakenSuccBB = BB.getConditionalSuccessor(false);
        
        if(TakenSuccBB && NotTakenSuccBB){
          Edge CFGEdge =
              std::make_pair(BB.getLabel(), NotTakenSuccBB->getLabel());
          CFGEdgeProbabilities[CFGEdge] = 0.5;

          CFGEdge = std::make_pair(BB.getLabel(), TakenSuccBB->getLabel());
          CFGEdgeProbabilities[CFGEdge] = 0.5; 

        }else if(NotTakenSuccBB){
          Edge CFGEdge =
              std::make_pair(BB.getLabel(), NotTakenSuccBB->getLabel());
          CFGEdgeProbabilities[CFGEdge] = 1.0;
        } else if (TakenSuccBB){
          Edge CFGEdge = std::make_pair(BB.getLabel(), TakenSuccBB->getLabel());
          CFGEdgeProbabilities[CFGEdge] = 1.0;         
        }
      }
    } else {
      BinaryBasicBlock *TakenSuccBB = BB.getConditionalSuccessor(true);
      if (TakenSuccBB) {
        Edge CFGEdge = std::make_pair(BB.getLabel(), TakenSuccBB->getLabel());
        CFGEdgeProbabilities[CFGEdge] = EdgeProbTaken;
        BB.setSuccessorBranchInfo(*TakenSuccBB, 0.0, 0.0);
      }

      BinaryBasicBlock *NotTakenSuccBB = BB.getConditionalSuccessor(false);
      if (NotTakenSuccBB) {
        Edge CFGEdge =
            std::make_pair(BB.getLabel(), NotTakenSuccBB->getLabel());
        CFGEdgeProbabilities[CFGEdge] = EdgeProbNotTaken;
        BB.setSuccessorBranchInfo(*NotTakenSuccBB, 0.0, 0.0);
      }
    }
  }
}

void StaticBranchProbabilities::computeHeuristicBasedProbabilities(
    BinaryFunction &Function) {

  Function.setExecutionCount(1);

  BinaryContext &BC = Function.getBinaryContext();
  auto Info = DataflowInfoManager(Function, nullptr, nullptr);
  auto &PDA = Info.getPostDominatorAnalysis();

  for (auto &BB : Function) {
    unsigned NumSucc = BB.succ_size();
    if (NumSucc == 0)
      continue;

    unsigned NumBackedges = BSI->countBackEdges(&BB);

    // If the basic block that conatins the branch has an exit call,
    // then we assume that its successors will never be reached.
    if (BSI->callToExit(&BB, BC)) {
      for (BinaryBasicBlock *SuccBB : BB.successors()) {
        double EdgeProb = 0.0;
        Edge CFGEdge = std::make_pair(BB.getLabel(), SuccBB->getLabel());
        CFGEdgeProbabilities[CFGEdge] = EdgeProb;
        BB.setSuccessorBranchInfo(*SuccBB, 0.0, 0.0);
      }

    } else if (NumBackedges > 0 && NumBackedges < NumSucc) {
      // Both back edges and exit edges
      for (BinaryBasicBlock *SuccBB : BB.successors()) {
        Edge CFGEdge = std::make_pair(BB.getLabel(), SuccBB->getLabel());

        if (BSI->isBackEdge(CFGEdge)) {
          double EdgeProb =
              BHI->getTakenProbability(LOOP_BRANCH_HEURISTIC) / NumBackedges;
          CFGEdgeProbabilities[CFGEdge] = EdgeProb;
        } else {
          double EdgeProb = BHI->getNotTakenProbability(LOOP_BRANCH_HEURISTIC) /
                            (NumSucc - NumBackedges);
          CFGEdgeProbabilities[CFGEdge] = EdgeProb;
        }
        BB.setSuccessorBranchInfo(*SuccBB, 0.0, 0.0);
      }

    } else if (NumBackedges > 0 || NumSucc != 2) {
      // Only back edges, or not a 2-way branch.
      for (BinaryBasicBlock *SuccBB : BB.successors()) {
        Edge CFGEdge = std::make_pair(BB.getLabel(), SuccBB->getLabel());
        CFGEdgeProbabilities[CFGEdge] = 1.0 / NumSucc;
        BB.setSuccessorBranchInfo(*SuccBB, 0.0, 0.0);
      }
    } else {
      assert(NumSucc == 2 && "Expected a two way conditional branch.");

      BinaryBasicBlock *TakenBB = BB.getConditionalSuccessor(true);
      BinaryBasicBlock *FallThroughBB = BB.getConditionalSuccessor(false);

      if (!TakenBB || !FallThroughBB)
        continue;

      Edge TakenEdge = std::make_pair(BB.getLabel(), TakenBB->getLabel());
      Edge FallThroughEdge =
          std::make_pair(BB.getLabel(), FallThroughBB->getLabel());

      // Consider that each edge is unbiased, thus each edge
      // has a likelihood of 50% of being taken.
      CFGEdgeProbabilities[TakenEdge] = 0.5f;
      CFGEdgeProbabilities[FallThroughEdge] = 0.5f;

      for (unsigned BHId = 0; BHId < BHI->getNumHeuristics(); ++BHId) {
        BranchHeuristics Heuristic = BHI->getHeuristic(BHId);
        PredictionInfo Prediction =
            BHI->getApplicableHeuristic(Heuristic, &BB, PDA);
        if (!Prediction.first)
          continue;

        /// If the heuristic applies then combines the probabilities and
        /// updates the edge weights

        BinaryBasicBlock *TakenBB = Prediction.first;
        BinaryBasicBlock *FallThroughBB = Prediction.second;

        Edge TakenEdge = std::make_pair(BB.getLabel(), TakenBB->getLabel());
        Edge FallThroughEdge =
            std::make_pair(BB.getLabel(), FallThroughBB->getLabel());

        double ProbTaken = BHI->getTakenProbability(Heuristic);
        double ProbNotTaken = BHI->getNotTakenProbability(Heuristic);

        double OldProbTaken = getCFGEdgeProbability(BB, *TakenBB);
        double OldProbNotTaken = getCFGEdgeProbability(BB, *FallThroughBB);

        double Divisor =
            OldProbTaken * ProbTaken + OldProbNotTaken * ProbNotTaken;

        CFGEdgeProbabilities[TakenEdge] = OldProbTaken * ProbTaken / Divisor;
        CFGEdgeProbabilities[FallThroughEdge] =
            OldProbNotTaken * ProbNotTaken / Divisor;
      }

      BB.setSuccessorBranchInfo(*TakenBB, 0.0, 0.0);
      BB.setSuccessorBranchInfo(*FallThroughBB, 0.0, 0.0);
    }
  }
}

} // namespace bolt

} // namespace llvm
