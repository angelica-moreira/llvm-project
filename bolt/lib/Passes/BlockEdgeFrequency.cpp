//===---- Passes/BlockEdgeFrequency.cpp - Block and Edge Frequencies
//-------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/BlockEdgeFrequency.h"
#include "bolt/Passes/DataflowInfoManager.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/Error.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Timer.h"

#include <cfloat>
#include <optional>
#include <string>

#undef DEBUG_TYPE
#define DEBUG_TYPE "bolt-block-edge-counts"

using namespace llvm;
using namespace object;
using namespace bolt;

namespace opts {

extern cl::OptionCategory BoltInferenceCategory;
extern cl::OptionCategory BoltStaleCategory;

extern cl::opt<bool> MLBased;
extern cl::opt<bool> WuLarusStaleProfile;
extern cl::opt<bool> VespaStaleProfile;
extern cl::opt<bool> TimeBuild;

extern cl::opt<bolt::StaticBranchProbabilities::HeuristicType> HeuristicBased;

cl::opt<std::string> ProbFilename("prob-file", cl::desc("<data file>"),
                                  cl::Optional, cl::cat(BoltInferenceCategory));

cl::opt<std::string> BeetleFilename("beetle-file", cl::desc("<data file>"),
                                    cl::Optional, cl::cat(BoltStaleCategory));

cl::opt<bool>
    BeetleBased("beetle-based",
                cl::desc("reads frequencies based on Beetle technique."),
                cl::ZeroOrMore, cl::Hidden, cl::cat(BoltStaleCategory));

cl::opt<bool> DisableBeetle("disable-beetle",
                            cl::desc("disable Beetle technique."),
                            cl::ZeroOrMore, cl::Hidden, cl::init(false),
                            cl::cat(BoltStaleCategory));

} // namespace opts

namespace llvm {

namespace bolt {

bool BlockEdgeFrequency::isVisited(const BinaryBasicBlock *BB) const {
  return !(ReachableBBs.count(BB));
}

void BlockEdgeFrequency::setVisited(const BinaryBasicBlock *BB) {
  ReachableBBs.erase(BB);
}

uint64_t BlockEdgeFrequency::getCFGEdgeFrequency(BinaryBasicBlock &SrcBB,
                                                 BinaryBasicBlock &DstBB) {
  double Frequency = static_cast<double>(SrcBB.getBranchInfo(DstBB).Count);
  if (static_cast<double>(UINT64_MAX) < Frequency || Frequency < 0.0) {
    SrcBB.setSuccessorBranchInfo(DstBB, 0, 0);
    return 0;
  }

  return static_cast<uint64_t>(Frequency);
}

uint64_t BlockEdgeFrequency::getBBExecutionCount(BinaryBasicBlock &BB) {
  double Count = static_cast<double>(BB.getKnownExecutionCount());
  if (static_cast<double>(UINT64_MAX) < Count || Count < 0.0) {
    BB.setExecutionCount(0);
    return 0;
  }

  return static_cast<uint64_t>(Count);
}

void BlockEdgeFrequency::updateCallFrequency(BinaryContext &BC, MCInst &Inst,
                                             StringRef CallAnnotation,
                                             double CallFreq,
                                             uint64_t TakenFreqEdge) {

  if (!BC.MIB->hasAnnotation(Inst, CallAnnotation)) {
    BC.MIB->addAnnotation<uint64_t>(Inst, CallAnnotation, TakenFreqEdge);
  } else {
    if (auto CountAnnt =
            BC.MIB->tryGetAnnotationAs<uint64_t>(Inst, CallAnnotation)) {
      double Count = ((*CountAnnt) / SCALING_FACTOR) * CallFreq;
      (*CountAnnt) = round(Count);
    }
  }
}

void BlockEdgeFrequency::getCallFrequency(BinaryContext &BC, MCInst &Inst,
                                          StringRef CallAnnotation,
                                          uint64_t &TakenFreqEdge) {

  if (!BC.MIB->hasAnnotation(Inst, CallAnnotation)) {
    BC.MIB->addAnnotation<uint64_t>(Inst, CallAnnotation, TakenFreqEdge);
  } else {
    if (auto CountAnnt =
            BC.MIB->tryGetAnnotationAs<uint64_t>(Inst, CallAnnotation)) {
      TakenFreqEdge = (*CountAnnt);
    }
  }
}

void BlockEdgeFrequency::tagReachableBlocks(BinaryBasicBlock *Head) {
  if (!Head)
    return;

  ReachableBBs.clear();

  SmallVector<BinaryBasicBlock *, 16> BBStack;
  BBStack.push_back(Head);

  // Traversing all children in depth-first fashion and
  // marking them as not visited.
  while (!BBStack.empty()) {
    BinaryBasicBlock *BB = BBStack.pop_back_val();
    if (!(ReachableBBs.insert(BB).second)) {
      continue;
    }
    // Adding the new successors into the call stack.
    for (BinaryBasicBlock *SuccBB : BB->successors()) {
      BBStack.push_back(SuccBB);
    }
  }
}

void BlockEdgeFrequency::propagateLoopFrequencies(BinaryLoop *Loop) {
  // Check if the loop has been visited.
  if (VisitedLoops.count(Loop))
    return;

  // Mark the current loop as visited.
  VisitedLoops.insert(Loop);

  // Process the inner-most loop first and use the cyclic probabilities
  // of the inner loops to compute frequencies for the outer loops.
  for (BinaryLoop *InnerLoop : *Loop)
    propagateLoopFrequencies(InnerLoop);

  // Get the loop header
  BinaryBasicBlock *LoopHeader = Loop->getHeader();

  // Mark all blocks reachable from the loop head as not visited.
  tagReachableBlocks(LoopHeader);

  // Propagate frequencies from the loop head.
  propagateFrequencies(LoopHeader, LoopHeader);
}

void BlockEdgeFrequency::propagateFrequencies(BinaryBasicBlock *BB,
                                              BinaryBasicBlock *Head) {
  LLVM_DEBUG(dbgs() << "===== Current Basic block " << BB->getName() << " -  "
                    << " Head " << Head->getName() << " =======\n";);

  // Checks if the basic block BB has been visited.
  if (isVisited(BB))
    return;

  /// 1. Finds the block frequency of BB

  // If the basic block BB is a loop head then assumes it executes only once.
  BB->setExecutionCount(SCALING_FACTOR);

  // If BB is not the loop head, computes the basic block frequency by
  // adding all the in edges (the edges that go to this basic block).
  // If there is a back edge adds all the in edges and offsets it by the
  // cyclic probability value.
  if (BB != Head) {

    // Checks if each BB's predecessor is reachable from the head (if it is
    // marked as not visited) and if the edge departing from each BB's
    // predecessor was previously processed.
    for (BinaryBasicBlock *PredBB : BB->predecessors()) {
      Edge CFGEdge = std::make_pair(PredBB->getLabel(), BB->getLabel());
      if (!isVisited(PredBB) && !BSI->isBackEdge(CFGEdge)) {
        // There is an unprocessed predecessor edge.
        return;
      }
    }

    // Holds the sum of the incoming edges frequencies for this block.
    BB->setExecutionCount(0);

    // Holds the cyclic probability of BB. The cyclic probability of BB
    // is the probability along all the paths that control goes from BB to BB.
    double CyclicProbability = 0.0;

    // Updates the block frequency of BB or computes the cyclic probability of
    // BB if the edge that goes from the predecessor to BB is a back edge.
    for (BinaryBasicBlock *PredBB : BB->predecessors()) {
      Edge CFGEdge = std::make_pair(PredBB->getLabel(), BB->getLabel());
      if (BSI->isBackEdge(CFGEdge) && BSI->isLoopHeader(BB)) {
        CyclicProbability += SBP->getCFGBackEdgeProbability(*PredBB, *BB);
        CyclicProbability =
            ((DBL_MAX - 1.0) <= CyclicProbability || CyclicProbability < 0.0)
                ? 0.0
                : CyclicProbability;
      } else {
        double BBExecCount = static_cast<double>(
            getBBExecutionCount(*BB) + getCFGEdgeFrequency(*PredBB, *BB));
        uint64_t UBBExecCount =
            (static_cast<double>(UINT64_MAX) < BBExecCount || BBExecCount < 0.0)
                ? 0
                : static_cast<uint64_t>(BBExecCount);

        BB->setExecutionCount(UBBExecCount);
      }
    }

    LLVM_DEBUG(dbgs() << "CURRENT BLOCK FREQUENCY:\n BlockFrequencies[ "
                      << BB->getName() << " ] = " << getBBExecutionCount(*BB)
                      << "\n CyclicProbability " << CyclicProbability << "\n";);

    // For a loop that terminates, the cyclic probability is less than one.
    // If a loop seems not to terminate the cyclic probability is higher than
    // one. Since the algorithm does not work as supposed to if the probability
    // is higher than one, we need to set it to the maximum value offset by
    // the  constant EPSILON.
    double UpperBound = SCALING_FACTOR - (EPSILON);
    uint64_t intCyclicProbability = CyclicProbability / SCALING_FACTOR;
    if (intCyclicProbability > UpperBound) {
      CyclicProbability = UpperBound;
    }

    double BBExecCount = round(
        (getBBExecutionCount(*BB) / (SCALING_FACTOR - CyclicProbability)) *
        SCALING_FACTOR);
    uint64_t UBBExecCount =
        (static_cast<double>(UINT64_MAX) < BBExecCount || BBExecCount < 0.0)
            ? 0
            : static_cast<uint64_t>(BBExecCount);

    BB->setExecutionCount(UBBExecCount);

    LLVM_DEBUG(dbgs() << "UPDATED BLOCK FREQUENCY\n BlockFrequencies[ "
                      << BB->getName()
                      << " ] = " << BB->getKnownExecutionCount()
                      << "\n CyclicProbability " << CyclicProbability << "\n";);
  }

  // Mark the basic block BB as visited.
  setVisited(BB);

  /// 2. Calculate the frequencies of BB's out edges
  for (BinaryBasicBlock *SuccBB : BB->successors()) {
    Edge CFGEdge = std::make_pair(BB->getLabel(), SuccBB->getLabel());
    double EdgeProb = SBP->getCFGEdgeProbability(*BB, *SuccBB);
    double EdgeFreq = EdgeProb * getBBExecutionCount(*BB);
    EdgeFreq = (EdgeFreq == DBL_MAX || EdgeFreq < 0.0) ? 0.0 : EdgeFreq;

    LLVM_DEBUG(dbgs() << "CURRENT EDGE FREQ INFO:\n " << BB->getName() << " -> "
                      << SuccBB->getName() << " : "
                      << getCFGEdgeFrequency(*BB, *SuccBB) << "\n";);

    BB->setSuccessorBranchInfo(*SuccBB, static_cast<int64_t>(round(EdgeFreq)),
                               0);

    LLVM_DEBUG(dbgs() << "UPDATED EDGE FREQ INFO:\n " << BB->getName() << " -> "
                      << SuccBB->getName() << " : "
                      << getCFGEdgeFrequency(*BB, *SuccBB) << "\n";);

    // Update back edge probability, in case the current successor is equal
    // to the head so it can be used by outer loops to calculate cyclic
    // probabilities of inner loops.
    if (SuccBB == Head) {
      LLVM_DEBUG(dbgs() << "CURRENT BACK EDGE PROB INFO:\n " << BB->getName()
                        << " -> " << SuccBB->getName() << " : "
                        << SBP->getCFGBackEdgeProbability(*BB, *SuccBB)
                        << "\n";);

      SBP->setCFGBackEdgeProbability(CFGEdge, EdgeFreq);
      LLVM_DEBUG(dbgs() << "UPDATED BACK EDGE PROB INFO:\n " << BB->getName()
                        << " -> " << SuccBB->getName() << " : "
                        << SBP->getCFGBackEdgeProbability(*BB, *SuccBB)
                        << "\n";);
    }
  }

  /// 3. Propagates the calculated frequencies to the successors of BB
  /// that are not back edges.
  for (BinaryBasicBlock *SuccBB : BB->successors()) {
    Edge CFGEdge = std::make_pair(BB->getLabel(), SuccBB->getLabel());
    if (!BSI->isBackEdge(CFGEdge)) {
      propagateFrequencies(SuccBB, Head);
    }
  }
}

bool BlockEdgeFrequency::checkPrecision(BinaryFunction &Function) const {

  // If the function has only one basic block the frequency
  // matches by definition.
  if (Function.size() == 1) {
    return true;
  }

  // Holds the sum of all edge frequencies that lead
  // to a terminator basic block.
  uint64_t OutFreq = 0;
  double DOutFreq = 0.0;

  for (BinaryBasicBlock &BB : Function) {
    // A basic block that does not have a successor
    // is a terminator basic block.
    if (BB.succ_size() == 0) {
      for (BinaryBasicBlock *PredBB : BB.predecessors()) {
        double BBCount = static_cast<double>(PredBB->getBranchInfo(BB).Count);
        if (static_cast<double>(UINT64_MAX) < BBCount || BBCount < 0.0) {
          PredBB->setSuccessorBranchInfo(BB, 0, 0);
          BBCount = 0.0;
        }

        DOutFreq += BBCount;
      }
    }
  }

  OutFreq = (static_cast<double>(UINT64_MAX) < DOutFreq || DOutFreq < 0.0)
                ? 0
                : static_cast<uint64_t>(DOutFreq);
  // Checks if the calculated frequency is within the defined boundary.
  bool Holds =
      (OutFreq >= static_cast<uint64_t>(round(SCALING_FACTOR - (LOOSEBOUND))) &&
       OutFreq <= static_cast<uint64_t>(round(SCALING_FACTOR + (LOOSEBOUND))));
       
  return Holds;
}

void BlockEdgeFrequency::updateLocalCallFrequencies(BinaryFunction &Function) {
  BinaryContext &BC = Function.getBinaryContext();
  for (BinaryBasicBlock &BB : Function) {
    uint64_t TakenFreqEdge = getBBExecutionCount(BB);
    for (MCInst &Inst : BB) {
      if (!BC.MIB->isCall(Inst))
        continue;

      StringRef CallAnnotation = "Count";
      if (BC.MIB->getConditionalTailCall(Inst)) {
        CallAnnotation = "CTCTakenCount";
      }

      if (!BC.MIB->hasAnnotation(Inst, CallAnnotation)) {
        BC.MIB->addAnnotation<uint64_t>(Inst, CallAnnotation, TakenFreqEdge);
      } else {
        if (auto CountAnnt =
                BC.MIB->tryGetAnnotationAs<uint64_t>(Inst, CallAnnotation)) {

          double Count = static_cast<double>(*CountAnnt) +
                         static_cast<double>(TakenFreqEdge);
          (*CountAnnt) += static_cast<uint64_t>(
              (static_cast<double>(UINT64_MAX) < Count || Count < 0.0)
                  ? 0
                  : static_cast<uint64_t>(Count));
        }
      }
    }
  }
}

void BlockEdgeFrequency::dumpProfileData(BinaryFunction &Function,
                                         raw_ostream &Printer) {
  BinaryContext &BC = Function.getBinaryContext();

  std::string FromFunName = Function.getPrintName();
  for (BinaryBasicBlock &BB : Function) {
    auto LastInst = BB.getLastNonPseudoInstr();
    for (auto &Inst : BB) {
      if (!BC.MIB->isCall(Inst))
        continue;

      auto Offset = BC.MIB->tryGetAnnotationAs<uint32_t>(Inst, "Offset");

      if (!Offset)
        continue;

      uint64_t TakenFreqEdge = 0;
      uint64_t NotTakenFreqEdge = 0;
      auto FromBb = Offset.get();
      std::string ToFunName;
      uint32_t ToBb;

      auto *CalleeSymbol = BC.MIB->getTargetSymbol(Inst);
      if (!CalleeSymbol)
        continue;

      ToFunName = CalleeSymbol->getName();
      ToBb = 0;

      StringRef CallAnnotation = "Count";
      if (BC.MIB->getConditionalTailCall(Inst)) {
        CallAnnotation = "CTCTakenCount";
      }

      if (!BC.MIB->hasAnnotation(Inst, CallAnnotation)) {
        BC.MIB->addAnnotation<uint64_t>(Inst, CallAnnotation, TakenFreqEdge);
      } else {
        if (auto CountAnnt =
                BC.MIB->tryGetAnnotationAs<uint64_t>(Inst, CallAnnotation)) {
          TakenFreqEdge = (*CountAnnt);
        }
      }

      if (TakenFreqEdge > 0)
        Printer << "1 " << FromFunName << " " << Twine::utohexstr(FromBb)
                << " 1 " << ToFunName << " " << Twine::utohexstr(ToBb) << " "
                << NotTakenFreqEdge << " " << TakenFreqEdge << "\n";
    }

    if (!LastInst)
      continue;

    auto Offset = BC.MIB->tryGetAnnotationAs<uint32_t>(*LastInst, "Offset");

    if (!Offset)
      continue;

    uint64_t TakenFreqEdge = 0;
    uint64_t NotTakenFreqEdge = 0;
    auto FromBb = Offset.get();
    std::string ToFunName;
    uint32_t ToBb;

    for (BinaryBasicBlock *SuccBB : BB.successors()) {
      TakenFreqEdge = getCFGEdgeFrequency(BB, *SuccBB);
      BinaryFunction *ToFun = SuccBB->getFunction();
      ToFunName = ToFun->getPrintName();
      ToBb = SuccBB->getInputOffset();

      if (TakenFreqEdge > 0)
        Printer << "1 " << FromFunName << " " << Twine::utohexstr(FromBb)
                << " 1 " << ToFunName << " " << Twine::utohexstr(ToBb) << " "
                << NotTakenFreqEdge << " " << TakenFreqEdge << "\n";
    }
  }
}

double BlockEdgeFrequency::getLocalEdgeFrequency(BinaryBasicBlock *SrcBB,
                                                 BinaryBasicBlock *DstBB) {
  double LocalEdgeFreq =
      static_cast<double>(SrcBB->getBranchInfo(*DstBB).Count);
  if (static_cast<double>(UINT64_MAX) < LocalEdgeFreq || LocalEdgeFreq < 0.0) {
    SrcBB->setSuccessorBranchInfo(*DstBB, 0, 0);
    return 0;
  }

  return static_cast<double>(LocalEdgeFreq / SCALING_FACTOR);
}

double BlockEdgeFrequency::getLocalBlockFrequency(BinaryBasicBlock *BB) {
  return static_cast<double>(getBBExecutionCount(*BB) / SCALING_FACTOR);
}

void BlockEdgeFrequency::computeBlockEdgeFrequencies(BinaryFunction &Function) {
  if (!Function.isLoopFree()) {
    // Discover all loops of this function.
    Function.calculateLoopInfo();
    const BinaryLoopInfo &LoopInfo = Function.getLoopInfo();
    // // Find all loop headers and loop back edges of this function.
    BSI->findLoopEdgesInfo(LoopInfo);
    for (BinaryLoop *BL : LoopInfo)
      propagateLoopFrequencies(BL);
  }

  /// Propagates the frequencies for all the basic blocks of the function
  /// making the entry block as the head of the function.
  BinaryBasicBlock *EntryBlock = &(*Function.begin());
  tagReachableBlocks(EntryBlock);
  propagateFrequencies(EntryBlock, EntryBlock);
}

void BlockEdgeFrequency::clear() {
  BSI->clear();
  SBP->clear();
  ReachableBBs.clear();
  VisitedLoops.clear();
  BlockFrequencies.clear();
  CFGEdgeFrequencies.clear();
}

bool BlockEdgeFrequency::computeFrequencies(BinaryFunction &Function,
                                            bool precision = true) {
  computeBlockEdgeFrequencies(Function);
  // Checks if the computed frequencies are within the precision
  // boundary.
  if (Function.getKnownExecutionCount() == 0)
    return false;
  bool Holds = (precision) ? checkPrecision(Function) : true;
  if (Holds) {
    updateLocalCallFrequencies(Function);
    Function.markProfiled(BinaryFunction::PF_STATIC);
    Function.setExecutionCount(SCALING_FACTOR);
  } else {
    Function.setExecutionCount(0);
    Function.clearProfile();
    Function.unsetProfileMatchRatio();
    LLVM_DEBUG(
        dbgs() << "BOLT-DEBUG: The local block and flow edge frequencies\n"
               << " for function: " << Function.getPrintName()
               << "BOLT-DEBUG: were calculated with accuracy bellow the "
               << "desirable boundary.\n "
               << "BOLT-DEBUG: Thus its CFG were dumpped in a dot and "
               << "txt formats\n.";
        Function.dumpGraphForPass("unchecked-block-edge-frequency");
        Function.dumpGraphToTextFile("unchecked-block-edge-frequency"););
  }

  clear();
  return Holds;
}

void BlockEdgeFrequency::updateStaleEdgesFrequencies(BinaryContext &BC) {
  outs() << "BOLT-INFO: based on probabilities infered by heuristics.\n";
  uint64_t Unchecked = 0;
  uint64_t Checked = 0;
  uint64_t test = 0;
  auto &BFs = BC.getBinaryFunctions();
  for (auto &BFI : BFs) {

    BinaryFunction &Function = BFI.second;
    if (Function.hasValidProfile() || Function.empty() ||
        !Function.isSimple() || Function.hasUnknownControlFlow() ||
        Function.isPLTFunction())

      continue;

    BinaryBasicBlock &EntryBB = (*Function.begin());
    if (!Function.hasProfile()) {
      Function.setExecutionCount(SCALING_FACTOR);
      EntryBB.setExecutionCount(SCALING_FACTOR);
    }

    if (opts::WuLarusStaleProfile) {
      Function.clearProfile();
      SBP->computeHeuristicBasedProbabilities(Function);
    }

    Function.setExecutionCount(SCALING_FACTOR);
    if (computeFrequencies(Function)) {
      Function.markProfiled(BinaryFunction::PF_STATIC);
      Checked++;
    } else {
      Unchecked++;
    }
    clear();
  }

  outs() << "================ Checked: " << Checked << "\n";
  outs() << "================ Unchecked: " << Unchecked << "\n";

  outs() << "BOLT-INFO: the BB counts and local Edge counts where updated"
         << " based on intraprodecural inference.\n";
}

void BlockEdgeFrequency::updateStaleEdgesFrequencies(
    BinaryContext &BC, std::set<uint64_t> &StaleFunctionsAddresses) {

  NamedRegionTimer T("updateStaleEdges", "gen static profile", "beetle",
                     "Improve stale profile", opts::TimeBuild);

  outs()
      << "BOLT-INFO: starting block and flow edge frequency inference pass\n";
  outs() << "BOLT-INFO: computing local static infered frequencies\n";

  auto &BFs = BC.getBinaryFunctions();
  for (auto &BFI : BFs) {

    BinaryFunction &Function = BFI.second;

    if (Function.empty() ||
        (StaleFunctionsAddresses.find(Function.getAddress()) ==
         StaleFunctionsAddresses.end()))
      continue;

    Function.setExecutionCount(0);
    Function.clearProfile();
  }
  std::set<uint64_t> MatchedStaleFuncAddresses;
  if (!opts::MLBased) {
    outs() << "BOLT-INFO: based on probabilities infered by heuristics.\n";
  } else {
    outs() << "BOLT-INFO: based on probabilities infered by a ML model.\n";
    outs() << "BOLT-INFO: processing the file " << opts::ProbFilename << "\n";

    auto MB = MemoryBuffer::getFileOrSTDIN(opts::ProbFilename);

    if (auto EC = MB.getError()) {
      errs() << "BOLT-ERROR: Cannot open " << opts::ProbFilename << ": "
             << EC.message() << "\n";
      exit(1);
    }

    SBP->parseProbabilitiesFile(std::move(MB.get()), BC,
                                StaleFunctionsAddresses,
                                MatchedStaleFuncAddresses);
  }

  uint64_t Unchecked = 0;
  uint64_t Checked = 0;
  uint64_t test = 0;
  for (auto &BFI : BFs) {

    BinaryFunction &Function = BFI.second;
    auto it = StaleFunctionsAddresses.find(Function.getAddress());
    if (Function.empty() || !Function.isSimple() ||
        (Function.getExecutionCount() != 1 && opts::MLBased) ||
        Function.hasUnknownControlFlow() || Function.isPLTFunction() ||
        (it == StaleFunctionsAddresses.end()))

      continue;

    auto it2 = MatchedStaleFuncAddresses.find(Function.getAddress());
    if (opts::MLBased && it2 == MatchedStaleFuncAddresses.end()) {
      continue;
    }

    BinaryBasicBlock &EntryBB = (*Function.begin());
    if (!Function.hasProfile()) {
      if (opts::MLBased && Function.size() != 1) {
        Function.setExecutionCount(0);
        Function.clearProfile();
        Function.unsetProfileMatchRatio();
        continue;
      }

      Function.setExecutionCount(SCALING_FACTOR);
      EntryBB.setExecutionCount(SCALING_FACTOR);
    }

    if (opts::WuLarusStaleProfile) {
      Function.clearProfile();
      SBP->computeHeuristicBasedProbabilities(Function);
    } else
      SBP->computeProbabilities(Function);

    Function.setExecutionCount(SCALING_FACTOR);
    if (computeFrequencies(Function)) {
      StaleFunctionsAddresses.erase(it);
      Function.markProfiled(BinaryFunction::PF_STATIC);
      Checked++;
    } else {
      Unchecked++;
    }
    clear();
  }

  outs() << "BOLT-INFO: Number of functions that CHECKS: " << Checked << "\n";
  outs() << "BOLT-INFO: Number of UNCHECKED functions: " << Unchecked << "\n";

  outs() << "BOLT-INFO: the BB counts and local Edge counts where updated"
         << " based on intraprodecural inference.\n";
}

void BlockEdgeFrequency::runOnStaleFunctions(
    BinaryContext &BC, std::set<uint64_t> &StaleFunctionsAddresses) {
  bool numValidProfile = 0;
  outs()
      << "BOLT-INFO: starting block and flow edge frequency inference pass\n";
  outs() << "BOLT-INFO: computing local static infered frequencies\n";

  if (!opts::DisableBeetle) {
    outs() << "BOLT-INFO: based on frequencies gathered by beetle technique.\n";
    outs() << "BOLT-INFO: processing the file " << opts::BeetleFilename << "\n";

    auto MB = MemoryBuffer::getFileOrSTDIN(opts::BeetleFilename);

    if (auto EC = MB.getError()) {
      errs() << "BOLT-ERROR: Cannot open " << opts::BeetleFilename << ": "
             << EC.message() << "\n";
      exit(EXIT_FAILURE);
    }

    SBP->parseBeetleProfileFile(std::move(MB.get()), BC,
                                StaleFunctionsAddresses);
  }

  outs() << "BOLT-INFO: the BB counts and local Edge counts where updated"
         << " based on beetle approach.\n";

  if ((opts::WuLarusStaleProfile || opts::VespaStaleProfile) &&
      StaleFunctionsAddresses.size() > 0) {
    opts::MLBased = (opts::VespaStaleProfile) ? true : false;
    updateStaleEdgesFrequencies(BC, StaleFunctionsAddresses);
  }
}

void BlockEdgeFrequency::runOnStaleFunctions(BinaryContext &BC) {
  outs()
      << "BOLT-INFO: starting block and flow edge frequency inference pass\n";
  outs() << "BOLT-INFO: computing local static infered frequencies\n";

  if (!opts::MLBased) {
    outs() << "BOLT-INFO: based on probabilities infered by heuristics.\n";
  } else {
    outs() << "BOLT-INFO: based on probabilities infered by a ML model.\n";
    outs() << "BOLT-INFO: processing the file " << opts::ProbFilename << "\n";

    auto MB = MemoryBuffer::getFileOrSTDIN(opts::ProbFilename);

    if (auto EC = MB.getError()) {
      errs() << "BOLT-ERROR: Cannot open " << opts::ProbFilename << ": "
             << EC.message() << "\n";
      exit(1);
    }

    SBP->parseProbabilitiesFile(std::move(MB.get()), BC);
  }

  const char *FileName = "localFrequencies.fdata";

  LLVM_DEBUG(
      dbgs() << "BOLT-DEBUG: dumping local static infered frequencies to "
             << FileName << "\n";
      std::error_code EC;
      raw_fd_ostream Printer(FileName, EC, sys::fs::OF_None); if (EC) {
        dbgs() << "BOLT-ERROR: " << EC.message() << ", unable to open "
               << FileName << " for output.\n";
        exit(1);
      });

  uint64_t Unchecked = 0;
  uint64_t Checked = 0;
  auto &BFs = BC.getBinaryFunctions();
  for (auto &BFI : BFs) {

    BinaryFunction &Function = BFI.second;

    if (Function.hasValidProfile() || !Function.isSimple() ||
        Function.isPLTFunction() || Function.hasUnknownControlFlow() ||
        Function.empty() ||
        (Function.getExecutionCount() != 1 && opts::MLBased))
      continue;

    for (BinaryBasicBlock &B : Function) {
      B.setExecutionCount(0);
    }
  }
  for (auto &BFI : BFs) {

    BinaryFunction &Function = BFI.second;
    if (Function.hasValidProfile() || !Function.isSimple() ||
        Function.isPLTFunction() || Function.hasUnknownControlFlow() ||
        Function.empty() ||
        (Function.getExecutionCount() != 1 && opts::MLBased))
      continue;

    BinaryBasicBlock &EntryBB = (*Function.begin());
    if (!Function.hasProfile()) {
      if (opts::MLBased && Function.size() != 1) {
        Function.setExecutionCount(0);
        Function.clearProfile();
        Function.unsetProfileMatchRatio();
        continue;
      }

      Function.setExecutionCount(SCALING_FACTOR);
      EntryBB.setExecutionCount(SCALING_FACTOR);
    }

    if (opts::HeuristicBased == bolt::StaticBranchProbabilities::H_WU_LARUS)
      SBP->computeHeuristicBasedProbabilities(Function);
    else
      SBP->computeProbabilities(Function);

    Function.setExecutionCount(SCALING_FACTOR);
    bool Holds = computeFrequencies(Function);
    if (Holds) {
      ++Checked;

      LLVM_DEBUG(std::error_code EC;
                 raw_fd_ostream Printer(FileName, EC, sys::fs::OF_Append);
                 if (EC) {
                   dbgs() << "BOLT-WARNING: " << EC.message()
                          << ", unable to open"
                          << " " << FileName << " for output.\n";
                   exit(1);
                 } dumpProfileData(Function, Printer););

    } else {
      ++Unchecked;
    }

    clear();
  }

  LLVM_DEBUG(
      dbgs() << "BOLT-DEBUG: Number of unchecked functions: " << Unchecked
             << "\nBOLT-DEBUG: Number of functions that cheked: " << Checked
             << "\nBOLT-DEBUG: Total number of functions that were processed: "
             << (Checked + Unchecked) << "\n";);

  outs() << "BOLT-INFO: the BB counts and local Edge counts where updated"
         << " based on intraprodecural inference.\n";
}

Error BlockEdgeFrequency::runOnFunctions(BinaryContext &BC) {
  outs()
      << "BOLT-INFO: starting block and flow edge frequency inference pass\n";
  outs() << "BOLT-INFO: computing local static infered frequencies\n";

  if (!opts::MLBased) {
    outs() << "BOLT-INFO: based on probabilities infered by heuristics.\n";
  } else {
    outs() << "BOLT-INFO: based on probabilities infered by a ML model.\n";
    outs() << "BOLT-INFO: processing the file " << opts::ProbFilename << "\n";

    auto MB = MemoryBuffer::getFileOrSTDIN(opts::ProbFilename);

    if (auto EC = MB.getError()) {
      errs() << "BOLT-ERROR: Cannot open " << opts::ProbFilename << ": "
             << EC.message() << "\n";
      exit(1);
    }

    SBP->parseProbabilitiesFile(std::move(MB.get()), BC);
  }

  const char *FileName = "localFrequencies.fdata";

  LLVM_DEBUG(
      dbgs() << "BOLT-DEBUG: dumping local static infered frequencies to "
             << FileName << "\n";
      std::error_code EC;
      raw_fd_ostream Printer(FileName, EC, sys::fs::OF_None); if (EC) {
        dbgs() << "BOLT-ERROR: " << EC.message() << ", unable to open "
               << FileName << " for output.\n";
        exit(1);
      });

  uint64_t Unchecked = 0;
  uint64_t Checked = 0;
  auto &BFs = BC.getBinaryFunctions();
  for (auto &BFI : BFs) {

    BinaryFunction &Function = BFI.second;

    if (Function.empty() ||
        (Function.getExecutionCount() != 1 && opts::MLBased))
      continue;

    for (BinaryBasicBlock &B : Function) {
      B.setExecutionCount(0);
    }
  }
  for (auto &BFI : BFs) {

    BinaryFunction &Function = BFI.second;
    if (Function.empty() ||
        (Function.getExecutionCount() != 1 && opts::MLBased))
      continue;

    BinaryBasicBlock &EntryBB = (*Function.begin());
    if (!Function.hasProfile()) {
      if (opts::MLBased && Function.size() != 1) {
        Function.setExecutionCount(0);
        Function.clearProfile();
        Function.unsetProfileMatchRatio();
        continue;
      }

      Function.setExecutionCount(SCALING_FACTOR);
      EntryBB.setExecutionCount(SCALING_FACTOR);
    }

    if (opts::HeuristicBased == bolt::StaticBranchProbabilities::H_WU_LARUS)
      SBP->computeHeuristicBasedProbabilities(Function);
    else
      SBP->computeProbabilities(Function);

    Function.setExecutionCount(SCALING_FACTOR);
    bool Holds = computeFrequencies(Function);
    if (Holds) {
      ++Checked;
      LLVM_DEBUG(std::error_code EC;
                 raw_fd_ostream Printer(FileName, EC, sys::fs::OF_Append);
                 if (EC) {
                   dbgs() << "BOLT-WARNING: " << EC.message()
                          << ", unable to open"
                          << " " << FileName << " for output.\n";
                   exit(1);
                 } dumpProfileData(Function, Printer););
    } else {
      ++Unchecked;
    }
    clear();
  }

  LLVM_DEBUG(
      dbgs() << "BOLT-DEBUG: Number of unchecked functions: " << Unchecked
             << "\nBOLT-DEBUG: Number of functions that cheked: " << Checked
             << "\nBOLT-DEBUG: Total number of functions that were processed: "
             << (Checked + Unchecked) << "\n";);

  outs() << "BOLT-INFO: the BB counts and local Edge counts where updated"
         << " based on intraprodecural inference.\n";

  return Error::success();
}

} // namespace bolt
} // namespace llvm
