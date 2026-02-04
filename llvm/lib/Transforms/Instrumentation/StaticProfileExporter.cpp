//===- StaticProfileExporter.cpp - Export Static Profile Info -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the StaticProfileExporterPass which exports statically
// inferred profile information from BlockFrequencyInfo in .profdata format 
// for use with llvm-cov and other profile tools.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/StaticProfileExporter.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/ProfileData/InstrProfWriter.h"
#include "llvm/Support/CRC.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "static-profile-export"

using namespace llvm;

namespace llvm {

// Fixed entry count for scaling static frequencies to absolute counts
static constexpr uint64_t DefaultEntryCount = 1000000;

// Compute function hash similar to PGOInstrumentation
static uint64_t computeFunctionHash(const Function &F) {
  uint32_t CRC = 0;
  for (const BasicBlock &BB : F) {
    for (const Instruction &I : BB) {
      CRC = llvm::crc32(CRC, I.getOpcode());
    }
  }
  
  uint64_t Hash = static_cast<uint64_t>(CRC);
  Hash &= NamedInstrProfRecord::FUNC_HASH_MASK;
  return Hash;
}

// Convert BlockFrequencyInfo frequencies to execution counts
static bool convertBFIToCounts(const Function &F, const BlockFrequencyInfo &BFI,
                                std::vector<uint64_t> &Counts) {
  const BasicBlock &EntryBB = F.getEntryBlock();
  BlockFrequency EntryFreq = BFI.getBlockFreq(&EntryBB);
  
  if (EntryFreq.getFrequency() == 0) {
    LLVM_DEBUG(dbgs() << "Entry block has zero frequency for "
                      << F.getName() << "\n");
    return false;
  }

  Counts.clear();
  
  // For each basic block, scale its frequency relative to entry
  for (const BasicBlock &BB : F) {
    BlockFrequency BBFreq = BFI.getBlockFreq(&BB);
    // Scale: count = (DefaultEntryCount * BBFreq) / EntryFreq
    uint64_t Count = (DefaultEntryCount * BBFreq.getFrequency()) / 
                     EntryFreq.getFrequency();
    Counts.push_back(Count);
    
    LLVM_DEBUG(dbgs() << "BB " << BB.getName() << " freq=" 
                      << BBFreq.getFrequency() << " count=" << Count << "\n");
  }
  
  return true;
}

PreservedAnalyses StaticProfileExporterPass::run(Module &M,
                                                  ModuleAnalysisManager &MAM) {
  if (ProfilePath.empty())
    return PreservedAnalyses::all();

  auto &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  InstrProfWriter Writer;
  unsigned FunctionsProcessed = 0;

  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    // Get BlockFrequencyInfo analysis (run if not cached)
    BlockFrequencyInfo &BFI = FAM.getResult<BlockFrequencyAnalysis>(F);

    // Convert BFI frequencies to execution counts
    std::vector<uint64_t> Counts;
    if (!convertBFIToCounts(F, BFI, Counts)) {
      LLVM_DEBUG(dbgs() << "Failed to convert profile for " << F.getName() << "\n");
      continue;
    }
    
    // Create NamedInstrProfRecord
    std::string FuncName = getIRPGOFuncName(F);
    uint64_t FuncHash = computeFunctionHash(F);
    
    NamedInstrProfRecord Record(FuncName, FuncHash, Counts);
    Writer.addRecord(std::move(Record), 1, [&](Error E) {
      errs() << "Warning adding profile record for " << F.getName() 
             << ": " << toString(std::move(E)) << "\n";
    });
    
    ++FunctionsProcessed;
  }

  if (FunctionsProcessed == 0) {
    errs() << "Warning: No functions processed for static profile dump\n";
    return PreservedAnalyses::all();
  }

  // Write profile data to file
  std::error_code EC;
  raw_fd_ostream Output(ProfilePath, EC, sys::fs::OF_None);
  if (EC) {
    errs() << "Error opening profile output file '" << ProfilePath
           << "': " << EC.message() << "\n";
    return PreservedAnalyses::all();
  }

  if (auto E = Writer.write(Output)) {
    errs() << "Error writing profile data: " << toString(std::move(E)) << "\n";
    return PreservedAnalyses::all();
  }

  LLVM_DEBUG(dbgs() << "Static profile written to '" << ProfilePath << "' (" 
                    << FunctionsProcessed << " functions)\n");

  return PreservedAnalyses::all();
}

} // namespace llvm
