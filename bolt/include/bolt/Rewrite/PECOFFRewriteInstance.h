//===- bolt/Rewrite/PECOFFRewriteInstance.h - PE/COFF rewriter --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Interface to control an instance of a PE/COFF binary rewriting process.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_REWRITE_PECOFF_REWRITE_INSTANCE_H
#define BOLT_REWRITE_PECOFF_REWRITE_INSTANCE_H

#include "bolt/Core/Linker.h"
#include "bolt/Rewrite/PDBRewriter.h"
#include "bolt/Rewrite/WinEHFuncInfoReader.h"
#include "bolt/Utils/NameResolver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include <memory>

namespace llvm {
class ToolOutputFile;
class raw_pwrite_stream;
namespace object {
class COFFObjectFile;
class RelocationRef;
} // namespace object

namespace bolt {

class BinaryContext;
class ProfileReaderBase;

/// Parsed SEH unwind information for a single function.
struct SEHUnwindInfo {
  uint32_t EndRVA = 0;
  uint8_t Version = 0;
  uint8_t Flags = 0;
  uint8_t PrologSize = 0;
  uint8_t FrameRegister = 0;
  uint8_t FrameOffset = 0;
  SmallVector<uint16_t, 8> UnwindCodes;
  uint32_t ExceptionHandlerRVA = 0;
  bool HasExceptionHandler = false;
  /// RVA of the MSVC C++ FuncInfo structure when the personality routine is
  /// __CxxFrameHandler3 (i.e. \c IsCxxEH is set), otherwise 0.
  uint32_t CxxFuncInfoRVA = 0;
  /// True when the handler data decodes as a classic __CxxFrameHandler3
  /// FuncInfo (recognized magic number).  __CxxFrameHandler4's compressed
  /// format does not match and leaves this false.
  bool IsCxxEH = false;
  bool IsChained = false;
  uint32_t ChainedBeginRVA = 0;
  uint32_t ChainedEndRVA = 0;
  uint32_t ChainedUnwindRVA = 0;
  /// RVA of the chained RUNTIME_FUNCTION record itself (inside .xdata), i.e.
  /// where ChainedBeginRVA/ChainedEndRVA are stored.  Used to repoint a
  /// funclet's chain when its primary function is moved out-of-place.
  uint32_t ChainedEntryRVA = 0;
};

class PECOFFRewriteInstance {
  object::COFFObjectFile *InputFile;
  StringRef ToolPath;
  std::unique_ptr<BinaryContext> BC;
  std::unique_ptr<BOLTLinker> Linker;

  NameResolver NR;

  std::unique_ptr<ToolOutputFile> Out;

  /// Cached PE ImageBase, read once during construction.
  uint64_t ImageBase = 0;

  /// Functions whose basic block layout was changed by optimization passes.
  DenseSet<uint64_t> ModifiedFunctions;

  /// Address translation for rewritten functions: maps old instruction
  /// offsets (within the function) to new offsets after BB reordering.
  /// Captured before CFG is released so PDB line tables can be remapped.
  /// Key: function VA.  Value: vector of {old_offset, new_offset} pairs.
  using OffsetMap = std::vector<std::pair<uint32_t, uint32_t>>;
  DenseMap<uint64_t, OffsetMap> FunctionOffsetMaps;

  /// Functions relocated out-of-place into the .bolt section, in ascending
  /// original-RVA order.  Used to build PDB OMAP address-translation tables.
  std::vector<BoltRelocatedFunc> RelocatedFuncs;

  /// RVA and byte size of the emitted .bolt section (0 if none was created).
  uint32_t BoltSectionRVA = 0;
  uint32_t BoltSectionSize = 0;

  std::unique_ptr<ProfileReaderBase> ProfileReader;

  /// SEH unwind info indexed by function begin RVA.
  DenseMap<uint32_t, SEHUnwindInfo> FunctionSEHInfo;

  /// Parsed MSVC C++ FuncInfo indexed by function begin RVA, populated for
  /// functions whose personality is __CxxFrameHandler3.
  DenseMap<uint32_t, WinEHFuncInfo> FunctionCxxEHInfo;

  /// Image sections retained for deferred C++ EH parsing.
  WinEHImageReader CxxEHImageReader;

  /// RVAs of out-of-line catch/cleanup funclets referenced by C++ EH FuncInfo.
  /// These are pinned in place so that funclet RVAs in the EH metadata stay
  /// valid when a parent function is reordered.
  DenseSet<uint32_t> CxxEHFuncletRVAs;

  /// RVAs of C++ EH functions eligible for reordering (currently: parsed
  /// __CxxFrameHandler3 personality, no try blocks, non-empty IPToState map).
  DenseSet<uint32_t> CxxEHCandidateRVAs;

  /// Regenerated, verified IPToState tables for reordered C++ EH functions,
  /// keyed by function begin RVA.  Populated by relocateCxxEHTables() in the
  /// non-dry-run path; entries hold image-relative IP RVAs and are written to a
  /// fresh table during rewriteFile(), which repoints the FuncInfo.
  DenseMap<uint32_t, SmallVector<WinEHFuncInfo::IPToStateEntry, 16>>
      RegeneratedEHTables;

  /// Number of functions skipped due to exception handlers.
  uint64_t NumFuncsWithHandlers = 0;

  /// Number of functions with a successfully parsed C++ FuncInfo.
  uint64_t NumCxxEHFuncs = 0;

  /// Number of functions skipped due to size overflow after optimization.
  uint64_t NumFuncsOverflow = 0;

  void preprocessProfileData();
  void processProfileDataPreCFG();
  void processProfileData();

  void adjustCommandLineOptions();
  void readSpecialSections();
  void readExceptionHandling();
  void readCxxEHIPToStateMaps();
  void discoverFileObjects();

  /// In lite mode, ignore functions without profile data before disassembly.
  void selectFunctionsToProcess();

  void disassembleFunctions();
  void buildFunctionsCFG();
  void postProcessFunctions();
  void runOptimizationPasses();
  void freezePrologInstructions();
  void emitAndLink();

  /// Regenerate the C++ EH IPToState table for reordered in-place functions
  /// (Phase 2). Runs after emitAndLink() so per-instruction output addresses
  /// are available via the IO address map. When \p DryRun is true, the
  /// regenerated table is verified and reported but nothing is written to the
  /// output binary, and the affected functions are reverted to their original
  /// layout so their EH metadata stays valid. When \p DryRun is false, each
  /// verified table is retained in RegeneratedEHTables for rewriteFile() to
  /// emit and repoint the FuncInfo, and the function is left reordered.
  void relocateCxxEHTables(bool DryRun);

  void mapCodeSections(BOLTLinker::SectionMapper MapSection);
  void rewriteFile();
  void identityRewriteFile();

  static StringRef getNewSecPrefix() { return ".bolt.new"; }
  static StringRef getOrgSecPrefix() { return ".bolt.org"; }

public:
  PECOFFRewriteInstance(object::COFFObjectFile *InputFile, StringRef ToolPath,
                        Error &Err);

  static Expected<std::unique_ptr<PECOFFRewriteInstance>>
  create(object::COFFObjectFile *InputFile, StringRef ToolPath);
  ~PECOFFRewriteInstance();

  Error setProfile(StringRef FileName);

  /// Run all the necessary steps to read, optimize and rewrite the binary.
  void run();
};

} // namespace bolt
} // namespace llvm

#endif
