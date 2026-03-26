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

#include "bolt/Utils/NameResolver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
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
  uint8_t Version = 0;
  uint8_t Flags = 0;
  uint8_t PrologSize = 0;
  uint8_t FrameRegister = 0;
  uint8_t FrameOffset = 0;
  std::vector<uint16_t> UnwindCodes;
  uint32_t ExceptionHandlerRVA = 0;
  bool HasExceptionHandler = false;
  bool IsChained = false;
  uint32_t ChainedBeginRVA = 0;
  uint32_t ChainedEndRVA = 0;
  uint32_t ChainedUnwindRVA = 0;
};

class PECOFFRewriteInstance {
  object::COFFObjectFile *InputFile;
  StringRef ToolPath;
  std::unique_ptr<BinaryContext> BC;

  NameResolver NR;

  std::unique_ptr<ToolOutputFile> Out;

  /// Holds the resolved (relocated) bytes for each emitted function.
  /// Populated by emitAndLink() and read by rewriteFile().
  std::vector<std::vector<uint8_t>> ResolvedFunctionBytes;

  /// Resolved jump table data sections.  Each entry owns its data.
  struct JTDataEntry {
    uint64_t VA;
    uint64_t OwnerVA; ///< VA of the function that owns this jump table.
    std::vector<uint8_t> Data;
  };
  std::vector<JTDataEntry> ResolvedJTData;

  /// Functions whose basic block layout was changed by optimization passes.
  /// Only these functions get their bytes replaced in the output binary.
  DenseSet<uint64_t> ModifiedFunctions;

  /// Address translation for rewritten functions: maps old instruction
  /// offsets (within the function) to new offsets after BB reordering.
  /// Captured before CFG is released so PDB line tables can be remapped.
  /// Key: function VA.  Value: vector of {old_offset, new_offset} pairs.
  using OffsetMap = std::vector<std::pair<uint32_t, uint32_t>>;
  DenseMap<uint64_t, OffsetMap> FunctionOffsetMaps;

  std::unique_ptr<ProfileReaderBase> ProfileReader;

  /// SEH unwind info indexed by function begin RVA.
  std::map<uint64_t, SEHUnwindInfo> FunctionSEHInfo;

  /// Number of functions skipped due to exception handlers.
  uint64_t NumFuncsWithHandlers = 0;

  /// Number of functions skipped due to size overflow after optimization.
  uint64_t NumFuncsOverflow = 0;

  void preprocessProfileData();
  void processProfileDataPreCFG();
  void processProfileData();

  void adjustCommandLineOptions();
  void readSpecialSections();
  void readExceptionHandling();
  void discoverFileObjects();
  void disassembleFunctions();
  void buildFunctionsCFG();
  void postProcessFunctions();
  void runOptimizationPasses();
  void emitAndLink();
  void rewriteFile();
  void identityRewriteFile();

  /// Look up the virtual address of a symbol referenced by a relocation in
  /// the emitted COFF object.  Defined symbols resolve through their section
  /// VA; external symbols fall back to BinaryContext lookups.
  uint64_t resolveRelocSymbol(const object::COFFObjectFile *Obj,
                              const object::RelocationRef &Rel,
                              const StringMap<uint64_t> &SectionNameToVA);

  /// Apply a single COFF x86_64 relocation to a writable section buffer.
  void applyCOFFRelocation(MutableArrayRef<uint8_t> Data, uint64_t SectionVA,
                           const object::RelocationRef &Rel, uint64_t SymVA);

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
