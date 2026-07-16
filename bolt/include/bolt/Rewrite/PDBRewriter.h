//===- bolt/Rewrite/PDBRewriter.h - PDB debug info updater ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Update PDB debug info after PE/COFF binary rewriting.  Patches function
// symbols (S_GPROC32) and line tables (DEBUG_S_LINES) to reflect the new
// code layout after block reordering.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_REWRITE_PDB_REWRITER_H
#define BOLT_REWRITE_PDB_REWRITER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <string>

namespace llvm {
namespace bolt {

class BinaryContext;

/// A function relocated out-of-place into the .bolt section.  Used to build
/// PDB OMAP address-translation tables so debuggers/profilers resolve the
/// moved code back to the original symbols.  All fields are image-relative
/// (RVA) except sizes.
struct BoltRelocatedFunc {
  uint32_t OrigRVA;  ///< Original function RVA.
  uint32_t OrigSize; ///< Original slot size in bytes.
  uint32_t NewRVA;   ///< New RVA in the .bolt section.
  uint32_t NewSize;  ///< Emitted size in bytes at the new location.
};

/// Rewrite PDB offsets and line tables after block reordering.
class PDBRewriter {
public:
  /// Patch the PDB to match the optimized binary.
  using OffsetMap = std::vector<std::pair<uint32_t, uint32_t>>;

  /// Update the PDB for the optimized binary.
  ///
  /// In-place block-reordered functions have their line tables remapped.
  /// When \p RelocatedFuncs is non-empty (functions moved out-of-place to the
  /// .bolt section), the PDB is rebuilt with OMAP address-translation tables
  /// (OmapToSrc/OmapFromSrc) plus updated section headers so the moved code
  /// resolves to the original symbols.
  static void rewritePDB(StringRef InputExe, StringRef OutputExe,
                         const BinaryContext &BC, uint64_t ImageBase,
                         const DenseSet<uint64_t> &ModifiedFunctions,
                         const DenseMap<uint64_t, OffsetMap> &OffsetMaps,
                         ArrayRef<BoltRelocatedFunc> RelocatedFuncs = {},
                         uint32_t BoltSectionRVA = 0,
                         uint32_t BoltSectionSize = 0);
};

} // namespace bolt
} // namespace llvm

#endif
