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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <string>

namespace llvm {
namespace bolt {

class BinaryContext;

/// Rewrite PDB debug info to match the optimized binary layout.
///
/// Reads the original PDB, updates function offsets and line tables
/// based on the address translation from block reordering, and writes
/// a new PDB alongside the optimized binary.
class PDBRewriter {
public:
  /// Rewrite the PDB associated with the input binary.
  /// \p InputExe is the original PE binary path (to find the PDB reference).
  /// \p OutputExe is the optimized binary path (PDB will be written next to it).
  /// \p BC provides the address translation for rewritten functions.
  /// \p ModifiedFunctions contains VAs of functions whose layout changed.
  /// \p OffsetMaps maps function VA to old_offset->new_offset pairs for
  ///    remapping line tables.
  using OffsetMap = std::vector<std::pair<uint32_t, uint32_t>>;
  static void rewritePDB(StringRef InputExe, StringRef OutputExe,
                         const BinaryContext &BC, uint64_t ImageBase,
                         const DenseSet<uint64_t> &ModifiedFunctions,
                         const DenseMap<uint64_t, OffsetMap> &OffsetMaps);
};

} // namespace bolt
} // namespace llvm

#endif
