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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <string>

namespace llvm {
namespace bolt {

class BinaryContext;

/// Rewrite PDB offsets and line tables after block reordering.
class PDBRewriter {
public:
  /// Patch the PDB to match the optimized binary.
  using OffsetMap = std::vector<std::pair<uint32_t, uint32_t>>;
  static void rewritePDB(StringRef InputExe, StringRef OutputExe,
                         const BinaryContext &BC, uint64_t ImageBase,
                         const DenseSet<uint64_t> &ModifiedFunctions,
                         const DenseMap<uint64_t, OffsetMap> &OffsetMaps);
};

} // namespace bolt
} // namespace llvm

#endif
