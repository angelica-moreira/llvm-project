//===- bolt/Rewrite/WinEHFuncInfoReader.h - MSVC C++ EH parser --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Reader for the MSVC C++ exception-handling tables referenced by the
// __CxxFrameHandler3/4 personality routine (the FuncInfo structure and its
// sub-tables).  The on-disk layout mirrors what LLVM emits in
// llvm/lib/CodeGen/AsmPrinter/WinException.cpp.
//
// The reader only decodes the metadata; it does not dereference the code RVAs
// it stores (IP boundaries, cleanup/catch funclet entry points).  All offsets
// are image-relative (RVAs) on x86_64.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_REWRITE_WINEH_FUNCINFO_READER_H
#define BOLT_REWRITE_WINEH_FUNCINFO_READER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include <cstdint>

namespace llvm {
namespace bolt {

/// Resolves image-relative addresses (RVAs) to bytes.  A binary's exception
/// tables may live in several sections (typically .rdata/.xdata), so the reader
/// is populated with one entry per section that could hold EH metadata.
class WinEHImageReader {
public:
  void addSection(uint32_t RVA, ArrayRef<uint8_t> Data) {
    Sections.push_back({RVA, Data});
  }

  /// Return a view of \p Size bytes starting at \p RVA, or an error if the
  /// range is not fully contained in a single known section.
  Expected<ArrayRef<uint8_t>> read(uint32_t RVA, uint32_t Size) const;

  Expected<uint32_t> readU32(uint32_t RVA) const;
  Expected<int32_t> readI32(uint32_t RVA) const;

private:
  struct Section {
    uint32_t RVA;
    ArrayRef<uint8_t> Data;
  };
  SmallVector<Section, 4> Sections;
};

/// Parsed MSVC C++ FuncInfo for a single function.  Field names follow the
/// layout documented in WinException.cpp.
struct WinEHFuncInfo {
  struct UnwindMapEntry {
    int32_t ToState = 0;
    /// RVA of the cleanup funclet, or 0.
    uint32_t Action = 0;
  };

  struct IPToStateEntry {
    /// RVA of the instruction boundary at which \c State becomes active.
    uint32_t IP = 0;
    int32_t State = 0;
  };

  struct HandlerType {
    uint32_t Adjectives = 0;
    uint32_t TypeDescriptor = 0;
    int32_t CatchObjOffset = 0;
    /// RVA of the catch funclet.
    uint32_t Handler = 0;
    int32_t ParentFrameOffset = 0;
  };

  struct TryBlock {
    int32_t TryLow = 0;
    int32_t TryHigh = 0;
    int32_t CatchHigh = 0;
    SmallVector<HandlerType, 1> Handlers;
  };

  uint32_t MagicNumber = 0;
  int32_t MaxState = 0;
  int32_t UnwindHelp = 0;
  uint32_t ESTypeListRVA = 0;
  int32_t EHFlags = 0;
  uint32_t IPToStateMapRVA = 0;
  uint32_t NumIPToStateEntries = 0;
  bool HasParsedIPToStateMap = false;

  SmallVector<UnwindMapEntry, 4> UnwindMap;
  SmallVector<TryBlock, 2> TryBlocks;
  SmallVector<IPToStateEntry, 8> IPToStateMap;
};

/// Decode the FuncInfo at \p FuncInfoRVA using \p Reader.
/// Defer the IPToState table when \p ParseIPToStateMap is false.
Expected<WinEHFuncInfo> parseWinEHFuncInfo(const WinEHImageReader &Reader,
                                           uint32_t FuncInfoRVA,
                                           bool ParseIPToStateMap = true);

/// Decode the deferred IPToState table in \p FI.
Error parseWinEHIPToStateMap(const WinEHImageReader &Reader, WinEHFuncInfo &FI);

/// Return true when \p IP lies inside the primary function body
/// [\p FuncBeginRVA, \p FuncEndRVA).  IPToState entries outside this range
/// belong to out-of-line catch/cleanup funclets placed elsewhere in the image.
inline bool isInBodyIP(uint32_t IP, uint32_t FuncBeginRVA,
                       uint32_t FuncEndRVA) {
  return IP >= FuncBeginRVA && IP < FuncEndRVA;
}

/// Base ("no active try/cleanup scope") EH state, in effect before the first
/// IPToState entry.
constexpr int WinEHNullState = -1;

/// Return the EH state in effect at IP RVA \p IP for an IPToState table: the
/// state of the last entry with IP <= \p IP, or WinEHNullState if none precedes
/// it.  \p Table must be sorted by ascending IP, as emitted by MSVC (see
/// WinException.cpp).
int stateAtIP(ArrayRef<WinEHFuncInfo::IPToStateEntry> Table, uint32_t IP);

/// Return the EH state in effect at the original IP RVA \p IP: the state of the
/// last IPToState entry with IP <= \p IP, or WinEHNullState if none precedes
/// it.  \c FI.IPToStateMap must be sorted by ascending IP, as emitted by MSVC
/// (see WinException.cpp).
int lookupEHState(const WinEHFuncInfo &FI, uint32_t IP);

/// One body instruction in the output (reordered) function, pairing its output
/// IP RVA with the EH state that was in effect at the corresponding input
/// instruction.
struct OutputInsnState {
  uint32_t OutputIP = 0;
  int State = WinEHNullState;
};

/// Recompute a function's IPToState table after in-place block reordering.
///
/// The MSVC representation stores one entry per point where the state changes
/// along the code layout, so a state region fragmented by reordering must be
/// described by several entries.  This mirrors WinException.cpp exactly: walking
/// \p BodyInsns (every body instruction in output-layout order, sorted by
/// ascending OutputIP), an entry is produced whenever the state differs from
/// the preceding instruction.  \p FuncletEntries are the original entries that
/// live in out-of-line funclets (which are not moved) and are carried over
/// unchanged.
///
/// The result is the regenerated body entries concatenated with the funclet
/// entries, sorted by ascending IP.  Its length may differ from the original
/// table, so callers must relocate the table and update the FuncInfo's
/// IPToStateMap RVA and entry count rather than patching in place.
SmallVector<WinEHFuncInfo::IPToStateEntry, 16>
regenerateIPToState(ArrayRef<OutputInsnState> BodyInsns,
                    ArrayRef<WinEHFuncInfo::IPToStateEntry> FuncletEntries);

} // namespace bolt
} // namespace llvm

#endif
