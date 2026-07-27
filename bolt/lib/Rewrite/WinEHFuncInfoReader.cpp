//===- bolt/Rewrite/WinEHFuncInfoReader.cpp - MSVC C++ EH parser ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Rewrite/WinEHFuncInfoReader.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Errc.h"
#include <utility>

namespace llvm {
namespace bolt {

// Recognized __CxxFrameHandler3/4 FuncInfo magic numbers.
static constexpr uint32_t CxxFrameMagicV1 = 0x19930520;
static constexpr uint32_t CxxFrameMagicV2 = 0x19930521;
static constexpr uint32_t CxxFrameMagicV3 = 0x19930522;

// Upper bounds used to reject implausible tables from mis-identified handler
// data before allocating or iterating.
static constexpr uint32_t MaxStates = 1u << 16;
static constexpr uint32_t MaxTryBlocks = 1u << 16;
static constexpr uint32_t MaxIPEntries = 1u << 20;
static constexpr uint32_t MaxHandlers = 1u << 12;

Expected<ArrayRef<uint8_t>> WinEHImageReader::read(uint32_t RVA,
                                                   uint32_t Size) const {
  for (const Section &S : Sections) {
    if (RVA < S.RVA)
      continue;
    uint32_t Offset = RVA - S.RVA;
    if (Offset <= S.Data.size() && Size <= S.Data.size() - Offset)
      return S.Data.slice(Offset, Size);
  }
  return createStringError(std::errc::result_out_of_range,
                           "RVA 0x%" PRIx32 " (%" PRIu32 " bytes) not mapped",
                           RVA, Size);
}

Expected<std::pair<uint32_t, ArrayRef<uint8_t>>>
WinEHImageReader::sectionContaining(uint32_t RVA) const {
  for (const Section &S : Sections) {
    if (RVA >= S.RVA && static_cast<uint64_t>(RVA) - S.RVA < S.Data.size())
      return std::make_pair(S.RVA, S.Data);
  }
  return createStringError(errc::invalid_argument,
                           "RVA 0x%x is outside the image", RVA);
}

Expected<uint32_t> WinEHImageReader::readU32(uint32_t RVA) const {
  Expected<ArrayRef<uint8_t>> Bytes = read(RVA, sizeof(uint32_t));
  if (!Bytes)
    return Bytes.takeError();
  return support::endian::read32le(Bytes->data());
}

Expected<int32_t> WinEHImageReader::readI32(uint32_t RVA) const {
  Expected<uint32_t> Value = readU32(RVA);
  if (!Value)
    return Value.takeError();
  return static_cast<int32_t>(*Value);
}

// Sizes of the on-disk x86_64 sub-table records (see WinException.cpp).
static constexpr uint32_t UnwindMapEntrySize = 8;    // ToState, Action
static constexpr uint32_t TryBlockMapEntrySize = 20; // 5 * i32
static constexpr uint32_t HandlerTypeSize = 20;      // 5 * i32
static constexpr uint32_t IPToStateEntrySize = 8;    // Ip, State

static Error parseUnwindMap(const WinEHImageReader &Reader, uint32_t RVA,
                            uint32_t Count, WinEHFuncInfo &FI) {
  for (uint32_t I = 0; I < Count; ++I) {
    uint32_t Base = RVA + I * UnwindMapEntrySize;
    Expected<int32_t> ToState = Reader.readI32(Base);
    if (!ToState)
      return ToState.takeError();
    Expected<uint32_t> Action = Reader.readU32(Base + 4);
    if (!Action)
      return Action.takeError();
    FI.UnwindMap.push_back({*ToState, *Action});
  }
  return Error::success();
}

static Error parseHandlers(const WinEHImageReader &Reader, uint32_t RVA,
                           uint32_t Count, WinEHFuncInfo::TryBlock &TB) {
  if (Count > MaxHandlers)
    return createStringError(std::errc::invalid_argument,
                             "implausible catch handler count %" PRIu32, Count);
  for (uint32_t I = 0; I < Count; ++I) {
    uint32_t Base = RVA + I * HandlerTypeSize;
    WinEHFuncInfo::HandlerType HT;
    Expected<uint32_t> Adjectives = Reader.readU32(Base);
    if (!Adjectives)
      return Adjectives.takeError();
    Expected<uint32_t> Type = Reader.readU32(Base + 4);
    if (!Type)
      return Type.takeError();
    Expected<int32_t> CatchObj = Reader.readI32(Base + 8);
    if (!CatchObj)
      return CatchObj.takeError();
    Expected<uint32_t> Handler = Reader.readU32(Base + 12);
    if (!Handler)
      return Handler.takeError();
    Expected<int32_t> ParentOff = Reader.readI32(Base + 16);
    if (!ParentOff)
      return ParentOff.takeError();
    HT.Adjectives = *Adjectives;
    HT.TypeDescriptor = *Type;
    HT.CatchObjOffset = *CatchObj;
    HT.Handler = *Handler;
    HT.ParentFrameOffset = *ParentOff;
    TB.Handlers.push_back(HT);
  }
  return Error::success();
}

static Error parseTryBlockMap(const WinEHImageReader &Reader, uint32_t RVA,
                              uint32_t Count, WinEHFuncInfo &FI) {
  for (uint32_t I = 0; I < Count; ++I) {
    uint32_t Base = RVA + I * TryBlockMapEntrySize;
    WinEHFuncInfo::TryBlock TB;
    Expected<int32_t> TryLow = Reader.readI32(Base);
    if (!TryLow)
      return TryLow.takeError();
    Expected<int32_t> TryHigh = Reader.readI32(Base + 4);
    if (!TryHigh)
      return TryHigh.takeError();
    Expected<int32_t> CatchHigh = Reader.readI32(Base + 8);
    if (!CatchHigh)
      return CatchHigh.takeError();
    Expected<uint32_t> NumCatches = Reader.readU32(Base + 12);
    if (!NumCatches)
      return NumCatches.takeError();
    Expected<uint32_t> HandlerArray = Reader.readU32(Base + 16);
    if (!HandlerArray)
      return HandlerArray.takeError();
    TB.TryLow = *TryLow;
    TB.TryHigh = *TryHigh;
    TB.CatchHigh = *CatchHigh;
    if (Error E = parseHandlers(Reader, *HandlerArray, *NumCatches, TB))
      return E;
    FI.TryBlocks.push_back(std::move(TB));
  }
  return Error::success();
}

static Expected<SmallVector<WinEHFuncInfo::IPToStateEntry, 8>>
readIPToStateMap(const WinEHImageReader &Reader, uint32_t RVA,
                 uint32_t Count) {
  SmallVector<WinEHFuncInfo::IPToStateEntry, 8> Entries;
  for (uint32_t I = 0; I < Count; ++I) {
    uint32_t Base = RVA + I * IPToStateEntrySize;
    Expected<uint32_t> IP = Reader.readU32(Base);
    if (!IP)
      return IP.takeError();
    Expected<int32_t> State = Reader.readI32(Base + 4);
    if (!State)
      return State.takeError();
    Entries.push_back({*IP, *State});
  }
  return Entries;
}

Error parseWinEHIPToStateMap(const WinEHImageReader &Reader,
                             WinEHFuncInfo &FI) {
  if (FI.HasParsedIPToStateMap)
    return Error::success();

  if (FI.NumIPToStateEntries > 0 && FI.IPToStateMapRVA) {
    Expected<SmallVector<WinEHFuncInfo::IPToStateEntry, 8>> Entries =
        readIPToStateMap(Reader, FI.IPToStateMapRVA, FI.NumIPToStateEntries);
    if (!Entries)
      return Entries.takeError();
    FI.IPToStateMap = std::move(*Entries);
  }

  FI.HasParsedIPToStateMap = true;
  return Error::success();
}

Expected<WinEHFuncInfo> parseWinEHFuncInfo(const WinEHImageReader &Reader,
                                           uint32_t FuncInfoRVA,
                                           bool ParseIPToStateMap) {
  WinEHFuncInfo FI;

  Expected<uint32_t> Magic = Reader.readU32(FuncInfoRVA);
  if (!Magic)
    return Magic.takeError();
  if (*Magic != CxxFrameMagicV1 && *Magic != CxxFrameMagicV2 &&
      *Magic != CxxFrameMagicV3)
    return createStringError(std::errc::invalid_argument,
                             "unrecognized FuncInfo magic 0x%" PRIx32, *Magic);
  FI.MagicNumber = *Magic;

  Expected<int32_t> MaxState = Reader.readI32(FuncInfoRVA + 4);
  if (!MaxState)
    return MaxState.takeError();
  Expected<uint32_t> UnwindMapRVA = Reader.readU32(FuncInfoRVA + 8);
  if (!UnwindMapRVA)
    return UnwindMapRVA.takeError();
  Expected<uint32_t> NumTryBlocks = Reader.readU32(FuncInfoRVA + 12);
  if (!NumTryBlocks)
    return NumTryBlocks.takeError();
  Expected<uint32_t> TryBlockMapRVA = Reader.readU32(FuncInfoRVA + 16);
  if (!TryBlockMapRVA)
    return TryBlockMapRVA.takeError();
  Expected<uint32_t> NumIPEntries = Reader.readU32(FuncInfoRVA + 20);
  if (!NumIPEntries)
    return NumIPEntries.takeError();
  Expected<uint32_t> IPToStateRVA = Reader.readU32(FuncInfoRVA + 24);
  if (!IPToStateRVA)
    return IPToStateRVA.takeError();
  Expected<int32_t> UnwindHelp = Reader.readI32(FuncInfoRVA + 28);
  if (!UnwindHelp)
    return UnwindHelp.takeError();
  Expected<uint32_t> ESTypeList = Reader.readU32(FuncInfoRVA + 32);
  if (!ESTypeList)
    return ESTypeList.takeError();
  Expected<int32_t> EHFlags = Reader.readI32(FuncInfoRVA + 36);
  if (!EHFlags)
    return EHFlags.takeError();

  if (*MaxState < 0 || static_cast<uint32_t>(*MaxState) > MaxStates)
    return createStringError(std::errc::invalid_argument,
                             "implausible MaxState %d", *MaxState);
  if (*NumTryBlocks > MaxTryBlocks)
    return createStringError(std::errc::invalid_argument,
                             "implausible NumTryBlocks %" PRIu32,
                             *NumTryBlocks);
  if (*NumIPEntries > MaxIPEntries)
    return createStringError(std::errc::invalid_argument,
                             "implausible NumIPEntries %" PRIu32,
                             *NumIPEntries);

  FI.MaxState = *MaxState;
  FI.UnwindHelp = *UnwindHelp;
  FI.ESTypeListRVA = *ESTypeList;
  FI.EHFlags = *EHFlags;
  FI.IPToStateMapRVA = *IPToStateRVA;
  FI.NumIPToStateEntries = *NumIPEntries;

  if (*MaxState > 0 && *UnwindMapRVA)
    if (Error E = parseUnwindMap(Reader, *UnwindMapRVA,
                                 static_cast<uint32_t>(*MaxState), FI))
      return std::move(E);
  if (*NumTryBlocks > 0 && *TryBlockMapRVA)
    if (Error E = parseTryBlockMap(Reader, *TryBlockMapRVA, *NumTryBlocks, FI))
      return std::move(E);
  if (ParseIPToStateMap)
    if (Error E = parseWinEHIPToStateMap(Reader, FI))
      return std::move(E);

  return FI;
}

int stateAtIP(ArrayRef<WinEHFuncInfo::IPToStateEntry> Table, uint32_t IP) {
  // Table is sorted by ascending IP; find the last entry whose IP is <= the
  // query.
  int State = WinEHNullState;
  for (const WinEHFuncInfo::IPToStateEntry &Entry : Table) {
    if (Entry.IP > IP)
      break;
    State = Entry.State;
  }
  return State;
}

int lookupEHState(const WinEHFuncInfo &FI, uint32_t IP) {
  return stateAtIP(FI.IPToStateMap, IP);
}

SmallVector<WinEHFuncInfo::IPToStateEntry, 16>
regenerateIPToState(ArrayRef<OutputInsnState> BodyInsns,
                    ArrayRef<WinEHFuncInfo::IPToStateEntry> FuncletEntries) {
  SmallVector<WinEHFuncInfo::IPToStateEntry, 16> Result;

  // Re-derive the body entries: emit one whenever the state differs from the
  // preceding instruction in output-layout order.
  bool First = true;
  int PrevState = WinEHNullState;
  for (const OutputInsnState &Insn : BodyInsns) {
    if (First || Insn.State != PrevState) {
      Result.push_back({Insn.OutputIP, Insn.State});
      PrevState = Insn.State;
      First = false;
    }
  }

  // Carry over funclet-region entries unchanged.
  Result.append(FuncletEntries.begin(), FuncletEntries.end());

  llvm::sort(Result, [](const WinEHFuncInfo::IPToStateEntry &A,
                        const WinEHFuncInfo::IPToStateEntry &B) {
    return A.IP < B.IP;
  });

  return Result;
}

} // namespace bolt
} // namespace llvm
