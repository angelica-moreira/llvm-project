//===- bolt/unittest/Core/WinEHFuncInfoReader.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Rewrite/WinEHFuncInfoReader.h"
#include "llvm/Support/Endian.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <vector>

using namespace llvm;
using namespace llvm::bolt;

namespace {

/// Small helper that lays out FuncInfo and its sub-tables in a single buffer
/// mapped at a fixed base RVA, matching the on-disk x86_64 layout.
class ImageBuilder {
public:
  explicit ImageBuilder(uint32_t BaseRVA) : BaseRVA(BaseRVA) {}

  uint32_t base() const { return BaseRVA; }

  /// RVA of the next byte that will be appended.
  uint32_t tell() const { return BaseRVA + Buffer.size(); }

  void u32(uint32_t V) {
    uint8_t Tmp[4];
    support::endian::write32le(Tmp, V);
    Buffer.insert(Buffer.end(), Tmp, Tmp + 4);
  }

  void i32(int32_t V) { u32(static_cast<uint32_t>(V)); }

  WinEHImageReader reader() const {
    WinEHImageReader R;
    R.addSection(BaseRVA, Buffer);
    return R;
  }

private:
  uint32_t BaseRVA;
  std::vector<uint8_t> Buffer;
};

// Reserve room for the fixed-size FuncInfo header so sub-tables can be laid out
// after it, then patch the header once the sub-table RVAs are known.
static constexpr uint32_t FuncInfoSize = 40;

TEST(WinEHFuncInfoReaderTest, RejectsBadMagic) {
  ImageBuilder Img(0x1000);
  Img.u32(0x12345678); // magic
  for (int I = 0; I < 9; ++I)
    Img.u32(0);
  WinEHImageReader R = Img.reader();
  EXPECT_THAT_EXPECTED(parseWinEHFuncInfo(R, 0x1000), Failed());
}

TEST(WinEHFuncInfoReaderTest, RejectsOutOfBounds) {
  ImageBuilder Img(0x1000);
  WinEHImageReader R = Img.reader();
  // Nothing mapped at this RVA.
  EXPECT_THAT_EXPECTED(parseWinEHFuncInfo(R, 0x9000), Failed());
}

TEST(WinEHFuncInfoReaderTest, RejectsImplausibleCounts) {
  ImageBuilder Img(0x1000);
  Img.u32(0x19930522);  // MagicNumber
  Img.i32(4);           // MaxState
  Img.u32(0);           // UnwindMap RVA
  Img.u32(0x7fffffff);  // NumTryBlocks (implausible)
  Img.u32(0);           // TryBlockMap RVA
  Img.u32(0);           // NumIPEntries
  Img.u32(0);           // IPToStateMap RVA
  Img.i32(0);           // UnwindHelp
  Img.u32(0);           // ESTypeList RVA
  Img.i32(0);           // EHFlags
  WinEHImageReader R = Img.reader();
  EXPECT_THAT_EXPECTED(parseWinEHFuncInfo(R, 0x1000), Failed());
}

// Build a complete FuncInfo with an unwind map, one try block with one catch
// handler, and an IPToState map, then verify every field round-trips.  Values
// mirror the shape of real MSVC output (magic 0x19930522, x86_64 records).
TEST(WinEHFuncInfoReaderTest, ParsesFullTable) {
  const uint32_t Base = 0x2000;
  // Sub-tables are laid out contiguously after the fixed-size header, so their
  // RVAs are known ahead of time.
  const uint32_t UnwindMapRVA = Base + FuncInfoSize;
  const uint32_t HandlerArrayRVA = UnwindMapRVA + 2 * 8;
  const uint32_t TryBlockMapRVA = HandlerArrayRVA + 1 * 20;
  const uint32_t IPToStateRVA = TryBlockMapRVA + 1 * 20;

  ImageBuilder Img(Base);
  Img.u32(0x19930522);       // MagicNumber
  Img.i32(2);                // MaxState (UnwindMap has 2 entries)
  Img.u32(UnwindMapRVA);     // UnwindMap RVA
  Img.u32(1);                // NumTryBlocks
  Img.u32(TryBlockMapRVA);   // TryBlockMap RVA
  Img.u32(3);                // NumIPEntries
  Img.u32(IPToStateRVA);     // IPToStateMap RVA
  Img.i32(400);              // UnwindHelp
  Img.u32(0);                // ESTypeList RVA
  Img.i32(0);                // EHFlags

  // UnwindMap: 2 entries {ToState, Action-RVA}.
  Img.i32(-1);
  Img.u32(0xAAAA);
  Img.i32(0);
  Img.u32(0xBBBB);

  // Handler array: 1 catch handler.
  Img.u32(0x40);   // Adjectives
  Img.u32(0xCCCC); // TypeDescriptor RVA
  Img.i32(0x20);   // CatchObjOffset
  Img.u32(0xDDDD); // Handler RVA
  Img.i32(0x10);   // ParentFrameOffset

  // TryBlockMap: 1 entry.
  Img.i32(0);      // TryLow
  Img.i32(1);      // TryHigh
  Img.i32(2);      // CatchHigh
  Img.u32(1);      // NumCatches
  Img.u32(HandlerArrayRVA);

  // IPToStateMap: 3 entries {IP-RVA, State}.
  Img.u32(0x156e0);
  Img.i32(-1);
  Img.u32(0x156f4);
  Img.i32(0);
  Img.u32(0x15700);
  Img.i32(1);

  WinEHImageReader R = Img.reader();
  auto FIOrErr = parseWinEHFuncInfo(R, Base);
  ASSERT_THAT_EXPECTED(FIOrErr, Succeeded());
  const WinEHFuncInfo &FI = *FIOrErr;

  EXPECT_EQ(FI.MagicNumber, 0x19930522u);
  EXPECT_EQ(FI.MaxState, 2);
  EXPECT_EQ(FI.UnwindHelp, 400);
  EXPECT_EQ(FI.EHFlags, 0);

  ASSERT_EQ(FI.UnwindMap.size(), 2u);
  EXPECT_EQ(FI.UnwindMap[0].ToState, -1);
  EXPECT_EQ(FI.UnwindMap[0].Action, 0xAAAAu);
  EXPECT_EQ(FI.UnwindMap[1].Action, 0xBBBBu);

  ASSERT_EQ(FI.TryBlocks.size(), 1u);
  const WinEHFuncInfo::TryBlock &TB = FI.TryBlocks[0];
  EXPECT_EQ(TB.TryLow, 0);
  EXPECT_EQ(TB.TryHigh, 1);
  EXPECT_EQ(TB.CatchHigh, 2);
  ASSERT_EQ(TB.Handlers.size(), 1u);
  EXPECT_EQ(TB.Handlers[0].Adjectives, 0x40u);
  EXPECT_EQ(TB.Handlers[0].TypeDescriptor, 0xCCCCu);
  EXPECT_EQ(TB.Handlers[0].CatchObjOffset, 0x20);
  EXPECT_EQ(TB.Handlers[0].Handler, 0xDDDDu);
  EXPECT_EQ(TB.Handlers[0].ParentFrameOffset, 0x10);

  ASSERT_EQ(FI.IPToStateMap.size(), 3u);
  EXPECT_EQ(FI.IPToStateMap[0].IP, 0x156e0u);
  EXPECT_EQ(FI.IPToStateMap[0].State, -1);
  EXPECT_EQ(FI.IPToStateMap[2].IP, 0x15700u);
  EXPECT_EQ(FI.IPToStateMap[2].State, 1);
}

// A function with no try blocks (MaxState > 0, NumTryBlocks == 0) must still
// parse its unwind and IPToState maps -- this matches RetrieveRecordPrivate.
TEST(WinEHFuncInfoReaderTest, ParsesNoTryBlocks) {
  const uint32_t Base = 0x3000;
  ImageBuilder Img(Base);
  Img.u32(0x19930522);        // MagicNumber
  Img.i32(1);                 // MaxState
  Img.u32(Base + FuncInfoSize); // UnwindMap RVA (immediately after header)
  Img.u32(0);                 // NumTryBlocks
  Img.u32(0);                 // TryBlockMap RVA
  Img.u32(1);                 // NumIPEntries
  Img.u32(Base + FuncInfoSize + 8); // IPToStateMap RVA
  Img.i32(400);               // UnwindHelp
  Img.u32(0);                 // ESTypeList RVA
  Img.i32(0);                 // EHFlags
  // UnwindMap (1 entry).
  Img.i32(-1);
  Img.u32(0x27fa2);
  // IPToStateMap (1 entry).
  Img.u32(0x156e0);
  Img.i32(-1);

  WinEHImageReader R = Img.reader();
  auto FIOrErr = parseWinEHFuncInfo(R, Base);
  ASSERT_THAT_EXPECTED(FIOrErr, Succeeded());
  const WinEHFuncInfo &FI = *FIOrErr;
  EXPECT_TRUE(FI.TryBlocks.empty());
  ASSERT_EQ(FI.UnwindMap.size(), 1u);
  EXPECT_EQ(FI.UnwindMap[0].Action, 0x27fa2u);
  ASSERT_EQ(FI.IPToStateMap.size(), 1u);
  EXPECT_EQ(FI.IPToStateMap[0].IP, 0x156e0u);
}

TEST(WinEHFuncInfoReaderTest, DefersIPToStateMap) {
  const uint32_t Base = 0x4000;
  ImageBuilder Img(Base);
  Img.u32(0x19930522);
  Img.i32(0);
  Img.u32(0);
  Img.u32(0);
  Img.u32(0);
  Img.u32(1);
  Img.u32(Base + FuncInfoSize);
  Img.i32(0);
  Img.u32(0);
  Img.i32(0);
  Img.u32(0x4010);
  Img.i32(2);

  WinEHImageReader R = Img.reader();
  auto FIOrErr = parseWinEHFuncInfo(R, Base, false);
  ASSERT_THAT_EXPECTED(FIOrErr, Succeeded());
  WinEHFuncInfo &FI = *FIOrErr;
  EXPECT_EQ(FI.NumIPToStateEntries, 1u);
  EXPECT_EQ(FI.IPToStateMapRVA, Base + FuncInfoSize);
  EXPECT_FALSE(FI.HasParsedIPToStateMap);
  EXPECT_TRUE(FI.IPToStateMap.empty());

  EXPECT_THAT_ERROR(parseWinEHIPToStateMap(R, FI), Succeeded());
  EXPECT_TRUE(FI.HasParsedIPToStateMap);
  ASSERT_EQ(FI.IPToStateMap.size(), 1u);
  EXPECT_EQ(FI.IPToStateMap[0].IP, 0x4010u);
  EXPECT_EQ(FI.IPToStateMap[0].State, 2);

  EXPECT_THAT_ERROR(parseWinEHIPToStateMap(R, FI), Succeeded());
  EXPECT_EQ(FI.IPToStateMap.size(), 1u);
}

TEST(WinEHFuncInfoReaderTest, DeferredIPToStateMapRejectsOutOfBounds) {
  const uint32_t Base = 0x5000;
  ImageBuilder Img(Base);
  Img.u32(0x19930522);
  Img.i32(0);
  Img.u32(0);
  Img.u32(0);
  Img.u32(0);
  Img.u32(2);
  Img.u32(Base + FuncInfoSize);
  Img.i32(0);
  Img.u32(0);
  Img.i32(0);
  Img.u32(0x5010);
  Img.i32(1);
  Img.u32(0x5020);

  WinEHImageReader R = Img.reader();
  auto FIOrErr = parseWinEHFuncInfo(R, Base, false);
  ASSERT_THAT_EXPECTED(FIOrErr, Succeeded());
  WinEHFuncInfo &FI = *FIOrErr;

  EXPECT_THAT_ERROR(parseWinEHIPToStateMap(R, FI), Failed());
  EXPECT_FALSE(FI.HasParsedIPToStateMap);
  EXPECT_TRUE(FI.IPToStateMap.empty());
}

// lookupEHState returns the state of the last entry with IP <= query.
TEST(WinEHFuncInfoReaderTest, LookupEHState) {
  WinEHFuncInfo FI;
  FI.IPToStateMap.push_back({0x1000, -1});
  FI.IPToStateMap.push_back({0x1010, 0});
  FI.IPToStateMap.push_back({0x1020, 1});
  FI.IPToStateMap.push_back({0x1030, -1});

  EXPECT_EQ(lookupEHState(FI, 0x0fff), WinEHNullState); // before first entry
  EXPECT_EQ(lookupEHState(FI, 0x1000), -1);
  EXPECT_EQ(lookupEHState(FI, 0x1018), 0);  // within [0x1010, 0x1020)
  EXPECT_EQ(lookupEHState(FI, 0x1020), 1);
  EXPECT_EQ(lookupEHState(FI, 0x2000), -1); // after last entry
}

// Reordering that keeps same-state code contiguous reproduces the original
// entry count.
TEST(WinEHFuncInfoReaderTest, RegenerateContiguous) {
  // Output order: state -1 for [0x1000,0x1010), then state 0 onward.
  OutputInsnState Insns[] = {
      {0x1000, -1}, {0x1008, -1}, {0x1010, 0}, {0x1018, 0}};
  auto Table = regenerateIPToState(Insns, {});
  ASSERT_EQ(Table.size(), 2u);
  EXPECT_EQ(Table[0].IP, 0x1000u);
  EXPECT_EQ(Table[0].State, -1);
  EXPECT_EQ(Table[1].IP, 0x1010u);
  EXPECT_EQ(Table[1].State, 0);
}

// The key correctness property: when reordering fragments a state region, the
// regenerated table gains entries to describe each fragment.
TEST(WinEHFuncInfoReaderTest, RegenerateFragmentedGrows) {
  // A state-0 region is split by a state -1 block in the new layout:
  // 0 . -1 . 0 . 1  -> four entries even though only three states appear.
  OutputInsnState Insns[] = {
      {0x1000, 0}, {0x1008, -1}, {0x1010, 0}, {0x1018, 1}};
  auto Table = regenerateIPToState(Insns, {});
  ASSERT_EQ(Table.size(), 4u);
  EXPECT_EQ(Table[0].State, 0);
  EXPECT_EQ(Table[1].State, -1);
  EXPECT_EQ(Table[2].State, 0);
  EXPECT_EQ(Table[3].State, 1);
  // Consecutive same-state instructions collapse into one entry.
  EXPECT_EQ(Table[2].IP, 0x1010u);
}

// Funclet-region entries are preserved verbatim and the whole table stays
// sorted by IP.
TEST(WinEHFuncInfoReaderTest, RegeneratePreservesFunclets) {
  OutputInsnState Insns[] = {{0x1000, -1}, {0x1008, 0}};
  WinEHFuncInfo::IPToStateEntry Funclets[] = {{0x3d000, 2}, {0x3d010, -1}};
  auto Table = regenerateIPToState(Insns, Funclets);
  ASSERT_EQ(Table.size(), 4u);
  EXPECT_EQ(Table[0].IP, 0x1000u);
  EXPECT_EQ(Table[1].IP, 0x1008u);
  EXPECT_EQ(Table[2].IP, 0x3d000u);
  EXPECT_EQ(Table[2].State, 2);
  EXPECT_EQ(Table[3].IP, 0x3d010u);
}

} // namespace
