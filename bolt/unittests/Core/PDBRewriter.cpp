//===- bolt/unittest/Core/PDBRewriter.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Rewrite/PDBRewriter.h"
#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/DebugLinesSubsection.h"
#include "llvm/Support/Endian.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

using namespace llvm;
using namespace llvm::bolt;
using namespace llvm::codeview;

namespace {

struct LineSpec {
  uint32_t Offset;
  uint32_t Flags;
  uint16_t StartColumn = 0;
  uint16_t EndColumn = 0;
};

class LineSubsectionBuilder {
public:
  LineSubsectionBuilder(uint32_t CodeSize, bool HasColumns)
      : HasColumns(HasColumns) {
    Bytes.resize(sizeof(LineFragmentHeader));
    support::endian::write16le(Bytes.data() +
                                   offsetof(LineFragmentHeader, Flags),
                               HasColumns ? uint16_t(LF_HaveColumns) : 0);
    support::endian::write32le(
        Bytes.data() + offsetof(LineFragmentHeader, CodeSize), CodeSize);
  }

  void addBlock(ArrayRef<LineSpec> Lines) {
    appendU32(1);
    appendU32(Lines.size());
    appendU32(sizeof(LineBlockFragmentHeader) +
              Lines.size() * sizeof(LineNumberEntry) +
              (HasColumns ? Lines.size() * sizeof(ColumnNumberEntry) : 0));
    for (const LineSpec &Line : Lines) {
      appendU32(Line.Offset);
      appendU32(Line.Flags);
    }
    if (HasColumns) {
      for (const LineSpec &Line : Lines) {
        appendU16(Line.StartColumn);
        appendU16(Line.EndColumn);
      }
    }
  }

  std::vector<uint8_t> take() { return std::move(Bytes); }

private:
  void appendU16(uint16_t Value) {
    const size_t Pos = Bytes.size();
    Bytes.resize(Pos + 2);
    support::endian::write16le(Bytes.data() + Pos, Value);
  }

  void appendU32(uint32_t Value) {
    const size_t Pos = Bytes.size();
    Bytes.resize(Pos + 4);
    support::endian::write32le(Bytes.data() + Pos, Value);
  }

  bool HasColumns;
  std::vector<uint8_t> Bytes;
};

uint32_t lineValue(ArrayRef<uint8_t> Bytes, uint32_t Index,
                   uint32_t FieldOffset) {
  const uint64_t Pos = sizeof(LineFragmentHeader) +
                       sizeof(LineBlockFragmentHeader) +
                       uint64_t(Index) * sizeof(LineNumberEntry) + FieldOffset;
  return support::endian::read32le(Bytes.data() + Pos);
}

uint16_t columnValue(ArrayRef<uint8_t> Bytes, uint32_t NumLines, uint32_t Index,
                     uint32_t FieldOffset) {
  const uint64_t Pos =
      sizeof(LineFragmentHeader) + sizeof(LineBlockFragmentHeader) +
      uint64_t(NumLines) * sizeof(LineNumberEntry) +
      uint64_t(Index) * sizeof(ColumnNumberEntry) + FieldOffset;
  return support::endian::read16le(Bytes.data() + Pos);
}

TEST(PDBRewriterTest, SortsRemappedLines) {
  LineSubsectionBuilder Builder(32, false);
  Builder.addBlock({{0, 1}, {4, 2}, {8, 3}, {12, 4}});
  std::vector<uint8_t> Input = Builder.take();
  std::vector<uint8_t> Output;
  uint32_t NumRemapped = 0;

  ASSERT_TRUE(pdb_detail::rewriteLineSubsection(Input, {{0, 16}, {8, 0}},
                                                Output, NumRemapped));
  EXPECT_EQ(NumRemapped, 4u);
  EXPECT_EQ(Output.size(), Input.size());
  EXPECT_TRUE(std::equal(Input.begin(),
                         Input.begin() + sizeof(LineFragmentHeader) +
                             sizeof(LineBlockFragmentHeader),
                         Output.begin()));
  EXPECT_EQ(lineValue(Output, 0, 0), 0u);
  EXPECT_EQ(lineValue(Output, 1, 0), 4u);
  EXPECT_EQ(lineValue(Output, 2, 0), 16u);
  EXPECT_EQ(lineValue(Output, 3, 0), 20u);
  EXPECT_EQ(lineValue(Output, 0, 4), 3u);
  EXPECT_EQ(lineValue(Output, 1, 4), 4u);
  EXPECT_EQ(lineValue(Output, 2, 4), 1u);
  EXPECT_EQ(lineValue(Output, 3, 4), 2u);
}

TEST(PDBRewriterTest, PermutesColumnsWithLines) {
  LineSubsectionBuilder Builder(32, true);
  Builder.addBlock({{0, 1, 10, 11}, {8, 2, 20, 21}});
  std::vector<uint8_t> Input = Builder.take();
  std::vector<uint8_t> Output;
  uint32_t NumRemapped = 0;

  ASSERT_TRUE(pdb_detail::rewriteLineSubsection(Input, {{0, 16}, {8, 0}},
                                                Output, NumRemapped));
  EXPECT_EQ(lineValue(Output, 0, 4), 2u);
  EXPECT_EQ(lineValue(Output, 1, 4), 1u);
  EXPECT_EQ(columnValue(Output, 2, 0, 0), 20u);
  EXPECT_EQ(columnValue(Output, 2, 0, 2), 21u);
  EXPECT_EQ(columnValue(Output, 2, 1, 0), 10u);
  EXPECT_EQ(columnValue(Output, 2, 1, 2), 11u);
}

TEST(PDBRewriterTest, PreservesEqualOffsetOrder) {
  LineSubsectionBuilder Builder(16, false);
  Builder.addBlock({{0, 1}, {4, 2}});
  std::vector<uint8_t> Output;
  uint32_t NumRemapped = 0;

  ASSERT_TRUE(pdb_detail::rewriteLineSubsection(
      Builder.take(), {{0, 4}, {4, 4}}, Output, NumRemapped));
  EXPECT_EQ(lineValue(Output, 0, 0), 4u);
  EXPECT_EQ(lineValue(Output, 1, 0), 4u);
  EXPECT_EQ(lineValue(Output, 0, 4), 1u);
  EXPECT_EQ(lineValue(Output, 1, 4), 2u);
}

TEST(PDBRewriterTest, PreservesTrailingBlockBytes) {
  LineSubsectionBuilder Builder(32, false);
  Builder.addBlock({{0, 1}, {8, 2}});
  std::vector<uint8_t> Input = Builder.take();
  constexpr uint32_t PaddingSize = 4;
  const size_t BlockSizeOffset =
      sizeof(LineFragmentHeader) + offsetof(LineBlockFragmentHeader, BlockSize);
  const uint32_t BlockSize =
      support::endian::read32le(Input.data() + BlockSizeOffset);
  support::endian::write32le(Input.data() + BlockSizeOffset,
                             BlockSize + PaddingSize);
  Input.insert(Input.end(), PaddingSize, 0xA5);

  std::vector<uint8_t> Output;
  uint32_t NumRemapped = 0;
  ASSERT_TRUE(pdb_detail::rewriteLineSubsection(Input, {{0, 16}, {8, 0}},
                                                Output, NumRemapped));

  EXPECT_EQ(NumRemapped, 2u);
  EXPECT_EQ(lineValue(Output, 0, 0), 0u);
  EXPECT_EQ(lineValue(Output, 1, 0), 16u);
  EXPECT_TRUE(std::equal(Input.end() - PaddingSize, Input.end(),
                         Output.end() - PaddingSize));
}

TEST(PDBRewriterTest, RejectsMalformedOrTruncatedBlock) {
  LineSubsectionBuilder Builder(16, false);
  Builder.addBlock({{0, 1}});
  std::vector<uint8_t> Malformed = Builder.take();
  support::endian::write32le(Malformed.data() + sizeof(LineFragmentHeader) +
                                 offsetof(LineBlockFragmentHeader, BlockSize),
                             19);
  std::vector<uint8_t> Output;
  uint32_t NumRemapped = 0;
  EXPECT_FALSE(pdb_detail::rewriteLineSubsection(Malformed, {{0, 0}}, Output,
                                                 NumRemapped));
  EXPECT_TRUE(Output.empty());

  LineSubsectionBuilder ColumnBuilder(16, true);
  ColumnBuilder.addBlock({{0, 1, 10, 11}});
  std::vector<uint8_t> Truncated = ColumnBuilder.take();
  Truncated.pop_back();
  EXPECT_FALSE(pdb_detail::rewriteLineSubsection(Truncated, {{0, 0}}, Output,
                                                 NumRemapped));
  EXPECT_TRUE(Output.empty());
}

TEST(PDBRewriterTest, RejectsOffsetOutsideFragment) {
  LineSubsectionBuilder Builder(8, false);
  Builder.addBlock({{4, 1}});
  std::vector<uint8_t> Output;
  uint32_t NumRemapped = 0;

  EXPECT_FALSE(pdb_detail::rewriteLineSubsection(Builder.take(), {{0, 8}},
                                                 Output, NumRemapped));
}

} // namespace
