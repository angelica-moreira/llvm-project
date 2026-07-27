//===- WinEHUnwindInfo.cpp - Windows unwind info tests --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Rewrite/WinEHUnwindInfo.h"
#include "bolt/Rewrite/WinEHHandlerInfo.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::bolt;

namespace {

void expectParseFailure(ArrayRef<uint8_t> Bytes) {
  Expected<ParsedSEHUnwindInfo> Parsed = parseWinEHUnwindInfo(Bytes, 0, 0x2000);
  const bool Succeeded = static_cast<bool>(Parsed);
  EXPECT_FALSE(Succeeded);
  if (!Parsed)
    consumeError(Parsed.takeError());
}

void expectValidUnwindCodes(ArrayRef<uint16_t> Codes, uint8_t Frame = 0) {
  SmallVector<uint8_t> Bytes = {0x01, 0x20, static_cast<uint8_t>(Codes.size()),
                                Frame};
  for (uint16_t Code : Codes) {
    Bytes.push_back(Code & 0xff);
    Bytes.push_back(Code >> 8);
  }
  if (Codes.size() & 1)
    Bytes.append(2, 0);

  Expected<ParsedSEHUnwindInfo> Parsed = parseWinEHUnwindInfo(Bytes, 0, 0x2000);
  ASSERT_TRUE(static_cast<bool>(Parsed));
}

TEST(WinEHHandlerInfoTest, RecognizesOnlyBareGSHandlerCheck) {
  EXPECT_EQ(classifyWinEHHandlerName("__GSHandlerCheck"),
            WinEHHandlerKind::GSHandlerCheck);
  EXPECT_EQ(classifyWinEHHandlerName("__GSHandlerCheck_SEH"),
            WinEHHandlerKind::Unknown);
  EXPECT_EQ(classifyWinEHHandlerName("__GSHandlerCheck_EH"),
            WinEHHandlerKind::Unknown);
  EXPECT_EQ(classifyWinEHHandlerName("__GSHandlerCheck_EH4"),
            WinEHHandlerKind::Unknown);
  EXPECT_EQ(classifyWinEHHandlerName("__C_specific_handler"),
            WinEHHandlerKind::Unknown);
  EXPECT_EQ(classifyWinEHHandlerName("prefix__GSHandlerCheck"),
            WinEHHandlerKind::Unknown);
}

TEST(WinEHUnwindInfoTest, AcceptsMatchingGSUnwindInfo) {
  SEHUnwindInfo Info;
  Info.IsValid = true;
  Info.Version = 1;
  Info.Flags = 1;
  Info.PrologSize = 3;
  Info.CountOfCodes = 1;
  Info.UnwindCodes.push_back(0x1234);
  Info.ExceptionHandlerRVA = 0x2000;
  Info.HandlerKind = WinEHHandlerKind::GSHandlerCheck;

  const uint8_t Original[] = {0x48, 0x83, 0xec, 0x20};
  const uint8_t Emitted[] = {0x48, 0x83, 0xec, 0x30};
  EXPECT_TRUE(isWinEHUnwindInfoReusable(Info, Original, Emitted));
}

TEST(WinEHUnwindInfoTest, RejectsChangedProlog) {
  SEHUnwindInfo Info;
  Info.IsValid = true;
  Info.Version = 1;
  Info.PrologSize = 4;

  const uint8_t Original[] = {0x48, 0x83, 0xec, 0x20};
  const uint8_t Emitted[] = {0x48, 0x83, 0xec, 0x30};
  EXPECT_FALSE(isWinEHUnwindInfoReusable(Info, Original, Emitted));
}

TEST(WinEHUnwindInfoTest, ParsesValidRecords) {
  const uint8_t GSRecord[] = {0x19, 0x03, 0x01, 0x00, 0x03, 0x32, 0x00, 0x00,
                              0x00, 0x10, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00};
  Expected<ParsedSEHUnwindInfo> GS = parseWinEHUnwindInfo(GSRecord, 0, 0x2000);
  ASSERT_TRUE(static_cast<bool>(GS));
  EXPECT_TRUE(GS->Info.IsValid);
  EXPECT_EQ(GS->Info.HandlerKind, WinEHHandlerKind::Unknown);
  EXPECT_EQ(GS->Info.ExceptionHandlerRVA, 0x1000u);
  EXPECT_EQ(GS->HandlerDataOffset, 8u);

  const uint8_t ChainRecord[] = {0x21, 0x00, 0x00, 0x00, 0x00, 0x10,
                                 0x00, 0x00, 0x10, 0x10, 0x00, 0x00,
                                 0x40, 0x20, 0x00, 0x00};
  Expected<ParsedSEHUnwindInfo> Chain =
      parseWinEHUnwindInfo(ChainRecord, 0, 0x2000);
  ASSERT_TRUE(static_cast<bool>(Chain));
  EXPECT_TRUE(Chain->Info.IsChained);
  EXPECT_EQ(Chain->Info.ChainedEntryRVA, 0x2004u);

  const uint8_t FrameRecord[] = {0x01, 0x18, 0x01, 0x35,
                                 0x18, 0x33, 0x00, 0x00};
  Expected<ParsedSEHUnwindInfo> Frame =
      parseWinEHUnwindInfo(FrameRecord, 0, 0x2000);
  ASSERT_TRUE(static_cast<bool>(Frame));
  EXPECT_EQ(Frame->Info.FrameRegister, 5u);
  EXPECT_EQ(Frame->Info.FrameOffset, 3u);
}

TEST(WinEHUnwindInfoTest, ParsesAllVersionOneOpcodes) {
  expectValidUnwindCodes({0x5001});
  expectValidUnwindCodes({0x0102, 0x0020});
  expectValidUnwindCodes({0x1103, 0x0020, 0x0000});
  expectValidUnwindCodes({0x3204});
  expectValidUnwindCodes({0x3305}, 0x35);
  expectValidUnwindCodes({0x5406, 0x0002});
  expectValidUnwindCodes({0x5507, 0x0002, 0x0000});
  expectValidUnwindCodes({0x7808, 0x0002});
  expectValidUnwindCodes({0x7909, 0x0002, 0x0000});
  expectValidUnwindCodes({0x0a0a});
  expectValidUnwindCodes({0x1a0a});
}

TEST(WinEHUnwindInfoTest, RejectsMalformedRecords) {
  expectParseFailure({});
  expectParseFailure({0x01, 0x00, 0x00});
  expectParseFailure({0x02, 0x00, 0x00, 0x00});
  expectParseFailure({0x19, 0x00, 0x00, 0x00});
  expectParseFailure({0x01, 0x00, 0x00, 0x10});
  expectParseFailure({0x01, 0x01, 0x01, 0x00});
  expectParseFailure({0x01, 0x01, 0x01, 0x00, 0x01, 0x21, 0x00, 0x00});
  expectParseFailure({0x01, 0x02, 0x02, 0x00, 0x01, 0x02, 0x02, 0x02});
  expectParseFailure({0x01, 0x00, 0x00, 0x05});
  expectParseFailure({0x21, 0x00, 0x00, 0x00});
  expectParseFailure({0x21, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
                      0x10, 0x00, 0x00, 0x40, 0x20, 0x00, 0x00});
  expectParseFailure({0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00});
}

TEST(WinEHUnwindInfoTest, RejectsMalformedMetadata) {
  const uint8_t Bytes[] = {0x48, 0x83, 0xec, 0x20};
  SEHUnwindInfo Info;
  Info.IsValid = true;
  Info.Version = 2;
  EXPECT_FALSE(isWinEHUnwindInfoReusable(Info, Bytes, Bytes));

  Info.Version = 1;
  Info.Flags = 0x08;
  EXPECT_FALSE(isWinEHUnwindInfoReusable(Info, Bytes, Bytes));

  Info.Flags = 0;
  Info.CountOfCodes = 1;
  EXPECT_FALSE(isWinEHUnwindInfoReusable(Info, Bytes, Bytes));

  Info.CountOfCodes = 0;
  Info.HandlerKind = WinEHHandlerKind::GSHandlerCheck;
  EXPECT_FALSE(isWinEHUnwindInfoReusable(Info, Bytes, Bytes));

  Info.HandlerKind = WinEHHandlerKind::None;
  Info.IsChained = true;
  EXPECT_FALSE(isWinEHUnwindInfoReusable(Info, Bytes, Bytes));

  Info.IsChained = false;
  Info.PrologSize = 5;
  EXPECT_FALSE(isWinEHUnwindInfoReusable(Info, Bytes, Bytes));
}

} // namespace
