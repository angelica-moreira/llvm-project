//===- bolt/Rewrite/WinEHUnwindInfo.h - Windows unwind info -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_REWRITE_WINEH_UNWIND_INFO_H
#define BOLT_REWRITE_WINEH_UNWIND_INFO_H

#include "bolt/Rewrite/WinEHHandlerInfo.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include <cstdint>

namespace llvm {
namespace bolt {

struct SEHUnwindInfo {
  bool IsValid = false;
  uint32_t EndRVA = 0;
  uint32_t UnwindInfoRVA = 0;
  uint8_t Version = 0;
  uint8_t Flags = 0;
  uint8_t PrologSize = 0;
  uint8_t FrameRegister = 0;
  uint8_t FrameOffset = 0;
  uint8_t CountOfCodes = 0;
  SmallVector<uint16_t, 8> UnwindCodes;
  uint32_t ExceptionHandlerRVA = 0;
  uint32_t HandlerDataMaxSize = 0;
  WinEHHandlerKind HandlerKind = WinEHHandlerKind::None;
  uint32_t CxxFuncInfoRVA = 0;
  bool IsChained = false;
  uint32_t ChainedBeginRVA = 0;
  uint32_t ChainedEndRVA = 0;
  uint32_t ChainedUnwindRVA = 0;
  uint32_t ChainedEntryRVA = 0;
};

struct ParsedSEHUnwindInfo {
  SEHUnwindInfo Info;
  uint32_t HandlerDataOffset = 0;
};

Expected<ParsedSEHUnwindInfo> parseWinEHUnwindInfo(ArrayRef<uint8_t> XData,
                                                   uint32_t Offset,
                                                   uint32_t XDataRVA);

bool isWinEHUnwindInfoReusable(const SEHUnwindInfo &Info,
                               ArrayRef<uint8_t> OriginalBytes,
                               ArrayRef<uint8_t> EmittedBytes);

} // namespace bolt
} // namespace llvm

#endif
