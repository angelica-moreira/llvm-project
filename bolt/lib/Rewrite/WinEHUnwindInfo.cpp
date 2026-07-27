//===- WinEHUnwindInfo.cpp - Windows unwind info validation ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Rewrite/WinEHUnwindInfo.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;
using namespace llvm::bolt;

namespace {

constexpr uint8_t UnwindHandlerFlags = 0x03;
constexpr uint8_t UnwindChainFlag = 0x04;
constexpr uint8_t KnownUnwindFlags = UnwindHandlerFlags | UnwindChainFlag;

static Error validateUnwindCodes(const SEHUnwindInfo &Info,
                                 bool AllowInheritedFrame) {
  bool HasSetFPReg = false;
  uint8_t PreviousOffset = UINT8_MAX;
  for (size_t Index = 0; Index < Info.UnwindCodes.size();) {
    const uint16_t Code = Info.UnwindCodes[Index];
    const uint8_t CodeOffset = Code & 0xff;
    const uint8_t UnwindOp = (Code >> 8) & 0x0f;
    const uint8_t OpInfo = (Code >> 12) & 0x0f;
    if (CodeOffset > Info.PrologSize || CodeOffset > PreviousOffset)
      return createStringError(errc::invalid_argument,
                               "invalid UNWIND_CODE ordering");
    PreviousOffset = CodeOffset;

    size_t Slots = 1;
    switch (UnwindOp) {
    case 0:
    case 2:
      break;
    case 1:
      if (OpInfo > 1)
        return createStringError(errc::invalid_argument,
                                 "invalid UWOP_ALLOC_LARGE");
      Slots = OpInfo ? 3 : 2;
      break;
    case 3:
      if (!Info.FrameRegister || HasSetFPReg)
        return createStringError(errc::invalid_argument,
                                 "invalid UWOP_SET_FPREG");
      HasSetFPReg = true;
      break;
    case 4:
    case 8:
      Slots = 2;
      break;
    case 5:
    case 9:
      Slots = 3;
      break;
    case 10:
      if (OpInfo > 1)
        return createStringError(errc::invalid_argument,
                                 "invalid UWOP_PUSH_MACHFRAME");
      break;
    default:
      return createStringError(errc::invalid_argument,
                               "unsupported UNWIND_CODE opcode");
    }
    if (Slots > Info.UnwindCodes.size() - Index)
      return createStringError(errc::invalid_argument,
                               "truncated multi-slot UNWIND_CODE");
    Index += Slots;
  }

  if (!AllowInheritedFrame && HasSetFPReg != (Info.FrameRegister != 0))
    return createStringError(errc::invalid_argument,
                             "inconsistent frame register metadata");
  return Error::success();
}

} // namespace

Expected<ParsedSEHUnwindInfo>
llvm::bolt::parseWinEHUnwindInfo(ArrayRef<uint8_t> XData, uint32_t Offset,
                                 uint32_t XDataRVA) {
  if (Offset & 3)
    return createStringError(errc::invalid_argument, "unaligned UNWIND_INFO");
  if (Offset > XData.size() || XData.size() - Offset < 4)
    return createStringError(errc::invalid_argument,
                             "truncated UNWIND_INFO header");

  ParsedSEHUnwindInfo Parsed;
  SEHUnwindInfo &Info = Parsed.Info;
  const uint8_t *Header = XData.data() + Offset;
  Info.Version = Header[0] & 0x7;
  Info.Flags = (Header[0] >> 3) & 0x1f;
  Info.PrologSize = Header[1];
  Info.CountOfCodes = Header[2];
  Info.FrameRegister = Header[3] & 0xf;
  Info.FrameOffset = (Header[3] >> 4) & 0xf;

  if (Info.Version != 1 || (Info.Flags & ~KnownUnwindFlags) != 0)
    return createStringError(errc::invalid_argument,
                             "unsupported UNWIND_INFO header");
  if (!Info.FrameRegister && Info.FrameOffset)
    return createStringError(errc::invalid_argument,
                             "invalid UNWIND_INFO frame offset");

  const uint8_t HandlerFlags = Info.Flags & UnwindHandlerFlags;
  const bool HasChain = (Info.Flags & UnwindChainFlag) != 0;
  if (HandlerFlags && HasChain)
    return createStringError(errc::invalid_argument,
                             "conflicting UNWIND_INFO flags");

  const uint64_t CodesEnd =
      static_cast<uint64_t>(Offset) + 4 + 2 * Info.CountOfCodes;
  if (CodesEnd > XData.size())
    return createStringError(errc::invalid_argument,
                             "truncated UNWIND_CODE array");

  for (uint32_t CodeOffset = Offset + 4; CodeOffset < CodesEnd; CodeOffset += 2)
    Info.UnwindCodes.push_back(
        support::endian::read16le(XData.data() + CodeOffset));
  if (Error E = validateUnwindCodes(Info, HasChain))
    return std::move(E);

  const uint64_t HandlerOffset = alignTo(CodesEnd, uint64_t(4));
  if (HandlerOffset > XData.size() || HandlerOffset > UINT32_MAX)
    return createStringError(errc::invalid_argument,
                             "truncated UNWIND_INFO padding");
  Parsed.HandlerDataOffset = static_cast<uint32_t>(HandlerOffset);

  if (HasChain) {
    if (XData.size() - Parsed.HandlerDataOffset < 12)
      return createStringError(errc::invalid_argument,
                               "truncated chained RUNTIME_FUNCTION");
    const uint8_t *Chain = XData.data() + Parsed.HandlerDataOffset;
    Info.IsChained = true;
    Info.ChainedBeginRVA = support::endian::read32le(Chain);
    Info.ChainedEndRVA = support::endian::read32le(Chain + 4);
    Info.ChainedUnwindRVA = support::endian::read32le(Chain + 8);
    if (!Info.ChainedBeginRVA || Info.ChainedEndRVA <= Info.ChainedBeginRVA ||
        !Info.ChainedUnwindRVA)
      return createStringError(errc::invalid_argument,
                               "invalid chained RUNTIME_FUNCTION");
    const uint64_t EntryRVA =
        static_cast<uint64_t>(XDataRVA) + Parsed.HandlerDataOffset;
    if (EntryRVA > UINT32_MAX)
      return createStringError(errc::invalid_argument,
                               "chained RUNTIME_FUNCTION RVA overflow");
    Info.ChainedEntryRVA = static_cast<uint32_t>(EntryRVA);
  } else if (HandlerFlags) {
    if (XData.size() - Parsed.HandlerDataOffset < 4)
      return createStringError(errc::invalid_argument,
                               "truncated exception handler RVA");
    Info.HandlerKind = WinEHHandlerKind::Unknown;
    Info.ExceptionHandlerRVA =
        support::endian::read32le(XData.data() + Parsed.HandlerDataOffset);
    if (!Info.ExceptionHandlerRVA)
      return createStringError(errc::invalid_argument,
                               "invalid exception handler RVA");
    Info.HandlerDataMaxSize =
        XData.size() - Parsed.HandlerDataOffset - sizeof(uint32_t);
  }

  Info.IsValid = true;
  return Parsed;
}

bool llvm::bolt::isWinEHUnwindInfoReusable(const SEHUnwindInfo &Info,
                                           ArrayRef<uint8_t> OriginalBytes,
                                           ArrayRef<uint8_t> EmittedBytes) {
  if (!Info.IsValid || Info.Version != 1 ||
      (Info.Flags & ~KnownUnwindFlags) != 0 ||
      Info.UnwindCodes.size() != Info.CountOfCodes)
    return false;

  const bool HasHandler = (Info.Flags & UnwindHandlerFlags) != 0;
  const bool HasChain = (Info.Flags & UnwindChainFlag) != 0;
  if (HasHandler && HasChain)
    return false;
  if (HasHandler != (Info.HandlerKind != WinEHHandlerKind::None))
    return false;
  if (HasHandler && !Info.ExceptionHandlerRVA)
    return false;
  if (HasChain != Info.IsChained)
    return false;

  if (Info.PrologSize > OriginalBytes.size() ||
      Info.PrologSize > EmittedBytes.size())
    return false;
  return OriginalBytes.take_front(Info.PrologSize) ==
         EmittedBytes.take_front(Info.PrologSize);
}
