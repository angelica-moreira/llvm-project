//===- bolt/Rewrite/WinEHHandlerInfo.h - Windows EH handlers ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_REWRITE_WINEH_HANDLER_INFO_H
#define BOLT_REWRITE_WINEH_HANDLER_INFO_H

#include "llvm/ADT/StringRef.h"
#include <cstdint>

namespace llvm {
namespace bolt {

enum class WinEHHandlerKind : uint8_t {
  None,
  Unknown,
  CxxFrameHandler3,
  GSHandlerCheck,
};

WinEHHandlerKind classifyWinEHHandlerName(StringRef Name);

} // namespace bolt
} // namespace llvm

#endif
