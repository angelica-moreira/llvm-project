//===- WinEHHandlerInfo.cpp - Windows EH handler classification ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Rewrite/WinEHHandlerInfo.h"

using namespace llvm;
using namespace llvm::bolt;

WinEHHandlerKind llvm::bolt::classifyWinEHHandlerName(StringRef Name) {
  return Name == "__GSHandlerCheck" ? WinEHHandlerKind::GSHandlerCheck
                                    : WinEHHandlerKind::Unknown;
}
