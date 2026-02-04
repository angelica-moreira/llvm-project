//===- StaticProfileExporter.h - Export Static Profile Info ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the StaticProfileExporterPass which exports statically
// inferred profile information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_STATICPROFILEEXPORTER_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_STATICPROFILEEXPORTER_H

#include "llvm/IR/PassManager.h"
#include <string>

namespace llvm {

class Module;

class StaticProfileExporterPass : public PassInfoMixin<StaticProfileExporterPass> {
  std::string ProfilePath;

public:
  explicit StaticProfileExporterPass(std::string Path = "")
      : ProfilePath(std::move(Path)) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_INSTRUMENTATION_STATICPROFILEEXPORTER_H
