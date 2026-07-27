//===- bolt/Rewrite/PDBInputFile.h - PE/COFF PDB input ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_REWRITE_PDB_INPUT_FILE_H
#define BOLT_REWRITE_PDB_INPUT_FILE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/CodeView/GUID.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <memory>
#include <string>

namespace llvm {

namespace object {
class COFFObjectFile;
}

namespace pdb {
class IPDBSession;
}

namespace bolt {

struct PDBInputFile {
  std::string Path;
  codeview::GUID Guid;
  uint32_t Age;
};

Expected<PDBInputFile> findPDBInputFile(const object::COFFObjectFile &Obj,
                                        StringRef ExecutablePath,
                                        StringRef ExplicitPDBPath = {});

Expected<PDBInputFile> findPDBInputFile(StringRef ExecutablePath,
                                        StringRef ExplicitPDBPath = {});

class PDBSymbolResolver {
public:
  ~PDBSymbolResolver();

  static Expected<std::unique_ptr<PDBSymbolResolver>>
  create(const PDBInputFile &Input);

  bool hasExactSymbolName(uint32_t RVA, StringRef Name);

private:
  explicit PDBSymbolResolver(std::unique_ptr<pdb::IPDBSession> Session);

  std::unique_ptr<pdb::IPDBSession> Session;
};

} // namespace bolt
} // namespace llvm

#endif
