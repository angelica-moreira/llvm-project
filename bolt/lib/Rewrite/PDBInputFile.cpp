//===- PDBInputFile.cpp - PE/COFF PDB input -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Rewrite/PDBInputFile.h"
#include "llvm/BinaryFormat/COFF.h"
#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/Native/NativeSession.h"
#include "llvm/DebugInfo/PDB/PDBSymbolExe.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFunc.h"
#include "llvm/DebugInfo/PDB/PDBSymbolPublicSymbol.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include <cstring>

using namespace llvm;
using namespace llvm::bolt;

namespace {

static std::string resolvePDBPath(StringRef ExecutablePath,
                                  StringRef EmbeddedPath,
                                  StringRef ExplicitPath) {
  if (!ExplicitPath.empty())
    return ExplicitPath.str();

  if (sys::fs::exists(EmbeddedPath))
    return EmbeddedPath.str();

  SmallString<128> AdjacentPath(ExecutablePath);
  sys::path::remove_filename(AdjacentPath);
  sys::path::append(AdjacentPath, sys::path::filename(EmbeddedPath));
  return AdjacentPath.str().str();
}

template <typename SymbolT>
static std::string findSymbolName(pdb::IPDBSession &Session, uint32_t RVA) {
  std::unique_ptr<pdb::PDBSymbol> Symbol =
      Session.findSymbolByRVA(RVA, SymbolT::Tag);
  if (!Symbol)
    return {};

  const auto *TypedSymbol = dyn_cast<SymbolT>(Symbol.get());
  if (!TypedSymbol || TypedSymbol->getRelativeVirtualAddress() != RVA)
    return {};
  return TypedSymbol->getName();
}

} // namespace

PDBSymbolResolver::PDBSymbolResolver(std::unique_ptr<pdb::IPDBSession> Session)
    : Session(std::move(Session)) {}

PDBSymbolResolver::~PDBSymbolResolver() = default;

Expected<PDBInputFile>
llvm::bolt::findPDBInputFile(const object::COFFObjectFile &Obj,
                             StringRef ExecutablePath,
                             StringRef ExplicitPDBPath) {
  const codeview::DebugInfo *DebugInfo;
  StringRef EmbeddedPath;
  if (Error E = Obj.getDebugPDBInfo(DebugInfo, EmbeddedPath))
    return std::move(E);
  if (!DebugInfo)
    return createStringError(errc::invalid_argument,
                             "executable has no CodeView PDB record");

  if (DebugInfo->Signature.CVSignature != OMF::Signature::PDB70)
    return createStringError(errc::not_supported,
                             "unsupported CodeView PDB signature");
  if (EmbeddedPath.empty() && ExplicitPDBPath.empty())
    return createStringError(errc::invalid_argument,
                             "executable has no PDB path");

  PDBInputFile Input;
  Input.Path = resolvePDBPath(ExecutablePath, EmbeddedPath, ExplicitPDBPath);
  if (!sys::fs::is_regular_file(Input.Path))
    return createStringError(errc::no_such_file_or_directory,
                             "PDB file not found: %s", Input.Path.c_str());

  std::memcpy(Input.Guid.Guid, DebugInfo->PDB70.Signature,
              sizeof(Input.Guid.Guid));
  Input.Age = DebugInfo->PDB70.Age;
  return Input;
}

Expected<PDBInputFile> llvm::bolt::findPDBInputFile(StringRef ExecutablePath,
                                                    StringRef ExplicitPDBPath) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> Buffer =
      MemoryBuffer::getFile(ExecutablePath);
  if (!Buffer)
    return errorCodeToError(Buffer.getError());

  Expected<std::unique_ptr<object::ObjectFile>> Object =
      object::ObjectFile::createObjectFile((*Buffer)->getMemBufferRef());
  if (!Object)
    return Object.takeError();

  const auto *COFF = dyn_cast<object::COFFObjectFile>(Object->get());
  if (!COFF)
    return createStringError(errc::invalid_argument,
                             "input is not a PE/COFF file");
  return findPDBInputFile(*COFF, ExecutablePath, ExplicitPDBPath);
}

Expected<std::unique_ptr<PDBSymbolResolver>>
PDBSymbolResolver::create(const PDBInputFile &Input) {
  std::unique_ptr<pdb::IPDBSession> Session;
  if (Error E = pdb::NativeSession::createFromPdbPath(Input.Path, Session))
    return std::move(E);

  std::unique_ptr<pdb::PDBSymbolExe> GlobalScope = Session->getGlobalScope();
  if (!GlobalScope || GlobalScope->getGuid() != Input.Guid ||
      GlobalScope->getAge() != Input.Age)
    return createStringError(errc::invalid_argument,
                             "PDB does not match the executable");

  return std::unique_ptr<PDBSymbolResolver>(
      new PDBSymbolResolver(std::move(Session)));
}

bool PDBSymbolResolver::hasExactSymbolName(uint32_t RVA,
                                           StringRef ExpectedName) {
  if (::findSymbolName<pdb::PDBSymbolPublicSymbol>(*Session, RVA) ==
      ExpectedName)
    return true;
  return ::findSymbolName<pdb::PDBSymbolFunc>(*Session, RVA) == ExpectedName;
}
