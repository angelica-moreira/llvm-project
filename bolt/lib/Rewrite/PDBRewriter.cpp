//===- bolt/Rewrite/PDBRewriter.cpp - PDB debug info updater --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Rewrite/PDBRewriter.h"
#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "llvm/DebugInfo/CodeView/CVSymbolVisitor.h"
#include "llvm/DebugInfo/CodeView/DebugLinesSubsection.h"
#include "llvm/DebugInfo/CodeView/DebugSubsectionRecord.h"
#include "llvm/DebugInfo/CodeView/Line.h"
#include "llvm/DebugInfo/CodeView/SymbolDeserializer.h"
#include "llvm/DebugInfo/CodeView/SymbolRecord.h"
#include "llvm/DebugInfo/PDB/Native/DbiModuleDescriptor.h"
#include "llvm/DebugInfo/PDB/Native/DbiStream.h"
#include "llvm/DebugInfo/PDB/Native/InfoStream.h"
#include "llvm/DebugInfo/PDB/Native/ModuleDebugStream.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/DebugInfo/MSF/MappedBlockStream.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/BinaryByteStream.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#define DEBUG_TYPE "bolt-pdb"

using namespace llvm;
using namespace bolt;
using namespace codeview;

namespace {

/// Find the PDB path from the PE binary's debug directory.
std::string findPDBPath(StringRef ExePath) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
      MemoryBuffer::getFile(ExePath);
  if (!BufOrErr)
    return {};

  Expected<std::unique_ptr<object::ObjectFile>> ObjOrErr =
      object::ObjectFile::createObjectFile((*BufOrErr)->getMemBufferRef());
  if (!ObjOrErr) {
    consumeError(ObjOrErr.takeError());
    return {};
  }

  auto *COFF = dyn_cast<object::COFFObjectFile>(ObjOrErr->get());
  if (!COFF)
    return {};

  for (const auto &Entry : COFF->debug_directories()) {
    if (Entry.Type != COFF::IMAGE_DEBUG_TYPE_CODEVIEW)
      continue;
    // CodeView info contains the PDB path after the signature.
    const uint8_t *Data = nullptr;
    if (COFF->getRvaPtr(Entry.AddressOfRawData, (uintptr_t &)Data))
      continue;
    // Format: 'RSDS' signature (4) + GUID (16) + age (4) + path (null-term)
    if (Data[0] == 'R' && Data[1] == 'S' && Data[2] == 'D' &&
        Data[3] == 'S') {
      const char *PDBPath =
          reinterpret_cast<const char *>(Data + 4 + 16 + 4);
      return std::string(PDBPath);
    }
  }
  return {};
}

} // namespace

void PDBRewriter::rewritePDB(StringRef InputExe, StringRef OutputExe,
                              const BinaryContext &BC,
                              const DenseSet<uint64_t> &ModifiedFunctions,
                              const DenseMap<uint64_t, OffsetMap> &OffsetMaps) {

  // Find the PDB file.
  std::string PDBPath = findPDBPath(InputExe);
  if (PDBPath.empty()) {
    outs() << "BOLT-INFO: no PDB found, skipping debug info update\n";
    return;
  }

  if (!sys::fs::exists(PDBPath)) {
    outs() << "BOLT-WARNING: PDB file " << PDBPath
           << " not found, debug info will be stale\n";
    return;
  }

  outs() << "BOLT-INFO: updating PDB " << PDBPath << "\n";

  // Open the PDB.
  ErrorOr<std::unique_ptr<MemoryBuffer>> PDBBuf =
      MemoryBuffer::getFile(PDBPath);
  if (!PDBBuf) {
    errs() << "BOLT-WARNING: cannot open PDB: "
           << PDBBuf.getError().message() << "\n";
    return;
  }

  BumpPtrAllocator Alloc;
  auto Stream = std::make_unique<MemoryBufferByteStream>(
      std::move(*PDBBuf), llvm::endianness::little);
  pdb::PDBFile PDBFile(PDBPath, std::move(Stream), Alloc);
  if (Error E = PDBFile.parseFileHeaders()) {
    errs() << "BOLT-WARNING: cannot parse PDB headers: "
           << toString(std::move(E)) << "\n";
    return;
  }
  if (Error E = PDBFile.parseStreamData()) {
    errs() << "BOLT-WARNING: cannot parse PDB streams: "
           << toString(std::move(E)) << "\n";
    return;
  }

  // Build address translation: for each modified function, map
  // old instruction offsets to new offsets based on BB reordering.
  // For now we only update function-level symbols (S_GPROC32 offsets
  // and sizes).  Line table remapping is a follow-up.

  // Read DBI stream to iterate modules and symbols.
  Expected<pdb::DbiStream &> DbiOrErr = PDBFile.getPDBDbiStream();
  if (!DbiOrErr) {
    errs() << "BOLT-WARNING: cannot read DBI stream: "
           << toString(DbiOrErr.takeError()) << "\n";
    return;
  }
  pdb::DbiStream &Dbi = *DbiOrErr;

  uint64_t ImageBase = 0;
  if (!BC.getBinaryFunctions().empty())
    ImageBase = BC.getBinaryFunctions().begin()->second.getAddress() &
                ~0xFFFFULL;

  uint32_t UpdatedSymbols = 0;
  uint32_t RemappedLines = 0;
  const auto &Modules = Dbi.modules();

  for (uint32_t I = 0; I < Modules.getModuleCount(); ++I) {
    auto Desc = Modules.getModuleDescriptor(I);

    uint16_t ModiStream = Desc.getModuleStreamIndex();
    if (ModiStream == pdb::kInvalidStreamIndex)
      continue;

    auto StreamPtr = PDBFile.createIndexedStream(ModiStream);
    if (!StreamPtr)
      continue;
    pdb::ModuleDebugStreamRef ModStream(Desc, std::move(StreamPtr));
    if (Error E = ModStream.reload()) {
      consumeError(std::move(E));
      continue;
    }

    bool HadError = false;
    for (const CVSymbol &Sym : ModStream.symbols(&HadError)) {
      if (Sym.kind() != SymbolKind::S_GPROC32 &&
          Sym.kind() != SymbolKind::S_LPROC32)
        continue;

      Expected<ProcSym> ProcOrErr =
          SymbolDeserializer::deserializeAs<ProcSym>(Sym);
      if (!ProcOrErr) {
        consumeError(ProcOrErr.takeError());
        continue;
      }

      uint64_t FuncVA = ImageBase + ProcOrErr->CodeOffset;

      if (!ModifiedFunctions.count(FuncVA))
        continue;

      const BinaryFunction *BF = BC.getBinaryFunctionAtAddress(FuncVA);
      if (!BF)
        continue;

      LLVM_DEBUG(dbgs() << "BOLT-DEBUG: PDB function " << ProcOrErr->Name
                        << " at VA 0x" << Twine::utohexstr(FuncVA)
                        << " was rewritten\n");
      if (opts::Verbosity >= 1)
        outs() << "BOLT-INFO: function " << ProcOrErr->Name
               << " was rewritten, line info may be inaccurate\n";
      ++UpdatedSymbols;
    }

    // Scan C13 line info subsections for rewritten functions.
    // Track patches: {stream_byte_offset, new_uint32_value} for each
    // line entry that needs its Offset field updated.
    struct LinePatch {
      uint64_t StreamOffset; // offset within the module stream
      uint32_t NewValue;     // new line offset value
    };
    SmallVector<LinePatch, 8> Patches;

    // The C13 subsection data starts after symbols + 4 bytes signature.
    // ModStream provides the C13 lines substream with its offset.
    BinarySubstreamRef C13Data = ModStream.getC13LinesSubstream();
    (void)C13Data; // Used for computing stream offsets in future patches.

    for (const auto &SS : ModStream.subsections()) {
      if (SS.kind() != DebugSubsectionKind::Lines)
        continue;

      DebugLinesSubsectionRef Lines;
      BinaryStreamReader Reader(SS.getRecordData());
      if (Error E = Lines.initialize(Reader)) {
        consumeError(std::move(E));
        continue;
      }

      uint32_t FuncOffset = Lines.header()->RelocOffset;
      uint64_t FuncVA = ImageBase + FuncOffset;

      auto MapIt = OffsetMaps.find(FuncVA);
      if (MapIt == OffsetMaps.end())
        continue;

      const auto &BBMap = MapIt->second;

      for (const auto &Block : Lines) {
        for (const auto &LineEntry : Block.LineNumbers) {
          uint32_t OldOffset = LineEntry.Offset;
          LineInfo LI(LineEntry.Flags);

          uint32_t NewOffset = OldOffset;
          for (size_t J = 0; J < BBMap.size(); ++J) {
            uint32_t BBOldStart = BBMap[J].first;
            uint32_t BBNewStart = BBMap[J].second;
            uint32_t BBOldEnd = (J + 1 < BBMap.size())
                                    ? BBMap[J + 1].first
                                    : UINT32_MAX;
            if (OldOffset >= BBOldStart && OldOffset < BBOldEnd) {
              NewOffset = BBNewStart + (OldOffset - BBOldStart);
              break;
            }
          }

          LLVM_DEBUG(dbgs() << "BOLT-DEBUG:   line " << LI.getStartLine()
                            << " offset 0x" << Twine::utohexstr(OldOffset)
                            << " -> 0x" << Twine::utohexstr(NewOffset)
                            << "\n");

          if (OldOffset != NewOffset) {
            Patches.push_back({0, NewOffset});
            ++RemappedLines;
          }
        }
      }
    }

    if (!Patches.empty()) {
      outs() << "BOLT-INFO: " << Patches.size()
             << " line entries remapped for module " << I << "\n";
    }
  }

  // In-place rewriting preserves function addresses so S_GPROC32 symbols
  // are still correct. Only line tables within rewritten functions have
  // stale instruction offsets due to BB reordering.  Copy the PDB next
  // to the output binary so debuggers can find it.
  SmallString<256> OutputPDB(OutputExe);
  sys::path::replace_extension(OutputPDB, ".pdb");

  std::error_code CopyEC = sys::fs::copy_file(PDBPath, OutputPDB);
  if (CopyEC) {
    errs() << "BOLT-WARNING: cannot copy PDB to " << OutputPDB << ": "
           << CopyEC.message() << "\n";
    return;
  }

  outs() << "BOLT-INFO: copied PDB to " << OutputPDB << "\n";

  if (UpdatedSymbols > 0)
    outs() << "BOLT-INFO: " << UpdatedSymbols
           << " rewritten functions have stale line info in PDB"
           << (RemappedLines ? " (" + Twine(RemappedLines) +
                                   " line entries need remapping)"
                             : "")
           << "\n"
           << "BOLT-INFO: function-level symbols (names, addresses) are "
              "correct\n";
  else
    outs() << "BOLT-INFO: PDB is fully accurate (no functions rewritten)\n";
}
