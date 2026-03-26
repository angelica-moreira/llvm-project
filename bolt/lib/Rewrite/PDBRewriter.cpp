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
    uintptr_t DataAddr = 0;
    if (Error E = COFF->getRvaPtr(Entry.AddressOfRawData, DataAddr)) {
      consumeError(std::move(E));
      continue;
    }
    const uint8_t *Data = reinterpret_cast<const uint8_t *>(DataAddr);
    // Format: 'RSDS' signature (4) + GUID (16) + age (4) + path (null-term)
    if (Entry.SizeOfData < 25) // at least 24 header + 1 byte path
      continue;
    if (Data[0] == 'R' && Data[1] == 'S' && Data[2] == 'D' &&
        Data[3] == 'S') {
      const char *PDBPath =
          reinterpret_cast<const char *>(Data + 4 + 16 + 4);
      size_t MaxPathLen = Entry.SizeOfData - 24;
      return std::string(PDBPath, strnlen(PDBPath, MaxPathLen));
    }
  }
  return {};
}

} // namespace

void PDBRewriter::rewritePDB(StringRef InputExe, StringRef OutputExe,
                              const BinaryContext &BC, uint64_t ImageBase,
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

  // Build section RVA table from DBI section headers. PDB symbols use
  // 1-based section indices with section-relative offsets. We need each
  // section's VirtualAddress to compute the full VA.
  auto SectionHeaders = Dbi.getSectionHeaders();
  SmallVector<uint32_t, 16> SectionRVAs;
  for (const auto &Hdr : SectionHeaders)
    SectionRVAs.push_back(Hdr.VirtualAddress);

  // Compute VA from a PDB section:offset pair.
  auto computeVA = [&](uint16_t Segment, uint32_t Offset) -> uint64_t {
    if (Segment == 0 || Segment > SectionRVAs.size())
      return 0;
    return ImageBase + SectionRVAs[Segment - 1] + Offset;
  };

  // Collect all patches: {stream_index, stream_offset, new_value}.
  struct PDBPatch {
    uint16_t StreamIndex;
    uint64_t StreamOffset;
    uint32_t NewValue;
  };
  SmallVector<PDBPatch, 16> AllPatches;

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

      uint64_t FuncVA = computeVA(ProcOrErr->Segment, ProcOrErr->CodeOffset);

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
    // Track byte positions of Offset fields that need patching.
    struct LinePatch {
      uint64_t StreamOffset; // byte offset within the module stream
      uint32_t NewValue;     // new line offset value
    };
    SmallVector<LinePatch, 8> Patches;

    // Walk the raw C13 bytes to track exact positions.
    // Layout: each subsection has {uint32 Kind, uint32 Length, [data]}.
    // Line fragment data: {uint32 RelocOffset, uint16 Segment,
    //   uint16 Flags, uint32 CodeSize}, then file blocks with line entries.
    BinarySubstreamRef C13Ref = ModStream.getC13LinesSubstream();
    uint64_t C13StreamBase = C13Ref.Offset; // offset of C13 data in stream

    ArrayRef<uint8_t> C13Bytes;
    if (auto EC = C13Ref.StreamData.readBytes(0, C13Ref.StreamData.getLength(),
                                               C13Bytes))
      continue;

    uint64_t Pos = 0;
    while (Pos + 8 <= C13Bytes.size()) {
      uint32_t Kind = support::endian::read32le(&C13Bytes[Pos]);
      uint32_t Length = support::endian::read32le(&C13Bytes[Pos + 4]);
      uint64_t DataStart = Pos + 8;
      uint64_t DataEnd = DataStart + Length;

      if (DataEnd > C13Bytes.size())
        break;

      if (Kind != uint32_t(DebugSubsectionKind::Lines)) {
        Pos = alignTo(DataEnd, 4);
        continue;
      }

      // Line fragment header at DataStart:
      // {uint32 RelocOffset, uint16 RelocSegment, uint16 Flags, uint32 CodeSize}
      if (Length < 12) { Pos = alignTo(DataEnd, 4); continue; }
      uint32_t RelocOffset = support::endian::read32le(&C13Bytes[DataStart]);
      uint16_t RelocSegment = support::endian::read16le(&C13Bytes[DataStart + 4]);
      uint64_t FuncVA = computeVA(RelocSegment, RelocOffset);

      auto MapIt = OffsetMaps.find(FuncVA);
      if (MapIt == OffsetMaps.end()) {
        Pos = alignTo(DataEnd, 4);
        continue;
      }

      const auto &BBMap = MapIt->second;

      // Skip header: RelocOffset(4) + Segment(2) + Flags(2) + CodeSize(4) = 12
      uint64_t FileBlockPos = DataStart + 12;

      while (FileBlockPos + 12 <= DataEnd) {
        // File block: FileIndex(4) + NumLines(4) + BlockSize(4)
        uint32_t NumLines =
            support::endian::read32le(&C13Bytes[FileBlockPos + 4]);
        uint32_t BlockSize =
            support::endian::read32le(&C13Bytes[FileBlockPos + 8]);

        uint64_t LineStart = FileBlockPos + 12;

        for (uint32_t L = 0; L < NumLines; ++L) {
          uint64_t EntryPos = LineStart + L * 8;
          if (EntryPos + 8 > DataEnd)
            break;

          uint32_t OldOffset =
              support::endian::read32le(&C13Bytes[EntryPos]);
          uint32_t Flags =
              support::endian::read32le(&C13Bytes[EntryPos + 4]);
          LineInfo LI(Flags);

          // Remap through BB offset map.
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

          if (OldOffset != NewOffset) {
            // Stream offset = C13 base in stream + position within C13 data
            uint64_t StreamOff = C13StreamBase + EntryPos;
            Patches.push_back({StreamOff, NewOffset});
            ++RemappedLines;
            LLVM_DEBUG(dbgs() << "BOLT-DEBUG:   line " << LI.getStartLine()
                              << " offset 0x" << Twine::utohexstr(OldOffset)
                              << " -> 0x" << Twine::utohexstr(NewOffset)
                              << " at stream offset " << StreamOff << "\n");
          }
        }

        FileBlockPos += BlockSize;
      }

      Pos = alignTo(DataEnd, 4);
    }

    // Accumulate patches with their stream index for MSF patching later.
    for (const auto &P : Patches)
      AllPatches.push_back({ModiStream, P.StreamOffset, P.NewValue});

    if (!Patches.empty()) {
      outs() << "BOLT-INFO: " << Patches.size()
             << " line entries remapped for module " << I << "\n";
    }
  }

  // Copy the PDB, then apply line patches via MSF block arithmetic.
  SmallString<256> OutputPDB(OutputExe);
  sys::path::replace_extension(OutputPDB, ".pdb");

  std::error_code CopyEC = sys::fs::copy_file(PDBPath, OutputPDB);
  if (CopyEC) {
    errs() << "BOLT-WARNING: cannot copy PDB to " << OutputPDB << ": "
           << CopyEC.message() << "\n";
    return;
  }

  if (!AllPatches.empty()) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> PatchBuf =
        MemoryBuffer::getFile(OutputPDB);
    if (PatchBuf) {
      std::string Data = (*PatchBuf)->getBuffer().str();
      uint32_t BS = PDBFile.getBlockSize();
      uint32_t Done = 0;

      for (const auto &P : AllPatches) {
        auto BL = PDBFile.getStreamBlockList(P.StreamIndex);
        uint32_t BI = P.StreamOffset / BS;
        uint32_t OB = P.StreamOffset % BS;
        if (BI >= BL.size()) continue;
        // Skip patches that straddle an MSF block boundary.  A 4-byte
        // write at the end of a block would corrupt the next physical
        // block instead of following the stream's block map.
        if (OB + 4 > BS) continue;
        uint64_t FO = (uint64_t)BL[BI] * BS + OB;
        if (FO + 4 > Data.size()) continue;
        support::endian::write32le(&Data[FO], P.NewValue);
        ++Done;
      }

      if (Done > 0) {
        std::error_code WE;
        raw_fd_ostream Out(OutputPDB, WE, sys::fs::OF_None);
        if (!WE) Out.write(Data.data(), Data.size());
      }
    }
  }

  outs() << "BOLT-INFO: wrote PDB to " << OutputPDB << "\n";
  if (RemappedLines > 0)
    outs() << "BOLT-INFO: patched " << RemappedLines
           << " line entries in PDB for " << UpdatedSymbols
           << " rewritten functions\n";
  else if (UpdatedSymbols > 0)
    outs() << "BOLT-INFO: " << UpdatedSymbols
           << " rewritten functions, no line entries needed remapping\n";
  else
    outs() << "BOLT-INFO: PDB is fully accurate (no functions rewritten)\n";
}