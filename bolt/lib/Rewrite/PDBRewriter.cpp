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
#include "llvm/ADT/STLExtras.h"
#include "llvm/DebugInfo/CodeView/CVSymbolVisitor.h"
#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/DebugLinesSubsection.h"
#include "llvm/DebugInfo/CodeView/DebugSubsectionRecord.h"
#include "llvm/DebugInfo/CodeView/SymbolDeserializer.h"
#include "llvm/DebugInfo/CodeView/SymbolRecord.h"
#include "llvm/DebugInfo/MSF/MSFBuilder.h"
#include "llvm/DebugInfo/MSF/MSFCommon.h"
#include "llvm/DebugInfo/MSF/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Native/DbiModuleDescriptor.h"
#include "llvm/DebugInfo/PDB/Native/DbiStream.h"
#include "llvm/DebugInfo/PDB/Native/InfoStream.h"
#include "llvm/DebugInfo/PDB/Native/ModuleDebugStream.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/DebugInfo/PDB/Native/RawConstants.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/BinaryByteStream.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/BinaryStreamWriter.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Timer.h"
#include <array>
#include <map>
#include <optional>

#define DEBUG_TYPE "bolt-pdb"

using namespace llvm;
using namespace bolt;
using namespace codeview;

namespace opts {
extern cl::opt<bool> TimeRewrite;
}

namespace llvm {
namespace bolt {
namespace pdb_detail {

static std::optional<uint32_t>
mapLineOffset(uint32_t OldOffset,
              ArrayRef<std::pair<uint32_t, uint32_t>> OffsetMap) {
  auto Upper = llvm::upper_bound(
      OffsetMap, OldOffset,
      [](uint32_t Value, const std::pair<uint32_t, uint32_t> &Entry) {
        return Value < Entry.first;
      });
  if (Upper == OffsetMap.begin())
    return OldOffset;

  --Upper;
  return checkedAddUnsigned<uint32_t>(Upper->second, OldOffset - Upper->first);
}

bool rewriteLineSubsection(ArrayRef<uint8_t> Contents,
                           ArrayRef<std::pair<uint32_t, uint32_t>> OffsetMap,
                           std::vector<uint8_t> &Output,
                           uint32_t &NumRemapped) {
  Output.clear();
  NumRemapped = 0;
  if (Contents.size() < sizeof(LineFragmentHeader))
    return false;

  const uint16_t Flags = support::endian::read16le(
      Contents.data() + offsetof(LineFragmentHeader, Flags));
  const uint32_t CodeSize = support::endian::read32le(
      Contents.data() + offsetof(LineFragmentHeader, CodeSize));
  const bool HasColumns = Flags & uint16_t(LF_HaveColumns);
  const uint64_t RecordSize =
      sizeof(LineNumberEntry) +
      (HasColumns ? sizeof(ColumnNumberEntry) : uint64_t(0));

  struct LineRecord {
    std::array<uint8_t, sizeof(LineNumberEntry)> Line;
    std::array<uint8_t, sizeof(ColumnNumberEntry)> Column;
    uint32_t NewOffset;
  };

  std::vector<uint8_t> Rewritten(Contents.begin(), Contents.end());
  uint32_t Remapped = 0;
  uint64_t BlockPos = sizeof(LineFragmentHeader);
  while (BlockPos < Contents.size()) {
    auto HeaderEnd =
        checkedAddUnsigned<uint64_t>(BlockPos, sizeof(LineBlockFragmentHeader));
    if (!HeaderEnd || *HeaderEnd > Contents.size())
      return false;

    const uint8_t *Block = Contents.data() + BlockPos;
    const uint32_t NumLines = support::endian::read32le(
        Block + offsetof(LineBlockFragmentHeader, NumLines));
    const uint32_t BlockSize = support::endian::read32le(
        Block + offsetof(LineBlockFragmentHeader, BlockSize));
    auto ExpectedSize = checkedMulAddUnsigned<uint64_t>(
        NumLines, RecordSize, sizeof(LineBlockFragmentHeader));
    if (!ExpectedSize || *ExpectedSize > BlockSize)
      return false;

    auto BlockEnd = checkedAddUnsigned<uint64_t>(BlockPos, BlockSize);
    if (!BlockEnd || *BlockEnd > Contents.size())
      return false;
    auto RecordsEnd = checkedAddUnsigned<uint64_t>(BlockPos, *ExpectedSize);
    if (!RecordsEnd)
      return false;

    const uint64_t LinesPos = *HeaderEnd;
    auto LinesSize =
        checkedMulUnsigned<uint64_t>(NumLines, sizeof(LineNumberEntry));
    if (!LinesSize)
      return false;
    auto ColumnsPos = checkedAddUnsigned<uint64_t>(LinesPos, *LinesSize);
    if (!ColumnsPos || *ColumnsPos > *BlockEnd)
      return false;

    std::vector<LineRecord> Records;
    Records.reserve(NumLines);
    for (uint32_t Index = 0; Index < NumLines; ++Index) {
      const uint64_t LinePos =
          LinesPos + uint64_t(Index) * sizeof(LineNumberEntry);
      LineRecord Record{};
      llvm::copy(Contents.slice(LinePos, sizeof(LineNumberEntry)),
                 Record.Line.begin());
      const uint32_t OldOffset = support::endian::read32le(Record.Line.data());
      std::optional<uint32_t> NewOffset = mapLineOffset(OldOffset, OffsetMap);
      if (!NewOffset || *NewOffset >= CodeSize)
        return false;
      Record.NewOffset = *NewOffset;
      if (OldOffset != *NewOffset) {
        auto NewCount = checkedAddUnsigned<uint32_t>(Remapped, 1);
        if (!NewCount)
          return false;
        Remapped = *NewCount;
      }
      support::endian::write32le(Record.Line.data(), *NewOffset);

      if (HasColumns) {
        const uint64_t ColumnPos =
            *ColumnsPos + uint64_t(Index) * sizeof(ColumnNumberEntry);
        llvm::copy(Contents.slice(ColumnPos, sizeof(ColumnNumberEntry)),
                   Record.Column.begin());
      }
      Records.push_back(Record);
    }

    llvm::stable_sort(Records,
                      [](const LineRecord &Left, const LineRecord &Right) {
                        return Left.NewOffset < Right.NewOffset;
                      });

    uint64_t LinePos = LinesPos;
    uint64_t ColumnPos = *ColumnsPos;
    for (const LineRecord &Record : Records) {
      llvm::copy(Record.Line, Rewritten.begin() + LinePos);
      LinePos += sizeof(LineNumberEntry);
      if (HasColumns) {
        llvm::copy(Record.Column, Rewritten.begin() + ColumnPos);
        ColumnPos += sizeof(ColumnNumberEntry);
      }
    }
    if (LinePos != *ColumnsPos || ColumnPos != *RecordsEnd)
      return false;
    BlockPos = *BlockEnd;
  }

  if (BlockPos != Contents.size())
    return false;
  Output = std::move(Rewritten);
  NumRemapped = Remapped;
  return true;
}

} // namespace pdb_detail
} // namespace bolt
} // namespace llvm

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
    if (Data[0] == 'R' && Data[1] == 'S' && Data[2] == 'D' && Data[3] == 'S') {
      const char *PDBPath = reinterpret_cast<const char *>(Data + 4 + 16 + 4);
      size_t MaxPathLen = Entry.SizeOfData - 24;
      return std::string(PDBPath, strnlen(PDBPath, MaxPathLen));
    }
  }
  return {};
}

struct PDBPatch {
  uint16_t StreamIndex;
  uint64_t StreamOffset;
  std::vector<uint8_t> Data;
};

bool applyPDBPatch(MutableArrayRef<uint8_t> Buffer, const PDBPatch &Patch) {
  auto End =
      checkedAddUnsigned<uint64_t>(Patch.StreamOffset, Patch.Data.size());
  if (!End || *End > Buffer.size())
    return false;
  llvm::copy(Patch.Data, Buffer.begin() + Patch.StreamOffset);
  return true;
}

bool writeBytesThroughBlocks(MutableArrayRef<uint8_t> FileData,
                             ArrayRef<support::ulittle32_t> BlockList,
                             uint32_t BlockSize, uint64_t StreamSize,
                             uint64_t StreamOffset, ArrayRef<uint8_t> Bytes) {
  if (!BlockSize)
    return false;
  auto StreamEnd = checkedAddUnsigned<uint64_t>(StreamOffset, Bytes.size());
  if (!StreamEnd || *StreamEnd > StreamSize)
    return false;

  auto visitChunks = [&](auto Visit) {
    uint64_t Offset = StreamOffset;
    size_t BytesPos = 0;
    while (BytesPos < Bytes.size()) {
      const uint64_t BlockIndex = Offset / BlockSize;
      const uint64_t OffsetInBlock = Offset % BlockSize;
      if (BlockIndex >= BlockList.size())
        return false;
      const size_t ChunkSize = std::min<size_t>(
          Bytes.size() - BytesPos, BlockSize - size_t(OffsetInBlock));
      auto FileOffset = checkedMulAddUnsigned<uint64_t>(
          uint64_t(BlockList[BlockIndex]), BlockSize, OffsetInBlock);
      if (!FileOffset)
        return false;
      auto FileEnd = checkedAddUnsigned<uint64_t>(*FileOffset, ChunkSize);
      if (!FileEnd || *FileEnd > FileData.size())
        return false;
      Visit(*FileOffset, BytesPos, ChunkSize);
      Offset += ChunkSize;
      BytesPos += ChunkSize;
    }
    return true;
  };

  if (!visitChunks([](uint64_t, size_t, size_t) {}))
    return false;
  return visitChunks([&](uint64_t FileOffset, size_t BytesPos,
                         size_t ChunkSize) {
    llvm::copy(Bytes.slice(BytesPos, ChunkSize), FileData.begin() + FileOffset);
  });
}

/// Rebuild the PDB with OMAP address-translation tables so functions moved
/// out-of-place into the .bolt section resolve back to their original
/// symbols.  Copies every input stream verbatim, applies the line-table
/// patches, extends the DBI optional-debug-header to reference three new
/// streams (OmapToSrc, OmapFromSrc, and a section-header stream that includes
/// .bolt), and writes the result to \p OutputPDB.  Returns true on success.
bool rebuildPDBWithOmap(pdb::PDBFile &File, pdb::DbiStream &Dbi,
                        StringRef OutputPDB, const BinaryContext &BC,
                        ArrayRef<PDBPatch> LinePatches,
                        ArrayRef<BoltRelocatedFunc> RelocatedFuncs,
                        ArrayRef<uint32_t> AnchorRVAs, uint32_t BoltSectionRVA,
                        uint32_t BoltSectionSize) {
  NamedRegionTimer T("rebuildPDB", "rebuild PDB", "rewrite", "Rewrite passes",
                     opts::TimeRewrite);
  using namespace llvm::msf;
  const uint32_t BlockSize = File.getBlockSize();
  const uint32_t N = File.getNumStreams();
  const uint32_t StreamDBI = uint32_t(pdb::StreamDBI); // == 3
  // Diagnostic: BOLT_OMAP_DIAG=copyonly rebuilds the PDB via MSFBuilder but
  // WITHOUT adding OMAP streams / patching the DBI debug header, to isolate
  // whether the MSF copy itself or the OMAP wiring affects symbol resolution.
  const char *DiagEnv = std::getenv("BOLT_OMAP_DIAG");
  const bool CopyOnly = DiagEnv && StringRef(DiagEnv) == "copyonly";

  // To keep peak memory bounded on multi-GB PDBs, streams are not all buffered
  // up front.  Only modified streams are materialized (the DBI stream and any
  // line-patched module streams); every other stream is copied straight from
  // the input to the output one at a time at write time.
  auto readStreamBytes = [&](uint32_t I, std::vector<uint8_t> &Out) -> bool {
    Out.clear();
    uint32_t Size = File.getStreamByteSize(I);
    if (Size == UINT32_MAX || Size == 0)
      return true;
    auto S = File.createIndexedStream(I);
    if (!S)
      return false;
    BinaryStreamReader R(*S);
    ArrayRef<uint8_t> Bytes;
    if (Error E = R.readBytes(Bytes, Size)) {
      consumeError(std::move(E));
      return false;
    }
    Out.assign(Bytes.begin(), Bytes.end());
    return true;
  };

  // Modified stream buffers.
  std::vector<uint8_t> DbiBuf;
  if (StreamDBI >= N || !readStreamBytes(StreamDBI, DbiBuf) ||
      DbiBuf.size() < 64) {
    BC.errs() << "BOLT-WARNING: PDB has no DBI stream; skipping OMAP\n";
    return false;
  }

  // Stream 0 is the old MSF directory and references stale block numbers.
  // MSFBuilder writes a fresh directory, so lld (and we) emit it empty;
  // dbghelp mis-resolves symbols if the stale one is copied verbatim.
  auto streamSize = [&](uint32_t I) -> uint32_t {
    if (I == 0)
      return 0;
    if (I == StreamDBI)
      return uint32_t(DbiBuf.size());
    uint32_t S = File.getStreamByteSize(I);
    return S == UINT32_MAX ? 0 : S;
  };

  // Values produced by the OMAP construction below (skipped in copy-only diag).
  std::vector<uint8_t> NewSH, OmapFromBytes, OmapToBytes;
  uint32_t NewSectionHdrIdx = N, OmapToIdx = N + 1, OmapFromIdx = N + 2;

  if (!CopyOnly) {
    // 2. Locate the optional debug header (last substream of the DBI stream).
    auto rd32 = [&](size_t Off) {
      return support::endian::read32le(&DbiBuf[Off]);
    };
    const uint64_t ModiSize = rd32(24);
    const uint64_t SecContrSize = rd32(28);
    const uint64_t SecMapSize = rd32(32);
    const uint64_t FileInfoSize = rd32(36);
    const uint64_t TypeServerSize = rd32(40);
    uint64_t OptDbgSize = rd32(48);
    const uint64_t ECSize = rd32(52);
    const uint64_t OptDbgOff = 64 + ModiSize + SecContrSize + SecMapSize +
                               FileInfoSize + TypeServerSize + ECSize;
    if (OptDbgOff + OptDbgSize != DbiBuf.size()) {
      BC.errs() << "BOLT-WARNING: unexpected DBI layout; skipping OMAP\n";
      return false;
    }

    // Ensure the optional debug header has entries through SectionHdrOrig (10).
    const uint32_t NeedEntries =
        uint32_t(pdb::DbgHeaderType::SectionHdrOrig) + 1; // 11
    uint32_t NumDbg = uint32_t(OptDbgSize / 2);
    if (NumDbg < NeedEntries) {
      // The optional debug header is the final substream, so appending new
      // 0xFFFF (absent) entries at the end assigns them the next indices.
      DbiBuf.insert(DbiBuf.end(), (NeedEntries - NumDbg) * 2, 0xFF);
      OptDbgSize += (NeedEntries - NumDbg) * 2;
      support::endian::write32le(&DbiBuf[48], uint32_t(OptDbgSize));
      NumDbg = NeedEntries;
    }

    auto dbgSlot = [&](pdb::DbgHeaderType T) -> size_t {
      return OptDbgOff + size_t(T) * 2;
    };
    uint16_t OldSectionHdrIdx = support::endian::read16le(
        &DbiBuf[dbgSlot(pdb::DbgHeaderType::SectionHdr)]);
    if (OldSectionHdrIdx == 0xFFFF || OldSectionHdrIdx >= N) {
      BC.errs() << "BOLT-WARNING: PDB has no section-header stream; skipping "
                   "OMAP\n";
      return false;
    }

    // New streams are appended after the existing N streams, in this order.
    NewSectionHdrIdx = N;
    OmapToIdx = N + 1;
    OmapFromIdx = N + 2;
    auto setDbg = [&](pdb::DbgHeaderType T, uint16_t V) {
      support::endian::write16le(&DbiBuf[dbgSlot(T)], V);
    };
    setDbg(pdb::DbgHeaderType::OmapToSrc, uint16_t(OmapToIdx));
    setDbg(pdb::DbgHeaderType::OmapFromSrc, uint16_t(OmapFromIdx));
    setDbg(pdb::DbgHeaderType::SectionHdr, uint16_t(NewSectionHdrIdx));
    setDbg(pdb::DbgHeaderType::SectionHdrOrig, OldSectionHdrIdx);

    // 3. New section-header stream = original headers + a .bolt entry.
    if (!readStreamBytes(OldSectionHdrIdx, NewSH)) {
      BC.errs() << "BOLT-WARNING: cannot read section-header stream; skipping "
                   "OMAP\n";
      return false;
    }
    {
      uint8_t Sec[40] = {};
      std::memcpy(Sec, ".bolt\0\0\0", 8);
      support::endian::write32le(Sec + 8, BoltSectionSize); // VirtualSize
      support::endian::write32le(Sec + 12, BoltSectionRVA); // VirtualAddress
      support::endian::write32le(Sec + 16, alignTo(BoltSectionSize, 512u));
      support::endian::write32le(Sec + 36, 0x60000020); // CODE|EXECUTE|READ
      NewSH.insert(NewSH.end(), Sec, Sec + 40);
    }

    // 4. Build OMAP tables.  Identity anchor at the lowest section RVA so that
    //    unchanged/in-place code maps to itself (rvaTo==0 means "unmapped").
    uint32_t AnchorRVA = UINT32_MAX;
    for (const object::coff_section &Hdr : Dbi.getSectionHeaders())
      AnchorRVA = std::min(AnchorRVA, uint32_t(Hdr.VirtualAddress));
    if (AnchorRVA == UINT32_MAX)
      AnchorRVA = 0x1000;

    std::map<uint32_t, uint32_t> From; // original RVA -> new RVA
    std::map<uint32_t, uint32_t> To;   // new RVA -> original RVA
    From[AnchorRVA] = AnchorRVA;
    To[AnchorRVA] = AnchorRVA;
    for (const BoltRelocatedFunc &F : RelocatedFuncs) {
      From[F.OrigRVA] = F.NewRVA;
      From.emplace(F.OrigRVA + F.OrigSize, F.OrigRVA + F.OrigSize);
      To[F.NewRVA] = F.OrigRVA;
      To.emplace(F.NewRVA + F.NewSize,
                 0u); // padding after moved body: unmapped
    }
    // OMAP drops symbols in identity gaps. Merge procedure RVAs with explicit
    // mappings, with explicit mappings taking precedence.
    auto serialize = [](const std::map<uint32_t, uint32_t> &M,
                        ArrayRef<uint32_t> Anchors) {
      std::vector<uint8_t> B;
      B.reserve((M.size() + Anchors.size()) * 8);
      auto put = [&](uint32_t K, uint32_t V) {
        uint8_t E[8];
        support::endian::write32le(E, K);
        support::endian::write32le(E + 4, V);
        B.insert(B.end(), E, E + 8);
      };
      auto MI = M.begin();
      size_t AI = 0;
      while (MI != M.end() || AI < Anchors.size()) {
        if (MI != M.end() &&
            (AI >= Anchors.size() || MI->first < Anchors[AI])) {
          put(MI->first, MI->second);
          ++MI;
        } else if (AI < Anchors.size() &&
                   (MI == M.end() || Anchors[AI] < MI->first)) {
          put(Anchors[AI], Anchors[AI]); // identity anchor
          ++AI;
        } else { // equal keys: the map entry wins, drop the identity anchor
          put(MI->first, MI->second);
          ++MI;
          ++AI;
        }
      }
      return B;
    };
    OmapFromBytes = serialize(From, AnchorRVAs);
    OmapToBytes = serialize(To, AnchorRVAs);
  } // end if (!CopyOnly)

  // 5. Apply line-table patches.  Each patch mutates a module stream; load
  // just those streams into an overrides map (on-demand), leaving all other
  // streams to be copied lazily at write time.
  std::map<uint32_t, std::vector<uint8_t>> Modified;
  bool ReadFailed = false;
  auto getModified = [&](uint32_t I) -> std::vector<uint8_t> & {
    auto It = Modified.find(I);
    if (It == Modified.end()) {
      std::vector<uint8_t> Buf;
      if (!readStreamBytes(I, Buf))
        ReadFailed = true;
      It = Modified.emplace(I, std::move(Buf)).first;
    }
    return It->second;
  };
  for (const PDBPatch &P : LinePatches) {
    if (P.StreamIndex >= N || P.StreamIndex == 0 || P.StreamIndex == StreamDBI)
      continue;
    std::vector<uint8_t> &Buf = getModified(P.StreamIndex);
    if (!applyPDBPatch(Buf, P))
      BC.errs() << "BOLT-WARNING: cannot apply PDB line-table patch; leaving "
                   "subsection unchanged\n";
  }
  if (ReadFailed) {
    BC.errs()
        << "BOLT-WARNING: cannot read patched PDB stream; skipping OMAP\n";
    return false;
  }

  // 6. Lay out and write the new MSF file.
  BumpPtrAllocator Alloc;
  auto MsfOrErr = MSFBuilder::create(Alloc, BlockSize);
  if (!MsfOrErr) {
    consumeError(MsfOrErr.takeError());
    BC.errs() << "BOLT-WARNING: cannot create MSF builder; skipping OMAP\n";
    return false;
  }
  MSFBuilder &Msf = *MsfOrErr;
  auto addStream = [&](uint32_t Size) -> bool {
    auto E = Msf.addStream(Size);
    if (!E) {
      consumeError(E.takeError());
      return false;
    }
    return true;
  };
  bool Ok = true;
  for (uint32_t I = 0; I < N && Ok; ++I)
    Ok = addStream(streamSize(I));
  if (!CopyOnly)
    Ok = Ok && addStream(uint32_t(NewSH.size())) &&
         addStream(uint32_t(OmapToBytes.size())) &&
         addStream(uint32_t(OmapFromBytes.size()));
  if (!Ok) {
    BC.errs() << "BOLT-WARNING: cannot add PDB streams; skipping OMAP\n";
    return false;
  }

  auto LayoutOrErr = Msf.generateLayout();
  if (!LayoutOrErr) {
    consumeError(LayoutOrErr.takeError());
    BC.errs() << "BOLT-WARNING: cannot generate MSF layout; skipping OMAP\n";
    return false;
  }
  MSFLayout Layout = std::move(*LayoutOrErr);
  auto BufOrErr = Msf.commit(OutputPDB, Layout);
  if (!BufOrErr) {
    consumeError(BufOrErr.takeError());
    BC.errs() << "BOLT-WARNING: cannot commit MSF file; skipping OMAP\n";
    return false;
  }
  FileBufferByteStream Buffer = std::move(*BufOrErr);

  auto writeStream = [&](uint32_t Idx, ArrayRef<uint8_t> Data) -> bool {
    if (Data.empty())
      return true;
    auto WS = WritableMappedBlockStream::createIndexedStream(Layout, Buffer,
                                                             Idx, Alloc);
    BinaryStreamWriter W(*WS);
    if (Error E = W.writeBytes(Data)) {
      consumeError(std::move(E));
      return false;
    }
    return true;
  };
  // Copy stream data into the committed file.  Modified streams write from
  // their buffers; every other stream is streamed from the input one at a
  // time and then released.  A read failure would leave a stream zero-filled,
  // so abort rather than emit a corrupt PDB.
  std::vector<uint8_t> Tmp;
  for (uint32_t I = 0; I < N; ++I) {
    if (I == 0)
      continue;
    if (I == StreamDBI) {
      writeStream(I, DbiBuf);
      continue;
    }
    auto MIt = Modified.find(I);
    if (MIt != Modified.end()) {
      writeStream(I, MIt->second);
      continue;
    }
    if (!readStreamBytes(I, Tmp)) {
      BC.errs() << "BOLT-WARNING: cannot read PDB stream " << I
                << "; skipping OMAP\n";
      return false;
    }
    writeStream(I, Tmp);
  }
  if (!CopyOnly) {
    writeStream(NewSectionHdrIdx, NewSH);
    writeStream(OmapToIdx, OmapToBytes);
    writeStream(OmapFromIdx, OmapFromBytes);
  }

  if (Error E = Buffer.commit()) {
    consumeError(std::move(E));
    BC.errs() << "BOLT-WARNING: cannot flush rebuilt PDB\n";
    return false;
  }
  return true;
}

} // namespace

void PDBRewriter::rewritePDB(StringRef InputExe, StringRef OutputExe,
                             const BinaryContext &BC, uint64_t ImageBase,
                             const DenseSet<uint64_t> &ModifiedFunctions,
                             const DenseMap<uint64_t, OffsetMap> &OffsetMaps,
                             ArrayRef<BoltRelocatedFunc> RelocatedFuncs,
                             uint32_t BoltSectionRVA,
                             uint32_t BoltSectionSize) {

  // Find the PDB file.
  std::string PDBPath = findPDBPath(InputExe);
  if (PDBPath.empty()) {
    BC.outs() << "BOLT-INFO: no PDB found, skipping debug info update\n";
    return;
  }

  if (!sys::fs::exists(PDBPath)) {
    // Try the PDB next to the input binary (common when copying both files).
    SmallString<256> Adjacent(sys::path::parent_path(InputExe));
    sys::path::append(Adjacent, sys::path::filename(PDBPath));
    if (sys::fs::exists(Adjacent)) {
      PDBPath = std::string(Adjacent);
    } else {
      BC.outs() << "BOLT-WARNING: PDB file " << PDBPath
                << " not found, debug info will be stale\n";
      return;
    }
  }

  BC.outs() << "BOLT-INFO: updating PDB " << PDBPath << "\n";

  // Open the PDB.
  ErrorOr<std::unique_ptr<MemoryBuffer>> PDBBuf =
      MemoryBuffer::getFile(PDBPath);
  if (!PDBBuf) {
    BC.errs() << "BOLT-WARNING: cannot open PDB: "
              << PDBBuf.getError().message() << "\n";
    return;
  }

  BumpPtrAllocator Alloc;
  auto Stream = std::make_unique<MemoryBufferByteStream>(
      std::move(*PDBBuf), llvm::endianness::little);
  pdb::PDBFile PDBFile(PDBPath, std::move(Stream), Alloc);
  if (Error E = PDBFile.parseFileHeaders()) {
    BC.errs() << "BOLT-WARNING: cannot parse PDB headers: "
              << toString(std::move(E)) << "\n";
    return;
  }
  if (Error E = PDBFile.parseStreamData()) {
    BC.errs() << "BOLT-WARNING: cannot parse PDB streams: "
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
    BC.errs() << "BOLT-WARNING: cannot read DBI stream: "
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

  SmallVector<PDBPatch, 16> AllPatches;
  std::vector<uint32_t> AnchorRVAs;

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

      if (!RelocatedFuncs.empty() && FuncVA)
        AnchorRVAs.push_back(SectionRVAs[ProcOrErr->Segment - 1] +
                             ProcOrErr->CodeOffset);

      if (!ModifiedFunctions.count(FuncVA))
        continue;

      const BinaryFunction *BF = BC.getBinaryFunctionAtAddress(FuncVA);
      if (!BF)
        continue;

      // Note: For in-place PE/COFF rewriting, we do NOT patch CodeSize.
      // The function still occupies the same memory footprint; only the
      // internal block order changed.  The original CodeSize remains valid.

      LLVM_DEBUG(dbgs() << "BOLT-DEBUG: PDB function " << ProcOrErr->Name
                        << " at VA 0x" << Twine::utohexstr(FuncVA)
                        << " was rewritten\n");
      if (opts::Verbosity >= 1)
        BC.outs() << "BOLT-INFO: function " << ProcOrErr->Name
                  << " was rewritten, line info may be inaccurate\n";
      ++UpdatedSymbols;
    }

    // Scan C13 line info subsections for rewritten functions.
    SmallVector<PDBPatch, 8> Patches;

    // Walk the raw C13 bytes to track exact positions.
    // Layout: each subsection has {uint32 Kind, uint32 Length, [data]}.
    // Line fragment data: {uint32 RelocOffset, uint16 Segment,
    //   uint16 Flags, uint32 CodeSize}, then file blocks with line entries.
    BinarySubstreamRef C13Ref = ModStream.getC13LinesSubstream();
    uint64_t C13StreamBase = C13Ref.Offset; // offset of C13 data in stream

    ArrayRef<uint8_t> C13Bytes;
    if (Error E = C13Ref.StreamData.readBytes(0, C13Ref.StreamData.getLength(),
                                              C13Bytes)) {
      consumeError(std::move(E));
      continue;
    }

    uint64_t Pos = 0;
    auto warnLineSubsection = [&] {
      BC.errs() << "BOLT-WARNING: cannot rewrite PDB line subsection for "
                   "module "
                << I << "; leaving it unchanged\n";
    };
    while (Pos + 8 <= C13Bytes.size()) {
      uint32_t Kind = support::endian::read32le(&C13Bytes[Pos]);
      uint32_t Length = support::endian::read32le(&C13Bytes[Pos + 4]);
      uint64_t DataStart = Pos + 8;
      auto DataEnd = checkedAddUnsigned<uint64_t>(DataStart, Length);

      if (!DataEnd || *DataEnd > C13Bytes.size()) {
        if (Kind == uint32_t(DebugSubsectionKind::Lines))
          warnLineSubsection();
        break;
      }

      if (Kind != uint32_t(DebugSubsectionKind::Lines)) {
        Pos = alignTo(*DataEnd, 4);
        continue;
      }

      // Line fragment header at DataStart:
      // {uint32 RelocOffset, uint16 RelocSegment, uint16 Flags, uint32
      // CodeSize}
      if (Length < 12) {
        warnLineSubsection();
        Pos = alignTo(*DataEnd, 4);
        continue;
      }
      uint32_t RelocOffset = support::endian::read32le(&C13Bytes[DataStart]);
      uint16_t RelocSegment =
          support::endian::read16le(&C13Bytes[DataStart + 4]);
      uint64_t FuncVA = computeVA(RelocSegment, RelocOffset);

      auto MapIt = OffsetMaps.find(FuncVA);
      if (MapIt == OffsetMaps.end()) {
        Pos = alignTo(*DataEnd, 4);
        continue;
      }

      ArrayRef<uint8_t> Contents = C13Bytes.slice(DataStart, Length);
      std::vector<uint8_t> Rewritten;
      uint32_t SubsectionRemapped = 0;
      if (!pdb_detail::rewriteLineSubsection(Contents, MapIt->second, Rewritten,
                                             SubsectionRemapped)) {
        warnLineSubsection();
        Pos = alignTo(*DataEnd, 4);
        continue;
      }

      if (!std::equal(Contents.begin(), Contents.end(), Rewritten.begin())) {
        auto StreamOffset =
            checkedAddUnsigned<uint64_t>(C13StreamBase, DataStart);
        if (!StreamOffset) {
          warnLineSubsection();
          Pos = alignTo(*DataEnd, 4);
          continue;
        }
        Patches.push_back({ModiStream, *StreamOffset, std::move(Rewritten)});
      }
      RemappedLines += SubsectionRemapped;
      Pos = alignTo(*DataEnd, 4);
    }

    if (!Patches.empty() && opts::Verbosity >= 1) {
      BC.outs() << "BOLT-INFO: " << Patches.size()
                << " line subsections rewritten for module " << I << "\n";
    }
    llvm::append_range(AllPatches, std::move(Patches));
  }
  if (!RelocatedFuncs.empty()) {
    llvm::sort(AnchorRVAs);
    AnchorRVAs.erase(llvm::unique(AnchorRVAs), AnchorRVAs.end());
  }

  // Copy the PDB, then apply line patches via MSF block arithmetic.
  SmallString<256> OutputPDB(OutputExe);
  sys::path::replace_extension(OutputPDB, ".pdb");

  // When functions have been relocated out-of-place into .bolt, rebuild the
  // PDB with OMAP address-translation tables so their moved code still
  // resolves back to the original symbols.
  if (!RelocatedFuncs.empty()) {
    if (rebuildPDBWithOmap(PDBFile, Dbi, OutputPDB, BC, AllPatches,
                           RelocatedFuncs, AnchorRVAs, BoltSectionRVA,
                           BoltSectionSize)) {
      BC.outs() << "BOLT-INFO: wrote PDB with OMAP tables to " << OutputPDB
                << " (" << RelocatedFuncs.size() << " relocated functions, "
                << AllPatches.size() << " line-table patches)\n";
      return;
    }
    BC.errs() << "BOLT-WARNING: OMAP rebuild failed; falling back to "
                 "copy+patch (OOP symbols will be stale)\n";
  }

  std::error_code CopyEC = sys::fs::copy_file(PDBPath, OutputPDB);
  if (CopyEC) {
    BC.errs() << "BOLT-WARNING: cannot copy PDB to " << OutputPDB << ": "
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
      MutableArrayRef<uint8_t> FileData(
          reinterpret_cast<uint8_t *>(Data.data()), Data.size());

      for (const auto &P : AllPatches) {
        auto BL = PDBFile.getStreamBlockList(P.StreamIndex);
        const uint32_t StreamSize = PDBFile.getStreamByteSize(P.StreamIndex);
        if (StreamSize != UINT32_MAX &&
            writeBytesThroughBlocks(FileData, BL, BS, StreamSize,
                                    P.StreamOffset, P.Data))
          ++Done;
        else
          BC.errs() << "BOLT-WARNING: cannot apply PDB line-table patch; "
                       "leaving subsection unchanged\n";
      }

      if (Done > 0) {
        std::error_code WE;
        raw_fd_ostream Out(OutputPDB, WE, sys::fs::OF_None);
        if (!WE)
          Out.write(Data.data(), Data.size());
      }
    }
  }

  BC.outs() << "BOLT-INFO: wrote PDB to " << OutputPDB << "\n";
  if (RemappedLines > 0)
    BC.outs() << "BOLT-INFO: patched " << RemappedLines
              << " line entries in PDB for " << UpdatedSymbols
              << " in-place rewritten functions\n";
  else if (UpdatedSymbols > 0)
    BC.outs()
        << "BOLT-INFO: " << UpdatedSymbols
        << " rewritten functions found in PDB (no line remapping needed)\n";
  else
    BC.outs() << "BOLT-INFO: PDB unchanged (no rewritten functions found)\n";
}
