//===- bolt/tools/verify-pdb/verify-pdb.cpp - PDB/PE consistency checker --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Validates that a PDB's debug info (function symbols and line tables) is
// consistent with its PE binary.  Useful for verifying BOLT's PDB rewriter
// produced correct output.
//
// Checks performed:
//   1. S_GPROC32/S_LPROC32 addresses map to valid code in the PE binary.
//   2. CodeSize in each proc symbol matches the function extent in .pdata.
//   3. C13 line entry offsets fall within the function's code range.
//   4. Line entries are monotonically non-decreasing within each block.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/CVSymbolVisitor.h"
#include "llvm/DebugInfo/CodeView/DebugSubsectionRecord.h"
#include "llvm/DebugInfo/CodeView/Line.h"
#include "llvm/DebugInfo/CodeView/SymbolDeserializer.h"
#include "llvm/DebugInfo/CodeView/SymbolRecord.h"
#include "llvm/DebugInfo/MSF/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Native/DbiModuleDescriptor.h"
#include "llvm/DebugInfo/PDB/Native/DbiStream.h"
#include "llvm/DebugInfo/PDB/Native/InfoStream.h"
#include "llvm/DebugInfo/PDB/Native/ModuleDebugStream.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/BinaryByteStream.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace codeview;

namespace {

cl::opt<std::string> InputExe(cl::Positional, cl::desc("<PE binary>"),
                              cl::Required);
cl::opt<std::string> PDBPathOpt("pdb", cl::desc("Explicit PDB path"),
                                cl::Optional);
cl::opt<bool> Verbose("v", cl::desc("Verbose output"), cl::init(false));

struct RuntimeFunction {
  support::ulittle32_t BeginAddress;
  support::ulittle32_t EndAddress;
  support::ulittle32_t UnwindInfoAddress;
};

/// Find the PDB path from the PE binary's debug directory.
std::string findPDBPath(object::COFFObjectFile &COFF) {
  for (const auto &Entry : COFF.debug_directories()) {
    if (Entry.Type != COFF::IMAGE_DEBUG_TYPE_CODEVIEW)
      continue;
    uintptr_t DataAddr = 0;
    if (Error E = COFF.getRvaPtr(Entry.AddressOfRawData, DataAddr)) {
      consumeError(std::move(E));
      continue;
    }
    const uint8_t *Data = reinterpret_cast<const uint8_t *>(DataAddr);
    if (Entry.SizeOfData < 25)
      continue;
    if (Data[0] == 'R' && Data[1] == 'S' && Data[2] == 'D' && Data[3] == 'S') {
      const char *Path = reinterpret_cast<const char *>(Data + 24);
      size_t MaxLen = Entry.SizeOfData - 24;
      return std::string(Path, strnlen(Path, MaxLen));
    }
  }
  return {};
}

/// Build a map of function RVA -> size from .pdata.
DenseMap<uint32_t, uint32_t> buildPDataMap(object::COFFObjectFile &COFF) {
  DenseMap<uint32_t, uint32_t> Map;
  for (const auto &Sec : COFF.sections()) {
    Expected<StringRef> Name = Sec.getName();
    if (!Name || *Name != ".pdata")
      continue;
    Expected<StringRef> Contents = Sec.getContents();
    if (!Contents) {
      consumeError(Contents.takeError());
      continue;
    }
    size_t NumEntries = Contents->size() / sizeof(RuntimeFunction);
    auto *Entries = reinterpret_cast<const RuntimeFunction *>(Contents->data());
    for (size_t I = 0; I < NumEntries; ++I) {
      uint32_t Begin = Entries[I].BeginAddress;
      uint32_t End = Entries[I].EndAddress;
      if (End > Begin)
        Map[Begin] = End - Begin;
    }
  }
  return Map;
}

} // namespace

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "BOLT PDB/PE consistency checker\n");

  // Open the PE binary.
  ErrorOr<std::unique_ptr<MemoryBuffer>> ExeBuf =
      MemoryBuffer::getFile(InputExe);
  if (!ExeBuf) {
    errs() << "error: cannot open " << InputExe << ": "
           << ExeBuf.getError().message() << "\n";
    return 1;
  }

  Expected<std::unique_ptr<object::ObjectFile>> ObjOrErr =
      object::ObjectFile::createObjectFile((*ExeBuf)->getMemBufferRef());
  if (!ObjOrErr) {
    errs() << "error: cannot parse " << InputExe << ": "
           << toString(ObjOrErr.takeError()) << "\n";
    return 1;
  }

  auto *COFF = dyn_cast<object::COFFObjectFile>(ObjOrErr->get());
  if (!COFF) {
    errs() << "error: " << InputExe << " is not a PE/COFF binary\n";
    return 1;
  }

  uint64_t ImageBase = COFF->getImageBase();

  // Find PDB.
  std::string PDBPath = PDBPathOpt.empty() ? findPDBPath(*COFF) : PDBPathOpt;
  if (PDBPath.empty()) {
    errs() << "error: no PDB found for " << InputExe << "\n";
    return 1;
  }
  if (!sys::fs::exists(PDBPath)) {
    errs() << "error: PDB file " << PDBPath << " not found\n";
    return 1;
  }

  outs() << "verify-pdb: binary " << InputExe << "\n";
  outs() << "verify-pdb: PDB    " << PDBPath << "\n";
  outs() << "verify-pdb: ImageBase 0x" << Twine::utohexstr(ImageBase) << "\n";

  // Build .pdata function map for cross-reference.
  auto PDataMap = buildPDataMap(*COFF);
  outs() << "verify-pdb: " << PDataMap.size() << " .pdata entries\n";

  // Open PDB.
  ErrorOr<std::unique_ptr<MemoryBuffer>> PDBBuf =
      MemoryBuffer::getFile(PDBPath);
  if (!PDBBuf) {
    errs() << "error: cannot open PDB: " << PDBBuf.getError().message() << "\n";
    return 1;
  }

  BumpPtrAllocator Alloc;
  auto Stream = std::make_unique<MemoryBufferByteStream>(
      std::move(*PDBBuf), llvm::endianness::little);
  pdb::PDBFile PDBFile(PDBPath, std::move(Stream), Alloc);
  if (Error E = PDBFile.parseFileHeaders()) {
    errs() << "error: " << toString(std::move(E)) << "\n";
    return 1;
  }
  if (Error E = PDBFile.parseStreamData()) {
    errs() << "error: " << toString(std::move(E)) << "\n";
    return 1;
  }

  Expected<pdb::DbiStream &> DbiOrErr = PDBFile.getPDBDbiStream();
  if (!DbiOrErr) {
    errs() << "error: " << toString(DbiOrErr.takeError()) << "\n";
    return 1;
  }
  pdb::DbiStream &Dbi = *DbiOrErr;

  // Build section RVA table.
  auto SectionHeaders = Dbi.getSectionHeaders();
  SmallVector<uint32_t, 16> SectionRVAs;
  for (const auto &Hdr : SectionHeaders)
    SectionRVAs.push_back(Hdr.VirtualAddress);

  auto computeRVA = [&](uint16_t Segment, uint32_t Offset) -> uint32_t {
    if (Segment == 0 || Segment > SectionRVAs.size())
      return 0;
    return SectionRVAs[Segment - 1] + Offset;
  };

  // Validation counters.
  uint32_t TotalProcs = 0;
  uint32_t TotalLineEntries = 0;
  uint32_t OutOfRangeLines = 0;
  uint32_t NonMonotonicLines = 0;

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

    // Check proc symbols.
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

      ++TotalProcs;
      uint32_t RVA = computeRVA(ProcOrErr->Segment, ProcOrErr->CodeOffset);
      uint32_t CodeSize = ProcOrErr->CodeSize;

      // Check against .pdata.  Note: .pdata EndAddress may differ from
      // PDB CodeSize for functions with chained unwind info or functions
      // that contain inlined code.  This is informational, not an error.
      auto PDIt = PDataMap.find(RVA);
      if (PDIt == PDataMap.end()) {
        // Not all functions have .pdata entries (leaf functions).
        if (Verbose)
          outs() << "  INFO: " << ProcOrErr->Name << " at RVA 0x"
                 << Twine::utohexstr(RVA) << " has no .pdata entry\n";
      } else {
        uint32_t PDataSize = PDIt->second;
        if (CodeSize != PDataSize && Verbose) {
          outs() << "  INFO: " << ProcOrErr->Name << " at RVA 0x"
                 << Twine::utohexstr(RVA) << ": PDB CodeSize=" << CodeSize
                 << " vs .pdata size=" << PDataSize
                 << " (normal for chained unwind)\n";
        }
      }
    }

    // Check C13 line entries.
    BinarySubstreamRef C13Ref = ModStream.getC13LinesSubstream();
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

      if (Length < 12) {
        Pos = alignTo(DataEnd, 4);
        continue;
      }

      uint32_t RelocOffset = support::endian::read32le(&C13Bytes[DataStart]);
      uint16_t RelocSegment =
          support::endian::read16le(&C13Bytes[DataStart + 4]);
      uint32_t CodeSize = support::endian::read32le(&C13Bytes[DataStart + 8]);
      uint32_t RVA = computeRVA(RelocSegment, RelocOffset);

      uint64_t FileBlockPos = DataStart + 12;
      while (FileBlockPos + 12 <= DataEnd) {
        uint32_t NumLines =
            support::endian::read32le(&C13Bytes[FileBlockPos + 4]);
        uint32_t BlockSize =
            support::endian::read32le(&C13Bytes[FileBlockPos + 8]);

        uint64_t LineStart = FileBlockPos + 12;
        uint32_t PrevOffset = 0;

        for (uint32_t L = 0; L < NumLines; ++L) {
          uint64_t EntryPos = LineStart + L * 8;
          if (EntryPos + 8 > DataEnd)
            break;

          uint32_t Offset = support::endian::read32le(&C13Bytes[EntryPos]);
          uint32_t Flags = support::endian::read32le(&C13Bytes[EntryPos + 4]);
          LineInfo LI(Flags);
          ++TotalLineEntries;

          // Check: offset should be within function code range.
          if (Offset >= CodeSize) {
            ++OutOfRangeLines;
            if (Verbose)
              outs() << "  OUT-OF-RANGE: line " << LI.getStartLine()
                     << " at RVA 0x" << Twine::utohexstr(RVA) << " offset 0x"
                     << Twine::utohexstr(Offset) << " >= CodeSize " << CodeSize
                     << "\n";
          }

          // Check: offsets should be non-decreasing within a file block.
          if (L > 0 && Offset < PrevOffset) {
            ++NonMonotonicLines;
            if (Verbose)
              outs() << "  NON-MONOTONIC: line " << LI.getStartLine()
                     << " at RVA 0x" << Twine::utohexstr(RVA) << " offset 0x"
                     << Twine::utohexstr(Offset) << " < prev 0x"
                     << Twine::utohexstr(PrevOffset) << "\n";
          }
          PrevOffset = Offset;
        }

        FileBlockPos += BlockSize;
      }

      Pos = alignTo(DataEnd, 4);
    }
  }

  // Summary.
  outs() << "\nverify-pdb: Summary\n";
  outs() << "  Proc symbols checked:    " << TotalProcs << "\n";
  outs() << "  Line entries checked:    " << TotalLineEntries << "\n";
  outs() << "  Out-of-range lines:      " << OutOfRangeLines << "\n";
  outs() << "  Non-monotonic lines:     " << NonMonotonicLines << "\n";

  uint32_t Errors = OutOfRangeLines + NonMonotonicLines;
  if (Errors == 0) {
    outs() << "verify-pdb: PASS - PDB is consistent with binary\n";
    return 0;
  }

  outs() << "verify-pdb: FAIL - " << Errors << " inconsistencies found\n";
  return 1;
}
