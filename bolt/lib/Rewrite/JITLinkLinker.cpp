//===- bolt/Rewrite/JITLinkLinker.cpp - BOLTLinker using JITLink ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Rewrite/JITLinkLinker.h"
#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryData.h"
#include "bolt/Core/BinarySection.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ExecutionEngine/JITLink/COFF_x86_64.h"
#include "llvm/ExecutionEngine/JITLink/ELF_riscv.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/ExecutionEngine/JITLink/x86_64.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "bolt"

namespace llvm {
namespace bolt {

namespace {

/// Liveness in JITLink is based on symbols so sections that do not contain
/// any symbols will always be pruned. This pass adds anonymous symbols to
/// needed sections to prevent pruning.
Error markSectionsLive(jitlink::LinkGraph &G) {
  // Collect all blocks that already own at least one symbol.  Without this
  // pre-pass we would call any_of(Section.symbols()) for every block, which
  // is O(blocks * symbols-per-section) and blows up on large COFF objects
  // where BOLT creates one section per function.
  DenseSet<const jitlink::Block *> HasSymbol;
  for (auto *Sym : G.defined_symbols())
    HasSymbol.insert(&Sym->getBlock());

  for (auto &Section : G.sections()) {
    if (Section.getMemLifetime() == orc::MemLifetime::NoAlloc)
      continue;

    if (JITLinkLinker::sectionSize(Section) == 0)
      continue;

    for (auto *Block : Section.blocks()) {
      if (HasSymbol.contains(Block))
        continue;

      G.addAnonymousSymbol(*Block, /*Offset=*/0, /*Size=*/0,
                           /*IsCallable=*/false, /*IsLive=*/true);
    }
  }

  return jitlink::markAllSymbolsLive(G);
}

void reassignSectionAddress(jitlink::LinkGraph &LG,
                            const BinarySection &BinSection, uint64_t Address) {
  auto *JLSection = LG.findSectionByName(BinSection.getSectionID());
  assert(JLSection && "cannot find section in LinkGraph");

  auto BlockAddress = Address;
  for (auto *Block : JITLinkLinker::orderedBlocks(*JLSection)) {
    // FIXME it would seem to make sense to align here. However, in
    // non-relocation mode, we simply use the original address of functions
    // which might not be aligned with the minimum alignment used by
    // BinaryFunction (2). Example failing test when aligning:
    // bolt/test/X86/addr32.s
    Block->setAddress(orc::ExecutorAddr(BlockAddress));
    BlockAddress += Block->getSize();
  }
}

} // anonymous namespace

struct JITLinkLinker::Context : jitlink::JITLinkContext {
  JITLinkLinker &Linker;
  JITLinkLinker::SectionsMapper MapSections;

  Context(JITLinkLinker &Linker, JITLinkLinker::SectionsMapper MapSections)
      : JITLinkContext(&Linker.Dylib), Linker(Linker),
        MapSections(MapSections) {}

  jitlink::JITLinkMemoryManager &getMemoryManager() override {
    return *Linker.MM;
  }

  bool shouldAddDefaultTargetPasses(const Triple &TT) const override {
    // The default passes manipulate DWARF sections in a way incompatible with
    // BOLT.
    // TODO check if we can actually use these passes to remove some of the
    // DWARF manipulation done in BOLT.
    return false;
  }

  Error modifyPassConfig(jitlink::LinkGraph &G,
                         jitlink::PassConfiguration &Config) override {
    Config.PrePrunePasses.push_back(markSectionsLive);
    Config.PostAllocationPasses.push_back([this](auto &G) {
      MapSections([&G](const BinarySection &Section, uint64_t Address) {
        reassignSectionAddress(G, Section, Address);
      });
      return Error::success();
    });

    if (G.getTargetTriple().isRISCV()) {
      Config.PostAllocationPasses.push_back(
          jitlink::createRelaxationPass_ELF_riscv());
    }

    // COFF objects use platform-specific edge kinds that must be lowered to
    // generic x86_64 kinds before the fixup phase.  The default JITLink passes
    // normally handle this, but BOLT disables them to avoid DWARF conflicts.
    // We add a minimal lowering pass here that covers the relocations BOLT
    // actually emits (PCRel32 and Pointer64).
    if (G.getTargetTriple().isOSBinFormatCOFF()) {
      Config.PreFixupPasses.push_back([](jitlink::LinkGraph &G) -> Error {
        using namespace jitlink;
        // COFF edge kinds start at x86_64::FirstPlatformRelocation.
        // These correspond to EdgeKind_coff_x86_64 in COFF_x86_64.cpp:
        //   PCRel32     = FirstPlatformRelocation + 0
        //   Pointer32NB = FirstPlatformRelocation + 1
        //   Pointer64   = FirstPlatformRelocation + 2
        //   SectionIdx16= FirstPlatformRelocation + 3
        //   SecRel32    = FirstPlatformRelocation + 4
        const Edge::Kind COFFPCRel32 = x86_64::FirstPlatformRelocation;
        const Edge::Kind COFFPointer32NB = x86_64::FirstPlatformRelocation + 1;
        const Edge::Kind COFFPointer64 = x86_64::FirstPlatformRelocation + 2;
        const Edge::Kind COFFSectionIdx16 = x86_64::FirstPlatformRelocation + 3;
        const Edge::Kind COFFSecRel32 = x86_64::FirstPlatformRelocation + 4;

        for (auto *B : G.blocks()) {
          for (auto &E : B->edges()) {
            switch (E.getKind()) {
            case COFFPCRel32:
              E.setKind(x86_64::PCRel32);
              break;
            case COFFPointer64:
              E.setKind(x86_64::Pointer64);
              break;
            case COFFPointer32NB:
              // Pointer32NB is image-base-relative.  Without an __ImageBase
              // symbol we cannot lower it properly, but BOLT does not emit
              // this relocation type so just leave it for now.
              // TODO: handle Pointer32NB if BOLT ever emits ADDR32NB relocs
              break;
            case COFFSectionIdx16:
              E.setKind(x86_64::Pointer16);
              break;
            case COFFSecRel32:
              // SecRel32 needs section-start subtraction which we do not
              // have here.  Not emitted by BOLT today.
              // TODO: handle SecRel32 if needed
              break;
            default:
              break;
            }
          }
        }
        return Error::success();
      });
    }

    return Error::success();
  }

  void notifyFailed(Error Err) override {
    errs() << "BOLT-ERROR: JITLink failed: " << Err << '\n';
    exit(1);
  }

  void
  lookup(const LookupMap &Symbols,
         std::unique_ptr<jitlink::JITLinkAsyncLookupContinuation> LC) override {
    jitlink::AsyncLookupResult AllResults;

    for (const auto &Symbol : Symbols) {
      std::string SymName = (*Symbol.first).str();
      LLVM_DEBUG(dbgs() << "BOLT: looking for " << SymName << "\n");

      if (auto SymInfo = Linker.lookupSymbolInfo(SymName)) {
        LLVM_DEBUG(dbgs() << "Resolved to address 0x"
                          << Twine::utohexstr(SymInfo->Address) << "\n");
        AllResults[Symbol.first] = orc::ExecutorSymbolDef(
            orc::ExecutorAddr(SymInfo->Address), JITSymbolFlags());
        continue;
      }

      if (const BinaryData *I = Linker.BC.getBinaryDataByName(SymName)) {
        uint64_t Address = I->isMoved() && !I->isJumpTable()
                               ? I->getOutputAddress()
                               : I->getAddress();
        LLVM_DEBUG(dbgs() << "Resolved to address 0x"
                          << Twine::utohexstr(Address) << "\n");
        AllResults[Symbol.first] = orc::ExecutorSymbolDef(
            orc::ExecutorAddr(Address), JITSymbolFlags());
        continue;
      }

      if (Linker.BC.isGOTSymbol(SymName)) {
        if (const BinaryData *I = Linker.BC.getGOTSymbol()) {
          uint64_t Address =
              I->isMoved() ? I->getOutputAddress() : I->getAddress();
          LLVM_DEBUG(dbgs() << "Resolved to address 0x"
                            << Twine::utohexstr(Address) << "\n");
          AllResults[Symbol.first] = orc::ExecutorSymbolDef(
              orc::ExecutorAddr(Address), JITSymbolFlags());
          continue;
        }
      }

      LLVM_DEBUG(dbgs() << "Resolved to address 0x0\n");
      // BOLT creates symbols like FUNCat0x<addr> and DATAat0x<addr> for
      // references into the original binary.  If the symbol was not registered
      // in BinaryData (e.g. a call target that is not a .pdata function), try
      // to parse the address from the name as a last resort.
      uint64_t ParsedAddr = 0;
      StringRef SN(SymName);
      // Look for the "0x" hex prefix that encodes the virtual address.
      size_t HexPos = SN.find("0x");
      if (HexPos != StringRef::npos &&
          !SN.substr(HexPos + 2).getAsInteger(16, ParsedAddr) &&
          ParsedAddr != 0) {
        LLVM_DEBUG(dbgs() << "BOLT: parsed address 0x"
                          << Twine::utohexstr(ParsedAddr)
                          << " from symbol name " << SymName << "\n");
        AllResults[Symbol.first] = orc::ExecutorSymbolDef(
            orc::ExecutorAddr(ParsedAddr), JITSymbolFlags());
        continue;
      }

      AllResults[Symbol.first] =
          orc::ExecutorSymbolDef(orc::ExecutorAddr(0), JITSymbolFlags());
    }

    LC->run(std::move(AllResults));
  }

  Error notifyResolved(jitlink::LinkGraph &G) override {
    for (auto *Symbol : G.defined_symbols()) {
      SymbolInfo Info{Symbol->getAddress().getValue(), Symbol->getSize()};
      auto Name =
          Symbol->hasName() ? (*Symbol->getName()).str() : std::string();
      Linker.Symtab.insert({std::move(Name), Info});
    }

    return Error::success();
  }

  void notifyFinalized(
      jitlink::JITLinkMemoryManager::FinalizedAlloc Alloc) override {
    if (Alloc)
      Linker.Allocs.push_back(std::move(Alloc));
    ++Linker.MM->ObjectsLoaded;
  }
};

JITLinkLinker::JITLinkLinker(BinaryContext &BC,
                             std::unique_ptr<ExecutableFileMemoryManager> MM)
    : BC(BC), MM(std::move(MM)) {}

JITLinkLinker::~JITLinkLinker() { cantFail(MM->deallocate(std::move(Allocs))); }

void JITLinkLinker::loadObject(MemoryBufferRef Obj,
                               SectionsMapper MapSections) {
  auto LG = jitlink::createLinkGraphFromObject(Obj, BC.getSymbolStringPool());
  if (auto E = LG.takeError()) {
    errs() << "BOLT-ERROR: JITLink failed: " << E << '\n';
    exit(1);
  }

  if ((*LG)->getTargetTriple().getArch() != BC.TheTriple->getArch()) {
    errs() << "BOLT-ERROR: linking object with arch "
           << (*LG)->getTargetTriple().getArchName()
           << " into context with arch " << BC.TheTriple->getArchName() << "\n";
    exit(1);
  }

  auto Ctx = std::make_unique<Context>(*this, MapSections);
  jitlink::link(std::move(*LG), std::move(Ctx));
}

std::optional<JITLinkLinker::SymbolInfo>
JITLinkLinker::lookupSymbolInfo(StringRef Name) const {
  auto It = Symtab.find(Name);
  if (It == Symtab.end())
    return std::nullopt;

  return It->second;
}

SmallVector<jitlink::Block *, 2>
JITLinkLinker::orderedBlocks(const jitlink::Section &Section) {
  SmallVector<jitlink::Block *, 2> Blocks(Section.blocks());
  llvm::sort(Blocks, [](const auto *LHS, const auto *RHS) {
    return LHS->getAddress() < RHS->getAddress();
  });
  return Blocks;
}

size_t JITLinkLinker::sectionSize(const jitlink::Section &Section) {
  size_t Size = 0;

  for (const auto *Block : orderedBlocks(Section)) {
    Size = jitlink::alignToBlock(Size, *Block);
    Size += Block->getSize();
  }

  return Size;
}

} // namespace bolt
} // namespace llvm
