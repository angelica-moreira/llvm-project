set(LLVM_LINK_COMPONENTS
  DebugInfoCodeView
  DebugInfoMSF
  DebugInfoPDB
  Object
  Support
  )

add_bolt_tool(verify-pdb
  verify-pdb.cpp

  DISABLE_LLVM_LINK_LLVM_DYLIB

  DEPENDS
  intrinsics_gen
)

add_dependencies(bolt verify-pdb)
