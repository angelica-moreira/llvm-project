set(LLVM_LINK_COMPONENTS
  Object
  Support
  )

add_bolt_tool(etw2bolt
  etw2bolt.cpp

  DISABLE_LLVM_LINK_LLVM_DYLIB

  DEPENDS
  intrinsics_gen
)

add_dependencies(bolt etw2bolt)
