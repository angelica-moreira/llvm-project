; RUN: llc < %s -mtriple=x86_64-pc-windows-msvc -function-sections -basic-block-address-map | FileCheck %s

define void @simple() {
  ret void
}

define i32 @branching(i1 %cond) {
entry:
  br i1 %cond, label %then, label %else
then:
  ret i32 1
else:
  ret i32 0
}

; CHECK-LABEL: simple:
; CHECK:       .section .llvm_bb_addr_map
; CHECK-NEXT:  .byte 5
; CHECK-NEXT:  .short 32

; CHECK-LABEL: branching:
; CHECK:       .section .llvm_bb_addr_map
; CHECK-NEXT:  .byte 5
; CHECK-NEXT:  .short 32
