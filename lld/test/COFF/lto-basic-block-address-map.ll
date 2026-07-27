; REQUIRES: x86
; Test that /lto-basic-block-address-map passes the option through to LTO
; and that .llvm_bb_addr_map sections are emitted in the output.

; RUN: llvm-as -o %t.obj %s
; RUN: lld-link /lto-basic-block-address-map /entry:main /subsystem:console %t.obj /out:%t.exe
; RUN: llvm-readobj --sections %t.exe | FileCheck %s

; Verify the negated form does not emit the section.
; RUN: lld-link /lto-basic-block-address-map:no /entry:main /subsystem:console %t.obj /out:%t2.exe
; RUN: llvm-readobj --sections %t2.exe | FileCheck %s --check-prefix=NO

; CHECK: Name: .llvm_bb_addr_map
; NO-NOT: .llvm_bb_addr_map

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

define void @main() {
entry:
  ret void
}
