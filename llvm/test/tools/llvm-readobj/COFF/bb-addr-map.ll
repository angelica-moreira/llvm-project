; REQUIRES: x86-registered-target
; Test that llvm-readobj can dump BB address maps from COFF objects.

; RUN: llc < %s -mtriple=x86_64-pc-windows-msvc -function-sections -basic-block-address-map -filetype=obj -o %t.o
; RUN: llvm-readobj --bb-addr-map %t.o | FileCheck %s

;; First function: simple (single basic block, returns).
; CHECK:      BBAddrMap [
; CHECK-NEXT:   Function {
; CHECK-NEXT:     At: 0x0
; CHECK:          BB Ranges [
; CHECK:            {
; CHECK:              Base Address: 0x0
; CHECK:              BB Entries [
; CHECK:                {
; CHECK:                  ID: 0
; CHECK:                  Offset: 0x0
; CHECK:                  Size: 0x1
; CHECK:                  HasReturn: Yes
; CHECK:                }
; CHECK:              ]
; CHECK:            }
; CHECK:          ]
; CHECK:        }
; CHECK:      ]

;; Second function: branching (three basic blocks).
; CHECK:      BBAddrMap [
; CHECK-NEXT:   Function {
; CHECK-NEXT:     At: 0x0
; CHECK:          BB Entries [
; CHECK:            {
; CHECK:              ID: 0
; CHECK:            }
; CHECK:            {
; CHECK:              ID: 1
; CHECK:              HasReturn: Yes
; CHECK:            }
; CHECK:            {
; CHECK:              ID: 2
; CHECK:              HasReturn: Yes
; CHECK:            }
; CHECK:          ]
; CHECK:        }
; CHECK:      ]

target triple = "x86_64-pc-windows-msvc"

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
