;;! threads = true

;; Out-of-bounds atomic accesses trap, including the read-modify-write,
;; compare-exchange and 64-bit forms that some targets implement with
;; load/store-exclusive loops.

(module
  (memory 1 1 shared)
  (func (export "load") (param i32) (result i32) (i32.atomic.load (local.get 0)))
  (func (export "load8") (param i32) (result i32) (i32.atomic.load8_u (local.get 0)))
  (func (export "store") (param i32) (i32.atomic.store (local.get 0) (i32.const 7)))
  (func (export "rmw") (param i32) (result i32) (i32.atomic.rmw.add (local.get 0) (i32.const 1)))
  (func (export "rmw8") (param i32) (result i32) (i32.atomic.rmw8.add_u (local.get 0) (i32.const 1)))
  (func (export "rmw16") (param i32) (result i32) (i32.atomic.rmw16.xchg_u (local.get 0) (i32.const 1)))
  (func (export "cas") (param i32) (result i32) (i32.atomic.rmw.cmpxchg (local.get 0) (i32.const 0) (i32.const 1)))
  (func (export "cas8") (param i32) (result i32) (i32.atomic.rmw8.cmpxchg_u (local.get 0) (i32.const 0) (i32.const 1)))
  (func (export "load64") (param i32) (result i64) (i64.atomic.load (local.get 0)))
  (func (export "store64") (param i32) (i64.atomic.store (local.get 0) (i64.const 7)))
  (func (export "rmw64") (param i32) (result i64) (i64.atomic.rmw.add (local.get 0) (i64.const 1)))
  (func (export "cas64") (param i32) (result i64) (i64.atomic.rmw.cmpxchg (local.get 0) (i64.const 0) (i64.const 1)))
)

(assert_trap (invoke "load" (i32.const 65536)) "out of bounds memory access")
(assert_trap (invoke "load8" (i32.const 65536)) "out of bounds memory access")
(assert_trap (invoke "store" (i32.const 65536)) "out of bounds memory access")
(assert_trap (invoke "rmw" (i32.const 65536)) "out of bounds memory access")
(assert_trap (invoke "rmw8" (i32.const 65536)) "out of bounds memory access")
(assert_trap (invoke "rmw16" (i32.const 65536)) "out of bounds memory access")
(assert_trap (invoke "cas" (i32.const 65536)) "out of bounds memory access")
(assert_trap (invoke "cas8" (i32.const 65536)) "out of bounds memory access")
(assert_trap (invoke "load64" (i32.const 65536)) "out of bounds memory access")
(assert_trap (invoke "store64" (i32.const 65536)) "out of bounds memory access")
(assert_trap (invoke "rmw64" (i32.const 65536)) "out of bounds memory access")
(assert_trap (invoke "cas64" (i32.const 65536)) "out of bounds memory access")

;; The last in-bounds locations still work, and the failed accesses above did
;; not modify memory.
(assert_return (invoke "rmw" (i32.const 65532)) (i32.const 0))
(assert_return (invoke "rmw64" (i32.const 65520)) (i64.const 0))
(assert_return (invoke "load" (i32.const 65532)) (i32.const 1))
