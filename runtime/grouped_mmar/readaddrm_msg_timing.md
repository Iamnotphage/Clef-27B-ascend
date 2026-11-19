# ReadAddrMsg timing derivation (read-only)

Source: 31 abi-preflight5 `disasm-ReadAddrMsg.txt` SHA `9ec8e134` (SUMS
`ae91a530`) and abi-preflight6 `disasm-ReadApiValidMsg.txt` SHA `57412224` (SUMS
`b8775ff6`), libccl_kernel.so `2281deca`; function range `0x7d8b6c-0x7d9360`.
No SSH, device, or code change. This note checks 33's timing claim
(20 s x w24 backoff, w24 1..128,0, 960 s cutoff, ~2560 s worst case, 180 s between
160/320 s).

## Clock and constants

| Address | Instruction | Meaning |
| --- | --- | --- |
| 0x7d8bcc | `mov w0,#4; bl clock_gettime` | clock id 4 = `CLOCK_MONOTONIC_RAW`; writes `timespec` at sp+64 |
| 0x7d8bd0/0x7d8bd8 | `mov x1,#0xca00; movk x1,#0x3b9a,lsl16` | multiplier 0x3b9aca00 = 1,000,000,000 (ns per second) |
| 0x7d8be4/0x7d8bf0 | `madd x0,x2,x1,x0`; `str x0,[sp,#24]` | `start_ns = tv_sec*1e9 + tv_nsec`; unit is **nanoseconds** |
| 0x7d8ca0-0x7d8ccc | second `clock_gettime(4)`; `sub x19,x19,x1` | `elapsed_ns = now_ns - start_ns`, same raw clock |
| 0x7d8cb0-0x7d8cc8 | constant `0x4a817c800` | 20,000,000,000 ns = **20 s** |
| 0x7d8d40-0x7d8d50 | constant `0xdf84758000` | 960,000,000,000 ns = **960 s** |

## Loop and w24

| Address | Instruction | Effect |
| --- | --- | --- |
| 0x7d8d60-0x7d8d7c | `add w20,w20,#1`; slot=`[this+8]+turn*64`; `bl ReadApiValidMsg(...,1)` | one non-blocking-looking poll; `w20` is an attempt counter |
| 0x7d8d80 | `tbz w0,#0,0x7d8c00` | failure (bit0=0) re-enters the loop |
| 0x7d8c90-0x7d8c94 | `cmp w20,#0x2711` (10001) | attempt counter reset/log path, independent of time |
| 0x7d8cd0-0x7d8cd4 | `cmp x0,x19; b.cs 0x7d8d60` | if `elapsed <= w24*20 s` continue immediately |
| 0x7d8d40-0x7d8d44 | `ubfiz w24,w24,#1,#7` | `w24 = (w24 & 0x7f) << 1`, a 7-bit field shifted to bits 7:1; sequence 1,2,4,8,16,32,64,128,0,... |
| 0x7d8d50-0x7d8d58 | `cmp x19,960e9; b.hi 0x7d9240` | hard cut only when a timeout-log branch is taken |
| 0x7d8d5c / 0x7d9240 | `SetErrToWarnSwitch(0/1); b 0x7d91c8` | 0x7d91c8 sets w19=0 and returns 0 |

There is **no sleep/nanosleep/usleep/select/poll call in the whole ReadAddrMsg
disassembly** (grep of the archived file finds none). The `w24*20 s` arithmetic is
the **timeout-log interval**, doubled after each timeout log; it is not a sleep.
The loop spins on `ReadApiValidMsg` and checks elapsed time after each failed poll.
The `ctx+0xF31F0` load at 0x7d8c00 is unrelated to the timing constants; no
`ctx+0xf31d` appears in this file.

## Per-round timeline (no message, assuming ReadApiValidMsg is non-blocking)

| Interval | w24 | Log/threshold | After the timeout log |
| --- | --- | --- | --- |
| 0-20 s | 1 | no log | poll |
| 20 s | 1->2 | timeout log at 20 s | elapsed>960? no |
| 40 s | 2->4 | log | no |
| 80 s | 4->8 | log | no |
| 160 s | 8->16 | log | no |
| 320 s | 16->32 | log | no |
| 640 s | 32->64 | log | no |
| 1280 s | 64->128 | log | **elapsed>960 -> return 0** |

The first threshold above 960 s is 1280 s, and the hard check sits inside that
branch; therefore the wall ceiling is approximately **1280 s plus one poll
iteration**, not 2560 s. If a single poll iteration happens to straddle the 960 s
point after the 640 s branch (for example elapsed jumps from <640 to 1000 s), the
return can occur between 960 and 1280 s. The 2560 s threshold (w24=128) is never
reached because the 960 s check returns at the 1280 s crossing.

180 s falls in the **160-320 s log interval** (one timeout log at 160 s, next at
320 s), which confirms 33's interval statement, but it is a log-interval boundary,
not a sleep and not a return point. An external 180 s or 240 s window truncates the
call long before ~1280 s; ReadAddrMsg itself does not return at 180 s.

## Callee check from abi-preflight6

`ReadApiValidMsg` is now available in abi-preflight6 (SUMS `b8775ff6`, file SHA
`57412224`, range `0x7d6f20-0x7d71cc`). The body contains **no
sleep/nanosleep/usleep/select/poll** call. It calls `TraceProfSubmit`,
`CopyAndCheckApiMsg@plt`, `CheckDebugMode`, `DlogRecord` and `syscall(178)` (gettid),
and returns 0/1. The final merged wording is: only the **two layers** ReadAddrMsg and
ReadApiValidMsg are checked, and they show **no explicit sleep/nanosleep/usleep/
select/poll** (syscall 178 = gettid). Deep helper durations (`CopyAndCheckApiMsg`,
`DlogRecord`, `TraceProfSubmit`, `CheckDebugMode`) are unknown; the derivation keeps
the **single-round return premise** that each helper returns and control re-enters
the loop. 31/33 checked `CopyAndCheckApiMsg` as a 64-byte copy/GenXor/memcmp with no
+50/+51 access.

This also corrects 33's sequence at the 640 s point: w24 is 32 before the 640 s
crossing and becomes **64** (not 128), so the next threshold is 1280 s. At the
1280 s crossing w24 becomes 128 and the same block checks elapsed > 960 s and
returns 0. The 2560 s threshold is therefore not reached; the ceiling remains about
1280 s plus one poll iteration.

Two separate questions must not be merged: "is the w24=128/2560 s threshold state
reachable by the CFG" and "does elapsed exceed 2560 s because a callee blocked".
The 960 s check lives inside the 1280 s branch, so the CFG reaches w24=128 and then
returns; the 2560 s branch is **not reachable by CFG**, and a long callee only raises
the wall time, it cannot be used as an exception that makes the threshold state
reachable. 33 confirmed the edges: w24 64->128, `cmp`, `b.hi 0x7d9240`, then
`SetErrToWarnSwitch(0) -> 0x7d91c8 (w19=0) -> 0x7d8fcc -> ret`; 0x7d9240 has **no
edge back to the read attempt 0x7d8d60**.

The "1280 s + one poll" figure is therefore a derivation **only under the
helper-returns/control-reenters premise**, not an actual upper bound for this cell.

## Conditional gap and merged request

The remaining uncertainty is helper duration (`DlogRecord`/`TraceProfSubmit`/
`CopyAndCheckApiMsg`/`CheckDebugMode`), not an explicit sleep; the single-round
return premise covers it. `CopyAndCheckApiMsg` is confirmed as copy/GenXor/memcmp by
31/33. The static path and this cell's actual path remain separate. No device action. The 20 s x w24
intervals are timeout-log intervals; 180 s is a log-interval point, not a sleep or
return.
