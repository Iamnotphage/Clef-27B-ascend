# Path-distinguishing log markers in libccl_kernel.so (read-only)

Scope: existing `HcclCheckLogLevel` + `DlogRecord` sites in the delivered abi3–abi11
windows, mapped to the AICPU RPC server branches relevant to the Grouped hang. This
note adds no code, runs nothing on device, and does not claim the logs appear at
runtime. **The current cells' plog shows the log env variables empty and
`HCCL_DEBUG_CONFIG=0x0`; that is a default-configuration fact, not evidence that
logging is disabled, because the default level/module semantics are not established
locally** (33's part; the public hiascend page returned empty and no local value
table exists). No "set X=DEBUG" commands are given or implied.

Sources: `abi-preflight3/disasm-RpcSrvLaunch.txt` (`d92fee9d`),
`disasm-CheckRcvWorkMsg.txt` (`4959ba3d`), `abi-preflight4/disasm-RunRpcServerApi.txt`,
`abi-preflight5/disasm-Init.txt` (`9cac74e1`), `disasm-ReadAddrMsg.txt`
(`9ec8e134`), `abi-preflight6/disasm-ReadApiValidMsg.txt` (`57412224`), abi8/abi9/
abi11 caller and body windows, all libccl `2281deca`.

## Observed call pattern and guard operands

- Gate: `HcclCheckLogLevel(w0, w1)` then `tbz/tbnz`. In the windows `w0` is the level
  and `w1` is a module/type word (`3`, or `0x01000003` with bit 24 set on several
  error-to-warn sites). Some sites also require `IsErrorToWarn() == true`.
- Log: `DlogRecord(w0=module/type, w1=level, x2=format string, x3=file/function
  string, w4=line, x5..x7 = first varargs)`. `x5` is normally `gettid()` via
  `syscall(178)`; `x6/x7` carry the fields below.
- The format/file text is in .rodata and is **not** in the delivered archives; the
  exact text is unknown until the minimal dump below. The guard's **default level
  semantics are likewise unproven**; the operands are reported exactly, without
  assuming which configuration enables them.
- Proven level-type mapping (cf9f1d6, same CANN 9.2.1 installation text) applies to
  the **global env variables only**: `ASCEND_GLOBAL_LOG_LEVEL` 0=DEBUG, 1=INFO,
  2=WARNING, 3=ERROR, 4=NULL; `ASCEND_GLOBAL_EVENT_ENABLE` 0/1;
  `ASCEND_PROCESS_LOG_PATH` is a directory path. `HcclCheckLogLevel`/`DlogRecord`
  operands are raw and there is no evidence they use that global enum directly
  (they may be a bitmask or another enum), so the w0/w1 values in this table are
  **not** mapped to 0–4. Module-level format and the `HCCL_DEBUG_CONFIG` bit
  semantics are unproven; the empty env and `HCCL_DEBUG_CONFIG=0x0` in the finished
  cells remain a default-configuration fact, not evidence that logging is off.

## Marker kinds

- **K-entry (entry-proximate, per call/turn)**: fires on every execution of the
  enclosing step, independent of elapsed time (subject to the gate). Useful as a
  "this step ran" marker; still not an absolute entry marker for the whole branch.
- **K-cond (conditional timeout/error)**: fires only when a time or value condition
  is met. Its absence in a short window proves nothing; its presence proves the
  condition was reached.
- **K-shared (shared helper)**: the enclosing method is called from several paths
  (caller set not enumerated in the delivered windows), so the marker identifies
  the operation, not the caller.

## Markers

| # | Kind | Branch / step | Dlog call | Guard operands (then branch) | Fields | Caller scope / exclusivity | 180–240 s window premise | Presence implies / absence implies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | K-entry (per turn) | ForApi / `RunRpcServerApi` after `WriteTurnCnt(turn)`, `[ctx+12756]=(turn+1)&0x3f` | 0x7b3a24 | `HcclCheckLogLevel(1,3)` at 0x7b39f0, `tbz -> 0x7b3918` | w0=3, w1=1, x2=0x14f8290, x3=0x14f71a8, w4=1715, x5=tid, x6=`[ctx+12756]` | `RunRpcServerApi` is the ForApi worker; its caller set is not enumerated here (K-shared for the caller question) | fires on each turn once the guard passes; first turn well within 180 s | presence: ForApi turn loop ran and reached the post-`WriteTurnCnt` point; absence: ForApi not reached, guard failed, or no turn completed — cannot separate without the guard semantics |
| 2 | K-cond (return 0x16) | OneStage `TryRunRpcServerOneStageWait` returned 22 | 0x762280 | `HcclCheckLogLevel(1,0x01000003)` at 0x76224c, `tbz -> 0x762284` | w0=0x01000003, w1=1, x2=0x14ee650, x3=0x14edee8, w4=98, x5=tid | call site 0x762228 in `RunAicpuRpcSrvLaunch`; OneStage-only | fires only if the OneStage call returns 22; may repeat per turn | presence: OneStage returned 22 (`b.ne 0x7619ac` not taken); absence: OneStage not called, returned another value, or guard failed — **not** proof that OneStage was not used |
| 3 | K-cond (arg0 NULL) | `RunAicpuRpcSrvLaunch` arg0 (`x19`) == 0 | 0x7615a4 | `HcclCheckLogLevel(3,3)` at 0x761568, `tbz -> 0x7614a0`, then `IsErrorToWarn` -> 0x761a78 | w0=0x01000003, w1=2, x2=0x14eb1b0, x3=0x14edee8, w4=433, x5=tid | RpcSrvLaunch only | immediate if the null-arg condition holds | presence: the null-arg branch ran; absence: not that condition (or guard failed) |
| 4 | K-cond (error/warn) | post-Init error/warn path; counter `[0x191fd04]`; exits 0x7620ec -> 0x7614a4 (clear + return) | 0x7620e8 | gate at 0x7620a4 (`mov w0,w1` at 0x7620a0) + `IsErrorToWarn` at 0x7620ac | w0=0x01000003, w1=2, x2=0x14ee728, x3=0x14edee8, w4=544, x5=tid, x6=`[0x191fd04]` | RpcSrvLaunch error path; gate module not fully resolved | immediate if the condition holds | presence: the error/warn branch ran and the flow is heading to the clear+return path; absence: not proven |
| 5 | K-entry (per Init4 call) | `Init(y,j,t,KFCTask*)` called, before the zero/non-zero branch | 0x7d67cc | `HcclCheckLogLevel(0,3)` at 0x7d6728, `tbnz -> 0x7d67a0` | w0=3, w1=0, x2=0x1613f20, x3=0x1613cc8, w4=27, x5=tid, x6=y | 4-arg Init has one direct caller, 0x76192c in `RunAicpuRpcSrvLaunch` (abi6) — so this marker is exclusive to the RpcSrvLaunch Init4 step among the fetched callers | fires on each Init4 call; per task/turn, well within 180 s | presence: Init4 ran with that `y`; absence: Init4 not called on this path or guard failed — **does not** distinguish zero vs non-zero branch |
| 6 | K-cond (wait timeout) | `ReadAddrMsg` wait; log when `elapsed > w24*20 s` (w24 doubles 1,2,4,...) | 0x7d8d20 | `HcclCheckLogLevel(2,0x01000003)` at 0x7d8ce4, `tbz -> 0x7d8d24` | w0=0x01000003, w1=2, x2=0x1614408, x3=0x1613cc8, w4=360, x5=tid, x6=turn (`[sp,#32]`), x7=20 | ReadAddrMsg is a shared method; callers not enumerated here | first log at 20 s, then 40/80/160 s; the 1280 s value is the CFG return ceiling, not a log. Within 180 s up to 4 logs; within 240 s the same 4 (next at 320 s) | presence: the wait exceeded a threshold (turn printed); after it `SetErrToWarnSwitch(1)` + `PrintAllHcclMsgArea` run. Absence in <20 s proves nothing; **this is not an entry marker and an empty log cannot exclude ReadAddrMsg** |
| 7 | K-cond (valid mismatch) | `ReadApiValidMsg` gate word != 0x5CDF123A, `TraceProfSubmit` return == 20 | 0x7d6fec | `HcclCheckLogLevel(2,3)` at 0x7d6f8c plus byte/bit checks | w0=3, w1=2, x2=0x1613bd0, x3=0x1613cc8, w4=226, x5=tid, x6=0x1613bc0, w7=ret | shared method; caller set not enumerated | immediate on mismatch when ret==20 | presence: invalid gate word observed on this path; absence: no mismatch of this kind, ret != 20, or guard failed |
| 8 | K-cond (valid mismatch) | same, ret != 20, `IsErrorToWarn` true | 0x7d7074 | `HcclCheckLogLevel(2,3)` at 0x7d6ffc + `IsErrorToWarn` at 0x7d7020 | w0=0x01000003, w1=2, x2=0x1613c00, x3=0x1613cc8, w4=226, x5=tid, x6=0x1613bc0, w7=ret | shared method | immediate on mismatch when ret != 20 and error-to-warn | as #7, plus the error-to-warn condition |
| 9 | K-cond (valid mismatch) | same, `IsErrorToWarn` false | 0x7d70b8 | same gate, branch 0x7d7030 -> 0x7d707c | w0=3, w1=3, x2=0x1613bd0, x3=0x1613cc8, w4=226, x5=tid, x6=0x1613bc0, w7=ret | shared method | immediate on mismatch when error-to-warn is false | as #7, different level |
| 10 | K-cond (valid word) | `ReadApiValidMsg` line-241 path, prints the actual `[slot+48]` word | 0x7d71bc | gate at 0x7d7160 (not fully read) | w0=3, w1=1, x2=0x1613ce0, x3=0x1613cc8, w4=241, x5=tid, x6=`[slot+48]` | shared method | immediate when that branch is taken | presence: prints the observed valid word; absence: branch not taken/guard, not proof |
| 11 | K-cond (6 s timeout) | `CheckRcvWorkMsg` budget expired with `sendCnt < turn` (e.g. turn > 255) | 0x7d744c | `HcclCheckLogLevel(3,3)` at 0x7d7410, `tbz -> 0x7d7450`, then `IsErrorToWarn` at 0x7d7418 | w0=0x01000003, w1=2, x2=0x1614060, x3=0x1613cc8, w4=285, x5=tid | shared method; called at 0x7c7f20 in TwoStageWait among others (caller set not enumerated) | 6 s per call; up to 30 events in 180 s if repeated | presence: the condition held for 6 s (the loop still reached the same continuation as the success jump); absence: no 6 s expiry, or guard failed. **Not a TwoStage entry marker; an empty log cannot exclude TwoStage** |
| 12 | K-entry (per response, not Dlog) | `PostMsg` response; `AivAicpuOpParam::PrintMsg(dest, stack string)` | 0x7d6868 | none | x0=dest, x1=&stack string built from global 0x1613c90 +704 (u16) / +706 (u8) | abi8 reports only two direct PostMsg callers (0x7c7f44/0x7c7f90), both in TwoStageWait | every PostMsg call | presence/visibility unknown: PrintMsg's body (abi11, 0x7c24e0/0x654) does not write dest +24/+44/+51, but whether it emits a Dlog and at which level is not established here |

## Positive-control rule for any observation run

Because the guard semantics are unproven, **absence of a marker is not evidence
until the configuration is shown to be effective on the target AICPU**. The
positive controls are the K-entry markers:

- Init4 Dlog 0x7d67cc, line 27 (per Init4 call), and
- ForApi Dlog 0x7b3a24, line 1715 (per turn).

If at least one of these appears, the guard path is live on that AICPU and the
presence/absence columns in the table above can be used. If neither appears, the
run has "missing measurement", not "branch excluded". 33's §77 candidate setting
(`ASCEND_GLOBAL_LOG_LEVEL=0` + `ASCEND_PROCESS_LOG_PATH=<dir>`) is exactly what this
positive control is meant to validate; this note does not assert that it works.
ATRACE KFC objects only separate KFC vs MRS and do not distinguish API / TwoStage /
MC2 (33 §73).

## What the four candidate paths have (and do not have)

- **API / address wait**: #6 is a conditional timeout log (20 s first), not an entry
  marker; the 1280 s figure is the CFG return ceiling, not a log event. #7–#10 are
  mismatch logs. None of these is a mandatory entry marker.
- **TwoStage**: #11 is a 6 s conditional timeout inside `CheckRcvWorkMsg`, which is
  also invoked from other paths; the only TwoStage-exclusive facts available are the
  callee prologue (`x19=x1`, abi11) and the two `PostMsg` call sites. There is **no
  full TwoStageWait disassembly** in abi3–abi11, so no internal entry marker is
  provable. Absence of #11 does not exclude TwoStage.
- **OneStage / Unfold**: #2 fires only when `TryRunRpcServerOneStageWait` returns 22;
  it is a conditional return marker, not an entry marker. No distinct "Unfold"
  window or marker exists in the delivered material.
- **MC2 / CommKfc**: not audited here (no MC2Trace/CommKfc windows in the marker
  set); 33's §73 covers it. Do not fill this row with a wish.

## Minimal rodata request (one dump, no full scan)

Dump NUL-terminated bytes around each address (0x60–0x100 per range is enough):

- 0x1613b80–0x1613d40 (covers 0x1613bc0, 0x1613bd0, 0x1613c00, 0x1613cc8, 0x1613ce0)
- 0x1613f00–0x1613f60 (0x1613f20)
- 0x1614040–0x16140a0 (0x1614060)
- 0x16143e0–0x1614440 (0x1614408)
- 0x14eb190–0x14eb200 (0x14eb1b0)
- 0x14ede80–0x14edf20 (0x14edee8)
- 0x14ee630–0x14ee690 (0x14ee650)
- 0x14ee700–0x14ee790 (0x14ee728)
- 0x14f7160–0x14f71f0 (0x14f71a8)
- 0x14f8270–0x14f82e0 (0x14f8290)

The file string 0x1613cc8 / 0x14edee8 is shared by several sites; once dumped, the
line numbers in this table identify each message. 33 merges this into the observation
plan; 31 performs the single remote read. No full-library scan, no installer change,
no runtime code change, no device action.

## Caveats

- Guard operands are reported exactly as the instructions set them; the default
  level/module semantics and the `0x01000000` bit meaning are unproven. The empty
  env in the finished cells is a default-configuration fact, not a claim that the
  logs were disabled.
- `IsErrorToWarn()` must also be true for #3, #4, #8 and #11.
- Sites #4 and #10 have gates that were only partially read in the delivered windows.
- The logs' destination (plog vs process log) is not established here.
