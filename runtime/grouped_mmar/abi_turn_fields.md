# ABI turn/counter fields in libccl_kernel.so (read-only analysis)

Source: 31 abi-preflight5 archive (`SHA256SUMS` file SHA `ae91a530…`), libccl_kernel.so
SHA `2281deca…`. Files used: `disasm-Init.txt` (`9cac74e1`),
`disasm-WriteFinishAndTurnCnt.txt` (`29027459`), and from abi-preflight4
`disasm-GetSendRecvCnt.txt` plus `nm-symbols.txt`. This note covers only the Init
overloads, WriteTurnCnt and WriteFinishWhenAllFinalize. The ReadAddrMsg /
CopyAndCheckApiMsg / RcvMsg consumer chain is 33's scope and is not repeated here.

abi-preflight8 (SUMS `d2a4043b…`, six files, libccl `2281deca`) adds the PostMsg
writer windows; abi-preflight9 (current SUMS `80926536…`; `correction.txt` supersedes
its first summary) adds getters, Init-4param and the ctx+352 / ctx+0xf31xx evidence;
abi-preflight3 `disasm-RpcSrvLaunch.txt` (SHA `d92fee9d…`, same libccl `2281deca`)
supplies the raw 0x76187c / 0x7618dc / 0x76192c instructions used below.

## Field read/write table

| Function (address) | Object/base | Offset / width | Access | Evidence |
| --- | --- | --- | --- | --- |
| `AicpuKfcRpcServer::Init(y,j,t,KFCTask*)` 0x7d66f0 | `*(void**)(KFCTask*+40)` | KFCTask+40, 8 B | read | `ldr x0,[x4,#40]` |
| same | `server+0x330` (816) | 8 B pointer | write | `str x0,[x19,#816]` |
| same, branch `*(ptr)==0` | `server+0` | 8 B = y | write | `str x20,[x19]` |
| same, branch `*(ptr)!=0` | `server+8` | 8 B = y | write | `str x20,[x19,#8]` |
| same, zero branch | `y + u32(j)` | 8 B | write to `server+0x10` | `add x0,x20,w22,uxtw; stp x0,x3,[x19,#16]` |
| same, zero branch | `y + u32(j) + (u16(t))*64` | 8 B | write to `server+0x18` | `umaddl x3,w23,#64,x0` |
| same, zero branch | `y+772` | 1 B zero | write | `strb wzr,[x20,#772]` |
| same, zero branch | `server+0x20` | 768 B zero | write | `memset(x19+0x20,0,0x300)` |
| same, both branches | `server+0x320` (800) | 8 B zero | write | `str xzr,[x19,#800]` |
| same, both branches | `server+0x328` (808) | 8 B = KFCTask* | write | `str x21,[x19,#808]` |
| `AicpuKfcRpcServer::Init(y)` 0x7d67e0 | `server+8` | 8 B = y | write | `str x1,[x0,#8]` |
| same | `server+0x320` (800) | 8 B zero | write | `str xzr,[x0,#800]` |
| same | `server+0x328`/`0x330` | 16 B zero | write | `stp xzr,xzr,[x0+0x400,#-216]` |
| `WriteFinishWhenAllFinalize(u32 turn)` 0x7d68ec | base=`[server+8]` | `0x5008 + 64*turn`, 8 B | write constant `0x112210f6ca9997ff` | `ubfiz/add #0x5000; str x3,[x2,#8]; dsb st` |
| `WriteTurnCnt(u32 turn)` 0x7d69a0 | base=`[server+8]` | `0x4008 + 64*turn`, 8 B | write zero | `str xzr,[x2,#16392]` |
| same | base=`[server+8]` | `0x5008 + 64*turn`, 8 B | read, +1, write (u64) | `ldr/add #1/str [x2,#20488]` |
| `AicpuKfcUtils::GetSendCnt(ctx)` 0x797450 | base=`[ctx+184]` + u32`[ctx+192]` | +50, 1 B | read (u8) | `ldr x1,[x0,#184]; ldr w0,[x0,#192]; ldrb w0,[x0,#50]` |
| `AicpuKfcUtils::GetRecvCnt(ctx)` 0x797470 | base=`[ctx+184]` + u32`[ctx+192]` + u16`[ctx+196]`*64 | +51, 1 B | read (u8) | `ldrh w0,[x0,#196]; add x0,x1,x0,lsl#6; ldrb w0,[x0,#51]` |

All offsets are hexadecimal except the explicitly decimal 772/800/808/816/16392/20488.
Widths follow the load/store opcodes (`ldrb`/`strb` = 1 B, `ldrh` = 2 B,
`ldr/str x` = 8 B).

## Alias and arithmetic findings

1. **Proven same-address alias**: WriteFinishWhenAllFinalize and WriteTurnCnt both
   address `base + 0x5008 + 64*turn` as an 8-byte word (`base = [server+8]`).
   WriteFinish stores the constant `0x112210f6ca9997ff`; WriteTurnCnt increments the
   same word. The intent (finalize flag vs counter) is not proven, but the address,
   width and slot arithmetic are.
2. **sendCnt/rcvCnt are literal 1-byte fields**: reads are at `+50` and `+51` of
   the record selected by `[ctx+184] + u32[ctx+192]` (recv adds
   `u16[ctx+196]<<6`). The server's turn word is 8 bytes at `+8` of a 64-byte slot
   in fixed regions `0x4008`/`0x5008`. The literal offsets and widths differ, but
   the base relation S vs C and the dynamic offset j (`u32[ctx+192]`) are unproven;
   for some j the byte ranges could overlap (for example if `C+j+50` equals
   `S+0x5008+64*t`, the 8-byte word covers +50). **No alias relation is established
   or excluded**; base, dynamic-offset and slot constraints are required.
3. **Shared stride, unproven base relation**: both use 64-byte turn slots, but the
   server uses `[server+8] + 0x5000 + 64*turn` while the counter path uses
   `[ctx+184] + u32[ctx+192] + u16[ctx+196]*64`. There is no evidence in abi5 that
   `[server+8] == [ctx+184]` or how `u32[ctx+192]` maps to `0x4000`/`0x5000`, and the
   dynamic j/t combinations could in principle shift a 1-byte field into the 8-byte
   word.
4. **Init overload ambiguity**: overload `Init(y,j,t,KFCTask*)` writes the base to
   `server+0` in the zero branch but to `server+8` in the non-zero branch;
   `Init(y)` writes `server+8`. WriteTurnCnt/WriteFinish read `[server+8]`, so which
   branch runs (and whether `Init(y)` runs afterwards) determines the base actually
   used. Call sites are not in the archive.
5. **ctx+0xf31d turn**: no `0xf31d` reference exists in the abi5 files, and no
   `0xf31dc` / `#12764` instruction exists in abi3–abi9. The nearest real field is
   `ctx+0xF31BC`: `x26 = x28 + 0xf0<<12` (0x761654, x28 = ctx from `AicpuGetComContext`
   at 0x7615ac/0x7615b0), `ldr w0,[x26,#12732]` at 0x761848 and 0x761978
   (0xF0000+0x31BC). abi9's `correction.txt` labels this `ctx+0xf31dc` but its own
   offset 12732 is 0x31BC, so the label is off by 0x20; the instruction-level
   expression is ctx+0xF31BC (u32). It cannot be equated with rcvCnt (+51, 1 byte).

## Proven vs unproven

Proven by disassembly: the offsets, widths and read/write directions in the table;
the WriteFinish/WriteTurn same-address `base+0x5008+64*turn` alias; sendCnt/rcvCnt
are 1-byte fields at +50/+51 with the ctx field arithmetic; the group tiling source
`sendCnt = curTurn` (rac_server.h) and the consumer `rcvCnt >= totalTurn` are
consistent with u8 counters. The consumer chain does not prove the same object as
the server slots. PostMsg's identity (0x7d6800), its dest chain
(`dest = *(void**)(this+0x18)`, stores only at +44 4 B and +51 1 B), the w1
provenance and u8 truncation, and the write order/`dsb st` are proven at instruction
level. server_A = 0x191fd10 and the non-reachability of 0x1920a10 are proven from
the raw adds plus the CFG. ctx+352 has literal clears at 0x76187c/0x7608c8; the
ctx+0xF31BC field expression is proven.

Unproven: whether S=C for the actual Grouped server (server_A 0x191fd10 and server_B
0x22c7cf0 are different static objects, so the same object is unproven); whether
`C+j+50/51` intersects the exact 8-byte sets for the actual j/T (byte-range alias not
established or excluded); which Init overload and branch Grouped uses; the meaning of
the WriteFinish constant and whether it is meant to overwrite the counter; the
value/`increment` source of ctx+352 (clears only) and the exact semantics of
ctx+0xF31BC; the AivAicpuOpParam population path (33's consumer chain); whether
PostMsg's `this` (TwoStageWait's second argument) is the same server_A object
(0x761994 passes x1 = 0x191fd10, but the TwoStageWait callee prologue and its callers
are not in the windows); who else writes server+0x18; what PrintMsg does to dest; and
the compiled layout because the installed headers lack CommKfcParamDesc/KFCTask.
The Init call sites and ctx 184/192/196 writers are resolved by abi-preflight6.

## abi-preflight6 call sites and exact alias sets

abi-preflight6 (SUMS `b8775ff6`, commit `433b88f`) provides the call-site evidence:

- `Init(y,j,t,KFCTask*)` is called at 0x76192c in `RunAicpuRpcSrvLaunch` with
  `x1=[ctx+184]`, `w2=[ctx+192]`, `w3=ldrh[ctx+196]`, `x0=r23+0xd10`, `x4=KFCTask*`.
- `Init(y)` is called at 0x7b67c4 in `AicpuKfcProcess::AicpuRunRpcServerForApi` with
  `x1=[x19+184]`, `x0=x20+0xb50+0x1a0 = x20+0xcf0`.
- ctx fields: `ctx+184` writer 0x7b9f50 (`str x2,[x20,#184]`), `ctx+192/200` read
  0x7ba23c (`ldp x5,x3,[x19,#192]`), `ctx+196` writer 0x761834 (`strh w0,[x28,#196]`).

Consequences: the 4-arg call's `y` is `[ctx+184]` (C) and `j=u32[ctx+192]`,
`T=u16[ctx+196]`; its zero branch stores C at server+0 and computes send base C+j,
recv base C+j+64T, matching `GetSendCnt/GetRecvCnt`. It does **not** set server+8.
`Init(y)` stores C at server+8. The two call sites use different server offsets
(+0xd10 vs +0xcf0), so S=C requires the same server object plus `Init(y)`; that
link is still unproven.

abi6 also shows the slot gate/consume fields, but the earlier "does not access
+50/+51" wording must be corrected by width: the accesses are **4-byte words**, and
a word at `[slot+48, slot+52)` physically covers +50 and +51.

| Function | Base / object | Start | Width | Covered bytes | Value / semantics |
| --- | --- | --- | --- | --- | --- |
| ReadApiValidMsg gate 0x7d6f40 | `x0` = slot pointer from ReadAddrMsg (`[server+8]+64*turn`) | +48 | 4 B load | 48,49,50,51 | compare to `0x5CDF123A`; LE bytes `3A 12 DF 5C`; valid-word gate |
| ReadApiValidMsg consume 0x7d7118-0x7d7124 | same slot | +48 | 4 B store | 48,49,50,51 | `0xA320EDC5`; LE bytes `C5 ED 20 A3`; consume marker |
| GenMsgByTaskParam 0x7d7f08/20/2c/30 | `x19` = `AivAicpuOpParam` | +44/+48/+49/+50 | 4 B/1 B/1 B/2 B | 44–47（valid）；合计 44–51 | valid word `0x5CDF123A`, isLast (48), funID from task+105 (49), u16 `0x1234` at +50 (sendCnt=0x34, rcvCnt=0x12); the four writes together cover 44–51 |
| CopyAndCheckApiMsg 0x7d6b24 | destination message | 0 | 64 B `memcpy_s` | 0-63 incl 50/51 | bulk copy, not an independent counter op |
| GetSendCnt/GetRecvCnt 0x797450/0x797470 | `[ctx+184]+u32[ctx+192]` (recv adds `u16[ctx+196]<<6`) | +50/+51 | 1 B load each | 50 / 51 | dedicated counter reads |
| `AicpuKfcRpcServer::PostMsg(uint) const` 0x7d6800 | `*(void**)(server+0x18)` (dest) | +44 | 4 B store | 44–47 | `str w4,[x0,#44]`, `0x5CDF123A` = `valid` (`AC_MSG_VALID_MASK`) |
| same PostMsg 0x7d6858 | same dest | +51 | 1 B store | 51 | `strb w1,[x0,#51]`, `rcvCnt` = low byte of the uint argument |

`0x7d67f4-0x7d67fc` are nops; PostMsg's real entry is `0x7d6800`
(`_ZNK17AicpuKfcRpcServer7PostMsgEj`), so the abi7 "unexported local function
between Init(y) and WriteFinish" description is superseded. The only stores to
dest in the PostMsg body are +44 and +51; it does not write +48/+49/+50. It calls
`AivAicpuOpParam::PrintMsg` with `x0 = dest` (body not fetched, so a write there is
not excluded) and returns 1. `x2` for the two source values is the global
`0x1613c90`: `ldrh [x2,#704]` (u16) and `ldrb [x2,#706]` (u8), stored only to the
stack, not to dest.

Provable statement: the gate/consume use an **integer valid word** and no
**independent byte counter store** was observed in that function; the 4-byte range
still physically covers +50/+51. Whether `slot` equals the `AivAicpuOpParam` base or
base-4 is unproven: GenMsg and the abi7 local writer both put `0x5CDF123A` at +44,
which is consistent with a 4-byte shift between the slot pointer and the struct, but
the layout relation is not established.

## abi-preflight8: PostMsg writer, destination chain, conditional same-byte

abi7 reported `0x7d6850 str w4,[x0,#44]` as a **local writer candidate** for the +44
valid magic; abi8 identifies the function and supersedes the "unexported local
function between Init(y) and WriteFinish" wording: `0x7d67f4-0x7d67fc` are nops and
the body starts at 0x7d6800, `_ZNK17AicpuKfcRpcServer7PostMsgEj`
(`AicpuKfcRpcServer::PostMsg(unsigned int) const`), 0x7d6800-0x7d68e8, returning 1.
This is not a library-wide uniqueness claim: GenMsgByTaskParam also writes the same
magic at +44.

Instruction-level dataflow inside PostMsg:

- entry `x0` = `this`; `0x7d6828 ldr x0,[x0,#24]` replaces it with
  `dest = *(void**)(this+0x18)`; no other write to `x0` occurs before 0x7d6858.
- `0x7d6838/0x7d6844 movz/movk w4 = 0x5CDF123A`; `0x7d6850 str w4,[x0,#44]`.
- `0x7d6858 strb w1,[x0,#51]`; `w1` is not modified anywhere in PostMsg before this
  store, so it is the caller's uint argument.
- `0x7d6868 bl AivAicpuOpParam::PrintMsg` with `x0 = dest` (body not fetched, so a
  write inside PrintMsg is not excluded); `0x7d6884 dsb st`; canary check; return 1.
- `x2` for the source values is the global `0x1613c90` (`ldrh [x2,#704]`,
  `ldrb [x2,#706]`), stored only to the stack, not to dest.

Callers and w1 provenance (both in
`AicpuKfcDeprecatedProcess::RunRpcServerTwoStageWait`, +0x198/+0x1e4):

- `0x7c7f44`: `ldr w1,[x20,#352]; sub w1,w1,#1; mov x0,x19; bl PostMsg`, so
  `w1 = (u32[ctx+352] - 1) mod 2^32`.
- `0x7c7f90`: `ldr w1,[x20,#352]; mov x0,x19; bl PostMsg`, so `w1 = u32[ctx+352]`.
- `strb` keeps only the low 8 bits: values >= 256 truncate mod 256, and a 0 value at
  the `-1` site becomes 0xFFFFFFFF, i.e. 0xFF.

Write order and barrier: program order is +44 (4 B) then +51 (1 B); there is no
barrier between the two stores; the only barrier is `dsb st` at 0x7d6884, after
PrintMsg and the string dispose. Both are plain stores; no `stlr`/`dmb`/release store
is present in the body.

Source-level field semantics (`rac_server.h`, `GMM_AR_RAC_SERVET_H_`; the binary
offsets match the header): `valid` u32 +44, `isLast` +48, `funID` +49, `sendCnt` +50,
`rcvCnt` +51, `everyTurnRsp` +52, `everyTurnWait` +53, `totalTurnCnt` +54, `res[9]`
+55..63 (size 64). `AC_MSG_VALID_MASK = 0x5CDF123A` and `~AC_MSG_VALID_MASK =
0xA320EDC5` (the consume marker). `TurnNotifyRunSetMsg` sets `sendCnt = curTurn` and
`valid = mask` on the send area; `TurnWait` polls `rcvCnt >= totalTurn`, then clears
`rcvCnt = 0` and sets `valid = ~mask` on the receive-response area. So PostMsg writes
the AICPU-to-AICore response: valid mask at +44 and `rcvCnt = (u8)turn` at +51.

Conditional same-byte with GetRecvCnt (all premises needed):

- Init4 zero branch (`[server+816] = *(u64*)(KFCTask+40) == 0`) writes
  `server+0x10 = C+j` and `server+0x18 = C+j+64T`, where `C = [ctx+184]`,
  `j = u32[ctx+192]`, `T = u16[ctx+196]` (abi6/abi9). The non-zero branch writes
  `server+8 = C` and leaves +0x18 untouched.
- `dest = *(void**)(server+0x18)`, so under (P1) PostMsg's `this` is the same server
  object, (P2) the zero branch ran, (P3) no later writer changed server+0x18, and
  (P4) the Init4 ctx and the GetRecvCnt ctx are the same object, `dest = C+j+64T`.
- GetRecvCnt reads the u8 at `C+j+64T+51`; therefore PostMsg's `strb w1,[dest+51]`
  and GetRecvCnt's `rcvCnt` byte are the **same byte** exactly under P1-P4, and the
  +44 word is the same valid word. If the non-zero branch ran, no relation holds.
- P1 is the open premise: PostMsg's `this` is x19 in TwoStageWait, and the 0x761994
  call passes x1 = 0x191fd10 (the same object Init4 receives, see below), but the
  TwoStageWait callee prologue and its callers are 33's graph and are not in these
  windows. P3 is not covered by a full store survey (only Init4 zero branch observed
  to write +0x18).

Gate relation: ReadAddrMsg's slot is `[server+8]+64*turn` and ReadApiValidMsg reads
the u32 at slot+48. If Init(y) set `[server+8] = C`, the gate word is
`C+64turn+48`, while PostMsg's valid word is `C+j+64T+44` under P1-P4. The two are
the same 4-byte word iff `j+64T = 64turn+4`, i.e. slot = dest-4. No value evidence
for that equation exists; the 4-byte displacement is compatible but not established,
and 33 keeps +44 vs +48 as different structures/offsets.

CheckRcvWorkMsg widths: `0x7d73b8 ldrb w0,[x19,#50]` (sendCnt, zero-extended u8),
`0x7d73bc ldr w1,[sp,#36]` (u32; `[sp,#36]` is `w3` saved at 0x7d7298, and the call
site passes `w3 = [ctx+352]` at 0x7c7f10/0x7c7f20), `0x7d73c0 cmp w0,w1`,
`0x7d73c4 b.cs 0x7d7450`. If ctx+352 > 255 the unsigned >= is never true; the loop
then runs to the 6 s budget (`x24 = 0x1_65a0_bc00` ns at 0x7d733c/0x7d7340, compare
at 0x7d7378, `b.hi 0x7d7408`), and the timeout path reaches the same continuation
0x7d7450 (`0x7d7414 tbz w0,0x7d7450`). It is a 6 s delay, not a deadlock; 33's
width/`>255` conclusion is confirmed at instruction level.

## Address arithmetic: server_A = 0x191fd10, not 0x1920a10

Raw sequence at the Init4 call:
`0x7618dc adrp x23,0x191f000`; `0x761918 add x0,x23,#0xd00`;
`0x761928 add x0,x0,#0x10`; `0x76192c bl Init(y,j,t,KFCTask*)`. So the `this`
argument is `0x191f000+0xd00+0x10 = 0x191fd10` (server_A). `0x191fd00` is the
enclosing struct base used for the counter at `[x2,#4]` (0x7618e8/0x7618f8) and for
`x25` (0x761904), not the Init4 object. `0x7620b4 add x23,x23,#0xd00` is a separate
post-Init error/log block; a local CFG parse of the full `disasm-RpcSrvLaunch.txt`
(SHA `d92fee9d…`) shows its exit `0x7620ec b 0x7614a4` and `0x7614a4`'s flow ending
at `0x76153c ret` without reaching 0x761904 or 0x76192c, so the bumped x23 value
never reaches the call and 0x1920a10 is not used (CFG covers direct branches and the
four `blr` calls as falling through; no `br` in the function). 31's "x0=r23+0xd10"
shorthand is valid only while r23 is the page base; the instructions are two adds on
x0. abi9's correction accepts server_A = 0x191fd10 and no back edge; independent
result matches.

## abi-preflight9 cross-check: ctx+352 clear and the ctx+0xF31BC field

- `0x76187c str wzr,[x28,#352]` is a real literal store to ctx+352, because
  `0x7615ac bl AicpuGetComContext; 0x7615b0 mov x28,x0` and x28 is written nowhere
  else in `RunAicpuRpcSrvLaunch`. abi9's first summary ("no ctx-relative #352
  store") was a grep truncation artifact; `correction.txt` now accepts
  0x76187c (and 0x7608c8 in `RunAicpuKfcClearOpRes`). Both are clears; the
  incrementing writer is still unlocated.
- The field labeled `ctx+0xf31dc` is actually **ctx+0xF31BC**:
  `0x761654 add x26,x28,#0xf0,lsl#12` gives `x26 = ctx_R+0xF0000`, and
  `0x761848 / 0x761978 ldr w0,[x26,#12732]`; 12732 decimal is 0x31BC, so the
  address is `ctx_R+0xF31BC` (u32), not 0xF31DC. 0x31DC would be 12764, which
  appears in no instruction in the delivered archives (scope negative). The
  dispatch uses the value: `0x761984 and w0,#0xfffffffd; 0x761988 cmp w0,#1;
  0x761990 b.ne 0x762228; 0x76198c mov x0,x28; 0x761994 bl RunRpcServerTwoStageWait`
  with `x1 = 0x191fd10` (0x76197c/0x761980). So TwoStageWait is taken iff
  `(v & ~2) == 1`, i.e. v in {1,3}; whether Grouped's value hits that is 33's
  handshake graph. The writers of 12732 are (a) a `CopyCtxInfo` field copy
  proven by the abi9 raw window: `0x7edfbc ldr w5,[x23,#12732]` then
  `0x7edfc0 str w5,[x7,#12732]` — source is its argument ctx, target a table
  context derived from x22; the only direct caller is `0x7ba9f0` in
  `AicpuKfcProcess::AicpuRpcResInit` (x0=x20, 0x7ba9ec), and object identity
  between these contexts and ctx_R is unproven; and (b) a static-initializer
  claim at `0x2ea7b8 str w8(=9),[x1,#12732]` with x1 = the static global
  0x22ce888+0xF0000 (not ctx_R), which is **summary-only in the delivered
  archives**: the abi10 `static-init-owner.txt` window covers 0x2ea700-0x2ea710
  and does not contain 0x2ea7b8, so this instruction is not raw-verified here.
  No increment site is observed; the runtime value of ctx_R+0xF31BC remains
  unproven.
- ReadAddrMsg has the same base pattern but a different field:
  `x23 = AicpuGetComContext()` (0x7d8ba0/0x7d8bb4), `x27 = x23+0xF0000`
  (0x7d8bec), `0x7d8c00 ldr x0,[x27,#12784]` = ctx+0xF31F0 (8 B, compared with 1).
  So ctx+0xF31BC, ctx+0xF31F0 and the nonexistent ctx+0xF31DC are distinct
  offsets; x28/x26 in RpcSrvLaunch and x23/x27 in ReadAddrMsg are written once and
  never adjusted. Both functions obtain the context via `AicpuGetComContext()`, but
  whether the two calls return the same object is not proven in these windows
  (CopyCtxInfo demonstrates that multiple contexts exist).

`GenMsgByTaskParam` writes a u16 `0x1234` at +50, covering both bytes; the
`CopyAndCheckApiMsg` 64-byte copy also covers them as a bulk operation. The **"libccl
has a unique +50/+51 writer" claim stays retracted**: no full-library store/alias/
wide-write survey exists; PostMsg writes +51 with a dedicated u8 store, GenMsg covers
it with a u16, and the bulk copy covers it as well. This does not imply a protocol
conflict; the actual object identity remains unproven. `rac_server.h` has
`sendCnt=curTurn` and `TurnWait` zeroing `rcvCnt`, but that is source semantics, not
proof of the runtime object.

Exact alias sets (S==C, t=0..63). The 8-byte writes are at `S+0x4008+64t` and
`S+0x5008+64t`; the continuous ranges `[0x4008,0x4FD0)` and `[0x5008,0x5FD0)` are
only envelopes (each 64-byte slot has an 8-byte live word and a 56-byte hole).

- send byte `j+50` hits the first array iff
  `j ∈ ∪_t [0x3FD6+64t, 0x3FDE+64t)`; second array is the same set +0x1000.
- recv byte `j+64T+51` hits the first array iff
  `j+64T ∈ ∪_t [0x3FD5+64t, 0x3FDD+64t)`; second array +0x1000.
- `T=u16[ctx+196]` is fixed per record and only adds multiples of 64, so for a given
  j only discrete 64T values can hit; it does not cover all j. The wide "T arbitrary"
  envelope cannot be used to exclude an alias. With S/C, j and T unproven, the alias
  remains neither established nor excluded.

## Minimal merged request (33 to merge; 31 sole remote fetch)

1. Compiled `AivAicpuOpParam` layout/size at `[ctx+184]+u32[ctx+192]`, confirming
   +50/+51 and the u16 slot index at ctx+196. abi7 shows the installed headers lack
   `CommKfcParamDesc`/`KFCTask`, so this must come from binary layout evidence, not
   source or `offsetof`; abi8/abi9 already confirm the +0..+64 field offsets against
   `rac_server.h`.
2. Which server object/branch the Grouped path uses (server_A 0x191fd10 and server_B
   0x22c7cf0 are different static objects; S=C needs the same object). 33's graph.
3. `CheckDebugMode` body only if a blocking/sleep question remains;
   `CopyAndCheckApiMsg` is confirmed as a 64-byte copy/GenXor/memcmp by 31/33.
4. PostMsg destination identity: the TwoStageWait callee prologue (does x19 = arg2?),
   the callers of TwoStageWait and which server object they pass, who else writes
   server+0x18, and the `AivAicpuOpParam::PrintMsg` body (can it write dest?). These
   are the only open premises for the PostMsg +51 / GetRecvCnt same-byte statement.
Do not fetch these outside 31; 33 merges and 31 executes the single remote read.
No new full-library store survey, no cross-DSO search, no device action.
