# ABI turn/counter fields in libccl_kernel.so (read-only analysis)

Source: 31 abi-preflight5 archive (`SHA256SUMS` file SHA `ae91a530…`), libccl_kernel.so
SHA `2281deca…`. Files used: `disasm-Init.txt` (`9cac74e1`),
`disasm-WriteFinishAndTurnCnt.txt` (`29027459`), and from abi-preflight4
`disasm-GetSendRecvCnt.txt` plus `nm-symbols.txt`. This note covers only the Init
overloads, WriteTurnCnt and WriteFinishWhenAllFinalize. The ReadAddrMsg /
CopyAndCheckApiMsg / RcvMsg consumer chain is 33's scope and is not repeated here.

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
2. **sendCnt/rcvCnt are different fields**: they are 1-byte reads at `+50` and `+51`
   of the record selected by `[ctx+184] + u32[ctx+192]` (recv adds
   `u16[ctx+196]<<6`). The server's turn field is 8 bytes at `+8` of a 64-byte slot
   in fixed regions `0x4008`/`0x5008`. Offsets 8 vs 50/51 and widths 8 B vs 1 B
   differ, so even if the bases coincided there is no same-address alias.
3. **Shared stride, unproven base relation**: both use 64-byte turn slots, but the
   server uses `[server+8] + 0x5000 + 64*turn` while the counter path uses
   `[ctx+184] + u32[ctx+192] + u16[ctx+196]*64`. There is no evidence in abi5 that
   `[server+8] == [ctx+184]` or how `u32[ctx+192]` maps to `0x4000`/`0x5000`.
4. **Init overload ambiguity**: overload `Init(y,j,t,KFCTask*)` writes the base to
   `server+0` in the zero branch but to `server+8` in the non-zero branch;
   `Init(y)` writes `server+8`. WriteTurnCnt/WriteFinish read `[server+8]`, so which
   branch runs (and whether `Init(y)` runs afterwards) determines the base actually
   used. Call sites are not in the archive.
5. **ctx+0xf31d turn**: no `0xf31d` reference exists in the abi5 files. Its base,
   width and read/write path are unproven; it cannot be equated with rcvCnt (+51,
   1 byte) from this evidence.

## Proven vs unproven

Proven by disassembly: the offsets, widths and read/write directions in the table;
the WriteFinish/WriteTurn same-address `base+0x5008+64*turn` alias; sendCnt/rcvCnt
are 1-byte fields at +50/+51 with the ctx field arithmetic; the group tiling source
`sendCnt = curTurn` (rac_server.h) and the consumer `rcvCnt >= totalTurn` are
consistent with u8 counters but do not prove the same object as the server slots.

Unproven: the relation `[server+8]` vs `[ctx+184]+u32[ctx+192]`; which Init overload
and branch Grouped uses; the meaning of the WriteFinish constant and whether it is
meant to overwrite the counter; `ctx+0xf31d`; the AivAicpuOpParam population path
(33's consumer chain).

## Minimal merged request (33 to merge; 31 sole remote fetch)

1. Call-site xrefs of `AicpuKfcRpcServer::Init(y,j,t,KFCTask*)` and `Init(y)` in
   libccl_kernel, with argument setup for the Grouped/MC2 path.
2. `AicpuComContext` layout offsets 184/192/196 (types and width) and any reference
   to `ctx+0xf31d`, plus the server-to-context link (is the server embedded at a
   known ctx offset?).
3. Struct layout/size for the record at `[ctx+184]+u32[ctx+192]` (AivAicpuOpParam)
   to confirm +50/+51 and the u16 slot index at ctx+196.
Do not fetch these outside 31; 33 merges and 31 executes the single remote read.
