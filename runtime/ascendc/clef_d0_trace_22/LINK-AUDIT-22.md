# adapter 对象符号/ABI 闭合（CPU 只读取证；未装载、未执行）

冻结证据目录（**不改原件**，副本已复制）：
`/workspace/yangchen/logs/perf-wave3-20261118/d0-link-audit-22/`

| 件 | SHA-256 |
| --- | --- |
| `d0_adapter.o`（SHA 已复核一致） | `8edf909bf8f8475caac39c7570e0bcc764cca8bb3734a82b010b955c91a8978d` |
| `d0_adapter.cpp`（源） | `cb816029d83f88d389a9e5058211dc097e4dc51a2c95db14bbe75b7455762454` |
| 生成头 `aclnn_clef_d0_trace.h` | `146daf7f29a7a8359217889949110d21cc2de2cafcf200680206712f42b0a395` |
| 包内 `libcust_opapi.so` | `3314922677b7b78164628f1bc22c7339ac63d59cf537c261533cbf740ac948ec` |

## 1) 对象未解析集合（完整，非抽样）
`nm -u d0_adapter.o`：**286 强 + 1 弱**。
- 强符号按类：`Py*` **95**、`_Py*` **7**、`at::_ops` **2**、`c10::` **19**、`acl*` **8**、其余 `_Z*` **142**
  （其余为 torch/torch_npu C++ 符号）
- 弱符号：`__cxa_pure_virtual`（虚函数纯虚桩，弱）

## 2) 提供者逐类
| 类 | 提供者 | 依据 | 状态 |
| --- | --- | --- | --- |
| `Py*` / `_Py*`（**102**） | **宿主 Python 解释器** | 只能由 `libpython3.12.so.1.0` 提供；21 未链、`-z defs` 下必然未解析 | **仅宿主可提供，保留未闭合** |
| `at::_ops`（2）、`c10::`（19） | `torch/lib/libtorch.so`、`libc10.so` | 严格链接中**已解析**（无相关报错） | **闭合** |
| 其余 `_Z*`（142） | `torch_npu/lib` + `torch/lib` | 同上，**已解析** | **闭合** |
| `acl*`（8） | `cann-9.2.1/aarch64-linux/lib64/libascendcl.so` | 显式 `-lascendcl` 后**已解析** | **闭合** |
| `Nnopbase*`（opapi 的 32 个） | `cann-9.2.1/aarch64-linux/lib64/libnnopbase.so` | 显式 `-lnnopbase` 后**已解析** | **闭合** |

## 3) 严格链接复核（**一次**，逐字记录）——结论已更正
```
g++ -shared -z defs -o d0_strict.so d0_adapter.o \
  -L<venv>/torch_npu/lib -L<venv>/torch/lib -ltorch_npu -ltorch -lc10 \
  -L<包内 op_api/lib> -lcust_opapi \
  -L/usr/local/Ascend/cann-9.2.1/aarch64-linux/lib64 -lnnopbase -lascendcl
```
- **RC = 1**；按 `undefined reference to` 右侧**完整引用**重解析：**265 条引用 / 115 唯一**
- **102 个 `Py*`/`_Py*` + 13 个真实非 Python 符号**

### 3.1 **更正**：先前"非 Python 全部解析"是**错的**
此前我用正则从 `_ZN2at…` 截断，把 13 项误当片段。**13 项是真实符号**：

| # | 符号（简写） | 在 `libtorch_cpu.so` 实际导出 |
| --- | --- | --- |
| 1-4 | `c10::{TensorType,FloatType,IntType,StringType}::get()` | ✅ |
| 5 | `c10::IValue::reportToTensorTypeError() const` | ✅ |
| 6 | `c10::detail::infer_schema::make_function_schema(...)` | ✅ |
| 7 | `torch::jit::parseSchema(...)` | ✅ |
| 8-10 | `torch::Library::{_def,_impl,Library}` | ✅ |
| 11 | `torch::CppFunction::~CppFunction()` | ✅ |
| 12-13 | `at::_ops::{empty_like,empty_memory_format}::call(...)` | ✅ |

按**完整签名 demangle 精确匹配**：**`libtorch_cpu.so` 13/13**；
`libtorch_python.so` 0/13、`libc10.so` 0/13、`libtorch.so` 0/13。
⇒ **缺失链接输入 = `libtorch_cpu`**（21 按设计未链）。

### 3.2 **更正**：`Py*` 并非"仅宿主、取不到提供者"
真实解释器：`/workspace/yangchen/python/cpython-3.12.16+20261110/bin/python3.12`
（**不是** `sysconfig` 记录的 `/install/lib`；构建期路径 ≠ 实际库路径）。

实际库：`/workspace/yangchen/python/cpython-3.12.16+20261110/lib/libpython3.12.so.1.0`
SHA-256 `212da0dac0371ea3d1f87cd24b6750b6db12d145280d9b4480cac66b16c3c169`
**定义符号 1801 个；本次需要的 102 个 `Py*`/`_Py*` 命中 102/102（缺失 0）**。

⇒ **缺失链接输入 = `libpython3.12`**。两项缺失都是**可指名的输入**，
非"只能运行期才知道"。

### 3.3 明确更正声明
**撤回**"非 Python 类全部解析"与"Py 仅宿主可提供"两条结论。
当前事实：**严格链接失败由两个具名缺失输入造成**（`libtorch_cpu`、`libpython3.12`），
是否做**一次有限修正**由 20 裁定；**本次未追加链接**。

## 5) 仍未闭合 / 未取得
1. ~~102 个 Python C-API 的宿主提供者不明~~ → **已定位**（`libpython3.12.so.1.0`，102/102）；
2. `sysconfig` 的 `/install/lib` 与实际库路径**不一致**：已按实际路径记录，运行期如何解析仍属**未取得**；
3. 运行装载、设备 binary、`-z defs` 之外的一切：**均未取得**。

**本文件只证明链接期解析结论**，不推运行装载与设备行为。
