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

## 3) 严格链接复核（**一次**，逐字记录）
```
g++ -shared -z defs -o d0_strict.so d0_adapter.o \
  -L<venv>/torch_npu/lib -L<venv>/torch/lib -ltorch_npu -ltorch -lc10 \
  -L<包内 op_api/lib> -lcust_opapi \
  -L/usr/local/Ascend/cann-9.2.1/aarch64-linux/lib64 -lnnopbase -lascendcl
```
- **RC = 1**；记录文件：`strict_link_cmd.txt`、`strict_link_rc.txt`、`strict_link.log`
- **268 条未解析引用，去重后 105 个名字：102 个是 `Py*`/`_Py*`**，其余 3 条是我正则从 `_ZN2at…/c10…/torch…` 误截的片段，**不是真实符号**
- 首错逐字：`d0_adapter.o: in function 'Py_SIZE': d0_adapter.cpp:(.text+0x2c): undefined reference to 'PyLong_Type'`

**结论**：**非 Python 类符号在本次严格链接中全部解析**；`-z defs` 失败的唯一原因是 **Python C-API 只能由宿主提供**（21 按设计未链 libpython）。这是**分类事实**，不是"接线缺陷"，故**不做修正**。

## 4) 路径歧义与源码内配置
- `libnnopbase.so` 同机两份不同 SHA：**lib64** `72cd43b4af1c9c451b566ccb7b32071793402949578f838d8345e369b056477c`（**本次显式选用**）；**devlib** `ea82573dc09180d70e9cbedf6ed66bca56a8caae547da4567af4f05c95358013`（**未选用**，分列）。
- `sysconfig`（**未导入 torch_npu**）：`LIBDIR=/install/lib`、`LDLIBRARY=libpython3.12.so`、`INSTSONAME=libpython3.12.so.1.0`。
  ⇒ **`/install/lib` 是构建期记录值，不一定是本机运行期实际路径**，属**未闭合项**。

## 5) 仍未闭合 / 未取得
1. 102 个 Python C-API 符号的**宿主实际提供者**（需在**运行期**由解释器 dlopen 解析；本次未装载、未执行）；
2. `/install/lib` 与运行期 libpython 实际路径的一致性；
3. 运行装载、设备 binary、`-z defs` 之外的一切：**均未取得**。

**本文件只证明链接期解析结论**，不推运行装载与设备行为。
