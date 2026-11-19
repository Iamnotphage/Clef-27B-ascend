# A12 build note (derived build, official source unchanged)

- 官方样例源：`test_aclnn_grouped_mat_mul_all_reduce.cpp`，SHA `6c1a90ab…`，来自
  `hicann/ops-transformer` commit `f5703e4…`，字节未改。
- 官方 `examples/CMakeLists.txt` **在 f5703e4 不存在**（33 目录实读仅 .cpp，raw 404；op 级 CMakeLists 不含 example target），因此不存在可补的官方文件；本目录的
  `compile_a12.sh` 按官方“编译与运行”配方的链接项**重建** CMakeLists glue，因此 A12 的构建是
  **派生构建**，不得声称“官方 as-is 构建”。
- 31 已在独立 stage 用等价 glue 构建：glue `22edb89f…`、binary `37929ec4…`（204,744 B）、
  stage manifest **`088f05f6…` 为权威**（`1d529ca8…` 为字段更新前早期版，已被取代）；
  ldd `not found` 0；可作为设备构建引用，但同样标注派生。
- A12 语义是**通路/存活控制**：样例无数值参考答案，成功只能是 API rc0 + 两 rank sync 返回；
  它相对 v4r1 有 C1–C5 多处差异（单进程 HcclCommInitAll vs torch_npu 多进程 comm、FP16 vs
  BF16+FP32 bias、链接 opapi_math/transformer vs dlopen opapi/nnopbase、2 group vs 1 group），
  因此 A12 返回不能单因证明 C1，A12 挂起则指向环境/SoC/API 通路。
- A12b（BF16+FP32 bias、v4r1-like shapes）只能在 A12 as-is 源臂之后作为明确派生 delta，
  且 C5 comm 创建差异仍保留。
