doc-envvar-scan-20261119 — CANN 9.2.1 安装树内日志/调试环境变量的同版文本级证据
================================================================================

范围与性质
----------
- 只读文本检索（grep -rIn -I，跳过二进制），无设备动作、无运行、无新反汇编。
- 安装树：/usr/local/Ascend/cann-9.2.1，目录 include tools conf compiler opp pkg_inc
  runtime python share fwkacllib toolkit compat asc bin。
- 另查：/usr/local/Ascend/driver/include、/usr/local/Ascend/driver/tools、
  /usr/local/Ascend/nnal（两变量各 0 命中）。
- 文档目录搜索（/usr/local/Ascend，maxdepth 5，doc/docs/documentation/doc_*/*_doc）：0 个。
- .pdf/.md（安装树 maxdepth 4）：仅 3 个 README（tools/hccl_test/README.md、
  README_en.md、tools/mspti/samples/README.md），均无相关变量；hccl_test README
  对 HCCL_DEBUG_CONFIG|LOG_LEVEL 命中数为 0。

版本同一性
----------
- compiler/version.info：Version=9.2.1；opp/version.info：Version=9.2.1
  （见 scope-and-version.txt）。

全量命中（未截断，grep-full.txt 共 52 行，26 个文件）
----------------------------------------------------
TOTAL 52
ASCEND_GLOBAL_LOG_LEVEL   36
ASCEND_MODULE_LOG_LEVEL    1
ASCEND_GLOBAL_EVENT_ENABLE 6
ASCEND_PROCESS_LOG_PATH    6
HCCL_DEBUG_CONFIG          3

取值/语义（同版安装树内可引用的原文）
------------------------------------
1) ASCEND_GLOBAL_LOG_LEVEL —— 有完整取值表
   python/site-packages/msopst/scripts/msopst.ini:16-22
     0: DEBUG；1: INFO；2: WARNING；3: ERROR（默认）；4: NULL（不输出日志）
   tools/ascend_system_advisor/asys/config/config_parser.py:44-49
     {"DEBUG":"0","INFO":"1","WARNING":"2","ERROR":"3","NULL":"4"}，asys 默认 "INFO"
   python/site-packages/pypto/lib/scripts/debug_aicore_error.py:300-305
     以 3（ERROR）为默认，非 3 提示单算子测试可能较慢
2) ASCEND_GLOBAL_EVENT_ENABLE —— 有布尔映射
   asys/config/config_parser.py:50-52 {"FALSE":"0","TRUE":"1"}，默认 "TRUE"
   asc_op_compile_base/common/utils/log_utils.py:151-155（tikcpp 同）
     仅当 int(值)==1 时执行 detail/event 打印
3) ASCEND_PROCESS_LOG_PATH —— 路径型变量（无取值表）
   asys/drv/env_var.py:26,34-43 要求路径存在且为目录，取 abspath
   asys/common/task_common.py:128-137 作为 CANN/plog 日志目录，缺省回退
   ASCEND_WORK_PATH/<log_type>
   msopst/tools 中用于为单算子日志设置目录（aicore_error_parser.py:1364,1678）
4) ASCEND_MODULE_LOG_LEVEL —— 无取值表/模块名格式文档
   安装树仅 1 处：pkg_inc/mmpa/sub_inc/mmpa_env_define.h:94 枚举 ID 6005
5) HCCL_DEBUG_CONFIG —— 无取值语义文档
   安装树 3 处：
     mmpa_env_define.h:142 枚举 ID 11025
     asc/impl/adv_api/detail/hccl/cc/src/aicpu_kfc/pub_inc/aicpu_operator_pub.h:660
       u64 debugConfig = 0; // 环境变量HCCL_DEBUG_CONFIG, 考虑兼容性放在结构体末尾
     opp/built-in/.../op_kernel/moe_distribute_base.h:240 同一结构体字段注释

结论与边界
----------
- 该版本安装树内可直接引用的只有 GLOBAL_LOG_LEVEL（0-4 表）、GLOBAL_EVENT_ENABLE
  （0/1）、PROCESS_LOG_PATH（目录路径）；不能据此写“MODULE_LOG_LEVEL/HCCL_DEBUG_CONFIG
  设某值=DEBUG”之类结论——两者在安装树中仅有枚举 ID 与结构体注释。
- “无文档”仅指上述检索范围（-I 跳过二进制、未检索 devlib 中的字符串）；显示未截断，
  grep-full.txt 为完整 52 行。
- 本目录不含任何设备运行或反汇编产物。

文件
----
grep-full.txt          全量 grep 结果（52 行）
var-counts.txt         分项计数（含文件数 26）
scope-and-version.txt  版本与检索范围/排除项计数
snippets.txt           关键文件原行片段
