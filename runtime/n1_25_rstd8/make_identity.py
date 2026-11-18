"""Hash existing build products; never loads shared libraries."""
import argparse
import json
from pathlib import Path
from contract import sha256

p = argparse.ArgumentParser()
p.add_argument("--build", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
b = a.build.resolve()
vendor = b / "build_out/_CPack_Packages/Linux/External/custom_opp_ubuntu_aarch64.run/packages/vendors/clef_native_ar_norm25_rstd8"
info = {"op": "ClefNativeArNorm25Rstd8", "vendor": "clef_native_ar_norm25_rstd8",
        "package_sha256": sha256(b / "build_out/custom_opp_ubuntu_aarch64.run"),
        "adapter_sha256": sha256(b / "n1_adapter.so"),
        "generated_header_sha256": sha256(b / "build_out/autogen/aclnn_clef_native_ar_norm25_rstd8.h"),
        "vendor_files": {str(f.relative_to(vendor)): sha256(f)
                         for f in sorted(vendor.rglob("*")) if f.is_file()},
        "source_files": {str(f.relative_to(b)): sha256(f)
                         for folder in ("op_host", "op_kernel", "torch_adapter")
                         for f in sorted((b / folder).rglob("*")) if f.is_file()}}
a.output.write_text(json.dumps(info, indent=2) + "\n")
print(json.dumps({k: v for k, v in info.items() if not isinstance(v, dict)}, indent=2))
