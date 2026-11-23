"""Install the private probe only when the isolated launcher enables it."""

import os

if os.environ.get("CLEF_PRIVATE_K32") in {"split", "cache"}:
    import importlib.abc
    import importlib.machinery
    import sys

    class _Loader(importlib.abc.Loader):
        def __init__(self, base):
            self.base = base

        def create_module(self, spec):
            return self.base.create_module(spec)

        def exec_module(self, module):
            self.base.exec_module(module)
            from ascend_probe import install
            install(module.NPUModelRunner)

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "vllm_ascend.worker.v2.model_runner":
                spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
                if spec is not None and spec.loader is not None:
                    spec.loader = _Loader(spec.loader)
                return spec
            return None

    sys.meta_path.insert(0, _Finder())
