"""CANN 9.2.1 compat: make glob(recursive=True) include hidden directories.

CANN's ``ascendc_kernel_cmake/legacy_modules/util/recompile_binary.py`` collects
host object files with::

    glob.glob(f"{add_dir}/**/*.o", recursive=True)

CMake installs OBJECT-library objects under a path that preserves the source
directory.  When that source lives under a hidden component (the project-local
``.venv`` for MemFabric's AccOffload kernel), Python's glob skips the hidden
directory and the kernel build fails with ``TooFewObj``.

Importing this module (via ``PYTHONPATH=scripts/compat/glob_hidden`` for the
``mfcli kernel install`` process tree only) flips the default to
``include_hidden=True``.  It does not change the interpreter default elsewhere.
"""

import glob as _glob
import inspect

if "include_hidden" in inspect.signature(_glob.glob).parameters:
    _orig_glob = _glob.glob

    def _glob_include_hidden(pattern, *args, **kwargs):
        kwargs.setdefault("include_hidden", True)
        return _orig_glob(pattern, *args, **kwargs)

    _glob.glob = _glob_include_hidden
