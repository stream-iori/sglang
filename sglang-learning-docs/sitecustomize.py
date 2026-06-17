import os
import sys

# Apply macOS ARM64 compatibility patches
try:
    import conftest
    conftest.patch_mac()
    print("[Antigravity] Apple Silicon Mac compatibility layer patched via sitecustomize.py")
except ImportError:
    pass

