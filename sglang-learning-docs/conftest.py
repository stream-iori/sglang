"""
Mac ARM64 兼容层 - 在 import sglang 之前自动 patch 缺失的依赖。

用法:
  1. 直接运行测试: cd sglang-learning-docs && python -m pytest
  2. 在脚本中使用: import conftest; conftest.patch_mac()  然后再 import sglang

此文件会自动被 pytest 加载 (conftest.py 机制)。
"""

import platform
import sys
import types

_patched = False


def patch_mac():
    """一次性 patch 所有 Mac CPU-only 环境缺失的模块。"""
    global _patched
    if _patched:
        return
    _patched = True

    is_mac_arm = platform.system() == "Darwin" and platform.machine() == "arm64"
    if not is_mac_arm:
        return

    # ---- 1. triton stub (Mac ARM 不支持 triton) ----
    if "triton" not in sys.modules:
        _install_triton_stub()

    # ---- 2. torch.mps.Stream (CPU-only torch 没有) ----
    try:
        import torch.mps

        if not hasattr(torch.mps, "Stream"):

            class _FakeStream:
                def __init__(self, *a, **k):
                    pass

                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    pass

            torch.mps.Stream = _FakeStream
    except ImportError:
        pass

    # ---- 3. sgl_kernel stub (需要 CUDA 编译, 用 __getattr__ 兜底) ----
    if "sgl_kernel" not in sys.modules:

        class _SglKernelModule(types.ModuleType):
            """sgl_kernel stub: 所有 kernel 函数返回 noop placeholder。"""

            def __getattr__(self, name):
                if name.startswith("__") and name.endswith("__"):
                    raise AttributeError(name)

                def _noop(*args, **kwargs):
                    raise RuntimeError(
                        f"sgl_kernel.{name}() is a CUDA kernel stub — "
                        "not available on Mac CPU. This is expected."
                    )

                return _noop

        sgl_kernel = _SglKernelModule("sgl_kernel")
        sgl_kernel.__version__ = "0.0.0.stub"
        sgl_kernel.__path__ = []
        sgl_kernel._is_stub = True
        sys.modules["sgl_kernel"] = sgl_kernel
        # sgl_kernel 子模块 (按需扩展)
        for sub in ["sgl_kernel.ops", "sgl_kernel.page", "sgl_kernel.kvcacheio"]:
            m = _SglKernelModule(sub)
            m.__path__ = []
            sys.modules[sub] = m
            setattr(sgl_kernel, sub.split(".")[-1], m)


def _make_stub(name, parent=None, **attrs):
    """创建一个 stub 模块并注册到 sys.modules，同时挂载到父模块属性上。"""
    m = types.ModuleType(name)
    # 让 Python 认为这是一个 package (有 __path__)
    m.__path__ = []
    m.__spec__ = None  # 避免 ValueError: __spec__ is None
    # 设置自定义属性
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    # 挂到父模块
    if parent is not None:
        short = name.rsplit(".", 1)[-1]
        setattr(parent, short, m)
    return m


def _install_triton_stub():
    """创建 triton 的完整内存 stub 模块树。"""
    from importlib.machinery import ModuleSpec

    class _TritonDtype:
        """Stub for triton dtype/function - 可被调用、可被用作装饰器。
        不支持字符串操作 (str/repr 不伪装成真实值)，确保代码在使用
        stub 返回值时能及时发现并走 fallback 路径。"""

        def __call__(self, *args, **kwargs):
            if args and callable(args[0]):
                return args[0]  # 用作装饰器
            return _TritonDtype()

        def __getattr__(self, name):
            if name.startswith("_"):
                raise AttributeError(name)
            return _TritonDtype()

        # 不提供 __str__/__hash__/__eq__ 等，保持 object 默认行为
        # 这确保 getattr(torch, triton_dtype_obj) 等操作会正常报错

    # --- triton 根模块 (带 __getattr__ 兜底) ---
    class _TritonRootModule(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("_"):
                raise AttributeError(name)
            # 对常见装饰器返回 identity decorator
            if name in ("jit", "autotune", "heuristics"):
                def _deco(*a, **k):
                    if a and callable(a[0]):
                        return a[0]
                    return lambda fn: fn
                return _deco
            # 返回一个可继承、可调用的 placeholder class
            return type(name, (_TritonPlaceholderBase,), {})

    triton = _TritonRootModule("triton")
    triton.__path__ = []
    triton.__spec__ = ModuleSpec("triton", None, is_package=True)
    triton.__version__ = "0.0.0"
    triton.jit = lambda *a, **k: (a[0] if a and callable(a[0]) else lambda fn: fn)
    triton.autotune = lambda *a, **k: lambda fn: fn
    triton.heuristics = lambda *a, **k: lambda fn: fn
    triton.cdiv = lambda a, b: (a + b - 1) // b
    triton.next_power_of_2 = lambda n: 1 << (n - 1).bit_length() if n > 0 else 1
    sys.modules["triton"] = triton

    # --- triton.language (使用 __getattr__ 兜底所有未定义属性) ---
    # 可继承的 placeholder 基类 (用于 KernelInterface 等)
    class _TritonPlaceholderBase:
        def __init__(self, *args, **kwargs):
            pass

        def __init_subclass__(cls, **kwargs):
            pass

    class _TritonLangModule(types.ModuleType):
        """triton 子模块 stub: 未定义属性自动返回 placeholder。
        大写开头的名字返回一个可继承的 class，其他返回 _TritonDtype 实例。"""

        def __getattr__(self, name):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            # 大写开头 → 可能是类 (如 KernelInterface, Config)，返回可继承的 class
            if name[0:1].isupper():
                return type(name, (_TritonPlaceholderBase,), {})
            return _TritonDtype()

    lang = _TritonLangModule("triton.language")
    lang.__path__ = []
    lang.__spec__ = None
    lang.constexpr = int
    lang.dtype = _TritonDtype
    sys.modules["triton.language"] = lang
    triton.language = lang
    for name in [
        "int1", "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64",
        "float8e4nv", "float8e5", "float16", "bfloat16", "float32", "float64",
        "void",
    ]:
        setattr(lang, name, _TritonDtype())

    # triton.language 子模块 (也用 __getattr__ 兜底)
    def _make_lang_sub(name, parent, **attrs):
        m = _TritonLangModule(name)
        m.__path__ = []
        m.__spec__ = None
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        setattr(parent, name.rsplit(".", 1)[-1], m)
        return m

    _make_lang_sub("triton.language.core", lang, dtype=_TritonDtype)
    _make_lang_sub("triton.language.math", lang)
    extra = _make_stub("triton.language.extra", parent=lang)
    cuda = _make_stub("triton.language.extra.cuda", parent=extra)
    _make_stub("triton.language.extra.cuda.libdevice", parent=cuda)
    _make_stub("triton.language.extra.libdevice", parent=extra)

    # --- triton.compiler (带 __getattr__ 兜底) ---
    compiler = _TritonLangModule("triton.compiler")
    compiler.__path__ = []
    compiler.__spec__ = None
    sys.modules["triton.compiler"] = compiler
    triton.compiler = compiler

    compiler_compiler = _TritonLangModule("triton.compiler.compiler")
    compiler_compiler.__path__ = []
    compiler_compiler.__spec__ = None
    compiler_compiler.AttrsDescriptor = type("AttrsDescriptor", (), {})
    sys.modules["triton.compiler.compiler"] = compiler_compiler
    compiler.compiler = compiler_compiler

    # torch._inductor uses Union[..., triton.compiler.CompiledKernel] which needs a type
    CompiledKernel = type("CompiledKernel", (), {})
    compiler.CompiledKernel = CompiledKernel

    # --- triton.runtime (带 __getattr__) ---
    runtime = _TritonLangModule("triton.runtime")
    runtime.__path__ = []
    runtime.__spec__ = None
    sys.modules["triton.runtime"] = runtime
    triton.runtime = runtime
    for rt_sub in ["autotuner", "jit", "interpreter", "cache"]:
        m = _TritonLangModule(f"triton.runtime.{rt_sub}")
        m.__path__ = []
        m.__spec__ = None
        sys.modules[f"triton.runtime.{rt_sub}"] = m
        setattr(runtime, rt_sub, m)

    # triton.runtime.driver 需要特殊处理:
    # 代码常用 triton.runtime.driver.active.xxx，必须让 .active 访问抛异常
    # 这样 sglang 的 try/except 才能正确走 fallback
    class _TritonDriverModule(types.ModuleType):
        @property
        def active(self):
            raise RuntimeError("Triton driver not available (Mac CPU stub)")

        def __getattr__(self, name):
            if name.startswith("_"):
                raise AttributeError(name)
            raise RuntimeError(f"Triton driver.{name} not available (Mac CPU stub)")

    driver = _TritonDriverModule("triton.runtime.driver")
    driver.__path__ = []
    driver.__spec__ = None
    sys.modules["triton.runtime.driver"] = driver
    runtime.driver = driver

    # --- triton.backends ---
    backends = _make_stub("triton.backends", parent=triton)
    _make_stub(
        "triton.backends.compiler",
        parent=backends,
        GPUTarget=type("GPUTarget", (), {}),
    )

    # --- triton.testing ---
    _make_stub("triton.testing", parent=triton)


# --- pytest 自动执行 ---
patch_mac()

# 确保 python/ 在 sys.path 中
import os

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_python_path = os.path.join(_project_root, "python")
if _python_path not in sys.path:
    sys.path.insert(0, _python_path)
