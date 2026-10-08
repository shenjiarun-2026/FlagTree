"""Register the SUPA device with PyTorch Inductor.

PyTorch treats cuda/xpu/mps as the only GPU backends. This module adds the
PrivateUse1 ``supa`` device to that registry so ``torch.compile`` can emit
Triton and hand it to the FlagTree SUPA backend. The C++ GPU wrapper is not
registered: it assumes a CUDA runtime.
"""

import threading
from contextlib import contextmanager

import torch

_REGISTERED = False
_LOCK = threading.Lock()


def _get_raw_supa_stream(device_idx):
    stream = torch.supa.current_stream(device_idx)
    for attr in ("supa_stream", "stream", "cuda_stream"):
        if hasattr(stream, attr):
            value = getattr(stream, attr)
            if hasattr(value, "value"):
                value = value.value
            return int(value)
    return int(stream)


def get_raw_stream(device_idx):
    return _get_raw_supa_stream(device_idx)


@contextmanager
def supa_device_guard(device_idx):
    old = torch.supa.current_device()
    torch.supa.set_device(device_idx)
    try:
        yield
    finally:
        torch.supa.set_device(old)


def _register_codegen():
    from torch._inductor import utils as inductor_utils
    from torch._inductor.codegen.common import register_backend_for_device
    from torch._inductor.codegen.triton import TritonScheduling
    from torch._inductor.codegen.wrapper import PythonWrapperCodegen

    if "supa" not in inductor_utils.GPU_TYPES:
        inductor_utils.GPU_TYPES.append("supa")
    inductor_utils.get_gpu_type.cache_clear()
    inductor_utils.GPU_KERNEL_BIN_EXTS["supa"] = ".birelf"
    register_backend_for_device(
        "supa",
        TritonScheduling,
        PythonWrapperCodegen,
    )


def _register_interface():
    from torch._dynamo.device_interface import (
        DeviceInterface,
        register_interface_for_device,
    )

    class SupaInterface(DeviceInterface):
        class Event(torch.supa.Event):
            pass

        class Stream(torch.supa.Stream):
            pass

        @staticmethod
        def current_device():
            return torch.supa.current_device()

        @staticmethod
        def set_device(device):
            return torch.supa.set_device(device)

        @staticmethod
        def device_count():
            if hasattr(torch.supa, "device_count"):
                return torch.supa.device_count()
            return 1

        @staticmethod
        def is_available():
            return torch.supa.is_available()

        @staticmethod
        def synchronize(device=None):
            if device is None:
                return torch.supa.synchronize()
            try:
                return torch.supa.synchronize(device)
            except TypeError:
                return torch.supa.synchronize()

        @staticmethod
        def get_raw_stream(device_idx):
            return _get_raw_supa_stream(device_idx)

        @staticmethod
        def get_compute_capability(device=None):
            del device
            return 100

        @staticmethod
        def is_bf16_supported(including_emulation=False):
            del including_emulation
            return True

        @staticmethod
        def is_triton_capable(device=None):
            del device
            return True

        @staticmethod
        def memory_allocated(device=None):
            if hasattr(torch.supa, "memory_allocated"):
                return torch.supa.memory_allocated(device)
            return 0

        @staticmethod
        def exchange_device(device):
            old = torch.supa.current_device()
            torch.supa.set_device(device)
            return old

        @staticmethod
        def maybe_exchange_device(device):
            old = torch.supa.current_device()
            if device is not None and device >= 0:
                torch.supa.set_device(device)
            return old

        class Worker:
            @staticmethod
            def current_device():
                return torch.supa.current_device()

            @staticmethod
            def set_device(device):
                torch.supa.set_device(device)

            @staticmethod
            def get_device_properties(device=None):
                if device is None:
                    device = torch.supa.current_device()
                if isinstance(device, torch.device):
                    device = device.index
                return torch.supa.get_device_properties(device)

    register_interface_for_device("supa", SupaInterface)
    for i in range(SupaInterface.device_count()):
        register_interface_for_device(f"supa:{i}", SupaInterface)


def _register_op_overrides():
    from torch._inductor.codegen.common import (
        DeviceOpOverrides,
        register_device_op_overrides,
    )

    class SupaDeviceOpOverrides(DeviceOpOverrides):
        def import_get_raw_stream_as(self, name):
            return (
                "from triton.backends.supa.inductor_bridge "
                f"import get_raw_stream as {name}, supa_device_guard"
            )

        def set_device(self, device_idx):
            return f"torch.supa.set_device({device_idx})"

        def synchronize(self):
            return "torch.supa.synchronize()"

        def device_guard(self, device_idx):
            # Extern-only graphs (linear/mm) never import the Triton stream
            # helper, so the guard has to be something the wrapper already has.
            return f"torch.supa.device({device_idx})"

    register_device_op_overrides("supa", SupaDeviceOpOverrides())


def _register_has_triton():
    import sys

    import torch.utils._triton as triton_utils

    original = triton_utils.has_triton

    def has_triton():
        try:
            if torch.supa.is_available():
                return True
        except Exception:
            pass
        return original()

    triton_utils.has_triton = has_triton
    # Inductor modules bind has_triton at import time.
    for mod in list(sys.modules.values()):
        if mod is not None and getattr(mod, "has_triton", None) is original:
            mod.has_triton = has_triton


def _materialize_for_sudnn(tensor):
    """Allocate a tensor whose torch_br physical shape matches ``tensor``.

    Inductor ``reinterpret_tensor`` updates PyTorch sizes and strides but
    leaves the Biren physical shape behind. SUDNN ``bmm`` then rejects the
    batch dimension (``matrix_b batch dim cannot meet broadcast condition``)
    even when ``is_contiguous()`` is already true, so ``require_contiguous``
    does not insert a copy. A fresh ``empty`` gets a physical shape SUDNN
    accepts; ``copy_`` moves the logical values across.
    """
    fresh = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
    fresh.copy_(tensor)
    return fresh


def _register_supa_bmm_materialize():
    """Copy SUPA ``bmm`` inputs and outputs into fresh tensors before SUDNN."""
    from torch._inductor.select_algorithm import extern_kernels

    current = extern_kernels.bmm
    if getattr(current, "_supa_materialized", False):
        return

    def bmm(mat1, mat2, *, out=None):
        if mat1.device.type != "supa":
            return current(mat1, mat2, out=out)
        mat1_f = _materialize_for_sudnn(mat1)
        mat2_f = _materialize_for_sudnn(mat2)
        if out is None:
            return current(mat1_f, mat2_f)
        out_f = torch.empty(out.shape, dtype=out.dtype, device=out.device)
        current(mat1_f, mat2_f, out=out_f)
        out.copy_(out_f)
        return out

    bmm._supa_materialized = True
    extern_kernels.bmm = bmm


def _register_supa_constant_fold():
    """Read Inductor's uniform folded scalars from host memory.

    Inference constant folding replaces a uniform tensor with
    ``aten.full(shape, tensor.item())``. On SUPA the first ``item()`` of a
    bool tensor returns False while a CPU copy of the same storage is True.
    The causal-mask identity is ``full([16], True)`` viewed to ``[16, 1]``.
    Folding that view with the bad scalar rewrites it to ``full([16, 1],
    False)``, and the compiled mask is entirely the dtype minimum.
    """
    from torch._inductor.fx_passes import joint_graph
    from torch.multiprocessing.reductions import StorageWeakRef

    folder = joint_graph.UniformValueConstantFolder
    current = folder.add_node_replacement
    if getattr(current, "_supa_host_scalar", False):
        return

    def add_node_replacement(self, node, tensor):
        value_tensor = tensor
        if tensor.device.type == "supa":
            value_tensor = tensor.detach().cpu()
        self.node_replacements[node] = value_tensor.flatten()[0].item()
        self.node_replacements_shapes[node] = node.meta["val"].shape
        self.constant_data_ptrs[node] = StorageWeakRef(tensor.untyped_storage())

    add_node_replacement._supa_host_scalar = True
    folder.add_node_replacement = add_node_replacement


def _eager_qwen_rope():
    """Keep Qwen3 RoPE eager under ``torch.compile``.

    A single Inductor kernel that stores both ``cos`` and ``sin`` as bf16
    returns ``cos(0) == 0.0078125``. Compiling either output alone matches
    eager bitwise, and ``torch.autocast(enabled=False)`` is not required to
    reproduce the failure. Outer SUPA autocast is off
    (``torch.is_autocast_enabled("supa")`` is False; the PrivateUse1 default
    dtype is still fp16 while ``get_amp_supported_dtype()`` is bf16), so
    leaving this forward eager does not drop the explicit ``.float()`` math.
    """
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    except Exception:
        return
    current = Qwen3RotaryEmbedding.forward
    if getattr(current, "_supa_eager_rope", False):
        return
    disabled = torch.compiler.disable(current)
    disabled._supa_eager_rope = True
    Qwen3RotaryEmbedding.forward = disabled


def register_supa_inductor():
    """Register SUPA once per process. Safe to call from every compile test."""
    global _REGISTERED
    with _LOCK:
        if _REGISTERED:
            return
        _register_codegen()
        _register_interface()
        _register_op_overrides()
        _register_has_triton()
        _register_supa_bmm_materialize()
        _register_supa_constant_fold()
        _eager_qwen_rope()
        # Inductor's async worker pickles the compile job. The SUPA compiler
        # keeps a process-wide BirenLLC lock, which cannot cross that boundary.
        from torch._inductor import config as inductor_config

        inductor_config.compile_threads = 1
        _REGISTERED = True
