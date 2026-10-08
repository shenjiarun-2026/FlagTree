import os
from pathlib import Path

from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase
from triton.runtime.build import compile_module_from_src


dirname = Path(__file__).resolve().parent


DEFAULT_BIREN_SDK = (
    "/usr/local/birensupa/"
    "sdk/1.10.0.0-24145.lkg"
)


def get_biren_sdk():
    sdk = Path(
        os.environ.get(
            "BIREN_SDK",
            DEFAULT_BIREN_SDK,
        )
    )

    if not sdk.exists():
        raise RuntimeError(
            f"BIREN SDK does not exist: {sdk}"
        )

    return sdk


def include_dirs():
    sdk = get_biren_sdk()

    return [
        str(
            sdk / "supa/include"
        )
    ]


def library_dirs():
    sdk = get_biren_sdk()

    return [
        str(
            sdk / "supa/lib"
        )
    ]


def ty_to_cpp(ty: str) -> str:
    if ty.startswith("*"):
        return "uint64_t"

    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",

        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",

        "fp16": "uint16_t",
        "bf16": "uint16_t",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",

        "constexpr": "int64_t",
    }[ty]


def ty_to_cpp_param(ty: str) -> str:
    # Kernel ABI type. Distinct from the Python C-API
    # parse type: Python floats arrive as double and are
    # then packed/cast to float/fp16/bf16.
    if ty.startswith("*"):
        return "uint64_t"

    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",

        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",

        "fp16": "uint16_t",
        "bf16": "uint16_t",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",

        "constexpr": "int64_t",
    }[ty]


class SupaUtils:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance

    def __init__(self):
        if getattr(
            self,
            "_initialized",
            False,
        ):
            return

        src = (
            dirname / "driver.c"
        ).read_text()

        mod = compile_module_from_src(
            src=src,
            name="supa_utils",

            include_dirs=include_dirs(),

            library_dirs=library_dirs(),

            libraries=[
                "supa-runtime",
            ],
        )

        self.load_binary = (
            mod.load_binary
        )
        self.unload_binary = (
            mod.unload_binary
        )

        self._initialized = True


    @staticmethod
    def get_device_properties(device):
        import torch
        import torch_br  # noqa: F401

        props = (
            torch.supa
            .get_device_properties(device)
        )

        return {
            "name":
                str(props.name),

            "total_memory":
                int(props.total_memory),

            "multiprocessor_count":
                int(
                    getattr(
                        props,
                        "multi_processor_count",
                        0,
                    )
                ),

            "max_compute_units":
                int(
                    getattr(
                        props,
                        "max_compute_units",
                        0,
                    )
                ),

            "max_threads_per_multi_processor":
                int(
                    getattr(
                        props,
                        "max_threads_per_multi_processor",
                        0,
                    )
                ),

            "die_count":
                int(
                    getattr(
                        props,
                        "die_count",
                        1,
                    )
                ),

            # CompiledKernel checks metadata.shared against this.
            "max_shared_mem":
                int(
                    getattr(
                        props,
                        "shared_memory_per_block",
                        getattr(
                            props,
                            "sharedMemPerBlock",
                            256 * 1024,
                        ),
                    )
                ),

            # BR166 SUPA SDK examples use warp_size=32.
            "warpSize": 32,
        }


def make_launcher(
    constants,
    signature,
    warp_size,
    multi_die,
):
    # Kernel ABI parameters. constexpr and specialization
    # constants are parsed from Python but not passed on.
    params = [
        i
        for i, ty in signature.items()
        if ty != "constexpr" and i not in constants
    ]

    arg_decls = ", ".join(
        f"{ty_to_cpp_param(signature[i])} arg{i}"
        for i in params
    )

    def parse_type(ty):
        if ty.startswith("*"):
            return "PyObject*"

        if ty == "constexpr":
            return "PyObject*"

        # Python float enters the C extension as double.
        if ty in (
            "fp16",
            "bf16",
            "fp32",
            "f32",
            "fp64",
        ):
            return "double"

        return ty_to_cpp_param(ty)

    def format_of(c_ty):
        return {
            "PyObject*": "O",

            "int8_t": "b",
            "int16_t": "h",
            "int32_t": "i",
            "int64_t": "L",

            "uint8_t": "B",
            "uint16_t": "H",
            "uint32_t": "I",
            "uint64_t": "K",

            "float": "f",
            "double": "d",
        }[c_ty]

    args_format = "".join(
        format_of(parse_type(ty))
        for ty in signature.values()
    )

    #
    # Triton runtime calling convention:
    #
    # gridX gridY gridZ
    # stream
    # function
    # kernel_metadata
    # launch_metadata
    # enter_hook
    # exit_hook
    #
    fmt = "iiiKKOOOO" + args_format

    args_list = ""
    if signature:
        args_list = ", " + ", ".join(
            f"&_arg{i}"
            for i in signature
        )

    packed_decls = []
    packed_inits = []
    launch_args = []

    for i in params:
        ty = signature[i]

        if ty.startswith("*"):
            launch_args.append(
                f"ptr_info{i}.dev_ptr"
            )
            continue

        if ty == "fp16":
            packed_decls.append(
                f"uint16_t arg{i};"
            )
            packed_inits.append(
                f"arg{i} = pack_fp16(_arg{i});"
            )
            launch_args.append(f"arg{i}")

        elif ty == "bf16":
            packed_decls.append(
                f"uint16_t arg{i};"
            )
            packed_inits.append(
                f"arg{i} = pack_bf16(_arg{i});"
            )
            launch_args.append(f"arg{i}")

        elif ty in ("fp32", "f32"):
            packed_decls.append(
                f"float arg{i} = (float)_arg{i};"
            )
            launch_args.append(f"arg{i}")

        else:
            launch_args.append(f"_arg{i}")

    packed_decls_src = "\n".join(
        f"    {x}"
        for x in packed_decls
    )

    packed_inits_src = "\n".join(
        f"    {x}"
        for x in packed_inits
    )

    unused_src = "\n".join(
        f"    (void)_arg{i};"
        for i, ty in signature.items()
        if ty == "constexpr" or i in constants
    )

    pointer_extract = "\n".join(
        f"""
    DevicePtrInfo ptr_info{i} =
        getPointer(_arg{i}, {i});

    if (!ptr_info{i}.valid)
        return NULL;
"""
        for i in params
        if signature[i].startswith("*")
    )

    if params:
        kernel_params_decl = (
            "void *params[] = { "
            + ", ".join(f"&arg{i}" for i in params)
            + " };"
        )
        kernel_params_arg = "params"
    else:
        # Zero-length arrays are a GCC extension.
        kernel_params_decl = "void **params = NULL;"
        kernel_params_arg = "params"

    launch_sig_extra = (
        ", " + arg_decls
        if arg_decls
        else ""
    )
    launch_call_extra = (
        ", " + ", ".join(launch_args)
        if launch_args
        else ""
    )
    arg_decls_c = " ".join(
        f"{parse_type(ty)} _arg{i};"
        for i, ty in signature.items()
    )

    if multi_die:
        multidie_code = """
    suLaunchAttribute launch_attr;
    memset(&launch_attr, 0, sizeof(launch_attr));

    launch_attr.id =
        suLaunchAttributeMultiDieDuplication;

    launch_attr.value.launchDuplication = 1;

    config.attrs = &launch_attr;
    config.numAttrs = 1;
"""
    else:
        multidie_code = """
    config.attrs = NULL;
    config.numAttrs = 0;
"""

    src = f"""
#define PY_SSIZE_T_CLEAN

#include <Python.h>

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include <supa_runtime.h>
#include <driver/supa_driver_base.h>


static inline uint16_t
pack_fp16(double val)
{{
    uint16_t result;

#if 0x030600B1 <= PY_VERSION_HEX && \\
    PY_VERSION_HEX <= 0x030B00A1 && \\
    !defined(PYPY_VERSION)

    _PyFloat_Pack2(
        val,
        (unsigned char *)&result,
        1
    );
#else
    PyFloat_Pack2(
        val,
        (char *)&result,
        1
    );
#endif

    return result;
}}


static inline uint16_t
pack_bf16(double val)
{{
    float f32 = (float)val;

    uint32_t u32;

    memcpy(
        &u32,
        &f32,
        sizeof(u32)
    );

    return (uint16_t)(
        u32 >> 16
    );
}}


static int
supa_check(
    suError_t code,
    const char *where
)
{{
    if (code == 0)
        return 1;

    const char *name =
        suGetErrorName(code);

    const char *msg =
        suGetErrorString(code);

    PyGILState_STATE gil =
        PyGILState_Ensure();

    PyErr_Format(
        PyExc_RuntimeError,

        "Triton SUPA error at %s: "
        "code=%d name=%s msg=%s",

        where,
        (int)code,

        name
            ? name
            : "<unknown>",

        msg
            ? msg
            : "<unknown>"
    );

    PyGILState_Release(gil);

    return 0;
}}


typedef struct {{
    uint64_t dev_ptr;
    bool valid;
}} DevicePtrInfo;


static inline DevicePtrInfo
getPointer(
    PyObject *obj,
    int idx
)
{{
    DevicePtrInfo out;

    out.dev_ptr = 0;
    out.valid = true;


    // Triton sometimes already passes a device address.
    if (PyLong_Check(obj)) {{
        out.dev_ptr =
            PyLong_AsUnsignedLongLong(obj);

        return out;
    }}


    if (obj == Py_None) {{
        return out;
    }}


    PyObject *method =
        PyObject_GetAttrString(
            obj,
            "data_ptr"
        );

    if (!method) {{
        PyErr_Format(
            PyExc_TypeError,
            "Pointer argument %d must "
            "be integer/None or expose data_ptr()",
            idx
        );

        out.valid = false;
        return out;
    }}


    PyObject *ret =
        PyObject_CallNoArgs(method);

    Py_DECREF(method);


    if (!ret) {{
        out.valid = false;
        return out;
    }}


    if (!PyLong_Check(ret)) {{
        Py_DECREF(ret);

        PyErr_Format(
            PyExc_TypeError,
            "data_ptr() for argument %d "
            "must return an integer",
            idx
        );

        out.valid = false;
        return out;
    }}


    out.dev_ptr =
        PyLong_AsUnsignedLongLong(ret);

    Py_DECREF(ret);

    if (PyErr_Occurred()) {{
        out.valid = false;
    }}

    return out;
}}


static void
launch_kernel(
    int gridX,
    int gridY,
    int gridZ,

    int num_warps,
    int num_ctas,
    int shared_memory,

    suStream_t stream,
    suDevFunc_t function{launch_sig_extra}
)
{{
    // First version does not support Triton cluster / multi-CTA.
    if (num_ctas != 1) {{
        PyGILState_STATE gil =
            PyGILState_Ensure();

        PyErr_Format(
            PyExc_NotImplementedError,
            "SUPA launcher currently requires "
            "num_ctas == 1, got %d",
            num_ctas
        );

        PyGILState_Release(gil);
        return;
    }}


    {kernel_params_decl}


    if (
        gridX == 0 ||
        gridY == 0 ||
        gridZ == 0
    ) {{
        return;
    }}


    suLaunchConfig config;

    memset(
        &config,
        0,
        sizeof(config)
    );


    config.gridDimX = gridX;
    config.gridDimY = gridY;
    config.gridDimZ = gridZ;


    // Triton num_warps -> SUPA G-Mode blockDimX.
    config.blockDimX =
        {warp_size} * num_warps;

    config.blockDimY = 1;
    config.blockDimZ = 1;


    config.sharedMemBytes =
        shared_memory;

    config.stream = stream;


    {multidie_code}


    suError_t status =
        sudrvLaunchKernelEx(
            &config,
            function,
            {kernel_params_arg},
            NULL
        );


    supa_check(
        status,
        "sudrvLaunchKernelEx"
    );
}}


static PyObject *
launch(
    PyObject *self,
    PyObject *args
)
{{
    int gridX;
    int gridY;
    int gridZ;

    uint64_t _stream;
    uint64_t _function;

    PyObject *kernel_metadata = NULL;
    PyObject *launch_metadata = NULL;

    PyObject *launch_enter_hook = NULL;
    PyObject *launch_exit_hook = NULL;


    {arg_decls_c}


    if (!PyArg_ParseTuple(
            args,
            "{fmt}",

            &gridX,
            &gridY,
            &gridZ,

            &_stream,
            &_function,

            &kernel_metadata,
            &launch_metadata,

            &launch_enter_hook,
            &launch_exit_hook

            {args_list}
    )) {{
        return NULL;
    }}


    int num_warps;
    int num_ctas;
    int shared_memory;


    if (!PyArg_ParseTuple(
            kernel_metadata,
            "iii",

            &num_warps,
            &num_ctas,
            &shared_memory
    )) {{
        PyErr_SetString(
            PyExc_TypeError,
            "kernel_metadata must be "
            "(num_warps, num_ctas, shared_memory)"
        );

        return NULL;
    }}


    if (
        launch_enter_hook != Py_None
    ) {{
        PyObject *hook_args =
            Py_BuildValue(
                "(O)",
                launch_metadata
            );

        PyObject *ret =
            PyObject_CallObject(
                launch_enter_hook,
                hook_args
            );

        Py_DECREF(hook_args);

        if (!ret)
            return NULL;

        Py_DECREF(ret);
    }}


{pointer_extract}

{packed_decls_src}
{packed_inits_src}
{unused_src}


    Py_BEGIN_ALLOW_THREADS;


    launch_kernel(
        gridX,
        gridY,
        gridZ,

        num_warps,
        num_ctas,
        shared_memory,

        (suStream_t)_stream,
        (suDevFunc_t)_function{launch_call_extra}
    );


    Py_END_ALLOW_THREADS;


    if (PyErr_Occurred()) {{
        return NULL;
    }}


    if (
        launch_exit_hook != Py_None
    ) {{
        PyObject *hook_args =
            Py_BuildValue(
                "(O)",
                launch_metadata
            );

        PyObject *ret =
            PyObject_CallObject(
                launch_exit_hook,
                hook_args
            );

        Py_DECREF(hook_args);

        if (!ret)
            return NULL;

        Py_DECREF(ret);
    }}


    Py_RETURN_NONE;
}}


static PyMethodDef ModuleMethods[] = {{
    {{
        "launch",
        launch,
        METH_VARARGS,
        "SUPA Triton launcher"
    }},
    {{
        NULL,
        NULL,
        0,
        NULL
    }}
}};


static struct PyModuleDef ModuleDef = {{
    PyModuleDef_HEAD_INIT,

    "__triton_launcher",

    NULL,

    -1,

    ModuleMethods
}};


PyMODINIT_FUNC
PyInit___triton_launcher(void)
{{
    return PyModule_Create(
        &ModuleDef
    );
}}
"""

    return src


def _supa_tensor_layout(tensor):
    """Biren physical layout, or None when this is not a SUPA tensor."""
    device = getattr(tensor, "device", None)
    if device is None or getattr(device, "type", None) != "supa":
        return None
    if not hasattr(tensor, "br_layout"):
        return None
    try:
        return tensor.br_layout
    except Exception:
        return None


def _linearize_supa_arg(tensor, seen, restores):
    """Hand Triton a BUFFER_ANY pointer with the tensor's logical strides.

    Eager SUPA ops such as embedding rewrite a tensor in place from
    BUFFER_ANY to COLMAJOR. ``stride`` stays ``(H, 1)`` and
    ``data_ptr()`` still looks like a base address, but the bytes are
    tiled. ``tl.load(table + token * H + col)`` then reads the wrong
    row. A fresh ``torch.empty`` is BUFFER_ANY, and ``copy_`` moves
    the logical values into that linear buffer.
    """
    key = id(tensor)
    if key in seen:
        return seen[key]
    layout = _supa_tensor_layout(tensor)
    if layout is None or layout == "BUFFER_ANY":
        seen[key] = tensor
        return tensor
    import torch

    # Match the strides the kernel was given. A contiguous COLMAJOR
    # tensor still reports stride (H, 1); the fresh buffer is the
    # linear storage those strides describe.
    if tensor.is_contiguous():
        linear = torch.empty(
            tensor.shape,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        linear = torch.empty_strided(
            tensor.shape,
            tensor.stride(),
            dtype=tensor.dtype,
            device=tensor.device,
        )
    linear.copy_(tensor)
    seen[key] = linear
    restores.append((tensor, linear))
    if os.environ.get("SUPA_LOG_LAYOUT") == "1":
        print(
            "[SUPA layout] linearize",
            layout,
            "shape",
            tuple(tensor.shape),
            "stride",
            tuple(tensor.stride()),
            "contig",
            tensor.is_contiguous(),
            "ptr",
            hex(tensor.data_ptr()),
            "storage",
            hex(tensor.untyped_storage().data_ptr()),
            flush=True,
        )
    return linear


class SupaLauncher:
    """
    Per-signature generated launcher.

    Triton builds this C extension at kernel-init time.
    MVP supports a flat signature only: no tuples, no
    tensor descriptors, no global scratch.
    """

    def __init__(
        self,
        src,
        metadata,
    ):
        #
        # Triton signature keys are sometimes an arg
        # index and sometimes a name.
        #
        def cst_key(key):
            if isinstance(key, str):
                return src.fn.arg_names.index(key)

            if (
                isinstance(key, tuple)
                and len(key) == 1
            ):
                return key[0]

            return key

        constants = (
            src.constants
            if hasattr(src, "constants")
            else {}
        )

        constants = {
            cst_key(k): v
            for k, v in constants.items()
        }

        signature = {
            cst_key(k): v
            for k, v in src.signature.items()
        }

        #
        # MVP: flat signature only.
        #
        for ty in signature.values():
            if not isinstance(ty, str):
                raise NotImplementedError(
                    "SUPA MVP launcher does not "
                    "support structured arguments "
                    f"yet: {ty!r}"
                )

            if ty.startswith("tensordesc"):
                raise NotImplementedError(
                    "SUPA MVP launcher does not "
                    "support tensor descriptors yet"
                )

        target = getattr(metadata, "target", None)
        warp_size = (
            int(target.warp_size)
            if (
                target is not None
                and getattr(target, "warp_size", None)
            )
            else 32
        )

        # Dual-die parts do not schedule every block unless the
        # launch is duplicated. Without it, program ids past the
        # first die stay at the fill value (a 96-block copy
        # returned 0 for ids 64..95).
        import triton

        active_driver = triton.runtime.driver.active
        device = active_driver.get_current_device()
        props = active_driver.utils.get_device_properties(
            device
        )
        die_count = int(props.get("die_count", 1))
        dup_env = os.environ.get("TRITON_SUPA_MULTI_DIE_DUP")
        if dup_env is None:
            multi_die = die_count > 1
        else:
            multi_die = bool(int(dup_env))

        launcher_src = make_launcher(
            constants=constants,
            signature=signature,
            warp_size=warp_size,
            multi_die=multi_die,
        )

        mod = compile_module_from_src(
            src=launcher_src,
            name="__triton_launcher",
            include_dirs=include_dirs(),
            library_dirs=library_dirs(),
            libraries=[
                "supa-runtime",
            ],
        )

        self.launch = mod.launch
        self.signature = signature
        self.warp_size = warp_size
        self.die_count = die_count
        # Same order as the kernel arguments appended after the
        # grid/stream/function prefix.
        self._arg_is_ptr = [
            isinstance(ty, str) and ty.startswith("*")
            for ty in signature.values()
        ]

    def __call__(
        self,
        *args,
        **kwargs,
    ):
        n_args = len(self._arg_is_ptr)
        if n_args == 0 or len(args) < n_args:
            return self.launch(*args, **kwargs)
        prefix = args[:-n_args]
        kernel_args = list(args[-n_args:])
        restores = []
        seen = {}
        for index, is_ptr in enumerate(self._arg_is_ptr):
            if not is_ptr:
                continue
            kernel_args[index] = _linearize_supa_arg(
                kernel_args[index],
                seen,
                restores,
            )
        result = self.launch(*prefix, *kernel_args, **kwargs)
        # detach() so a leaf parameter (the embedding table) can be
        # written back. copy_ is layout-aware: linear bytes go back
        # into COLMAJOR storage as logical values.
        for original, linear in restores:
            original.detach().copy_(linear)
        return result


class SupaDriver(DriverBase):

    def __init__(self):
        import torch
        import torch_br  # noqa: F401

        if not hasattr(
            torch,
            "supa",
        ):
            raise RuntimeError(
                "torch.supa is unavailable "
                "after importing torch_br"
            )

        self._torch = torch

        self.utils = SupaUtils()

        self.launcher_cls = (
            SupaLauncher
        )


    @classmethod
    def is_active(cls):
        try:
            import torch
            import torch_br  # noqa: F401

            return (
                hasattr(torch, "supa")
                and torch.supa.is_available()
            )

        except Exception:
            return False


    def map_python_to_cpp_type(
        self,
        ty: str,
    ) -> str:
        return ty_to_cpp(ty)


    def get_current_target(self):
        #
        # Important distinction:
        #
        # Physical product: BR166M
        # Compiler target: br100
        #
        # BRCC -### showed:
        #   -target-cpu br100
        #
        arch_env = os.environ.get(
            "TRITON_SUPA_ARCH",
            "100",
        )

        if arch_env.startswith("br"):
            arch_env = arch_env[2:]

        arch = int(arch_env)

        warp_size = int(
            os.environ.get(
                "TRITON_SUPA_WARP_SIZE",
                "32",
            )
        )

        return GPUTarget(
            "supa",
            arch,
            warp_size,
        )


    def get_active_torch_device(self):
        return self._torch.device(
            "supa",
            self.get_current_device(),
        )


    def get_current_device(self):
        return (
            self._torch
            .supa
            .current_device()
        )


    def set_current_device(
        self,
        device,
    ):
        self._torch.supa.set_device(
            device
        )


    def get_current_stream(
        self,
        device,
    ):
        stream = (
            self._torch
            .supa
            .current_stream(device)
        )

        # torch_br versions may expose the raw
        # stream handle under different names.
        for attr in (
            "supa_stream",
            "stream",
            "cuda_stream",
        ):
            if hasattr(stream, attr):
                value = getattr(
                    stream,
                    attr,
                )

                if hasattr(
                    value,
                    "value",
                ):
                    value = value.value

                return int(value)

        try:
            return int(stream)

        except Exception as exc:
            raise RuntimeError(
                "Cannot extract raw SUPA "
                f"stream handle from {stream!r}"
            ) from exc


    def get_device_interface(self):
        return self._torch.supa


    def get_benchmarker(self):
        from triton.testing import do_bench

        return do_bench


    def get_empty_cache_for_benchmark(
        self,
    ):
        # Same rough strategy as other GPU
        # backends. Not performance-critical now.
        size = 64 * 1024 * 1024

        return self._torch.empty(
            size // 4,
            dtype=self._torch.int32,
            device=self.get_active_torch_device(),
        )


    def clear_cache(
        self,
        cache,
    ):
        cache.zero_()
