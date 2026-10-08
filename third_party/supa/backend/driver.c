#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdint.h>
#include <stdio.h>

#include <supa_runtime.h>
#include <driver/supa_driver_base.h>


static int set_supa_error(
    suError_t code,
    const char *where
) {
    if (code == 0) {
        return 1;
    }

    const char *name = suGetErrorName(code);
    const char *msg = suGetErrorString(code);

    PyErr_Format(
        PyExc_RuntimeError,
        "Triton SUPA error at %s: "
        "code=%d, name=%s, msg=%s",
        where,
        (int)code,
        name ? name : "<unknown>",
        msg ? msg : "<unknown>"
    );

    return 0;
}


/*
 * Python:
 *
 *   load_binary(
 *       kernel_name: str,
 *       binary: bytes,
 *       shared: int,
 *       device: int,
 *   )
 *
 * Returns:
 *
 *   (
 *       module_handle,
 *       function_handle,
 *       n_regs,
 *       n_spills,
 *       n_max_threads,
 *   )
 */
static PyObject *
loadBinary(
    PyObject *self,
    PyObject *args
) {
    const char *name = NULL;
    const char *data = NULL;

    Py_ssize_t data_size = 0;

    int shared = 0;
    int device = 0;

    if (!PyArg_ParseTuple(
            args,
            "ss#ii",
            &name,
            &data,
            &data_size,
            &shared,
            &device)) {
        return NULL;
    }

    (void)shared;
    (void)device;

    if (data_size <= 0) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "SUPA binary is empty"
        );
        return NULL;
    }


    /*
     * IMPORTANT:
     *
     * Do not create another SUPA context here.
     *
     * torch_br owns the current context and Triton kernels
     * must use the same context as torch.supa tensors.
     */
    suContext ctx = 0;

    suError_t status =
        sudrvContextGetCurrent(&ctx);

    if (!set_supa_error(
            status,
            "sudrvContextGetCurrent")) {
        return NULL;
    }

    if (ctx == 0) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "No current SUPA context. "
            "Initialize torch.supa before loading "
            "a Triton kernel."
        );
        return NULL;
    }


    suModule_t module = 0;
    suDevFunc_t function = 0;


    /*
     * binary must already be ET_EXEC here.
     *
     * LLVM IR -> BirenLLC -> ET_REL -> BirenLink
     *
     * belongs to compiler.py, not driver.c.
     */
    status = sudrvModuleLoadData(
        &module,
        (const void *)data
    );

    if (!set_supa_error(
            status,
            "sudrvModuleLoadData")) {
        return NULL;
    }


    status = sudrvModuleGetFunction(
        &function,
        module,
        name
    );

    if (status != 0) {
        /*
         * Module was created successfully, but the
         * kernel lookup failed. Clean it up.
         */
        sudrvModuleUnload(module);

        set_supa_error(
            status,
            "sudrvModuleGetFunction"
        );

        return NULL;
    }


    /*
     * TODO:
     * Query actual register/spill/max-thread attributes
     * through sudrvFuncGetAttribute().
     *
     * Triton CompiledKernel rejects the kernel when
     * num_warps * warp_size > n_max_threads, so 0 would
     * make every launch fail OutOfResources.
     */
    int n_regs = 0;
    int n_spills = 0;
    int n_max_threads = 1024;


    return Py_BuildValue(
        "(KKiii)",
        (uint64_t)module,
        (uint64_t)function,
        n_regs,
        n_spills,
        n_max_threads
    );
}


static PyObject *
unloadBinary(PyObject *self, PyObject *args)
{
    unsigned long long module = 0;

    if (!PyArg_ParseTuple(args, "K", &module)) {
        return NULL;
    }

    suError_t status =
        sudrvModuleUnload((suModule_t)module);

    if (!set_supa_error(
            status,
            "sudrvModuleUnload")) {
        return NULL;
    }

    Py_RETURN_NONE;
}


static PyMethodDef ModuleMethods[] = {
    {
        "load_binary",
        loadBinary,
        METH_VARARGS,
        "Load a linked Biren ELF and resolve a SUPA kernel"
    },
    {
        "unload_binary",
        unloadBinary,
        METH_VARARGS,
        "Unload a SUPA module"
    },

    {
        NULL,
        NULL,
        0,
        NULL
    }
};


static struct PyModuleDef ModuleDef = {
    PyModuleDef_HEAD_INIT,

    "supa_utils",

    NULL,

    -1,

    ModuleMethods
};


PyMODINIT_FUNC
PyInit_supa_utils(void) {
    return PyModule_Create(
        &ModuleDef
    );
}
