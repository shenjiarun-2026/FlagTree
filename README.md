# FlagTree SUPA：壁仞 GPU 适配

本分支在 [FlagTree](https://github.com/flagos-ai/FlagTree) 上增加壁仞 GPU 的 SUPA backend。

Python 侧的 backend 在 [`third_party/supa/backend/`](third_party/supa/backend/)。`FLAGTREE_BACKEND=supa` 时，FlagTree 从这里加载 driver、compiler 和 PyTorch Inductor 桥接。`python/triton/backends/` 由安装流程生成，不入库；本地开发时 `python/triton/backends/supa` 指向 `third_party/supa/backend`。

## 环境

在已安装壁仞 SDK 和 `torch_br` 的环境里执行：

```bash
source /usr/local/birensupa/sdk/1.10.0.0-24145.lkg/scripts/brsw_set_env.sh
export PYTHONPATH=/Your_PATH_to/FlagTree/python:${PYTHONPATH}
export FLAGTREE_BACKEND=supa
export TRITON_ALWAYS_COMPILE=1
export TORCHINDUCTOR_COMPILE_THREADS=1
```

`PYTHONPATH` 要加在原有路径前面，不要整段替换，否则 SUDNN 会找不到 `br_generator`。

## 运行 Qwen3-Embedding 编译测试

脚本在 [`third_party/supa/python/test/test_qwen3_compile.py`](third_party/supa/python/test/test_qwen3_compile.py)。它用 `Qwen/Qwen3-Embedding-0.6B`（bfloat16）比较 eager 和 `torch.compile` 的 pooled embedding。

设置上面的环境变量后运行：

```bash
python3 third_party/supa/python/test/test_qwen3_compile.py
```

通过时脚本会打印 `QWEN3 EMBEDDING TORCH.COMPILE PASS`。
