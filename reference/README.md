# Development oracle

This directory is not part of the qwen38 runtime. It pins a Transformers
revision with Qwen3.5 support, dumps FP32 intermediate tensors, and compares
them with `qwen38_08b --trace`.

```bash
python3 -m pip install --target reference/.deps -r reference/requirements.txt
scripts/test_08b_oracle.sh models/Qwen3.5-0.8B
```

`dump_qwen35_trace.py` evaluates one token at a time with `past_key_values`,
matching qwen38 prefill/decode state evolution. It writes only the final
token's trace. `compare_traces.py` uses raw little-endian FP32 arrays and a
default absolute tolerance of `1e-3`.

`test_tokenizer.py` checks several Unicode, whitespace, and special-token
cases against a CMake qwen38 build using the tokenizer C ABI.
