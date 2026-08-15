# Contributing

qwen38.cpp is intentionally narrow. A useful contribution makes the fixed
Qwen execution easier to read, more correct, or measurably faster without
turning the project into a generic inference framework.

Before opening a change, run the dependency-free checks:

```bash
make test
make tiny-test
make dev-test
```

If a local Qwen3.5-0.8B checkpoint and the development oracle are available,
also run:

```bash
scripts/test_08b_oracle.sh models/Qwen3.5-0.8B
make qwen38_08b_cuda
./qwen38_08b_cuda --cuda-compare models/Qwen3.5-0.8B 248044,198,198
```

Keep the CPU reference plain and explicit. A faster Metal or CUDA kernel may
be added beside it, but must retain a direct numerical comparison path back to
the CPU/oracle. Do not add a generic tensor type, virtual backend hierarchy,
model registry, computation graph, or unrelated model compatibility as an
incidental refactor.

Downloaded checkpoints and Python oracle dependencies are local test fixtures;
they are ignored by Git and must not be committed. Please explain the source
and tolerance for any new numerical regression test.
