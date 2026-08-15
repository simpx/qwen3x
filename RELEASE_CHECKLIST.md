# Pre-release checklist

This repository is ready for a public **development preview** once the owner
chooses and adds a code license. It is not yet a Qwen3.8-27B performance
release: Metal remains unimplemented and 27B needs a larger validation host.

## Required before the first public push

- [ ] Choose the repository code license (the code and separately downloaded
      model weights must be considered independently).
- [ ] Confirm that no `models/`, `build/`, `reference/.deps/`, binaries, or
      local trace directories are staged.
- [ ] Enable the included GitHub Actions workflow and verify it on the remote.
- [ ] Run `make test`, `make tiny-test`, and `make dev-test`.
- [ ] On a development machine with the local checkpoint, run
      `make oracle-test MODEL=models/Qwen3.5-0.8B` and
      `make tokenizer-test MODEL=models/Qwen3.5-0.8B`.
- [ ] On an NVIDIA machine, run `make tiny-cuda-test` and
      `make cuda-dev-test MODEL=models/Qwen3.5-0.8B`.

## Required before claiming Qwen3.8-27B support is validated

- [ ] Run the strict schema check and a full CPU intermediate trace against
      the official Qwen3.8-27B checkpoint on a host that can hold its BF16
      weights.
- [ ] Repeat the trace and decode comparisons for the CUDA implementation.
- [ ] Implement and compare a Metal path on Apple Silicon.
- [ ] Record exact hardware, model revision, prompt IDs, tolerances, and
      generated IDs in the release notes.

## Scope guardrails

Do not delay this checklist by adding general model compatibility, GGUF,
quantization formats, serving, distributed execution, multimodality, or a
generic tensor/backend framework. Those are separate projects.
