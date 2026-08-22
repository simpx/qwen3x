PYTHON ?= python3
MODEL ?= ../models/Qwen3.5-0.8B
ENGINE ?= ../02-cpu-0.8b/qwen35
WEIGHTS ?= ../02-cpu-0.8b/build/qwen35-0.8b.bin

.PHONY: test e2e run

test:
	$(PYTHON) -m unittest -v test_runtime.py

e2e:
	$(PYTHON) test_e2e.py --model "$(MODEL)" --engine "$(ENGINE)" --weights "$(WEIGHTS)"

run:
	$(PYTHON) runtime.py --model "$(MODEL)" --engine "$(ENGINE)" --weights "$(WEIGHTS)"
