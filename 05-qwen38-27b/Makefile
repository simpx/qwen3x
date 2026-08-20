# Stage 5 的第一道 gate：只下载官方 metadata（约几十 KB），绝不下载 27B 权重。
PYTHON ?= python3
CURL ?= curl
BUILD ?= build
DEVICE_GIB ?= 16
EXPECT_RAW_FITS ?= no
REPO ?= https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main

.PHONY: all metadata inspect test clean

all: test

metadata:
	mkdir -p $(BUILD)
	$(CURL) -L --fail --silent --show-error $(REPO)/config.json -o $(BUILD)/config.json
	$(CURL) -L --fail --silent --show-error $(REPO)/model.safetensors.index.json -o $(BUILD)/model.safetensors.index.json

inspect: metadata
	$(PYTHON) qwen38_preflight.py --config $(BUILD)/config.json --index $(BUILD)/model.safetensors.index.json --device-gib $(DEVICE_GIB)

# On this 16 GiB development GPU, raw BF16 must not fit.  This is a passing capability
# check, not an attempt to start a 56 GB download.  A large GPU can use `make inspect`.
test: metadata
	$(PYTHON) test_preflight.py --checker qwen38_preflight.py --config $(BUILD)/config.json --index $(BUILD)/model.safetensors.index.json --device-gib $(DEVICE_GIB) --expect-raw-fits $(EXPECT_RAW_FITS)

clean:
	rm -rf $(BUILD)
