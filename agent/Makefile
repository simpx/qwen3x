BUN ?= bun
BUILD ?= build

.PHONY: build check test

build:
	mkdir -p $(BUILD)
	$(BUN) build ./main.ts --compile --minify --no-compile-autoload-dotenv \
		--no-compile-autoload-bunfig --outfile $(BUILD)/q3x

check:
	$(BUN) run check

test: check
	$(BUN) test
