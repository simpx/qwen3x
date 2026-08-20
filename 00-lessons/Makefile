# 第 0 章目录是自包含的：进入 00-lessons/ 后运行 `make`，逐课得到 00 到 08 的 bin。

.PHONY: all test model-test clean

all: 00_toy_logits 01_rmsnorm_linear 02_swiglu_residual 03_rope 04_attention 05_gqa_kv_cache 06_deltanet_recurrence 07_deltanet_layer 08_hybrid_qwen 09_qwen35_0_8b

00_toy_logits: 00_toy_logits.cpp
	c++ -O3 -std=c++17 -Wall -Wextra -Wpedantic 00_toy_logits.cpp -o 00_toy_logits

01_rmsnorm_linear: 01_rmsnorm_linear.cpp
	c++ -O3 -std=c++17 -Wall -Wextra -Wpedantic 01_rmsnorm_linear.cpp -o 01_rmsnorm_linear

02_swiglu_residual: 02_swiglu_residual.cpp
	c++ -O3 -std=c++17 -Wall -Wextra -Wpedantic 02_swiglu_residual.cpp -o 02_swiglu_residual

03_rope: 03_rope.cpp
	c++ -O3 -std=c++17 -Wall -Wextra -Wpedantic 03_rope.cpp -o 03_rope

04_attention: 04_attention.cpp
	c++ -O3 -std=c++17 -Wall -Wextra -Wpedantic 04_attention.cpp -o 04_attention

05_gqa_kv_cache: 05_gqa_kv_cache.cpp
	c++ -O3 -std=c++17 -Wall -Wextra -Wpedantic 05_gqa_kv_cache.cpp -o 05_gqa_kv_cache

06_deltanet_recurrence: 06_deltanet_recurrence.cpp
	c++ -O3 -std=c++17 -Wall -Wextra -Wpedantic 06_deltanet_recurrence.cpp -o 06_deltanet_recurrence

07_deltanet_layer: 07_deltanet_layer.cpp
	c++ -O3 -std=c++17 -Wall -Wextra -Wpedantic 07_deltanet_layer.cpp -o 07_deltanet_layer

08_hybrid_qwen: 08_hybrid_qwen.cpp
	c++ -O3 -std=c++17 -Wall -Wextra -Wpedantic 08_hybrid_qwen.cpp -o 08_hybrid_qwen

# 第 09 课不再是 toy：它 mmap 经过 09_pack_weights.py 转换的真实 Qwen3.5-0.8B 权重。
# 它仍是纯 CPU、单文件、可从上读到下的 reference，不需要 CUDA。
09_qwen35_0_8b: 09_qwen35_0_8b.cpp
	c++ -O3 -std=c++17 -Wall -Wextra -Wpedantic 09_qwen35_0_8b.cpp -o 09_qwen35_0_8b

test: all
	@for lesson in 00_toy_logits 01_rmsnorm_linear 02_swiglu_residual 03_rope 04_attention 05_gqa_kv_cache 06_deltanet_recurrence 07_deltanet_layer 08_hybrid_qwen 09_qwen35_0_8b; do ./$$lesson; done

model-test: 09_qwen35_0_8b
	@test -n "$(MODEL)" || (echo "usage: make model-test MODEL=/path/to/Qwen3.5-0.8B" >&2; exit 2)
	./09_test.sh "$(MODEL)"

clean:
	rm -f 00_toy_logits 01_rmsnorm_linear 02_swiglu_residual 03_rope 04_attention 05_gqa_kv_cache 06_deltanet_recurrence 07_deltanet_layer 08_hybrid_qwen 09_qwen35_0_8b
