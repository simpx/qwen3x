# 最小构建说明：`make` 只编译全部；每个可执行文件就是下面一条 c++ 命令。

.PHONY: all test lesson-test course-test course-oracle-test clean

all: qwen38 lessons/00_toy_logits lessons/01_rmsnorm_linear lessons/02_swiglu_residual lessons/03_rope lessons/04_attention lessons/05_gqa_kv_cache lessons/06_deltanet_recurrence lessons/07_deltanet_layer lessons/08_hybrid_qwen

qwen38: capstone/qwen38.cpp
	c++ -O3 -std=c++17 -Wall -Wextra -Wpedantic capstone/qwen38.cpp -o qwen38

lessons/00_toy_logits: lessons/00_toy_logits.cpp
	c++ -O3 -std=c++17 -Wall -Wextra -Wpedantic lessons/00_toy_logits.cpp -o lessons/00_toy_logits

lessons/01_rmsnorm_linear: lessons/01_rmsnorm_linear.cpp
	c++ -O3 -std=c++17 -Wall -Wextra -Wpedantic lessons/01_rmsnorm_linear.cpp -o lessons/01_rmsnorm_linear

lessons/02_swiglu_residual: lessons/02_swiglu_residual.cpp
	c++ -O3 -std=c++17 -Wall -Wextra -Wpedantic lessons/02_swiglu_residual.cpp -o lessons/02_swiglu_residual

lessons/03_rope: lessons/03_rope.cpp
	c++ -O3 -std=c++17 -Wall -Wextra -Wpedantic lessons/03_rope.cpp -o lessons/03_rope

lessons/04_attention: lessons/04_attention.cpp
	c++ -O3 -std=c++17 -Wall -Wextra -Wpedantic lessons/04_attention.cpp -o lessons/04_attention

lessons/05_gqa_kv_cache: lessons/05_gqa_kv_cache.cpp
	c++ -O3 -std=c++17 -Wall -Wextra -Wpedantic lessons/05_gqa_kv_cache.cpp -o lessons/05_gqa_kv_cache

lessons/06_deltanet_recurrence: lessons/06_deltanet_recurrence.cpp
	c++ -O3 -std=c++17 -Wall -Wextra -Wpedantic lessons/06_deltanet_recurrence.cpp -o lessons/06_deltanet_recurrence

lessons/07_deltanet_layer: lessons/07_deltanet_layer.cpp
	c++ -O3 -std=c++17 -Wall -Wextra -Wpedantic lessons/07_deltanet_layer.cpp -o lessons/07_deltanet_layer

lessons/08_hybrid_qwen: lessons/08_hybrid_qwen.cpp
	c++ -O3 -std=c++17 -Wall -Wextra -Wpedantic lessons/08_hybrid_qwen.cpp -o lessons/08_hybrid_qwen

lesson-test: all
	@for lesson in lessons/00_toy_logits lessons/01_rmsnorm_linear lessons/02_swiglu_residual lessons/03_rope lessons/04_attention lessons/05_gqa_kv_cache lessons/06_deltanet_recurrence lessons/07_deltanet_layer lessons/08_hybrid_qwen; do ./$$lesson; done

course-test: qwen38
	./qwen38 --self-test

test: lesson-test course-test

course-oracle-test: qwen38
	@test -n "$(MODEL)" || (echo "usage: make course-oracle-test MODEL=/path/to/Qwen3.5-0.8B" >&2; exit 2)
	scripts/test_course_08b.sh "$(MODEL)"

clean:
	rm -f qwen38 lessons/00_toy_logits lessons/01_rmsnorm_linear lessons/02_swiglu_residual lessons/03_rope lessons/04_attention lessons/05_gqa_kv_cache lessons/06_deltanet_recurrence lessons/07_deltanet_layer lessons/08_hybrid_qwen
