#!/usr/bin/env python3
"""Run the fixed service tasks on pinned mlx-lm, using qwen3x-render token IDs.

Run separately from the C++ service so two copies never compete for the GPU.
"""
import argparse
import gc
import json
from pathlib import Path
import time

from mlx_long_fixture import render
from mlx_service import TASKS, messages, task_pass


def run(args):
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step

    mx.set_cache_limit(1024**3)
    model, tokenizer = load(str(args.model))
    args.output.mkdir(parents=True, exist_ok=True)
    results = []

    def generate(name, request, expected=None, pattern=None):
        tokens = render(args, request)
        cache = model.make_cache()
        mx.reset_peak_memory()
        start = time.monotonic()
        output = []
        first = None
        generator = generate_step(mx.array(tokens), model, max_tokens=128,
                                  prompt_cache=cache, prefill_step_size=args.chunk)
        for token, _ in generator:
            if first is None:
                first = time.monotonic()
            if token in tokenizer.eos_token_ids:
                break
            output.append(token)
        generator.close()
        content = tokenizer.decode(output)
        passed = (all(x in content for x in expected) if expected else
                  task_pass(name, content, pattern))
        row = dict(case=name, passed=passed, content=content,
                   prompt_tokens=len(tokens), output_tokens=len(output),
                   seconds=time.monotonic()-start, ttft=first-start,
                   active=mx.get_active_memory(), peak=mx.get_peak_memory(),
                   chunk=args.chunk)
        results.append(row)
        (args.output/'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(json.dumps(row, ensure_ascii=False), flush=True)
        del generator, cache
        gc.collect()
        mx.clear_cache()

    if not args.long_only:
        for name, text, pattern in TASKS:
            generate(name, dict(messages=messages(text), enable_thinking=False), pattern=pattern)
        tool={'type':'function','function':{'name':'add','description':'Add two integers.',
              'parameters':{'type':'object','properties':{'a':{'type':'integer'},'b':{'type':'integer'}},
                            'required':['a','b']}}}
        prompt=messages('必须调用 add 工具计算 17 加 25，不要自己计算。')
        generate('tool_call',dict(messages=prompt,tools=[tool],enable_thinking=False),
                 pattern=r'(?s)<function=add>.*<parameter=a>\s*17\s*</parameter>.*<parameter=b>\s*25\s*</parameter>')
        prompt += [{'role':'assistant','content':None,'tool_calls':[{'id':'fixture_add','type':'function',
                   'function':{'name':'add','arguments':'{"a":17,"b":25}'}}]},
                   {'role':'tool','tool_call_id':'fixture_add','content':'42'}]
        generate('tool_result',dict(messages=prompt,tools=[tool],enable_thinking=False),pattern=r'42')
    for path in args.long_request:
        fixture = json.loads(path.read_text())
        generate(path.stem, {k: fixture[k] for k in ['messages', 'enable_thinking']},
                 expected=fixture['expected'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=Path('build/models/mlx-community-Qwen3.6-35B-A3B-4bit'))
    p.add_argument('--renderer', type=Path, default=Path('build/render-test'))
    p.add_argument('--render', type=Path, default=Path('build/qwen3x-render.bin'))
    p.add_argument('--chunk', type=int, default=2048)
    p.add_argument('--output', type=Path, default=Path('build/qwen36-night/behavior-reference'))
    p.add_argument('--long-request', type=Path, nargs='*', default=[])
    p.add_argument('--long-only', action='store_true')
    run(p.parse_args())
