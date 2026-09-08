#!/usr/bin/env python3
"""Same-weight mlx-lm oracle, C++ Session verification, and reproducible timings.

Run reference and C++ in separate processes: never load both 35B copies together.
The short oracle stores every vocabulary logit, not just selected token scores.
"""
import argparse
import ctypes
from contextlib import closing
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODEL = ROOT / "build/models/mlx-community-Qwen3.6-35B-A3B-4bit"
REVISION = "38740b847e4cb78f352aba30aa41c76e08e6eb46"
CASES = [
    {"name": "raw", "tokens": [9707, 11, 358, 1079, 264, 1720, 13]},
    {"name": "chat", "tokens": [248045, 846, 198, 95826, 110827, 97431, 24167,
                                   6738, 1710, 248046, 198, 248045, 74455, 198,
                                   248068, 271, 248069, 271]},
    {"name": "boundary", "tokens": [10, 42, 99, 7, 123, 456, 10, 42, 99, 7]},
]


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def metrics(actual, expected):
    a, b = np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError("invalid logits")
    d = np.abs(a-b)
    at = np.argsort(a)[-8:][::-1]; bt = np.argsort(b)[-8:][::-1]
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
            "relative_l2": float(np.linalg.norm(d)/max(np.linalg.norm(b), 1e-12)),
            "argmax_equal": bool(a.argmax() == b.argmax()),
            "actual_top": int(a.argmax()), "reference_top": int(b.argmax()),
            "actual_top8":at.tolist(), "reference_top8":bt.tolist(),
            "top8_overlap":len(set(at.tolist()) & set(bt.tolist()))}


def reference(args):
    import mlx.core as mx
    from mlx_lm import load
    mx.set_cache_limit(1024**3)
    model, tokenizer = load(str(args.model))
    print("reference loaded", mx.get_active_memory()/1024**3, "GiB", flush=True)
    cases = list(CASES)
    for name, text in [("chinese", "用一句话解释什么是递归。"),
                       ("code", "Write a Python function to add two numbers.")]:
        cases.append({"name": name, "tokens": tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False)})
    rows, repeat_results, chunk_results, split_rows, chunk_rows = [], [], [], [], []
    for case in cases:
        cache = model.make_cache()
        begin = len(rows)
        for token in case["tokens"]:
            y = model(mx.array([[token]]), cache=cache)
            mx.eval(y, [c.state for c in cache])
            rows.append(np.array(y[0, -1].astype(mx.float32)))
        # Repeat identical operation order to establish reproducibility before C++ tuning.
        second = model.make_cache()
        for i, token in enumerate(case["tokens"]):
            y = model(mx.array([[token]]), cache=second)
            mx.eval(y, [c.state for c in second])
            repeat_results.append(metrics(np.array(y[0,-1].astype(mx.float32)),rows[begin+i]))
        greedy = []
        for _ in range(12):
            token = int(mx.argmax(y[0,-1]).item()); greedy.append(token)
            y = model(mx.array([[token]]), cache=second)
            mx.eval(y, [c.state for c in second])
        case["greedy"] = greedy
        case["generation"] = tokenizer.decode(greedy)
        chunk_cache = model.make_cache()
        chunk_h = model.language_model.model(mx.array([case["tokens"]]),cache=chunk_cache)
        chunk_y = model.language_model.lm_head(chunk_h)
        mx.eval(chunk_y,[c.state for c in chunk_cache])
        for i in range(len(case["tokens"])):
            chunk_results.append(metrics(np.array(chunk_y[0,i].astype(mx.float32)),rows[begin+i]))
        # The production backend projects only the last position, as required
        # for bounded long-context memory. Match that GEMV/GEMM shape explicitly.
        chunk_rows.append(np.array(model.language_model.lm_head(chunk_h[:,-1:])[0,-1].astype(mx.float32)))
        split_cache = model.make_cache()
        cut=len(case["tokens"])//2
        prefix_h=model.language_model.model(mx.array([case["tokens"][:cut]]),cache=split_cache)
        prefix=model.language_model.lm_head(prefix_h[:,-1:])
        mx.eval(prefix,[c.state for c in split_cache])
        suffix_h=model.language_model.model(mx.array([case["tokens"][cut:]]),cache=split_cache)
        suffix=model.language_model.lm_head(suffix_h[:,-1:])
        mx.eval(suffix,[c.state for c in split_cache])
        split_rows.extend([np.array(prefix[0,-1].astype(mx.float32)),
                           np.array(suffix[0,-1].astype(mx.float32))])
        split_greedy=[]
        for _ in range(12):
            token=int(mx.argmax(suffix[0,-1]).item());split_greedy.append(token)
            suffix=model(mx.array([[token]]),cache=split_cache)
            mx.eval(suffix,[c.state for c in split_cache])
        case['split_greedy']=split_greedy
        print(case["name"],len(case["tokens"]),case["generation"],flush=True)
    np.save(args.output / "logits.npy", np.stack(rows))
    np.save(args.output / "split-logits.npy",np.stack(split_rows))
    np.save(args.output / "chunk-logits.npy",np.stack(chunk_rows))
    # BF16 kernels can differ in reductions; strict same-order equality is a
    # diagnostic, while chunked comparisons use separately stated bounds.
    write(args.output / "oracle.json", {"revision": REVISION, "cases": cases,
          "dtype": "BF16 activations, FP32 recurrent state", "repeat": repeat_results,
          "reference_chunk_vs_token":chunk_results,
          "versions": {"mlx": mx.__version__}, "logits_sha256": hashlib.sha256(
              (args.output/"logits.npy").read_bytes()).hexdigest()})


def check(args):
    from reference.qwen3x import Engine
    meta = json.loads((args.output/"oracle.json").read_text())
    rows = np.load(args.output/"logits.npy")
    splits = np.load(args.output/"split-logits.npy")
    chunks = np.load(args.output/"chunk-logits.npy")
    results = []; row = 0
    model_path=args.model/"config.json" if args.model.is_dir() else args.model
    with Engine(args.library, model_path) as engine:
        for ci,case in enumerate(meta["cases"]):
            tokens = case["tokens"]; expected = rows[row:row+len(tokens)]; row += len(tokens)
            with closing(engine.create_session(65536)) as s:
                for i, token in enumerate(tokens):
                    s.eval(token)
                    r = metrics(s.copy_logits(), expected[i]);r.update(case=case["name"],step=i,path="token")
                    results.append(r)
                checkpoint = len(tokens)//2
                s.reset();s.sync(tokens,checkpoint_at=checkpoint)
                results.append(dict(metrics(s.copy_logits(),splits[ci*2+1]),case=case["name"],path="chunk"))
                assert s.sync(tokens[:checkpoint],checkpoint_at=checkpoint)==checkpoint
                results.append(dict(metrics(s.copy_logits(),splits[ci*2]),case=case["name"],path="restore"))
                assert s.sync(tokens,checkpoint_at=checkpoint)==checkpoint
                fresh = np.array(s.copy_logits())
                assert s.sync(tokens,checkpoint_at=checkpoint)==len(tokens)
                assert np.array_equal(s.copy_logits(),fresh)
                results.append(dict(metrics(fresh,splits[ci*2+1]),case=case["name"],path="restored_suffix"))
                generated=[]
                for _ in case["greedy"]:
                    token=s.argmax();generated.append(token);s.eval(token)
                print(case["name"],"greedy",generated==case["split_greedy"],flush=True)
                results.append({"case":case["name"],"path":"greedy", "equal":generated==case["split_greedy"],"tokens":generated})
                s.reset();s.sync(tokens)
                results.append(dict(metrics(s.copy_logits(),chunks[ci]),case=case["name"],path="full_chunk"))
    write(args.output/"cpp-check.json",results)
    print("worst",max(r.get("max_abs",0) for r in results),flush=True)
    # Fixed before performance tuning: matched schedules and last-position head
    # produced bit-identical BF16 logits in the baseline. Keep that contract.
    for r in results:
        if "max_abs" in r:
            assert r["max_abs"] == 0 and r["argmax_equal"],r
        if r["path"]=="greedy":assert r["equal"],r


def bench(args):
    records=[]
    start_load=time.perf_counter()
    source_revision=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    swap_before=subprocess.run(['sysctl','vm.swapusage'],capture_output=True,text=True).stdout.strip()
    # Fixed non-special token sequence, identical between implementations.
    base=[9707,11,358,1079,264,1720,13,198,785,279,1787,374,220,16,13]
    if args.backend=="python":
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.generate import generate_step
        mx.set_cache_limit(1024**3)
        model,_=load(str(args.model))
    else:
        from reference.qwen3x import Engine
        os.environ['Q3X_MLX_CHUNK']=str(args.chunk)
        model_path=args.model/"config.json" if args.model.is_dir() else args.model
        engine=Engine(args.library,model_path)
        stats=engine._library.q3x_mlx_memory_stats
        stats.argtypes=[ctypes.POINTER(ctypes.c_uint64),ctypes.c_bool]
        stats.restype=None
    write(args.output/'environment.json',dict(source_revision=source_revision,
          dirty=subprocess.run(['git','status','--short'],capture_output=True,text=True).stdout,
          command=sys.argv, model=str(args.model), checkpoint_revision=REVISION,
          library=str(args.library),library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest()
              if args.backend=='cpp' else None,
          source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
              ['arch/mlx/engine.cpp','runtime.cpp','internal.h','Makefile','tests/mlx_reference.py']},
          load_seconds=time.perf_counter()-start_load, swap_before=swap_before,
          checkpoint_retained=args.backend=='cpp' and not args.no_checkpoint,
          timeout_policy='caller must bound the complete experiment; HTTP requests use 600s'))
    for length in args.lengths:
        tokens=(base*((length+len(base)-1)//len(base)))[:length]
        for run in range(args.repeats+1):
            if args.backend=='cpp':stats((ctypes.c_uint64*3)(),True)
            else:mx.reset_peak_memory()
            start=time.perf_counter()
            generated=[]
            if args.backend=="python":
                cache=model.make_cache()
                gen=generate_step(mx.array(tokens),model,max_tokens=args.decode,
                                  prompt_cache=cache,prefill_step_size=args.chunk)
                generated.append(next(gen)[0]); first=time.perf_counter()
                for _ in range(args.decode-1): generated.append(next(gen)[0])
                end=time.perf_counter()
                memory={"active":mx.get_active_memory(),"peak":mx.get_peak_memory(),"cache":mx.get_cache_memory()}
                gen.close();del gen,cache;gc.collect();mx.clear_cache()
            else:
                s=engine.create_session(max(65536,length+args.decode))
                s.sync(tokens,checkpoint_at=-1 if args.no_checkpoint else length)
                token=s.argmax();generated.append(token);first=time.perf_counter()
                for _ in range(args.decode-1):
                    s.eval(token);token=s.argmax();generated.append(token)
                end=time.perf_counter()
                values=(ctypes.c_uint64*3)();stats(values,False)
                memory=dict(zip(['active','peak','cache'],values))
                probe_start=time.perf_counter()
                for _ in range(100):s.argmax()
                memory['repeated_argmax_us']=(time.perf_counter()-probe_start)*1e6/100
                s.close()
            r={"backend":args.backend,"prompt_tokens":length,"output_tokens":args.decode,
               "checkpoint_retained":args.backend=='cpp' and not args.no_checkpoint,
               "generated_tokens":generated,
               "run":run,"warmup":run==0,"chunk":args.chunk,"prefill_seconds":first-start,
               "prefill_tps":length/(first-start),"decode_tps":(args.decode-1)/(end-first),
               "maxrss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**memory}
            records.append(r);print(json.dumps(r),flush=True)
            write(args.output/f"bench-{args.backend}.json",records)
    if args.backend=="cpp":engine.close()


def checkpoint_bench(args):
    """Isolate snapshot retention cost with identical long teacher-forced decode.

    Prefill once per condition, then one warmup and three adjacent decode blocks.
    This is a diagnostic, not a substitute for the fresh-prompt benchmark protocol.
    """
    from reference.qwen3x import Engine
    os.environ['Q3X_MLX_CHUNK']=str(args.chunk)
    path=args.model/'config.json' if args.model.is_dir() else args.model
    base=[9707,11,358,1079,264,1720,13,198,785,279,1787,374,220,16,13]
    rows=[]
    with Engine(args.library,path) as engine:
        stats=engine._library.q3x_mlx_memory_stats
        stats.argtypes=[ctypes.POINTER(ctypes.c_uint64),ctypes.c_bool]
        stats.restype=None
        for n in args.lengths:
            tokens=(base*((n+len(base)-1)//len(base)))[:n]
            final=[]
            for keep in [False,True]:
                capacity=n+args.decode*(args.repeats+1)
                with closing(engine.create_session(capacity)) as session:
                    session.sync(tokens,checkpoint_at=n if keep else -1)
                    stats((ctypes.c_uint64*3)(),True)
                    for run in range(args.repeats+1):
                        predicted=[];start=time.perf_counter()
                        for i in range(args.decode):
                            session.eval(base[(run*args.decode+i)%len(base)])
                            predicted.append(session.argmax())
                        elapsed=time.perf_counter()-start
                        values=(ctypes.c_uint64*3)();stats(values,False)
                        row=dict(prompt_tokens=n,checkpoint_retained=keep,run=run,warmup=run==0,
                                 decode_tokens=args.decode,decode_tps=args.decode/elapsed,
                                 position=n+(run+1)*args.decode,chunk=args.chunk,
                                 active=values[0],peak=values[1],cache=values[2],predicted=predicted)
                        rows.append(row)
                        write(args.output/'checkpoint-bench.json',rows)
                        print(json.dumps({k:v for k,v in row.items() if k!='predicted'}),flush=True)
                    final.append(np.array(session.copy_logits()))
            assert np.array_equal(*final),'checkpoint retention changed final logits'
            off=[r['predicted'] for r in rows if r['prompt_tokens']==n and not r['checkpoint_retained']]
            on=[r['predicted'] for r in rows if r['prompt_tokens']==n and r['checkpoint_retained']]
            assert off==on,'checkpoint retention changed decode argmax'


def range_check(args):
    """Exercise real chunk boundaries with identical Python/C++ schedules."""
    base=[9707,11,358,1079,264,1720,13,198,785,279,1787,374,220,16,13]
    cases=[]
    if args.mode=='range-reference':
        import mlx.core as mx
        from mlx_lm import load
        mx.set_cache_limit(1024**3)
        model,_=load(str(args.model))
        def forward(tokens,cache):
            for i in range(0,len(tokens),args.chunk):
                h=model.language_model.model(mx.array([tokens[i:i+args.chunk]]),cache=cache)
                y=model.language_model.lm_head(h[:,-1:]) if i+args.chunk>=len(tokens) else None
                mx.eval([c.state for c in cache], [] if y is None else [y])
            return np.array(y[0,-1].astype(mx.float32))
        arrays=[]
        for n in args.lengths:
            tokens=(base*((n+len(base)-1)//len(base)))[:n];cut=n//2
            cache=model.make_cache()
            prefix=forward(tokens[:cut],cache);final=forward(tokens[cut:],cache)
            arrays.extend([prefix,final]);cases.append({'length':n,'checkpoint':cut,'tokens':tokens})
            print('range reference',n,flush=True)
        np.save(args.output/'range-logits.npy',np.stack(arrays))
        write(args.output/'ranges.json',{'chunk':args.chunk,'cases':cases,'revision':REVISION})
    else:
        from reference.qwen3x import Engine
        meta=json.loads((args.output/'ranges.json').read_text());rows=np.load(args.output/'range-logits.npy')
        os.environ['Q3X_MLX_CHUNK']=str(meta['chunk'])
        path=args.model/'config.json' if args.model.is_dir() else args.model
        results=[]
        with Engine(args.library,path) as e:
            for i,c in enumerate(meta['cases']):
                with closing(e.create_session(max(65536,c['length']+128))) as s:
                    s.sync(c['tokens'],checkpoint_at=c['checkpoint'])
                    r=metrics(s.copy_logits(),rows[2*i+1]);results.append(dict(r,length=c['length'],path='prefill'))
                    s.sync(c['tokens'][:c['checkpoint']],checkpoint_at=c['checkpoint'])
                    r=metrics(s.copy_logits(),rows[2*i]);results.append(dict(r,length=c['length'],path='restore'))
                    s.sync(c['tokens'],checkpoint_at=c['checkpoint'])
                    r=metrics(s.copy_logits(),rows[2*i+1]);results.append(dict(r,length=c['length'],path='suffix'))
                write(args.output/'range-check.json',results)
                assert all(r['max_abs']==0 and r['argmax_equal'] for r in results),results
                print('range C++',c['length'],r,flush=True)
        write(args.output/'range-check.json',results)
        assert all(r['max_abs']==0 and r['argmax_equal'] for r in results),results


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["reference","check","bench","checkpoint-bench","range-reference","range-check"])
    p.add_argument("--model",type=Path,default=MODEL)
    p.add_argument("--library",type=Path,default=ROOT/"build/mlx/libqwen3x-mlx.dylib")
    p.add_argument("--output",type=Path,default=ROOT/"build/qwen36-night/reference")
    p.add_argument("--backend",choices=["python","cpp"],default="python")
    p.add_argument("--lengths",type=int,nargs="+",default=[512,2048,8192])
    p.add_argument("--decode",type=int,default=128)
    p.add_argument("--chunk",type=int,default=256)
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--no-checkpoint",action="store_true",help="C++ compute-only comparison; default retains prompt checkpoint")
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    {"reference":reference,"check":check,"bench":bench,
     "checkpoint-bench":checkpoint_bench,
     "range-reference":range_check,"range-check":range_check}[args.mode](args)
