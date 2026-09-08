#!/usr/bin/env python3
"""Real-model HTTP behavior, streaming, tools and a paced single-slot soak.

Only fixture tools are simulated; no model-selected external operation executes.
Start qwen3x separately, then run this against its localhost endpoint.
"""
import argparse
import ast
import http.client
import json
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit

TASKS = [
    ("arithmetic", "17 加 25 等于多少？只输出数字。", r"^42[。.]?$"),
    ("translate", "把英文 apple 翻译为中文，只输出译文。", r"苹果"),
    ("sort", "把 9、1、4 从小到大排列，仅输出 JSON 数组。", r"\[\s*1\s*,\s*4\s*,\s*9\s*\]"),
    ("extract", "记录：姓名林林，编号 A42，城市杭州。仅输出编号。", r"A42"),
    ("summary", "用一句话总结：会议地点改为上海，时间改为周五。", r"(?s)(?=.*上海)(?=.*周五)"),
    ("python_add", "Write only a Python function add(a, b) that returns their sum.", r"(?s)def add\(a, b\).*return a \+ b"),
    ("python_reverse", "Write a Python function reverse(s) using slicing to reverse a string.", r"\[\s*::\s*-1\s*\]"),
    ("sql", "Write SQL selecting name from users where age is greater than 18.", r"(?is)select\s+name\s+from\s+users\s+where\s+age\s*>\s*18"),
    ("json", "只输出合法 JSON 对象，键 ok 为布尔 true，键 count 为整数 3。不要代码围栏。", r"(?s)(?=.*\"ok\"\s*:\s*true)(?=.*\"count\"\s*:\s*3)"),
    ("unicode", "逐字复制这三个词，不添加解释：苹果 café 日本語", r"苹果 café 日本語"),
    ("count", "字符串 banana 中字母 a 出现几次？只输出数字。", r"^3[。.]?$"),
    ("units", "2 米等于多少厘米？只输出数字。", r"^200[。.]?$"),
    ("logic", "如果所有猫都是动物，所有动物都需要水，那么猫需要水吗？回答是或否。", r"^是"),
    ("code_explain", "Python 中 len([1, 2, 3]) 的结果是什么？只输出数字。", r"^3[。.]?$"),
    ("csv", "CSV 数据：name,qty\na,2\nb,3\n数量总和是多少？只输出数字。", r"^5[。.]?$"),
    ("context", "档案记载：项目代号是蓝鲸，负责人是小林。只回答项目代号。", r"蓝鲸"),
    ("format", "只输出下面的两行，不加其他内容：第一行 HELLO，第二行 WORLD。", r"^HELLO\s+WORLD$"),
    ("subtraction", "100 减去 37 等于多少？只输出数字。", r"^63[。.]?$"),
    ("boolean", "Python 表达式 bool(0) 的值是什么？只输出值。", r"^False[。.]?$"),
    ("recursion", "用一句中文解释递归，必须提到函数调用自身。", r"(?s)(?=.*函数)(?=.*自身)"),
]


def call(args, model, messages, **extra):
    body={"model":model,"messages":messages,"temperature":0,"max_tokens":128,
          "chat_template_kwargs":{"enable_thinking":False},**extra}
    endpoint=urlsplit(args.url)
    conn=http.client.HTTPConnection(endpoint.hostname,endpoint.port,timeout=600)
    started=time.monotonic()
    conn.request('POST','/v1/chat/completions',json.dumps(body,ensure_ascii=False).encode(),
                 {'Content-Type':'application/json'})
    response=conn.getresponse()
    if not body.get('stream'):
        data=response.read();conn.close()
        return response.status,json.loads(data),time.monotonic()-started
    chunks=[];ttft=None;done=False
    while True:
        line=response.readline()
        if not line:break
        if not line.startswith(b'data:'):continue
        payload=line[5:].strip()
        if payload==b'[DONE]':done=True;break
        item=json.loads(payload);chunks.append(item)
        if ttft is None and any(c.get('delta',{}).get('content') for c in item.get('choices',[])):
            ttft=time.monotonic()-started
    conn.close()
    return response.status,{'chunks':chunks,'done':done,'ttft':ttft},time.monotonic()-started


def messages(text):return [{'role':'user','content':text}]


def cached(data):
    return data.get('usage',{}).get('prompt_tokens_details',{}).get('cached_tokens',0)


def streamed_completion(data):
    content='';finish=None;usage={}
    for row in data.get('chunks',[]):
        usage=row.get('usage') or usage
        for choice in row.get('choices',[]):
            content+=choice.get('delta',{}).get('content') or ''
            finish=choice.get('finish_reason') or finish
    return {'choices':[{'message':{'role':'assistant','content':content},'finish_reason':finish}],
            'usage':usage,'stream':data}


def task_pass(name,content,pattern):
    if re.search(pattern,content.strip()) is None:return False
    if name=='json':
        try:return json.loads(content)=={'ok':True,'count':3}
        except ValueError:return False
    if name.startswith('python_'):
        # Execute only an expression-only function after validating every AST
        # node. Imports, calls, attributes, loops and side effects are forbidden.
        match=re.search(r'```(?:python)?\s*\n(.*?)```',content,re.S)
        code=match.group(1) if match else content
        allowed=(ast.Module,ast.FunctionDef,ast.arguments,ast.arg,ast.Return,ast.BinOp,
                 ast.Add,ast.Name,ast.Load,ast.Constant,ast.Expr,ast.Subscript,ast.Slice,
                 ast.UnaryOp,ast.USub)
        try:
            tree=ast.parse(code)
            if not all(isinstance(n,allowed) for n in ast.walk(tree)):return False
            scope={'__builtins__':{},'str':str,'int':int,'float':float}
            exec(compile(tree,'<validated fixture>','exec'),scope)
            if name=='python_add':return scope['add'](3,4)==7 and scope['add'](-2,5)==3
            return scope['reverse']('abc')=='cba' and scope['reverse']('')==''
        except Exception:return False
    return True


def footprint(args, label):
    path=args.output/f'footprint-{label}.txt'
    sample=subprocess.run(['sample',str(args.pid),'1','10','-file',str(path)],
                          capture_output=True,text=True,timeout=20)
    lines=([line.strip() for line in path.read_text().splitlines()
            if line.startswith('Physical footprint')] if path.exists() else [])
    return {'footprint_sample_status':sample.returncode,'footprint':lines}


def run(args):
    endpoint=urlsplit(args.url)
    conn=http.client.HTTPConnection(endpoint.hostname,endpoint.port,timeout=30)
    conn.request('GET','/v1/models');response=conn.getresponse()
    assert response.status==200
    model=json.loads(response.read())['data'][0]['id'];conn.close()
    args.output.mkdir(parents=True,exist_ok=True)
    records=[];started=time.monotonic()
    swap_before=subprocess.run(['sysctl','vm.swapusage'],capture_output=True,text=True).stdout.strip()
    (args.output/'tasks.json').write_text(json.dumps(TASKS,ensure_ascii=False,indent=2))
    def record(r):
        if args.pid:
            sample=subprocess.run(['ps','-o','rss=','-p',str(args.pid)],capture_output=True,text=True)
            r['rss_kib']=int(sample.stdout.strip() or 0)
        if len(records)%10==0:
            r['swap']=subprocess.run(['sysctl','vm.swapusage'],capture_output=True,text=True).stdout.strip()
            vm=subprocess.run(['vm_stat'],capture_output=True,text=True).stdout
            r['vm_stat']=[line.strip() for line in vm.splitlines() if any(
                key in line for key in ['page size','Pageouts:','Swapins:','Swapouts:','occupied by compressor'])]
        if args.pid and len(records)%25==0:
            r.update(footprint(args,len(records)))
        records.append(r)
        with (args.output/'requests.jsonl').open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        print(r.get('case'),r.get('pass'),round(r.get('seconds',0),3),flush=True)
    for name,text,pattern in TASKS:
        status,data,elapsed=call(args,model,messages(text))
        content=data.get('choices',[{}])[0].get('message',{}).get('content','') or ''
        ok=status==200 and task_pass(name,content,pattern)
        record({'case':name,'pass':bool(ok),'status':status,'content':content,'seconds':elapsed,'response':data})
    # A benign tool is simulated, with exact argument and continuation checks.
    tool={'type':'function','function':{'name':'add','description':'Add two integers.',
          'parameters':{'type':'object','properties':{'a':{'type':'integer'},'b':{'type':'integer'}},'required':['a','b']}}}
    prompt=messages('必须调用 add 工具计算 17 加 25，不要自己计算。')
    status,data,elapsed=call(args,model,prompt,tools=[tool],max_tokens=256)
    calls=data.get('choices',[{}])[0].get('message',{}).get('tool_calls',[])
    ok=False
    if status==200 and len(calls)==1 and data['choices'][0]['finish_reason']=='tool_calls':
        fn=calls[0]['function']
        try:ok=fn['name']=='add' and json.loads(fn['arguments'])=={'a':17,'b':25}
        except (ValueError,KeyError):pass
    record({'case':'tool_call','pass':ok,'seconds':elapsed,'response':data})
    if ok:
        prompt += [data['choices'][0]['message'],{'role':'tool','tool_call_id':calls[0]['id'],'content':'42'}]
        status,data,elapsed=call(args,model,prompt,tools=[tool])
        record({'case':'tool_result','pass':status==200 and '42' in (data['choices'][0]['message'].get('content') or ''),
                'seconds':elapsed,'response':data})
    status,data,elapsed=call(args,model,messages('必须调用 add 工具计算 17 加 25，不要自己计算。'),
                           tools=[tool],max_tokens=256,stream=True)
    streamed={};tool_finish=False
    for row in data.get('chunks',[]):
        for choice in row.get('choices',[]):
            tool_finish |= choice.get('finish_reason')=='tool_calls'
            for item in choice.get('delta',{}).get('tool_calls',[]):
                fn=streamed.setdefault(item['index'],{'name':'','arguments':''})
                for key in fn:fn[key]+=item.get('function',{}).get(key,'')
    ok=status==200 and data.get('done') and tool_finish and len(streamed)==1
    if ok:
        fn=next(iter(streamed.values()))
        try:ok=fn['name']=='add' and json.loads(fn['arguments'])=={'a':17,'b':25}
        except ValueError:ok=False
    record({'case':'tool_sse','pass':bool(ok),'seconds':elapsed,'response':data})
    status,data,elapsed=call(args,model,messages('只回答 OK。'),stream=True,
                           stream_options={'include_usage':True})
    record({'case':'sse','pass':status==200 and data['done'] and any(
        c.get('finish_reason') is not None for row in data['chunks'] for c in row.get('choices',[])) and
        any((row.get('usage') or {}).get('completion_tokens',0)>0 for row in data['chunks']),
        'seconds':elapsed,'response':data})
    stop_prompt=messages('只输出 HELLO WORLD。')
    status,baseline,elapsed=call(args,model,stop_prompt)
    uncut=baseline.get('choices',[{}])[0].get('message',{}).get('content') or ''
    record({'case':'stop_baseline','pass':status==200 and 'WORLD' in uncut,
            'seconds':elapsed,'response':baseline})
    status,data,elapsed=call(args,model,stop_prompt,stop=['WORLD'])
    record({'case':'stop','pass':status==200 and data['choices'][0]['finish_reason']=='stop' and
            'WORLD' in uncut and data['choices'][0]['message']['content']==uncut.split('WORLD')[0],
            'seconds':elapsed,'response':data})
    status,data,elapsed=call(args,model,messages('只回答 OK。'))
    usage=data.get('usage',{})
    record({'case':'eos_usage','pass':status==200 and data['choices'][0]['finish_reason']=='stop' and
            usage.get('total_tokens',-1)==usage.get('prompt_tokens',0)+usage.get('completion_tokens',0) and
            usage.get('completion_tokens',0)>0,'seconds':elapsed,'response':data})
    status,data,elapsed=call(args,model,messages('请详细解释为什么 17 加 25 等于 42。'),
                           chat_template_kwargs={'enable_thinking':True},max_tokens=4096)
    record({'case':'thinking','pass':status==200 and bool(data['choices'][0]['message'].get('reasoning_content')) and
            '42' in (data['choices'][0]['message'].get('content') or '') and data['choices'][0]['finish_reason']=='stop',
            'seconds':elapsed,'response':data})
    status,data,elapsed=call(args,model,messages('从 1 开始一直数数。'),max_tokens=1)
    record({'case':'max_tokens','pass':status==200 and data['choices'][0]['finish_reason']=='length',
            'seconds':elapsed,'response':data})
    for path in args.prefill_request:
        fixture=json.loads(path.read_text())
        status,data,elapsed=call(args,model,fixture['messages'],max_tokens=128,stream=True,
                               stream_options={'include_usage':True})
        normalized=streamed_completion(data)
        answer=normalized['choices'][0]['message']['content']
        record({'case':'fresh_'+path.stem,'pass':status==200 and data.get('done') and
                all(x in answer for x in fixture['expected']) and cached(normalized)==0,
                'seconds':elapsed,'response':normalized})
    if args.long_request:
        request=json.loads(args.long_request.read_text())
        status,data,elapsed=call(args,model,request['messages'],max_tokens=128,stream=True)
        content=''.join(c.get('delta',{}).get('content','') or '' for row in data.get('chunks',[])
                        for c in row.get('choices',[]))
        record({'case':'long_retrieval','pass':status==200 and all(x in content for x in request['expected']),
                'seconds':elapsed,'response':data,'expected':request['expected']})
        if status==200:
            # Exact request repeats restore the prompt checkpoint after generation.
            status,repeat,elapsed=call(args,model,request['messages'],max_tokens=128)
            record({'case':'long_checkpoint','pass':status==200 and cached(repeat)>0 and
                    repeat['choices'][0]['message']['content']==content,
                    'seconds':elapsed,'response':repeat})
            follow=request['messages']+[repeat['choices'][0]['message'],{'role':'user','content':'仅重复中间记录的代码。'}]
            status,data,elapsed=call(args,model,follow,max_tokens=64,stream=True,
                                   stream_options={'include_usage':True})
            data=streamed_completion(data)
            record({'case':'long_append','pass':status==200 and cached(data)>0 and request['expected'][1] in (data['choices'][0]['message'].get('content') or ''),
                    'seconds':elapsed,'response':data})
            # A tool result appended to the long history; only local fixture data.
            tool_follow=follow+[data['choices'][0]['message'],
                {'role':'assistant','content':None,'tool_calls':[{'id':'fixture_add','type':'function',
                    'function':{'name':'add','arguments':'{"a":17,"b":25}'}}]},
                {'role':'tool','tool_call_id':'fixture_add','content':'42'},
                {'role':'user','content':'Give the tool result and the EARLY_RECORD access code.'}]
            status,data,elapsed=call(args,model,tool_follow,max_tokens=128,stream=True,
                                   stream_options={'include_usage':True})
            data=streamed_completion(data)
            answer=data.get('choices',[{}])[0].get('message',{}).get('content') or ''
            record({'case':'long_tool_result','pass':status==200 and cached(data)>0 and '42' in answer and request['expected'][0] in answer,
                    'seconds':elapsed,'response':data})
        status,data,elapsed=call(args,model,messages('新的独立对话，只输出 ISOLATED。'))
        record({'case':'prefix_isolation','pass':status==200 and cached(data)==0 and
                'ISOLATED' in data['choices'][0]['message']['content'],'seconds':elapsed,'response':data})
        oversized=messages(request['messages'][0]['content']*2)
        status,data,elapsed=call(args,model,oversized)
        record({'case':'context_overflow','pass':status==400 and bool(data.get('error')),
                'seconds':elapsed,'response':data})
    # The soak timer includes the initial behavioral/long requests. Subsequent
    # calls are paced; no GPU inference runs concurrently with a benchmark.
    target=max(100,args.requests) if args.minutes else len(records)
    duration=args.minutes*60
    while len(records)<target or time.monotonic()-started<duration:
        i=len(records);name,text,pattern=TASKS[i%len(TASKS)]
        if args.long_request and i%25==0:
            request=json.loads(args.long_request.read_text())
            status,data,elapsed=call(args,model,request['messages'],max_tokens=128)
            content=data.get('choices',[{}])[0].get('message',{}).get('content') or ''
            ok=status==200 and all(x in content for x in request['expected'])
            name='long_retrieval'
        else:
            status,data,elapsed=call(args,model,messages(text),stream=i%7==0)
            content=(''.join(c.get('delta',{}).get('content','') or '' for row in data.get('chunks',[])
                     for c in row.get('choices',[])) if i%7==0 else
                     data.get('choices',[{}])[0].get('message',{}).get('content') or '')
            ok=status==200 and task_pass(name,content,pattern)
        r={'case':f'soak_{i}_{name}','pass':ok,'seconds':elapsed,'status':status,'content':content}
        record(r)
        next_due=started+duration*min(len(records)/target,1)
        time.sleep(max(0,min(18,next_due-time.monotonic())))
    summary={'requests':len(records),'elapsed_seconds':time.monotonic()-started,
             'behavior_passed':sum(r['pass'] for r in records[:len(TASKS)]),'behavior_total':len(TASKS),
             'failures':[r['case'] for r in records if not r['pass']],
             'swap_before':swap_before,
             'swap_after':subprocess.run(['sysctl','vm.swapusage'],capture_output=True,text=True).stdout.strip()}
    if args.pid:summary['final_memory']=footprint(args,'final')
    summary['vm_stat_final']=subprocess.run(['vm_stat'],capture_output=True,text=True).stdout
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)
    assert summary['behavior_passed']/len(TASKS)>=0.8,summary
    assert not [r for r in records[len(TASKS):] if not r['pass']],summary
    if args.pid:
        assert summary['final_memory']['footprint_sample_status']==0 and summary['final_memory']['footprint'],summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:18080')
    p.add_argument('--output',type=Path,default=Path('build/qwen36-night/service'))
    p.add_argument('--minutes',type=float,default=0)
    p.add_argument('--requests',type=int,default=100)
    p.add_argument('--pid',type=int)
    p.add_argument('--long-request',type=Path)
    p.add_argument('--prefill-request',type=Path,nargs='*',default=[])
    run(p.parse_args())
