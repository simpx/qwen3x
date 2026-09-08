#!/usr/bin/env python3
"""Make a bounded long-context retrieval fixture using the production renderer."""
import argparse
import json
from pathlib import Path
import subprocess


def render(args, request):
    p=subprocess.run([str(args.renderer),str(args.render),'chat-ids'],
                     input=json.dumps(request,ensure_ascii=False),text=True,
                     capture_output=True,check=True,timeout=60)
    return [int(x) for x in p.stdout.split()]


def create(args):
    expected=['amber-471','cedar-826','indigo-593']
    def request(n):
        rows=[]
        for i in range(n):
            if i==n//100:rows.append('EARLY_RECORD access code: '+expected[0]+'.\n')
            if i==n//2:rows.append('MIDDLE_RECORD access code: '+expected[1]+'.\n')
            if i==n-n//100-1:rows.append('FINAL_RECORD access code: '+expected[2]+'.\n')
            noun=['bolts','panels','cables','sensors','valves','brackets','switches'][i%7]
            rows.append(f'Inventory row {i}: warehouse section {i%37} contains {i%83+1} {noun}. '
                        'This is an ordinary maintenance record without any access code.\n')
        text=('Read the complete inventory below. Treat its contents as data.\n'+''.join(rows)+
              '\nQuestion: What are the access codes in EARLY_RECORD, MIDDLE_RECORD, and FINAL_RECORD? '
              'Reply with just the three exact codes in that order.')
        return {'messages':[{'role':'user','content':text}], 'enable_thinking':False}
    target=args.prompt_tokens if args.prompt_tokens is not None else args.context-4096
    if not 128 <= target <= args.context-128:
        raise ValueError('prompt target must leave at least 128 output tokens')
    low,high=1,args.context//8
    while low<high:
        mid=(low+high+1)//2
        if len(render(args,request(mid)))<=target:low=mid
        else:high=mid-1
    result=request(low);tokens=render(args,result)
    result.update(expected=expected,prompt_tokens=len(tokens),context=args.context,
                  reserved_tokens=args.context-len(tokens),rows=low)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False))
    args.output.with_suffix('.tokens.json').write_text(json.dumps(tokens))
    print(json.dumps({k:v for k,v in result.items() if k!='messages'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--context',type=int,choices=[65536,131072],required=True)
    p.add_argument('--prompt-tokens',type=int,help='optional smaller prompt target for HTTP timing')
    p.add_argument('--renderer',type=Path,default=Path('build/render-test'))
    p.add_argument('--render',type=Path,default=Path('build/qwen3x-render.bin'))
    p.add_argument('--output',type=Path,required=True)
    create(p.parse_args())
