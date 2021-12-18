# encoding: utf-8
import json
with open('test_public.jsonl','r') as f:
    data = f.readlines()
data = [json.loads(line) for line in data]
with open('test_placeholder.jsonl','w') as f:
    for d in data:
        d['answer'] = 0
        f.write(json.dumps(d,ensure_ascii=False)+'\n')
