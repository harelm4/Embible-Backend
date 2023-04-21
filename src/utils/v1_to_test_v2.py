import json

with open('../../data/test.json', 'r') as r:
    test_data=json.load(r)
res=[]
for entry in test_data:
    missing = list(dict(entry['missing']).items())
    txt=entry['text']
    tmp=''
    new_missing={}

    start_idx=0
    end_idx=0
    missing_idx=0
    for i,c in enumerate(txt):
        if i<end_idx:
            continue
        if c == '?':
            start_idx=i
            for j in range(i,i+len(txt[i:])):
                if txt[j]=='?':
                    try:
                     tmp=tmp+missing[missing_idx][1]
                    except:
                        break
                    missing_idx+=1
                else:
                    end_idx=j
                    break
            new_missing[start_idx]=tmp
            tmp=''

    res.append({'text':entry['text'],'missing':new_missing})

json_object = json.dumps(res)

# Writing to sample.json
with open("../../data/test_v2.json", "w") as outfile:
    outfile.write(json_object)