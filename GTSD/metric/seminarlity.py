from transformers import AutoTokenizer, AutoModel
import torch
import jsonlines
from sklearn.metrics.pairwise import cosine_similarity


def get_max(p):
    maxx=0
    o=0
    index=0
    for i in p:
        if i>maxx:
            maxx=i
            index=o
        o+=1
    return index

def get_number(text1,text2,text3, sum):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')

    k=[]
    minn=99999
    index=0
    last=text1[0]
    p=[]
    d=[]
    pm=[]
    dm=[]
    textt=[]
    for i in range(0, sum):
        s1 = text1[i]
        s2 = text2[i]

        inputs1 = tokenizer(s1, return_tensors='pt')
        inputs2 = tokenizer(s2, return_tensors='pt')

        with torch.no_grad():
            embeddings1 = model(**inputs1).pooler_output
            embeddings2 = model(**inputs2).pooler_output

        similarity = cosine_similarity(embeddings1.numpy(), embeddings2.numpy())
        k.append(similarity[0][0])
        if similarity[0][0]<minn:
            minn=similarity[0][0]
            index=i

        if last==s1:
            p.append(similarity[0][0])
            d.append(i)
        else:
            pm.append(p[get_max(p)])
            dm.append(text2[d[get_max(p)]])
            textt.append(text3[d[get_max(p)]])
            last=s1
            p=[]
            d=[]
            p.append(similarity[0][0])
            d.append(i)

    print("min similarity :",minn," index =",index)
    o=0
    with open("similarity-text-choose.txt", "w") as f:
        for i in dm:
            f.write(i +'        similarity:'+ str(pm[o])+'\n')  # 自带文件关闭功能，不需要再写f.close()
            o+=1

    with open("generate-y.txt", "w") as f:
        for i in textt:
            f.write(i+'\n')  # 自带文件关闭功能，不需要再写f.close()
    return k
text1=[]
text2=[]
text3=[]
with jsonlines.open("original.jsonl", 'r') as rp:
    for line in rp:
        text1.append(line)
print(text1)
f = open("rank0_seed_101_x.txt")
sum=0
line = f.readline()
while line:
    text2.append(line.replace('\n', ''))
    sum += 1
    line = f.readline()

f.close()
print(text2)
# read txt method one
f = open("rank0_seed_101_y.txt")
sum=0
line = f.readline()
while line:
    text3.append(line.replace('\n', ''))
    sum += 1
    line = f.readline()

f.close()
print(text3)

k=get_number(text1,text2,text3,sum)

with open("similarity.txt","w") as f:
    for i in k:
        f.write(str(i)+'\n')  # 自带文件关闭功能，不需要再写f.close()
