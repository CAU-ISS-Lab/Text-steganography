import math

import jsonlines
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForMaskedLM


def read_jsonal(path):
    s=[]
    with jsonlines.open("original.jsonl", 'r') as rp:
        for line in rp:
            s.append(line)
    return s

def read_text(path,start,end):
    s=[]
    f = open(path)
    sum=0
    line = f.readline()
    while line:
        sum += 1
        if sum>=start and sum<=end:
            s.append(line.replace('\n', ''))
        line = f.readline()
        if sum>end:
            break
    f.close()
    return s,end-start+1

def get_ppl(model,tokenizer,s,text,sum):
    kk=0.0
    d=[]
    ppl_index=[]
    ppl_all=[]
    d_ppl=[]
    with torch.no_grad():
        min_ppl = 999999
        index=-1
        dd=0
        for j in range(0, int(sum)):
            print(j,"/",int(sum))
            sentence = s[j]
            tokenize_input = tokenizer.tokenize(sentence)
            print(len(tokenize_input))
            tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
            sen_len = len(tokenize_input)
            sentence_loss = 0.
            for i, word in enumerate(tokenize_input):
                # add mask to i-th character of the sentence
                tokenize_input[i] = '[MASK]'
                mask_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])

                output = model(mask_input)
                #print(output)

                prediction_scores = output[0]
                softmax = nn.Softmax(dim=0)
                ps = softmax(prediction_scores[0, i]).log()
                word_loss = ps[tensor_input[0, i]]
                sentence_loss += word_loss.item()

                tokenize_input[i] = word

            ppl = math.exp(-sentence_loss / sen_len)
            print(ppl)
            d_ppl.append(ppl)
            #ppl_all.append([s[j],ppl])
            if j == 0:
                min_ppl=ppl
                index=j
            elif (j+1)%128==0:
                ddd=torch.topk(torch.Tensor(d_ppl),5,largest = False)
                print(ddd)
                for kkk in range(0,5):
                    ppl_all.append([s[ddd.indices[kkk]], d_ppl[ddd.indices[kkk]]])
                dd+=1
                d.append(s[index])
                ppl_index.append(min_ppl)
                kk+=min_ppl
                min_ppl=ppl
                index=j
            else:
                if min_ppl>ppl:
                    min_ppl=ppl
                    index=j
        kk = kk / (sum/128)
        print(kk)
        return d,ppl_index,ppl_all
def write_text(path,s):
    sum=0
    sum_dpw=0.
    with open(path, "a") as f:
        for i in s:
            #d=i.split(" ")
            #sum_dpw+=float(32)/float(len(d))
            #sum+=1
            #print(i,float(32)/float(len(d)))
            f.write(str(i)+'\n')  # 自带文件关闭功能，不需要再写f.close()
    #print(sum_dpw/sum)

def write_ppl_all(path,s):
    with open(path, "a") as f:
        for i in s:
            f.write(i[0]+"    "+str(i[1])+'\n')  # 自带文件关闭功能，不需要再写f.close()


if __name__ == '__main__':
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model.eval()
    for i in range(0,1000):
        start=1+i*128
        end=128+i*128
        generate_text,sum=read_text("rank0_seed_101_y.txt",start,end)
        print(sum)
        orignal_text=read_jsonal("original.jsonl")
        choosen_text,ppl,ppl_all=get_ppl(model,tokenizer,generate_text,orignal_text,sum)
        path_ppl="generate_ppl.txt"
        path_text="generate.txt"
        path_ppl_all= "generate_ppl_all.txt"
        write_text(path_ppl,ppl)
        write_text(path_text,choosen_text)
        write_ppl_all(path_ppl_all, ppl_all)



