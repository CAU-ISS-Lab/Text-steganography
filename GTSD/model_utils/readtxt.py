import torch
import numpy as np
def read_data(x, path, mode):
    x = x.cpu()
    f = open(path, encoding='utf-8')
    if mode==3:
        sum = 0
        for line in f:
            line = line.strip().split(",")
            npp = []
            # print(text.shape,len(line))
            for i in range(0, len(line) - 1):
                npp.append(float(line[i]))
            if sum == x.shape[1]:
                break
            for i in range(0, x.shape[0]):
                for k in range(0, x.shape[2]):
                    d = torch.from_numpy(np.array(npp))
                    x[i][sum][k] = d[k]
            sum += 1
        x = x.cuda()
    elif mode==2:
        sum = 0
        for line in f:
            line = line.strip().split(",")
            npp = []
            for i in range(0, len(line) - 1):
                npp.append(float(line[i]))
            if sum == x.shape[0]:
                break
            for i in range(0, x.shape[0]):
                    d = torch.from_numpy(np.array(npp))
                    x[sum][i] = d[i]
            sum += 1
        x = x.cuda()
    return x