import torch
import torch.nn as nn
torch.manual_seed(42)  
x = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 0.0]
])
query = nn.Linear(4, 4)
key   = nn.Linear(4, 4)
value = nn.Linear(4, 4)
Q = query(x)
K = key(x)
V = value(x)
scores = torch.matmul(Q, K.T)
scores = scores / (Q.shape[-1] ** 0.5)     
attention = torch.softmax(scores, dim=-1)
output = torch.matmul(attention, V)
