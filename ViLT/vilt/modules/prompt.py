import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import get_activation
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform


class ReOrder(nn.Module):
    def __init__(self, length, hidden_size):
        super().__init__()
        self.tokeners = nn.Parameter(torch.FloatTensor(1, length, hidden_size))
        nn.init.trunc_normal_(self.tokeners.data, mean=0.0, std=0.02)
        self.ratio = hidden_size ** -0.5

        self.fq = nn.Linear(hidden_size, hidden_size)
        self.fk = nn.Linear(hidden_size, hidden_size)
        self.fv = nn.Linear(hidden_size, hidden_size)

    def forward(self, text_embedding):
        B = text_embedding.shape[0]
        tokeners = self.tokeners.repeat(B, 1, 1)
        oreder = tokeners[:, 1:, :]

        reordered_express = torch.bmm(self.fq(oreder), self.fk(text_embedding).transpose(1, 2))
        reordered_express = reordered_express * self.ratio
        reordered_express = reordered_express.softmax(dim=-1)
        reordered_express = oreder + torch.bmm(reordered_express, self.fv(text_embedding))

        reordered_express = torch.cat([tokeners[:, 0:1, :], reordered_express], dim=1)

        return reordered_express


class ExtMemory(nn.Module):
    def __init__(self, reader_num, hidden_size):
        super().__init__()

        self.reader_num = reader_num
        self.reader = nn.Parameter(torch.FloatTensor(1, reader_num, hidden_size))
        nn.init.trunc_normal_(self.reader.data, mean=0.0, std=1.0)

    def forward(self, embedding, mask):
        B = embedding.shape[0]
        readers = self.reader.repeat(B, 1, 1)
        embedding = torch.cat([embedding, readers], dim=1)

        mem_mask = torch.ones(
            B, self.reader_num, device=mask.device, dtype=torch.long
        )
        mask = torch.cat([mem_mask, mask], dim=1)

        return embedding, mask


class RepMemory(nn.Module):
    def __init__(self, reader_num, length, times, hidden_size):
        super().__init__()

        self.tokeners = nn.Parameter(torch.FloatTensor(1, length * times, hidden_size))
        nn.init.trunc_normal_(self.tokeners.data, mean=0.0, std=1.0)

        self.fc = nn.Sequential(
            nn.Linear(reader_num, length * 2),
            nn.GELU(),
            nn.Linear(length * 2, length)
        )

        # self.fc1 = nn.Parameter(torch.FloatTensor(1, hidden_size, hidden_size // 16))
        # self.fc2 = nn.Parameter(torch.FloatTensor(1, hidden_size // 16, hidden_size))

        self.reader_num = reader_num
        self.length = length
        self.scale = hidden_size ** -0.5

    def forward(self, embedding, res):
        
        B = embedding.shape[0]

        readers = embedding[:, :self.reader_num, :] # B x R x C
        readers = self.fc(readers.transpose(-1, -2)) # B x C x L
        
        tokeners = self.tokeners.repeat(B, 1, 1) # B x TL x C

        # rel = torch.bmm(torch.matmul(tokeners, self.fc1), torch.matmul(self.fc2, readers)).transpose(-1, -2) # B x L x TL
        rel = torch.bmm(tokeners, readers).transpose(-1, -2) # B x L x TL
        rel = rel * self.scale
        rel = rel.softmax(dim=-1)

        tokeners = torch.bmm(rel, tokeners) # B x L x C

        if res:
            embedding[:, -self.length:, :] += tokeners
        else:
            embedding = torch.cat([embedding, tokeners], dim=1)

        return embedding

    
class Adapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.input_dim = dim
        # reduction_factor = 8
        # self.down_sample_size = self.input_dim // reduction_factor
        self.down_sample_size = 96
        self.activation = nn.ReLU(inplace=True)
        self.down_sampler = nn.Linear(self.input_dim, self.down_sample_size) 
        nn.init.normal_(self.down_sampler.weight, std=1e-2)
        nn.init.zeros_(self.down_sampler.bias)
        self.up_sampler = nn.Linear(self.down_sample_size, self.input_dim)
        nn.init.normal_(self.up_sampler.weight, std=1e-2)
        nn.init.zeros_(self.up_sampler.bias)

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        z = self.up_sampler(z)
        output = x + z
        return output
