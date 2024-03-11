# define GPT model

import math
import inspect
from dataclasses import dataclass #decorator that adds __init__

import torch
import torch.nn as nn
from torch.nn import functional as F 

class LayerNorm(nn.Module):
    """ Layer normalization with optional bias. """

    def __init__(self, ndim, bias):
        super().init()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config): # config is a dict
        assert config.n_embd % config.n_head == 0
        # key, query, value for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias = config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias = config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention
        self.flash = hasattr(torch.nn.function, "scaled_dot_product_attention") # hasattr checks wheter the object has an attribute of the provided name
        if not self.flash:
            print("Warning: slow attention.")
            # causal attention (save them in buffer so to have the param in state_dict but not update by optim)
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                 .view(1,1, config.block_size, config.block_size)) # view -->to reshape


        def forward(self, x):
            B, T, C = x.size() # batch size, sequence length, embedding dim (n_embd)

            # calculate query, key and value vectors for all heads in batch and move head forward to the batch dim (?)
            # output of c_attn(x) is n_embd * 3 !
            q, k, v = self.c_attn(x).split(self.n_embd, dim = 2)
            # // floor division (why n_head dim?)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #(B, nh, T, hs) # hs is --> emdedding dim/num of heads
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #(B, nh, T, hs)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #(B, nh, T, hs)

            # casual self-attention (B, nh, T, hs) x (B, nh, hs, T) --> (B, nh, T, T)
            #                           Query X Key --> Attention Scores/Weights

            if self.flash:
                # efficient attention
                y = torch.nn.function.scaled_dot_product_attention(q, k, v, attn_mask = None, dropout_p = self.dropout if self.training else 0, is_causal = True)

            else:
                # manual implementation of self-attention
                att = (q @ k.transpose(-2,-1)) * (1.0/math.sqrt(k.size(-1))) #scaled self-attention
                att = att.mask_fill(self.bias[:, :, :T, :T] == 0, float("-inf")) # mask
                att = F.softmax(att, dim = -1) #over token
                att = self.attn_dropout(att) # randomly ignore some tokens ?
                y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            y = y.transpose(1,2).contiguous().view(B, T, C) # reassemble heads side-by-side ; contiguous is for memory format

            # output projection (to combine the heads into a single representation) + dropout
            y = self.resid_dropout(self.c_proj(y))

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4* config.n_embd, bias = config.bias) # * 4 to expand dim of hidden state
        self.gelu = nn.GELU() # Gaussian Error Linear Units function
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd, bias = config.bias)
        self.dropout = nn.Dropout(config.dropout)


    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias = config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias = config.bias)
        self.mlp = MLP(config)

    def forward(self, x):

        x = x + self.attn(self.ln_1(x)) # pre-layer norm formulation!
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass # makes the following varibales a class variable in the init statement
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 
    n_layer: int = 12
    n_head: int = 12 # multi-head attention
    n_embd: int = 768 # embedding dimensionality
    dropout: float = 0.0 # drop-out rate
    bias: bool = True  # True: bias in Linear Layer and Layer Norm; False: a bit better and faster

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # ModuleDict is an ordered dict that works like a python dict
        self.transformer = nn.ModuleDict( dict( 
            wte = nn.Embedding(config.vocab_size, config.n_embd)
            wpe  = nn.Embedding(config.block_size, config.n_embd) # positional is learned
            drop = nn.Dropout(config.dropout)
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias = config.biasp)
        ))

        self.lm_head = nn.Linear(config.n_emdn, config.vocab_size, bias = False) # read out layer
        # some comments

        self.transformer.wte.weight = self.lm_head.weight # weight-tying, same weight for mapping from vocab to vector and viceverse

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projection, as per GPT2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean = 0.0, std = 0.02/math.sqrt(2*config.n_layer))


        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))


    def get_num_params(self, non_embedding = True):
        




















