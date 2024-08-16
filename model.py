import torch 
import torch.nn as nn


class InputEnbedding(nn.Module):
    def __init__(self,d_model:int,vocab_size:int) -> None:
        '''
            d_model: int: the dimension of each embedding => 512
            nn.Embedding: dict which maps numbers(words) to the vector representation which is trainable 
        '''
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)
        self.d_model = d_model
    
    def forward(self,x):
        '''
            x: torch.tensor: input tensor of shape (batch_size,seq_len)
        '''
        
        # (batch_size,seq_len) --> (batch_size,seq_len,d_model)
        return self.embedding(x) * torch.sqrt(torch.tensor(self.d_model,dtype=torch.float32))  # multiply weights   by sqrt(d_model) as per the paper
    

class PositionalEncoding(nn.Module):
    def __init__(self,d_model:int, seq_len: int, dropout: float) -> None:
        '''
            seq_len: maximum length of the sentence 
        '''
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # Create a matrix of shape (seq_len, d_model) so that be can add to the embedding 
        pe = torch.zeros(seq_len,d_model)
        # Create a vector of shape (seq_len,1)
        position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(10000.0) / d_model)) # tensor([1.0000, 0.9988, 0.9976 .... ]) , len = 512/2
        # Apply sin to even positions and cos to odd
        # Broadcasting
        # position => (seq_len,1) -> (seq_len,d_model)
        # div_term => (d_model) -> (1,d_model) -> (seq_len,d_model)
        # then position * div_term
        pe[:,0::2] = torch.sin(position * div_term) 
        pe[:,1::2] = torch.cos(position * div_term) 
        
        pe = pe.unsqueeze(0) # (1,seq_len,d_model)
        
        self.register_buffer('pe',pe) # Tensor will be saved in the file along with the state of the model but not as a learnt parameter
        
    def forward(self, x:torch.Tensor):
        '''
        '''
        # Add the posistional encoding to every word
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
        
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        '''
            eps : very small term for error term in denominator
        '''
        super().__init__()
        self.eps = eps
        # We also introduce two parameters, usually called gamma (multiplicative) and beta (additive) that introduce some fluctuations in the data, because maybe having all values between 0 and 1 may be too restrictive for the network. The network will learn to tune these two parameters to introduce fluctuations when necessary.
        self.alpha = nn.Parameter(torch.ones(1))  # multiplied 
        self.bias = nn.Parameter(torch.zeros(1))  # added
        # nn.Parameter makes the tensor learn able
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        
        return self.alpha*(x-mean)/(std*self.eps)+self.bias
        
# FFN(x) = max(0,xW1 + B1)W2 + B2
class FeedForwardBlock:
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model,d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff,d_model) # W2 and B2
    
    def forward(self,x):
        # (batch, Seq_len, d_model) --> (batch,seq_len,d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self,d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        
        assert d_model % h == 0, "d_model is not divisible by h"    
        
        self.d_k = d_model // h     # each head will have the whole sequence but different part of the embeddings
        self.w_q = nn.Linear(d_model, d_model)   
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # (batch, h, seq_len, d_k) @ (batch, h, d_k, seq_len) --> (batch, h, seq_len, seq_len)              # attention for every word with every word(seq_len,seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / torch.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

        
        
    def forward(self, q, k, v, mask):
        query = self.w_q(q)   # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k)     # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v)   # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        
        
        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)   
        # .contiguous() puts the tensors in continuous memory so that the transformation can happen in place.
        # operations like transpose and permute can potentailly change the memory layout
        
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)