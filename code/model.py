import torch 
import torch.nn as nn


class InputEmbedding(nn.Module):
    def __init__(self,d_model:int,vocab_size:int) -> None:
        '''
            d_model: int: the dimension of each embedding => 512
            nn.Embeddig: dict which maps numbers(words) to the vector representation which is trainable 
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
        return self.embedding(x) * torch.sqrt(torch.tensor(self.d_model,dtype=torch.float32))  # multiply weights by sqrt(d_model) as per the paper
    

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
        '''
            h : Number of heads
        '''
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

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) ->None:
        super().__init__()
        self.dropout = nn.dropout(dropout)
        self.norm = LayerNormalization()
    
    def forward(self, x, sublayer):   # sublayer will be attention block in this case.
        return x + self.dropout(sublayer(self.norm(x)))    # add and norm
        # although in paper it first passes through sublayer and then norm


class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    
    def forward(self, x, src_mask):
        # x first passed though the attention block then residual-connect(add norm) the output with x
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self,x , mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self.self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x,lambda x: self.self_attention_block(x,x,x,tgt_mask))
        x = self.residual_connections[1](x,lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x) 


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) ->None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # (batch, seq_len, d_model) => (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)


class Transformer(nn.Module):
    def __init__(self, 
            encoder: Encoder, 
            decoder: Decoder, 
            src_embed: InputEmbedding, 
            tgt_embed: InputEmbedding,
            src_pos: PositionalEncoding,
            tgt_pos: PositionalEncoding,
            projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
    
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src,src_mask)
    
    def decode(self, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decode(tgt, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)



def build_transformer(
        src_vocab_size: int, 
        tgt_vocab_size: int, 
        src_seq_len: int, 
        tgt_seq_len: int,
        d_model: int = 512,
        N: int = 6,
        h: int = 8,
        dropout: float = 0.1,
        d_ff: int = 2048,
    ) -> Transformer:
    
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)
    
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    encoder_blocks = []
    for _ in range(N):
        encoder_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer
