import torch 
from torch import nn
import torch.nn.functional as F
import numpy as np 

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, p, d_input=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        if d_input is None:
            d_xq = d_xk = d_xv = d_model
        else:
            d_xq, d_xk, d_xv = d_input
            
        # Make sure that the embedding dimension of model is a multiple of number of heads
        assert d_model % self.num_heads == 0

        self.d_k = d_model // self.num_heads
        
        # These are still of dimension d_model. They will be split into number of heads 
        self.W_q = nn.Linear(d_xq, d_model, bias=False)
        self.W_k = nn.Linear(d_xk, d_model, bias=False)
        self.W_v = nn.Linear(d_xv, d_model, bias=False)
        
        # Outputs of all sub-layers need to be of dimension d_model
        self.W_h = nn.Linear(d_model, d_model)
        self.dropout1 = nn.Dropout(p)
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                # m.weight.data.normal_(0.0, 0.02)
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            
    def scaled_dot_product_attention(self, Q, K, V):
        # batch_size = Q.size(0) 
        # k_length = K.size(-2) 
        
        # Scaling by d_k so that the soft(arg)max doesnt saturate
        Q = Q / np.sqrt(self.d_k)                         # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(Q, K.transpose(2,3).contiguous())          # (bs, n_heads, q_length, k_length)
        
        A = F.softmax(scores, dim=3)   # (bs, n_heads, q_length, k_length)
        
        # Get the weighted average of the values
        H = torch.matmul(A, V)     # (bs, n_heads, q_length, dim_per_head)

        H = self.dropout1(H)
        return H, A 

        
    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (heads X depth)
        Return after transpose to put in shape (batch_size X num_heads X seq_length X d_k)
        """
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2).contiguous()

    def group_heads(self, x, batch_size):
        """
        Combine the heads again to get (batch_size X seq_length X (num_heads times d_k))
        """
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
    

    def forward(self, X_q, X_k, X_v):
        batch_size, seq_length, dim = X_q.size()

        # After transforming, split into num_heads 
        Q = self.split_heads(self.W_q(X_q), batch_size)  # (bs, n_heads, q_length, dim_per_head)
        K = self.split_heads(self.W_k(X_k), batch_size)  # (bs, n_heads, k_length, dim_per_head)
        V = self.split_heads(self.W_v(X_v), batch_size)  # (bs, n_heads, v_length, dim_per_head)
        
        # Calculate the attention weights for each of the heads
        H_cat, A = self.scaled_dot_product_attention(Q, K, V)
        
        # Put all the heads back together by concat
        H_cat = self.group_heads(H_cat, batch_size)    # (bs, q_length, dim)
        
        # Final linear layer  
        H = self.W_h(H_cat)          # (bs, q_length, dim)
        
        return H, A


class CNN(nn.Module):
    def __init__(self, d_model, hidden_dim, p):
        super().__init__()
        self.k1convL1 = nn.Linear(d_model, hidden_dim)
        self.k1convL2 = nn.Linear(hidden_dim, d_model)
        self.activation = nn.ReLU(True)
        self.dropout = nn.Dropout(p=p)
        self.dropout2 = nn.Dropout(p=p)
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                m.weight.data.normal_(0.0, 0.04)
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                # nn.init.xavier_uniform_(m.weight)
                m.weight.data.normal_(0.0, 0.04)
                if m.bias is not None:                    
                    nn.init.constant_(m.weight, 1)
        
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            m.weight.data.normal_(0.0, 0.04)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            # m.weight.data.normal_(0.0, 0.02)
            with torch.no_grad():
                if m.bias is not None:                    
                    nn.init.constant_(m.weight, 1)
        
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            m.weight.data.normal_(0.0, 0.04)
    
    def forward(self, x):
        x = self.k1convL1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.k1convL2(x)
        x = self.dropout2(x)
        return x

class TFEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, conv_hidden_dim, p=0.1):
        super().__init__()

        self.mha = MultiHeadAttention(d_model, num_heads, p)
        self.cnn = CNN(d_model, conv_hidden_dim, p)

        self.layernorm1 = nn.LayerNorm(normalized_shape=d_model, eps=1e-5)
        self.layernorm2 = nn.LayerNorm(normalized_shape=d_model, eps=1e-5)
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                if m.bias is not None:
                    # m.weight.data.normal_(0.0, 0.02)
                    nn.init.constant_(m.weight, 1.)
                    nn.init.constant_(m.bias, 0.)
                    # nn.init.constant_(m.weight, 1)
        
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            m.weight.data.normal_(0.0, 0.04)
    
    def forward(self, x):
        
        # Multi-head attention 
        attn_output, _ = self.mha(x, x, x)  # (batch_size, input_seq_len, d_model)
        
        # Layer norm after adding the residual connection 
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        
        # Feed forward 
        cnn_output = self.cnn(out1)  # (batch_size, input_seq_len, d_model)
        
        #Second layer norm after adding residual connection 
        out2 = self.layernorm2(out1 + cnn_output)  # (batch_size, input_seq_len, d_model)

        return out2

class TFEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, ff_hidden_dim, p=0.1, norm=None):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers        

        self.enc_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.enc_layers.append(TFEncoderLayer(d_model, num_heads, ff_hidden_dim, p))        
        
        self.norm = nn.LayerNorm(normalized_shape=d_model, eps=1e-5) if norm is not None else norm
        
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                if m.bias is not None:                    
                    nn.init.constant_(m.weight, 1.)
                    nn.init.constant_(m.bias, 0.)
                    # m.weight.data.normal_(0.0, 0.02)
                    # nn.init.constant_(m.weight, 1)
        
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)    

    def forward(self, x):        

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        if self.norm is not None:
            x = self.norm(x)
        return x  # (batch_size, input_seq_len, d_model)