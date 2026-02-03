import torch
import math
import einops
from jaxtyping import Bool, Float, Int
from torch import Tensor

class Linear(torch.nn.Module):
    def __init__(self, 
        in_features:int, #final dimension of the input
        out_features:int, #final dimension of the output
        device:torch.device | None = None, #Device to store the parameters on
        dtype:torch.dtype | None = None #Data type of the parameters
    ):
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty(in_features,out_features,device=device, dtype=dtype))
        sigma:float = math.sqrt(2.0/(in_features+out_features))
        torch.nn.init.trunc_normal_(self.W,0,sigma,-3*sigma,3*sigma)
    def forward(self, x: Tensor) -> Tensor:
        return x @ self.W

class Embedding(torch.nn.Module):
    def __init__(self,
        num_embeddings: int, #Size of the vocabulary
        embedding_dim: int, #Dimension of the embedding vectors
        device: torch.device | None = None, #Device to store the parameters on
        dtype: torch.dtype | None = None #Data type of the parameters
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight,0,1,-3,3)
    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]

class RMSNorm(torch.nn.Module):
    def __init__(self,
        d_model: int, # Hidden dimension of the model
        eps: float = 1e-5, #Epsilon value for numerical stability
        device: torch.device | None = None, #Device to store the parameters on
        dtype: torch.dtype | None = None #Data type of the parameters
    ):
        super().__init__()
        self.d_model:int = d_model
        self.eps:float = eps
        self.gamma = torch.nn.Parameter(torch.ones(d_model,device=device,dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        # Process an input tensor of shape (batch_size, sequence_length, d_model) 
        # and return a tensor of the same shape.
        in_dtype = x.dtype
        y_pow:Tensor = x ** 2
        mean_y_pow:Tensor = y_pow.mean(dim=-1,keepdim=True)
        rms:Tensor = torch.sqrt(mean_y_pow + self.eps)
        normalized_x:Tensor = x / rms
        result:Tensor = normalized_x * self.gamma
        return result.to(in_dtype)

def silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    return in_features / (1 + (-in_features).exp())

class SwiGLU(torch.nn.Module):
    def __init__(self,
        d_model: int, #Dimensionality of the feedforward input and output.
        d_ff: int, #Dimensionality of the up-project happening internally to swiglu
        device: torch.device | None = None, #Device to store the parameters on
        dtype: torch.dtype | None = None #Data type of the parameters
    ):
        super().__init__()
        self.d_model:int=d_model
        self.d_ff:int = d_ff
        self.W1 = torch.nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.W2 = torch.nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        self.W3 = torch.nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        pass
    def forward(self,x:Tensor) -> Tensor:
        w1_x:Tensor = einops.einsum(x,self.W1,"... d_model,d_ff d_model -> ... d_ff")
        silu_w1x:Tensor = w1_x*torch.sigmoid(w1_x)
        w3_x:Tensor = einops.einsum(x,self.W3,"... d_model,d_ff d_model -> ... d_ff")
        ans:Tensor = einops.einsum(silu_w1x*w3_x,self.W2,"... d_ff,d_model d_ff -> ... d_model")
        return ans

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self,
        theta: float, # Θ value for the RoPE
        d_k: int, # dimension of query and key vectors
        max_seq_len: int, # Maximum sequence length that will be inputted
        device: torch.device | None = None # Device to store the buffer on
    ):
        super().__init__()
        self.d_k:int = d_k
        self.theta:float = theta
        self.max_seq_len:int = max_seq_len
        frequencies:Tensor = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        positions:Tensor = torch.arange(max_seq_len, device=device).float().unsqueeze(1)
        angles:Tensor = positions * frequencies
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
    def forward(self, 
        x: Float[Tensor, " ... sequence_length d_k"],
        token_positions: Int[Tensor, " ... sequence_length"]
    ) -> Tensor:
        seq_len = x.size(-2)
        cos = self.cos_cached[token_positions] # type: ignore
        sin = self.sin_cached[token_positions] # type: ignore
        x_reshaped = x.view(*x.shape[:-1], -1, 2)
        x1 = x_reshaped[..., 0]
        x2 = x_reshaped[..., 1]
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos
        x_rotated = torch.stack([x1_rot, x2_rot], dim=-1)
        return x_rotated.view(*x.shape)

def softmax(x: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    max_x = x.max(dim,keepdim=True).values
    x1 = x - max_x
    exp_x1 = x1.exp()
    return exp_x1 / exp_x1.sum(dim,keepdim=True)

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.size()[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask==False, float('-inf'))
    attention_weights = softmax(scores, dim=-1) 
    attention = attention_weights @ V
    return attention

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        use_rope: bool,
        q_proj_weight: Float[Tensor, "d_model d_in"],
        k_proj_weight: Float[Tensor, "d_model d_in"],
        v_proj_weight: Float[Tensor, "d_model d_in"],
        o_proj_weight: Float[Tensor, "d_model d_model"],
        device:str|None=None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryPositionalEmbedding(theta,d_model//num_heads,max_seq_len,device=device)
        self.q_proj_weight = torch.nn.Parameter(q_proj_weight)
        self.k_proj_weight = torch.nn.Parameter(k_proj_weight)
        self.v_proj_weight = torch.nn.Parameter(v_proj_weight)
        self.o_proj_weight = torch.nn.Parameter(o_proj_weight)
    def forward(self,
        in_features: Float[Tensor, "... sequence_length d_in"],
        token_positions: Int[Tensor, " ... sequence_length"]|None
    ) -> Float[Tensor, " ... sequence_length d_out"]:
        seq_len = in_features.shape[-2]
        Q_all = einops.einsum(in_features, self.q_proj_weight, "... seq d_in, d_model d_in -> ... seq d_model")
        K_all = einops.einsum(in_features, self.k_proj_weight, "... seq d_in, d_model d_in -> ... seq d_model")
        V_all = einops.einsum(in_features, self.v_proj_weight, "... seq d_in, d_model d_in -> ... seq d_model")
        Q = einops.rearrange(Q_all, "... seq (head d_k) -> ... head seq d_k", head=self.num_heads)
        K = einops.rearrange(K_all, "... seq (head d_k) -> ... head seq d_k", head=self.num_heads)
        V = einops.rearrange(V_all, "... seq (head d_k) -> ... head seq d_k", head=self.num_heads)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=in_features.device), diagonal=1).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = ~mask
        if self.use_rope:
            Q = self.rope.forward(Q,token_positions[:,None,:]) #type: ignore
            K = self.rope.forward(K,token_positions[:,None,:]) #type: ignore
        head_output = scaled_dot_product_attention(Q,K,V,mask)
        combined = einops.rearrange(head_output, "... head seq d_k -> ... seq (head d_k)")
        output = einops.einsum(combined, self.o_proj_weight,"... seq d_model, d_model_out d_model -> ... seq d_model_out")
        return output

class Transformer_block(torch.nn.Module):
    def __init__(self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        weights: dict[str, Tensor],
        device:str|None = None
    ):
        super().__init__()
        self.rms1:RMSNorm = RMSNorm(d_model,device=device)
        self.rms1.load_state_dict({"gamma":weights["ln1.weight"]})
        self.mhsa:MultiHeadSelfAttention = MultiHeadSelfAttention(
            d_model,num_heads,max_seq_len,theta,True,
            weights["attn.q_proj.weight"],weights["attn.k_proj.weight"],
            weights["attn.v_proj.weight"],weights["attn.output_proj.weight"],
            device=device
        )
        self.rms2:RMSNorm = RMSNorm(d_model,device=device)
        self.rms2.load_state_dict({"gamma":weights["ln2.weight"]})
        self.swiglu:SwiGLU = SwiGLU(d_model,d_ff,device=device)
        self.swiglu.load_state_dict({"W1":weights["ffn.w1.weight"],"W2":weights["ffn.w2.weight"],"W3":weights["ffn.w3.weight"]})
    def forward(self,
        in_features: Float[Tensor, " batch sequence_length d_model"]
    ) -> Float[Tensor, " batch sequence_length d_model"]:
        normed_input = self.rms1.forward(in_features)
        
        batch_size, seq_len, _ = in_features.shape
        positions_base = torch.arange(seq_len, device=in_features.device, dtype=torch.int32)
        positions = positions_base.unsqueeze(0).expand(batch_size, seq_len)

        after_mhsa = self.mhsa.forward(normed_input,positions)
        input_residual = in_features + after_mhsa
        normed_output = self.rms2.forward(input_residual)
        return self.swiglu.forward(normed_output)+input_residual

class Transformer_LM(torch.nn.Module):
    def __init__(self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        weights: dict[str, Tensor],
        device:str|None=None
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model,device=device)
        self.token_embeddings.weight.data = weights["token_embeddings.weight"]
        
        self.layers = torch.nn.ModuleList([
            Transformer_block(
                d_model, num_heads, d_ff, context_length, rope_theta,
                {
                    "ln1.weight":weights[f"layers.{i}.ln1.weight"],
                    "ln2.weight":weights[f"layers.{i}.ln2.weight"],
                    "attn.q_proj.weight":weights[f"layers.{i}.attn.q_proj.weight"],
                    "attn.k_proj.weight":weights[f"layers.{i}.attn.k_proj.weight"],
                    "attn.v_proj.weight":weights[f"layers.{i}.attn.v_proj.weight"],
                    "attn.output_proj.weight":weights[f"layers.{i}.attn.output_proj.weight"],
                    "ffn.w1.weight":weights[f"layers.{i}.ffn.w1.weight"],
                    "ffn.w2.weight":weights[f"layers.{i}.ffn.w2.weight"],
                    "ffn.w3.weight":weights[f"layers.{i}.ffn.w3.weight"],
                },
                device=device
            )
            for i in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model,device=device)
        self.ln_final.load_state_dict({"gamma": weights["ln_final.weight"]})
        self.lm_head = Linear(d_model, vocab_size,device=device)
        self.lm_head.load_state_dict({"W":weights["lm_head.weight"].t()})
        # self.lm_head.weight = self.token_embeddings.weight
    def forward(self,
        in_indices: Int[Tensor, " batch_size sequence_length"]
    )-> Float[Tensor, " batch_size sequence_length vocab_size"]:
        after_embedding = self.token_embeddings.forward(in_indices)
        tmp = after_embedding
        for i in self.layers:
            tmp = i.forward(tmp)
        after_rmsnorm = self.ln_final.forward(tmp)
        return self.lm_head.forward(after_rmsnorm)

def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    inputs_changed = inputs - inputs.max(dim=1,keepdim=True).values
    log_softmax = inputs_changed - inputs_changed.exp().sum(dim=1,keepdim=True).log()
    
    targets_expanded = targets.unsqueeze(1)
    log_probs_true = torch.gather(log_softmax, 1, targets_expanded).squeeze(1) 
    loss = -log_probs_true.mean()
    return loss
