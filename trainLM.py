from typing import Callable, Iterable, Iterator, Optional,BinaryIO,IO
import torch
from torch import Tensor

import math
import einops
from jaxtyping import Bool, Float, Int

import numpy.typing as npt
import numpy as np
import os

class AdamW(torch.optim.Optimizer):
    def __init__(self,
        params:Iterator[torch.nn.Parameter],
        weight_decay:float=0.01,
        lr:float=1e-3,
        betas:tuple[float,float]=(0.9,0.999),
        eps:float=1e-8
    ):
        defaults = {
            "weight_decay":weight_decay,
            "lr":lr,
            "betas":betas,
            "eps":eps
        }
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None,outlet_lr:float|None=None): # type: ignore
        loss = None if closure is None else closure()
        for group in self.param_groups:
            if outlet_lr is not None:
                lr = outlet_lr
            else:
                lr = group["lr"]
            beta1,beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                state['step'] += 1
                step:int = state['step']
                
                state_m:Tensor = state['m']
                state_m = state['m'] = beta1 * state_m + (1 - beta1) * grad
                
                state_v:Tensor = state['v']
                state_v = state['v'] = beta2 * state_v + (1 - beta2) * (grad**2)
                
                alpha_t = lr * (math.sqrt(1-beta2**step)/(1-beta1**step))
                p.data.sub_(alpha_t * ( state_m / (state_v.sqrt() + eps)))
                p.data.mul_(1-lr*weight_decay)

def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    total_norm = math.sqrt(sum(torch.norm(parameter.grad.data)**2 for parameter in parameters if parameter.grad is not None))
    if total_norm > max_l2_norm:
        times = max_l2_norm / (total_norm + 1e-6)
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.data.mul_(times)
    return

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start_idx = len(dataset) - context_length
    start_indices = np.random.randint(0, max_start_idx, size=batch_size)
    input_sequences = np.zeros((batch_size, context_length), dtype=np.int64)
    target_sequences = np.zeros((batch_size, context_length), dtype=np.int64)
    for i, start_idx in enumerate(start_indices):
        input_sequences[i] = dataset[start_idx:start_idx + context_length]
        target_sequences[i] = dataset[start_idx + 1:start_idx + context_length + 1]
    inputs = torch.from_numpy(input_sequences).long()
    targets = torch.from_numpy(target_sequences).long()
    inputs = inputs.to(device)
    targets = targets.to(device)
    return inputs,targets

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes]
) -> None:
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str = None
) -> int:
    checkpoint = torch.load(src, map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']

if __name__ == "__main__":
    import base_layers
    import datetime
    start_new = True
    data_path = "trained_values/tiny_story_2/encoded_txt.npy"
    check_point_reload_path = "tmp/train_loss_changes"
    lm_valid_path = "trained_values/tiny_story_2/encoded_valid_txt.npy"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    max_lr = lr = 3e-4
    min_lr = 3e-5
    warmup_iter = 200
    cosine_cycle_iters = 4000

    # simple:(0.9,0.999)
    eps = 1e-8
    betas = (0.9, 0.98)
    weight_decay = 0.01

    vocab_size = 10000
    context_length = 256

    d_model = 512
    num_heads = 16
    d_ff = 1344

    num_layers = 4

    rope_theta = 10000
    epochs = 5000

    grad_accum_steps = 1
    batch_size = 256
    check_size = 32

    save_dir = "tmp"
    save_checkpoint_delta = 5000

    batch_time_loss_list = []

    print("==start training==")
    print(f"load data:{data_path}")
    memmap = np.memmap(data_path, mode="r", dtype=np.uint16)  # 添加dtype
    valid_memmap = np.memmap(lm_valid_path, mode="r", dtype=np.uint16)  # 添加dtype
    
    embed_std = 0.02
    proj_std = 0.02 / math.sqrt(num_layers)   # residual-aware
    ffn_std  = 0.02
    lm_head_std = 0.08                        # 🔥 关键：放大 logits

    # 初始化权重（放在 GPU 上）
    weights = {
        # token embedding：正常
        "token_embeddings.weight": torch.randn(
            (vocab_size, d_model), device=device
        ) * embed_std,

        # final LN
        "ln_final.weight": torch.ones((d_model,), device=device),

        # 🔥 lm head：显式放大
        "lm_head.weight": torch.randn(
            (vocab_size, d_model), device=device
        ) * lm_head_std,
    }

    for i in range(num_layers):
        # ===== Attention =====
        weights[f"layers.{i}.attn.q_proj.weight"] = (
            torch.randn((d_model, d_model), device=device) * proj_std
        )
        weights[f"layers.{i}.attn.k_proj.weight"] = (
            torch.randn((d_model, d_model), device=device) * proj_std
        )
        weights[f"layers.{i}.attn.v_proj.weight"] = (
            torch.randn((d_model, d_model), device=device) * proj_std
        )

        # output proj：**最重要的 residual 控制点**
        weights[f"layers.{i}.attn.output_proj.weight"] = (
            torch.randn((d_model, d_model), device=device) * proj_std
        )

        # ===== LayerNorm =====
        weights[f"layers.{i}.ln1.weight"] = torch.ones((d_model,), device=device)
        weights[f"layers.{i}.ln2.weight"] = torch.ones((d_model,), device=device)

        # ===== FFN (SwiGLU) =====
        weights[f"layers.{i}.ffn.w1.weight"] = (
            torch.randn((d_ff, d_model), device=device) * ffn_std
        )
        weights[f"layers.{i}.ffn.w3.weight"] = (
            torch.randn((d_ff, d_model), device=device) * ffn_std
        )

        # 🔥 FFN output：同样是 residual，必须缩
        weights[f"layers.{i}.ffn.w2.weight"] = (
            torch.randn((d_model, d_ff), device=device) * proj_std
        )
    
    Model = base_layers.Transformer_LM(vocab_size, context_length, d_model, num_layers,
                                       num_heads, d_ff, rope_theta, weights,device=device)
    
    adamw = AdamW(Model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    total_params = sum(p.numel() for p in Model.parameters())
    print("total params:", total_params)
    epoch_start = 0
    if not start_new:
        # 修改load_checkpoint函数以支持GPU
        epoch_start = load_checkpoint(check_point_reload_path, Model, adamw, device)
    
    print("Start epoch!!!!")
    scaler = torch.amp.GradScaler()
    for epoch in range(epoch_start, epochs):

        adamw.zero_grad(set_to_none=True)

        # ======== train (gradient accumulation) ========
        for micro in range(grad_accum_steps):

            learn_input, learn_target = get_batch(
                memmap, batch_size, context_length, device
            )

            with torch.amp.autocast(dtype=torch.bfloat16, device_type='cuda'):
                logits = Model.forward(learn_input)
                logits = logits[:, :-1, :]
                targets = learn_target[:, :-1]

                loss = base_layers.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    targets.reshape(-1)
                )
                # 🔥 关键：均分 loss
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            # 🔥 立刻释放大张量
            del logits, loss

        # ======== optimizer step ========
        scaler.unscale_(adamw)
        gradient_clipping(Model.parameters(), max_l2_norm=1.0)

        current_lr = lr_cosine_schedule(
            epoch + 1,
            max_lr,
            min_lr,
            warmup_iter,
            cosine_cycle_iters
        )
        for param_group in adamw.param_groups:
            param_group["lr"] = current_lr

        scaler.step(adamw)
        scaler.update()

        if epoch % 20 == 19:
            # ======== validation ========
            check_input, check_target = get_batch(
                valid_memmap, check_size, context_length, device
            )

            with torch.no_grad():
                with torch.amp.autocast(dtype=torch.bfloat16, device_type='cuda'):
                    check_logits = Model.forward(check_input)
                    check_logits = check_logits[:, :-1, :]
                    check_targets = check_target[:, :-1]

                    check_loss = base_layers.cross_entropy(
                        check_logits.reshape(-1, vocab_size),
                        check_targets.reshape(-1)
                    )
            
            print(
                f"epoch {epoch + 1} | "
                f"val loss {check_loss.item():.4f} | "
                f"std {check_logits.std().item():.4f} | "
                f"lr {current_lr:.2e}"
            )

            batch_time_loss_list.append([
                epoch,
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                check_loss.item()
            ])

        if epoch % save_checkpoint_delta == save_checkpoint_delta - 1:
            save_checkpoint(
                Model,
                adamw,
                epoch,
                os.path.join(
                    save_dir, f"check_point_epoch{epoch+1}.sav"
                )
            )
    import pickle
    with open(os.path.join(save_dir, f"train_loss_changes"), "wb") as f:
        pickle.dump(batch_time_loss_list, f)
