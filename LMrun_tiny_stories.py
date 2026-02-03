import math
import torch
import torch.nn.functional as F
from cs336_basics import base_layers,tokenizer,trainLM

@torch.no_grad()
def generate_reply(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int | None = None,
):
    model.eval()
    device = input_ids.device
    eos_id = 256

    for _ in range(max_new_tokens):
        logits = model(input_ids)          # (1, T, V)
        logits = logits[:, -1, :]          # (1, V)

        logits = logits / temperature

        if top_k is not None:
           v, _ = torch.topk(logits, top_k)
           logits[logits < v[:, [-1]]] = -float("inf")

        probs = base_layers.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)

        input_ids = torch.cat([input_ids, next_id], dim=1)

        if next_id.item() == eos_id:
            break

    return input_ids

def build_prompt(history):
    #prompt = ""
    #for role, text in history:
    #    if role == "user":
    #        prompt += f"User: {text}\n"
    #    else:
    #        prompt += f"Assistant: {text}\n"
    #prompt += "Assistant: "
    #return prompt
    prompt = ""
    for role,text in history:
        prompt += text
    return prompt

def chat(model, tokenizer_):
    device = next(model.parameters()).device
    history = []

    print("🤖 Chat started. Type 'exit' to quit.\n")

    while True:
        user_input = input("🧑 User: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        history.append(("user", user_input))

        prompt = build_prompt(history)
        input_ids = tokenizer_.encode(prompt)
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

        output_ids = generate_reply(
            model,
            tokenizer_,
            input_ids,
            max_new_tokens=128,
            temperature=0.8,
            top_k=10,
        )

        # 只取新生成的部分
        new_tokens = output_ids[0, input_ids.size(1):].tolist()
        reply = tokenizer_.decode(new_tokens)

        # 简单清洗
        reply = reply.split("<|endoftext|>")[0].strip()

        print(f"🤖 Assistant: {reply}\n")

        history.append(("assistant", reply))
if __name__ == "__main__":
    check_point_addr = "remote/lr_3e-4_3e-5/tmp_batch_128/check_point_epoch10000.sav"
    tokenizer_vocab = "trained_values/tiny_story_2/vocab_final.pkl"
    tokenizer_merge = "trained_values/tiny_story_2/merges_final.pkl"
    tokenizer_special_tokens = ["<|endoftext|>"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vocab_size = 10000
    context_length = 256
    d_model = 512
    num_heads = 16
    d_ff = 1344
    num_layers = 4
    rope_theta = 10000

    weights = {
        "token_embeddings.weight": torch.zeros((vocab_size, d_model), device=device),
        "ln_final.weight": torch.zeros((d_model,), device=device),
        "lm_head.weight": torch.zeros((vocab_size, d_model), device=device),
    }
    for i in range(num_layers):
        weights[f"layers.{i}.attn.q_proj.weight"] = (
            torch.zeros((d_model, d_model), device=device)
        )
        weights[f"layers.{i}.attn.k_proj.weight"] = (
            torch.zeros((d_model, d_model), device=device)
        )
        weights[f"layers.{i}.attn.v_proj.weight"] = (
            torch.zeros((d_model, d_model), device=device)
        )
        weights[f"layers.{i}.attn.output_proj.weight"] = (
            torch.zeros((d_model, d_model), device=device)
        )
        weights[f"layers.{i}.ln1.weight"] = torch.zeros((d_model,), device=device)
        weights[f"layers.{i}.ln2.weight"] = torch.zeros((d_model,), device=device)

        weights[f"layers.{i}.ffn.w1.weight"] = (
            torch.zeros((d_ff, d_model), device=device)
        )
        weights[f"layers.{i}.ffn.w3.weight"] = (
            torch.zeros((d_ff, d_model), device=device)
        )
        weights[f"layers.{i}.ffn.w2.weight"] = (
            torch.zeros((d_model, d_ff), device=device)
        )
    Model = base_layers.Transformer_LM(vocab_size, context_length, d_model, num_layers,
                                       num_heads, d_ff, rope_theta, weights,device=device)
    adams = trainLM.AdamW(Model.parameters())
    epochs = trainLM.load_checkpoint(check_point_addr,Model,adams,device)
    this_tokenizer = tokenizer.from_files(None,tokenizer_vocab,tokenizer_merge,tokenizer_special_tokens)
    chat(Model,this_tokenizer)
