"""从本地训练 checkpoint 生成文本；不实现 KV Cache。"""

import argparse

import torch
from tokenizers import Tokenizer

from .config import GPTConfig
from .model import GPT


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default="out/gpt2/ckpt.pt")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()
    if args.tokens <= 0 or args.temperature <= 0 or args.top_k < 0:
        parser.error("tokens 和 temperature 必须为正数；top-k >= 0")
    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    cfg = GPTConfig(**checkpoint["config"])
    model = GPT(cfg)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    tokenizer = Tokenizer.from_str(checkpoint["tokenizer_json"])
    eot = tokenizer.token_to_id("<|endoftext|>")
    if tokenizer.get_vocab_size() != cfg.vocab_size or eot is None:
        raise ValueError("checkpoint 的 tokenizer 与模型配置不匹配")
    ids = tokenizer.encode(args.prompt, add_special_tokens=False).ids
    # 空 prompt 从文档分隔符开始。
    x = torch.tensor([ids or [eot]], dtype=torch.long, device=device)
    for _ in range(args.tokens):
        logits, _ = model(x[:, -cfg.block_size :])
        logits = logits[:, -1, :] / args.temperature
        if args.top_k:
            cutoff = torch.topk(logits, min(args.top_k, cfg.vocab_size)).values[:, [-1]]
            logits = logits.masked_fill(logits < cutoff, float("-inf"))
        token = torch.multinomial(logits.softmax(dim=-1), 1)
        x = torch.cat((x, token), dim=1)
        if token.item() == eot:
            break
    print(tokenizer.decode(x[0].tolist(), skip_special_tokens=True))


if __name__ == "__main__":
    main()
