import cProfile
import pstats
from cs336_basics.train_bpe import train_bpe_streaming_read

def main():
    train_bpe_streaming_read(
        input_path="tests/fixtures/tinystories_sample_5M.txt",
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],  # 按你的真实情况
    )

if __name__ == "__main__":
    cProfile.run(
        "main()",
        filename="tokenizer.prof"
    )