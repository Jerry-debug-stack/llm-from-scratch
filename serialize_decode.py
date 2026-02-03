from cs336_basics import tokenizer
import numpy as np

my_tokenizer = tokenizer.from_files(
    None,
    "trained_values/tiny_story_2/vocab_final.pkl",
    "trained_values/tiny_story_2/merges_final.pkl",
    ["<|endoftext|>"]
)

text_addr = "../assignment1-data/TinyStoriesV2-GPT4-valid.txt"
target_addr = "trained_values/tiny_story_2/encoded_valid_txt"

def encode_text_to_file(input_file: str, output_file: str, chunk_size: int = 1024*1024):
    
    # 分块读取文件以避免内存问题
    with open(input_file, 'r', encoding='utf-8') as f:
        a = f.read()
    all_tokens = my_tokenizer.encode(a)
    # 转换为numpy数组（uint16）
    token_array = np.array(all_tokens, dtype=np.uint16)
    
    # 保存到文件
    np.save(output_file, token_array)
    print(f"编码完成。共 {len(token_array)} 个tokens，已保存到 {output_file}")

# 使用方法
if __name__ == "__main__":
    # 直接调用函数
    encode_text_to_file(text_addr, target_addr)
    
    # 或者如果你想直接加载验证
    # loaded_array = np.load(target_addr)
    # print(f"加载的数组形状: {loaded_array.shape}")
    # print(f"前10个tokens: {loaded_array[:10]}")
