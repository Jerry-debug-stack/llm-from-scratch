import heapq
import os
import pickle
from collections import defaultdict
from typing import Dict, Generator, List, Tuple
import regex as re
import codecs

DEFAULT_BUFFER_SIZE = 1024*1024*1

WHITESPACE = set(" \t\r\n")

class ShortestTrieNode:
    __slots__ = ("children", "terminal")
    def __init__(self):
        self.children = {}
        self.terminal = False

class ShortestTrie:
    """
    插入 special tokens。搜索时从当前位置向右逐字符走，
    一旦发现 terminal 节点立即返回（保证最短匹配，和你的原实现等价）。
    """
    def __init__(self):
        self.root = ShortestTrieNode()
        self.min_len = None
        self.max_len = 0

    def add(self, s: str):
        if s == "":
            return
        node = self.root
        L = 0
        for ch in s:
            L += 1
            node = node.children.setdefault(ch, ShortestTrieNode())
        node.terminal = True
        if self.min_len is None or len(s) < self.min_len:
            self.min_len = len(s)
        if len(s) > self.max_len:
            self.max_len = len(s)

    def find_from(self, s: str, i: int) -> int:
        """
        在 s[i:] 查找最短 match。若找到，返回匹配长度 L (>0)；否则返回 0。
        """
        node = self.root
        n = len(s)
        j = i
        # 只搜索到 max_len 或到字符串结尾
        max_end = min(n, i + self.max_len)
        while j < max_end:
            ch = s[j]
            node = node.children.get(ch)
            if node is None:
                return 0
            j += 1
            if node.terminal:
                # 立即返回（最短匹配）
                return j - i
        return 0

def train_bpe_streaming_read(input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str], use_pre:bool = False):
    if use_pre:
        with open(input_path, "rb") as f:
            data_dict:dict = pickle.load(f)
        vocab :dict[int,bytes] = data_dict["vocab"]
        words_dict :dict[str,int] = data_dict["words_dict"]
        vocab_total = len(vocab.keys())
    else:
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        vocab_total = 256
        for x in special_tokens:
            vocab[vocab_total] = x.encode()
            vocab_total += 1
        if len(special_tokens) != 0:
            escaped_tokens = [re.escape(token) for token in special_tokens]
            special_pattern = '|'.join(escaped_tokens)
            PAT = rf"""(?:{special_pattern})|'(?:[sdmt]|ll|ve|re)| ?\p{{L}}+| ?\p{{N}}+| ?[^\s\p{{L}}\p{{N}}]+|\s+(?!\S)|\s+"""
        else:
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        token_re = re.compile(PAT)
        words_dict: dict[str, int] = {}
        print("!!!streaming token-count start!!!")

        N_BUCKETS = 256
        BUCKET_MASK = N_BUCKETS - 1
        MAX_BUCKET_SIZE = 500_000

        buckets: list[dict[str, int]] = [
            defaultdict(int) for _ in range(N_BUCKETS)
        ]

        def flush_bucket(bid: int):
            with open(f"bucket_{bid}.pkl", "ab") as f:
                pickle.dump(dict(buckets[bid]), f)
            buckets[bid].clear()
        
        
        decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")
        buffer = ""
        if len(special_tokens) == 0:
            def flush_segment(seg: str):
                if not seg:
                    return
                toks = token_re.findall(seg)
                for t in toks:
                    bid = hash(t) & BUCKET_MASK
                    buckets[bid][t] += 1
                    if len(buckets[bid]) >= MAX_BUCKET_SIZE:
                        flush_bucket(bid)
            with open(input_path, "rb") as f:
                special_start_chars = set(t[0] for t in special_tokens)
                l = 0
                while True:
                    chunk = f.read(DEFAULT_BUFFER_SIZE)
                    if not chunk:
                        buffer += decoder.decode(b"", final=True)
                        flush_segment(buffer)
                        break
                    
                    buffer += decoder.decode(chunk)
                    if buffer and buffer[-1] not in WHITESPACE:
                        # 从后往前找到最后一个空白字符的位置
                        last_ws_pos = -1
                        for i in range(len(buffer) - 1, -1, -1):
                            if buffer[i] in WHITESPACE:
                                last_ws_pos = i
                                break
                        if last_ws_pos != -1:
                            # 处理到最后一个空白字符之前的内容
                            flush_segment(buffer[:last_ws_pos])
                            # 保留剩余部分（包括空白字符）
                            buffer = buffer[last_ws_pos:]
                    print(l)
                    l += 1
            words_dict: dict[str, int] = defaultdict(int)

            for bid in range(N_BUCKETS):
                # 先 merge 内存中的
                for tok, cnt in buckets[bid].items():
                    words_dict[tok] += cnt
                buckets[bid].clear()

                # 再 merge 落盘的
                try:
                    with open(f"bucket_{bid}.pkl", "rb") as f:
                        while True:
                            part = pickle.load(f)
                            for tok, cnt in part.items():
                                words_dict[tok] += cnt
                except FileNotFoundError:
                    pass
                except EOFError:
                    pass
            with open("l.txt","wb") as f:
                pickle.dump({
                    "words_dict":words_dict,
                    "vocab":vocab,
                },f)
        else:
            def flush_segment(seg: str):
                if not seg:
                    return
                toks = token_re.findall(seg)
                for t in toks:
                    if t in special_set:
                        continue
                    bid = hash(t) & BUCKET_MASK
                    buckets[bid][t] += 1
                    if len(buckets[bid]) >= MAX_BUCKET_SIZE:
                        flush_bucket(bid)
            trie = ShortestTrie()
            special_set = set(special_tokens)
            if special_tokens:
                for t in special_tokens:
                    trie.add(t)
            min_sp_local = min(map(len, special_tokens)) if special_tokens else 0
            max_sp_local = max(map(len, special_tokens)) if special_tokens else 0
            with open(input_path, "rb") as f:
                special_start_chars = set(t[0] for t in special_tokens)
                l = 0
                while True:
                    chunk = f.read(DEFAULT_BUFFER_SIZE)
                    if not chunk:
                        buffer += decoder.decode(b"", final=True)
                        flush_segment(buffer)
                        break
                    
                    buffer += decoder.decode(chunk)
                    if buffer and buffer[-1] not in WHITESPACE:
                        i = 0
                        start = 0
                        buf = buffer
                        n = len(buf)
                        special_set_local = special_set
                        while i < n:
                            if buffer[i] not in special_start_chars:
                                i += 1
                                continue
                            L = trie.find_from(buffer, i)
                            if L:
                                flush_segment(buffer[start:i])
                                i += L
                                start = i
                            else:
                                i += 1
                        buffer = buffer[start:]
                    print(l)
                    l += 1
            words_dict: dict[str, int] = defaultdict(int)

            for bid in range(N_BUCKETS):
                # 先 merge 内存中的
                for tok, cnt in buckets[bid].items():
                    words_dict[tok] += cnt
                buckets[bid].clear()

                # 再 merge 落盘的
                try:
                    with open(f"bucket_{bid}.pkl", "rb") as f:
                        while True:
                            part = pickle.load(f)
                            for tok, cnt in part.items():
                                words_dict[tok] += cnt
                except FileNotFoundError:
                    pass
                except EOFError:
                    pass
            with open("l.txt","wb") as f:
                pickle.dump({
                    "words_dict":words_dict,
                    "vocab":vocab,
                },f)

    vocab,merges = train_bpe_core(vocab,vocab_size,words_dict)

    byte_merges: list[tuple[bytes, bytes]] = [(vocab[a], vocab[b]) for (a, b) in merges]
    with open("vocab_final.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("merges_final.pkl", "wb") as f:
        pickle.dump(byte_merges, f)
    
    return vocab, byte_merges

def train_bpe_core(
    vocab :dict[int,bytes],
    vocab_size: int,
    words_dict :dict[str,int],
    word_tokens_input:list[list[int]] | None = None,
    merges_input:list[tuple[int,int]] | None = None
):

    print("!!!begin merging!!!")
    vocab_total = len(vocab)
    
    # 转换单词列表为索引形式
    words = list(words_dict.keys())
    word_freqs = [words_dict[word] for word in words]
    
    # 1. word_tokens: 每个单词的当前token序列
    if word_tokens_input is None:
        word_tokens = [list(word.encode()) for word in words]
    else:
        word_tokens = word_tokens_input
    
    # 2. token_in_words: 映射token id到包含该token的单词索引集合
    token_in_words = defaultdict(set)
    for word_idx, tokens in enumerate(word_tokens):
        for token in tokens:
            token_in_words[token].add(word_idx)
    
    # 3. neighbor_pairs: 每个单词的相邻对统计
    neighbor_pairs = []
    for word_idx, tokens in enumerate(word_tokens):
        pairs = defaultdict(int)
        for j in range(len(tokens) - 1):
            pair = (tokens[j], tokens[j + 1])
            pairs[pair] += 1
        neighbor_pairs.append(pairs)
    
    # 4. global_pair_counts: 全局相邻对频率统计
    global_pair_counts = defaultdict(int)
    for word_idx, pairs in enumerate(neighbor_pairs):
        freq = word_freqs[word_idx]
        for pair, count in pairs.items():
            global_pair_counts[pair] += count * freq
    
    if merges_input is not None:
        merges:list[tuple[int,int]] = merges_input
    else:
        merges:list[tuple[int,int]] = []
    
    # 调试输出控制
    p = 10
    
    while vocab_total < vocab_size:
        if not global_pair_counts:
            break
        
        # 找到最频繁的相邻对
        max_freq = max(global_pair_counts.values())
        max_pairs = [pair for pair, freq in global_pair_counts.items() if freq == max_freq]
        
        # 如果有多个相同频率的，按词汇表中的字节值排序
        a, b = max(max_pairs, key=lambda pair: (vocab[pair[0]], vocab[pair[1]])) # type: ignore
        
        merges.append((a, b))
        new_token_id = vocab_total
        
        # 创建新token
        vocab[new_token_id] = vocab[a] + vocab[b]
        vocab_total += 1
        
        # 初始化新token的相关数据结构
        token_in_words[new_token_id] = set()
        
        # 关键修复：找到所有包含token a或token b的单词
        # 因为这些单词中可能有(a, b)对，或者包含a或b作为其他对的一部分
        affected_words = token_in_words[a].union(token_in_words[b])
        
        # 更新受影响的单词
        for word_idx in affected_words:
            tokens = word_tokens[word_idx]
            freq = word_freqs[word_idx]
            
            # 保存旧的相邻对统计，用于更新全局计数
            old_pairs = neighbor_pairs[word_idx]
            
            # 构建新的token序列
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    # 合并a和b为new_token_id
                    new_tokens.append(new_token_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            
            # 更新单词的token序列
            word_tokens[word_idx] = new_tokens
            
            # 更新token_in_words
            # 从相关集合中移除旧的token
            if a in tokens and a not in new_tokens:
                token_in_words[a].discard(word_idx)
            if b in tokens and b not in new_tokens:
                token_in_words[b].discard(word_idx)
            
            # 添加新token到相关集合
            if new_token_id in new_tokens:
                token_in_words[new_token_id].add(word_idx)
            
            # 重新计算这个单词的相邻对统计
            new_pairs = defaultdict(int)
            for j in range(len(new_tokens) - 1):
                pair = (new_tokens[j], new_tokens[j + 1])
                new_pairs[pair] += 1
            neighbor_pairs[word_idx] = new_pairs
            
            # 更新全局计数：减去旧的，加上新的
            for pair, count in old_pairs.items():
                global_pair_counts[pair] -= count * freq
                if global_pair_counts[pair] <= 0:
                    del global_pair_counts[pair]
            
            for pair, count in new_pairs.items():
                global_pair_counts[pair] += count * freq
        
        p -= 1
        if p == 0:
            print(f"已合并 {len(merges)} 次，当前词汇表大小: {vocab_total}")
            p = 10
    return vocab,merges

def train_bpe_restart(
        vocab_filepath:str, 
        merges_filepath:str,
        words_dict_path:str,
        special_tokens:list[str],
        vocab_size:int):
    with open(vocab_filepath,"rb") as f:
        vocab = pickle.load(f)
    with open(merges_filepath,"rb") as f:
        merges = pickle.load(f)
    with open(words_dict_path,"rb") as f:
        data_dict:dict = pickle.load(f)
    words_dict :dict[str,int] = data_dict["words_dict"]

    from cs336_basics.tokenizer import Tokenizer
    my_tokenizer:Tokenizer = Tokenizer(vocab,merges,special_tokens)
    word_tokens_input:list[list[int]] = [my_tokenizer.encode(word) for word in list(words_dict.keys())]
    vocab,merges = train_bpe_core(vocab,vocab_size,words_dict,word_tokens_input,merges)
    with open("merges.pkl","wb") as f:
        pickle.dump(merges,f)
    with open("vocab_final.pkl", "wb") as f:
        pickle.dump(vocab, f)
    byte_merges: list[tuple[bytes, bytes]] = [(vocab[a], vocab[b]) for (a, b) in merges]
    with open("merges_final.pkl", "wb") as f:
        pickle.dump(byte_merges, f)

    pass

if __name__ == "__main__":
    vocab, merges = train_bpe_streaming_read(
        "../assignment1-data/TinyStoriesV2-GPT4-train.txt",
        10000,
        ["<|endoftext|>"],
        use_pre=False
    )
    pass
