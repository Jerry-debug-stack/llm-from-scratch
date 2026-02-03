from typing import Iterable, Iterator
import regex as re
import heapq
import pickle

def split_by_matches(s: str, a: list[str]) -> list[str]:
    if not s:
        return [""] if s == "" else []
    if not a:
        return [s]
    patterns = [p for p in a if p]
    if not patterns:
        return [s]
    patterns.sort(key=len, reverse=True)
    n = len(s)
    result = []
    i = 0
    while i < n:
        matched = False
        for pattern in patterns:
            pattern_len = len(pattern)
            if i + pattern_len <= n and s[i:i + pattern_len] == pattern:
                result.append(pattern)
                i += pattern_len
                matched = True
                break
        if not matched:
            start = i
            while i < n:
                local_matched = False
                for pattern in patterns:
                    pattern_len = len(pattern)
                    if i + pattern_len <= n and s[i:i + pattern_len] == pattern:
                        local_matched = True
                        break
                if local_matched:
                    break
                i += 1
            if start < i:
                result.append(s[start:i])
    return result

class Tokenizer:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []
    special_tokens: list[str] = []
    reverse_vocab: dict[bytes,int] = {}
    special_tokens_longest:int = 0
    def __init__(self,
                 vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(self.merges)
        }
        self.special_tokens = special_tokens if (special_tokens) else []
        for key,value in vocab.items():
            self.reverse_vocab[value] = key
        if special_tokens:
            self.special_tokens = special_tokens
            self.special_tokens_longest = max((len(i) for i in self.special_tokens))
        else:
            self.special_tokens = []
        
        self.BYTE_TABLE = [bytes([i]) for i in range(256)]

    def _bpe_merge(self, word_bytes: bytes) -> list[bytes]:
        if not word_bytes:
            return []
        n = len(word_bytes)
        tokens = [self.BYTE_TABLE[b] for b in word_bytes]
        if n == 1:
            return tokens
        ranks = self.merge_ranks
        prev = list(range(-1, n-1))
        next = list(range(1, n+1))
        next[-1] = -1
        alive = [True] * n
        heap = []
        for i in range(n - 1):
            pair = (tokens[i], tokens[i + 1])
            r = ranks.get(pair)
            if r is not None:
                heapq.heappush(heap, (r, i))
        while heap:
            r, i = heapq.heappop(heap)
            if not alive[i]:
                continue
            j = next[i]
            if j == -1 or not alive[j]:
                continue
            pair = (tokens[i], tokens[j])
            if ranks.get(pair) != r:
                continue
            merged = tokens[i] + tokens[j]
            tokens[i] = merged
            alive[j] = False
            nxt = next[j]
            next[i] = nxt
            if nxt != -1:
                prev[nxt] = i
            pi = prev[i]
            if pi != -1 and alive[pi]:
                ppair = (tokens[pi], tokens[i])
                pr = ranks.get(ppair)
                if pr is not None:
                    heapq.heappush(heap, (pr, pi))
            if next[i] != -1 and alive[next[i]]:
                npair = (tokens[i], tokens[next[i]])
                nr = ranks.get(npair)
                if nr is not None:
                    heapq.heappush(heap, (nr, i))
        res = []
        cur = 0
        while cur != -1:
            if alive[cur]:
                res.append(tokens[cur])
            cur = next[cur]
        return res
    def encode(self, text: str) -> list[int]:
        texts:list[str] = split_by_matches(text,self.special_tokens)
        ans :list[int]=[]
        length:int = len(self.merges)
        for i in texts:
            if i in self.special_tokens:
                ans.append(self.reverse_vocab[bytes(i,"utf-8")])
            else:
                words:list[str]= re.findall(self.PAT,i)
                for word in words:
                    bytes_list:list[bytes] = self._bpe_merge(word.encode())
                    for m in bytes_list:
                        ans.append(self.reverse_vocab[m])
        return ans
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        for text in iterable:
            buffer += text
            r_max:int = len(buffer) - self.special_tokens_longest
            if (r_max < 0):
                continue
            words:list[str] = split_by_matches(buffer,self.special_tokens)
            l:int = 0
            r:int = 0
            end_r:int = 0
            reach_r_boundary = False
            for i in words:
                r = l + len(i)
                if i in self.special_tokens:
                    if r <= r_max:
                        yield self.encode(i)[0]
                        end_r = r
                        l = r
                        continue
                    else:
                        end_r = l
                        reach_r_boundary = True
                        break
                else:
                    # 对于一般的部分
                    for j in re.finditer(self.PAT,i):
                        inner_r = l + j.end()
                        inner_l = l + j.start()
                        if inner_r > r_max:
                            reach_r_boundary = True
                            end_r = inner_l
                            break
                        else:
                            inner_str:str = buffer[inner_l:inner_r:]
                            for k in self.encode(inner_str):
                                yield k
                            end_r = inner_r
                    if reach_r_boundary:
                        break
                l = r
                end_r = r
            buffer = buffer[end_r::]
        if buffer:
            for i in self.encode(buffer):
                yield i
    def decode(self, ids: list[int]) -> str:
        ans_utf8 = bytes([]).join((self.vocab[i] for i in ids))
        return ans_utf8.decode(errors='replace')

def from_files(cls, vocab_filepath:str, merges_filepath:str, special_tokens=None):
    with open(vocab_filepath,"rb") as f:
        vocab = pickle.load(f)
    with open(merges_filepath,"rb") as f:
        merges = pickle.load(f)
    return Tokenizer(vocab,merges,special_tokens)
