import json
import os
from functools import cache, partial
from typing import Optional

import regex as re

Ids = list[int]
Tokens = list[str]
Pair = tuple[str, str]
Ranks = dict[Pair, int]


@cache
def bytes_to_unicode() -> dict[int, str]:
    d, n = {}, 0
    for i in range(256):
        if i < 33 or 126 < i < 161 or i == 173:
            d[i] = chr(256 + n)
            n += 1
        else:
            d[i] = chr(i)
    return d


bytes_enc = bytes_to_unicode()
bytes_dec = {v: k for k, v in bytes_enc.items()}


def utf_encode_text(text: str) -> str:
    return "".join(bytes_enc[b] for b in text.encode("utf-8"))


def utf_decode_text(text: str) -> str:
    return bytearray([bytes_dec[c] for c in text]).decode("utf-8")


def index_of_pair(tokens: Tokens, pair: Pair) -> Optional[int]:
    for i, p in enumerate(zip(tokens[:-1], tokens[1:])):
        if pair == p:
            return i
    return None


def get_min_pair(tokens: Tokens, merge_ranks: Ranks) -> Optional[Pair]:
    min_pair, min_rank = None, float("inf")
    for pair in zip(tokens[:-1], tokens[1:]):
        if pair in merge_ranks and merge_ranks[pair] < min_rank:
            min_pair, min_rank = pair, merge_ranks[pair]
    return min_pair


def merge_tokens(tokens: Tokens, pair: Pair) -> Tokens:
    def merge(t):
        i = index_of_pair(t, pair)
        if i is not None:
            return t[:i] + ["".join(pair)] + merge(t[i + 2 :])
        else:
            return t

    return merge(tokens)


def bpe(text: str, merge_ranks: Ranks) -> Tokens:
    assert text != ""

    tokens = list(text)
    while True:
        pair = get_min_pair(tokens, merge_ranks)
        if not pair:
            break
        tokens = merge_tokens(tokens, pair)

    return tokens


class BPETokenizer:
    def __init__(self, vocab: dict[str, int], merge_ranks: Ranks):
        self.tkn_to_id = vocab
        self.id_to_tok = {v: k for k, v in vocab.items()}
        self.merge_ranks = merge_ranks
        self.pat = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        self.bpe = cache(partial(bpe, merge_ranks=self.merge_ranks))

    def encode(self, text: str) -> Ids:
        return [
            self.tkn_to_id[bpe_token]
            for subtext in re.findall(self.pat, text)
            for bpe_token in self.bpe(utf_encode_text(subtext))
        ]

    def decode(self, ids: Ids) -> str:
        return utf_decode_text("".join(self.id_to_tok[i] for i in ids))

    @classmethod
    def load(cls, dirpath: str):
        with open(os.path.join(dirpath, "encoder.json")) as f:
            vocab = json.load(f)

        with open(os.path.join(dirpath, "vocab.bpe")) as f:
            merge_ranks = {
                tuple(s.strip().split(" ")): i
                for i, s in enumerate(f.readlines()[1:-1])
            }

        return cls(vocab, merge_ranks)
