"""
Tokenizers for pocketmodel.

  - CharTokenizer      : character-level, vocabulary built from training data.
  - TiktokenTokenizer  : byte-level BPE via tiktoken — fixed GPT-2/GPT-4 vocab,
                         no training step required, ~10× faster encoding.

Use `load_tokenizer(path)` to reload whichever type was saved.
"""

import os
import json


# ---------------------------------------------------------------------------
# Character-level tokenizer
# ---------------------------------------------------------------------------

class CharTokenizer:
    def __init__(self):
        self.char_to_id: dict[str, int] = {}
        self.id_to_char: dict[int, str] = {}
        self.n_vocab = 0

    def build_from_text(self, text: str) -> 'CharTokenizer':
        chars = sorted(set(text))
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for ch, i in self.char_to_id.items()}
        self.n_vocab = len(chars)
        print(f"  Vocab size: {self.n_vocab} unique characters")
        return self

    def encode(self, text: str) -> list[int]:
        unknown = [ch for ch in text if ch not in self.char_to_id]
        if unknown:
            print(f"  [Warning] {len(unknown)} unknown character(s) skipped: "
                  f"{sorted(set(unknown))[:10]}")
        return [self.char_to_id[ch] for ch in text if ch in self.char_to_id]

    def decode(self, ids: list[int]) -> str:
        return ''.join(self.id_to_char.get(i, '') for i in ids)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'tokenizer.json'), 'w') as f:
            json.dump({'type': 'char', 'char_to_id': self.char_to_id,
                       'n_vocab': self.n_vocab}, f)

    @classmethod
    def load(cls, path: str) -> 'CharTokenizer':
        with open(os.path.join(path, 'tokenizer.json')) as f:
            data = json.load(f)
        tok = cls()
        tok.char_to_id = data['char_to_id']
        tok.id_to_char = {int(i): ch for ch, i in tok.char_to_id.items()}
        tok.n_vocab = data['n_vocab']
        return tok


# ---------------------------------------------------------------------------
# Tiktoken tokenizer (byte-level BPE, GPT-2 / GPT-4 vocab)
# ---------------------------------------------------------------------------

class TiktokenTokenizer:
    """
    Byte-level BPE using tiktoken — same vocabulary as GPT-2/3/4.

    Compared with CharTokenizer:
      - Fixed, production-quality vocabulary (no training step needed).
      - gpt2        : 50,257 tokens — matches GPT-2/3 weights exactly.
      - cl100k_base : 100,277 tokens — GPT-4, better for code & multilingual text.
      - o200k_base  : 200,019 tokens — GPT-4o, largest coverage.

    tiktoken caches the vocab file locally after the first download.
    """

    ENCODINGS = ('gpt2', 'cl100k_base', 'o200k_base')

    def __init__(self, encoding: str = 'gpt2'):
        import tiktoken
        if encoding not in self.ENCODINGS:
            raise ValueError(
                f"Unknown encoding '{encoding}'. Choose from: {self.ENCODINGS}"
            )
        self._enc = tiktoken.get_encoding(encoding)
        self._encoding_name = encoding
        self.n_vocab = self._enc.n_vocab

    def encode(self, text: str) -> list[int]:
        # encode_ordinary skips special tokens (BOS/EOS) — correct for LM training.
        return self._enc.encode_ordinary(text)

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'tokenizer.json'), 'w') as f:
            json.dump({'type': 'tiktoken', 'encoding': self._encoding_name,
                       'n_vocab': self.n_vocab}, f)

    @classmethod
    def load(cls, path: str) -> 'TiktokenTokenizer':
        with open(os.path.join(path, 'tokenizer.json')) as f:
            meta = json.load(f)
        return cls(encoding=meta.get('encoding', 'gpt2'))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_tokenizer(path: str):
    """Reload whichever tokenizer was saved in `path` (char or tiktoken)."""
    meta_path = os.path.join(path, 'tokenizer.json')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"tokenizer.json not found in '{path}'.")
    with open(meta_path) as f:
        kind = json.load(f).get('type', 'char')
    if kind == 'tiktoken':
        return TiktokenTokenizer.load(path)
    return CharTokenizer.load(path)
