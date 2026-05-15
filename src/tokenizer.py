"""
Tokenizers for pocketmodel.

Two backends are available:
  - CharTokenizer  : character-level, built from training data, no extra deps.
  - BPETokenizer   : subword BPE via sentencepiece, learns word-piece merges.

Use `load_tokenizer(path)` to transparently reload whichever type was saved.
"""

import os
import json
import tempfile


# ---------------------------------------------------------------------------
# Character-level tokenizer (original)
# ---------------------------------------------------------------------------

class CharTokenizer:
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.n_vocab = 0

    def build_from_text(self, text):
        chars = sorted(set(text))
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for ch, i in self.char_to_id.items()}
        self.n_vocab = len(chars)
        print(f"  Vocab size: {self.n_vocab} unique characters")
        return self

    def encode(self, text):
        unknown = [ch for ch in text if ch not in self.char_to_id]
        if unknown:
            unique_unknown = sorted(set(unknown))
            print(f"  [Warning] {len(unknown)} unknown character(s) skipped "
                  f"(not in training vocab): {unique_unknown[:10]}")
        return [self.char_to_id[ch] for ch in text if ch in self.char_to_id]

    def decode(self, ids):
        return ''.join(self.id_to_char.get(i, '') for i in ids)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'tokenizer.json'), 'w') as f:
            json.dump({'type': 'char', 'char_to_id': self.char_to_id, 'n_vocab': self.n_vocab}, f)

    @classmethod
    def load(cls, path):
        with open(os.path.join(path, 'tokenizer.json')) as f:
            data = json.load(f)
        tok = cls()
        tok.char_to_id = data['char_to_id']
        tok.id_to_char = {int(i): ch for ch, i in tok.char_to_id.items()}
        tok.n_vocab = data['n_vocab']
        return tok


# ---------------------------------------------------------------------------
# BPE tokenizer (sentencepiece)
# ---------------------------------------------------------------------------

class BPETokenizer:
    """
    Subword BPE tokenizer built from training data using sentencepiece.

    Compared with CharTokenizer:
      - Tokens correspond to common subwords/words, not individual characters.
      - The model learns faster because each token carries more semantic signal.
      - vocab_size is configurable (default 1000); must be > number of unique chars.
    """

    def __init__(self):
        self._sp = None
        self._model_data = None
        self.n_vocab = 0

    def build_from_text(self, text, vocab_size=1000):
        import sentencepiece as spm

        # sentencepiece trains from a file path
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                         delete=False, encoding='utf-8')
        tmp.write(text)
        tmp.close()

        prefix = tmp.name + '_sp'
        spm.SentencePieceTrainer.train(
            input=tmp.name,
            model_prefix=prefix,
            vocab_size=vocab_size,
            model_type='bpe',
            character_coverage=1.0,   # cover every char in the training text
            hard_vocab_limit=False,   # allow smaller vocab when text is tiny
            pad_id=0,
            unk_id=1,
            bos_id=-1,                # no BOS/EOS — we handle context windows ourselves
            eos_id=-1,
        )

        with open(prefix + '.model', 'rb') as f:
            self._model_data = f.read()

        os.unlink(tmp.name)
        os.unlink(prefix + '.model')
        os.unlink(prefix + '.vocab')

        self._sp = spm.SentencePieceProcessor()
        self._sp.load_from_serialized_proto(self._model_data)
        self.n_vocab = self._sp.get_piece_size()
        print(f"  BPE vocab size: {self.n_vocab} subword pieces")
        return self

    def encode(self, text):
        if self._sp is None:
            raise RuntimeError("BPETokenizer not loaded — call build_from_text or load first.")
        return self._sp.encode(text, out_type=int, add_bos=False, add_eos=False)

    def decode(self, ids):
        if self._sp is None:
            raise RuntimeError("BPETokenizer not loaded — call build_from_text or load first.")
        return self._sp.decode(ids)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'tokenizer.model'), 'wb') as f:
            f.write(self._model_data)
        with open(os.path.join(path, 'tokenizer.json'), 'w') as f:
            json.dump({'type': 'bpe', 'n_vocab': self.n_vocab}, f)

    @classmethod
    def load(cls, path):
        import sentencepiece as spm
        tok = cls()
        with open(os.path.join(path, 'tokenizer.model'), 'rb') as f:
            tok._model_data = f.read()
        tok._sp = spm.SentencePieceProcessor()
        tok._sp.load_from_serialized_proto(tok._model_data)
        tok.n_vocab = tok._sp.get_piece_size()
        return tok


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_tokenizer(path):
    """Load whichever tokenizer was saved in `path` (char or bpe)."""
    meta_path = os.path.join(path, 'tokenizer.json')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"tokenizer.json not found in '{path}'.")
    with open(meta_path) as f:
        meta = json.load(f)
    kind = meta.get('type', 'char')
    if kind == 'bpe':
        return BPETokenizer.load(path)
    return CharTokenizer.load(path)
