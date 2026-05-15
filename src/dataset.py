import os
import glob
import numpy as np


def read_all_text(path):
    """Return all text from a file or directory of .txt/.md files."""
    paths = []
    if os.path.isfile(path):
        paths = [path]
    elif os.path.isdir(path):
        paths = glob.glob(os.path.join(path, '**/*.txt'), recursive=True)
        paths += glob.glob(os.path.join(path, '**/*.md'), recursive=True)
        if not paths:
            raise ValueError(f"No .txt or .md files found in {path}")
    else:
        raise ValueError(f"Path does not exist: {path}")

    text = ''
    for p in sorted(paths):
        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
            text += f.read()
        print(f"  Loaded: {p}")
    return text


def train_val_split(tokens, val_fraction=0.1):
    """Split a flat token list into train and validation portions.

    The split is done at a fixed boundary (not shuffled) so that validation
    loss reflects generalisation to unseen *continuations* of the text.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    split = int(len(tokens) * (1.0 - val_fraction))
    return tokens[:split], tokens[split:]


def make_chunks(tokens, chunk_size, stride=None):
    """Split a flat token list into overlapping chunks of chunk_size+1.

    Each chunk has length chunk_size+1: the first chunk_size tokens are
    model inputs and the last chunk_size tokens are prediction targets
    (shifted by one).

    Args:
        stride: Step between chunk start positions.  Defaults to chunk_size
                (non-overlapping).  Set stride < chunk_size for a sliding
                window that exposes more sequence boundaries to the model.
    """
    if stride is None:
        stride = chunk_size
    stride = max(1, stride)
    chunks = []
    for i in range(0, len(tokens) - chunk_size, stride):
        chunks.append(tokens[i : i + chunk_size + 1])
    return np.array(chunks, dtype=np.int32)


class Sampler:
    """Randomly samples batches of chunks without replacement per epoch."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.indices = np.arange(len(chunks))
        self.pos = len(self.indices)  # trigger shuffle on first call

    def sample(self, batch_size):
        if self.pos + batch_size > len(self.indices):
            np.random.shuffle(self.indices)
            self.pos = 0
        idx = self.indices[self.pos : self.pos + batch_size]
        self.pos += batch_size
        return self.chunks[idx]
