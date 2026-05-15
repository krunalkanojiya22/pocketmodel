import os
import glob
import numpy as np


# ---------------------------------------------------------------------------
# Text loading
# ---------------------------------------------------------------------------

def read_all_text(path: str) -> str:
    """Return all text from a file or directory of .txt/.md files."""
    paths = []
    if os.path.isfile(path):
        paths = [path]
    elif os.path.isdir(path):
        paths  = glob.glob(os.path.join(path, '**/*.txt'), recursive=True)
        paths += glob.glob(os.path.join(path, '**/*.md'),  recursive=True)
        if not paths:
            raise ValueError(f'No .txt or .md files found in {path}')
    else:
        raise ValueError(f'Path does not exist: {path}')

    text = ''
    for p in sorted(paths):
        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
            text += f.read()
        print(f'  Loaded: {p}')
    return text


# ---------------------------------------------------------------------------
# Splitting and chunking
# ---------------------------------------------------------------------------

def train_val_split(tokens: list, val_fraction: float = 0.1):
    """
    Fixed-boundary split — validation is the tail of the token sequence.
    Reflects generalisation to unseen *continuations*, not random positions.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f'val_fraction must be in (0, 1), got {val_fraction}')
    split = int(len(tokens) * (1.0 - val_fraction))
    return tokens[:split], tokens[split:]


def make_chunks(tokens, chunk_size: int, stride: int | None = None) -> np.ndarray:
    """
    Slice a flat token list into overlapping windows of length chunk_size+1.

    Each chunk[i] is the input; chunk[i+1:] is the prediction target (shifted by 1).
    stride < chunk_size creates overlapping windows for more training signal.
    """
    if stride is None:
        stride = chunk_size
    stride = max(1, stride)
    chunks = [tokens[i: i + chunk_size + 1]
              for i in range(0, len(tokens) - chunk_size, stride)]
    return np.array(chunks, dtype=np.int32)


# ---------------------------------------------------------------------------
# In-memory sampler (small / medium datasets)
# ---------------------------------------------------------------------------

class Sampler:
    """Randomly samples batches from chunks without replacement per epoch."""

    def __init__(self, chunks: np.ndarray):
        self.chunks  = chunks
        self.indices = np.arange(len(chunks))
        self.pos     = len(self.indices)    # triggers shuffle on first call

    def sample(self, batch_size: int) -> np.ndarray:
        if self.pos + batch_size > len(self.indices):
            np.random.shuffle(self.indices)
            self.pos = 0
        idx       = self.indices[self.pos: self.pos + batch_size]
        self.pos += batch_size
        return self.chunks[idx]


# ---------------------------------------------------------------------------
# Memory-mapped sampler (large / pre-tokenised datasets)
# ---------------------------------------------------------------------------

class MemoryMappedSampler:
    """
    Random-access sampler for pre-tokenised .npy files too large to fit in RAM.

    The numpy array is memory-mapped — the OS pages in only the regions that
    are actually accessed, keeping resident memory proportional to batch size
    rather than dataset size.

    Usage:
        sampler = MemoryMappedSampler('data/pile_train.npy', chunk_size=2048)
        batch   = sampler.sample(batch_size=8)   # np.ndarray (8, 2049)
    """

    def __init__(self, path: str, chunk_size: int, stride: int | None = None):
        self.data       = np.load(path, mmap_mode='r')   # memory-mapped read-only
        self.chunk_size = chunk_size
        self.stride     = stride or chunk_size
        self.n_chunks   = max(0, (len(self.data) - chunk_size) // self.stride)
        if self.n_chunks == 0:
            raise ValueError(
                f'Dataset at {path} has {len(self.data)} tokens — '
                f'not enough for even one chunk of size {chunk_size}.'
            )

    def __len__(self) -> int:
        return self.n_chunks

    def sample(self, batch_size: int) -> np.ndarray:
        indices = np.random.randint(0, self.n_chunks, size=batch_size)
        return np.stack([
            self.data[i * self.stride: i * self.stride + self.chunk_size + 1]
            for i in indices
        ]).astype(np.int32)


# ---------------------------------------------------------------------------
# Pre-tokenisation utility
# ---------------------------------------------------------------------------

def pretokenize(text_path: str, tokenizer, output_path: str) -> int:
    """
    Tokenise all text files at `text_path` and save as a flat uint32 .npy file.

    Storing tokens as uint32 instead of int32 doubles the maximum vocabulary
    size (4 B tokens vs 2 B) while using the same memory per token.

    Example:
        from dataset import pretokenize
        from tokenizer import TiktokenTokenizer
        tok = TiktokenTokenizer('gpt2')
        pretokenize('data/books/', tok, 'data/books_gpt2.npy')
    """
    text  = read_all_text(text_path)
    ids   = tokenizer.encode(text)
    arr   = np.array(ids, dtype=np.uint32)
    np.save(output_path, arr)
    print(f'  Saved {len(arr):,} tokens ({arr.nbytes / 1e9:.3f} GB) → {output_path}')
    return len(arr)
