#!/usr/bin/env python3
"""
Pre-tokenise a text dataset and save as a flat .npy binary.

Running this once converts raw text to tokens so training can start
instantly without re-tokenising on every run — essential for datasets
larger than a few GB where tokenisation takes minutes.

Usage:
    # Character tokenizer (builds vocab from the data)
    uv run python src/pretokenize.py \\
        --dataset data/shakespeare.txt \\
        --output  data/shakespeare_char.npy \\
        --tokenizer char

    # tiktoken GPT-2 vocab
    uv run python src/pretokenize.py \\
        --dataset data/books/ \\
        --output  data/books_gpt2.npy \\
        --tokenizer tiktoken

Then train from the binary:
    uv run python src/train.py \\
        --dataset_bin data/books_gpt2.npy \\
        --n_vocab 50257 \\
        --model llama3-8b
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import dataset as ds
import tokenizer as tok_module


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument('--dataset',   required=True,
                   help='Text file or directory of .txt/.md files')
    p.add_argument('--output',    required=True,
                   help='Output path for the .npy token file')
    p.add_argument('--tokenizer', default='tiktoken',
                   choices=['char', 'tiktoken'],
                   help='Tokenizer to use (default: tiktoken)')
    p.add_argument('--tiktoken_encoding', default='gpt2',
                   choices=['gpt2', 'cl100k_base', 'o200k_base'],
                   help='tiktoken encoding (default: gpt2 = 50,257 tokens)')
    p.add_argument('--save_tokenizer', default=None,
                   help='Also save the tokenizer to this directory '
                        '(needed for char tokenizer; tiktoken is self-describing)')
    return p.parse_args()


def main():
    args = parse_args()

    print(f'Loading text from: {args.dataset}')
    text = ds.read_all_text(args.dataset)
    print(f'  {len(text):,} characters')

    print(f'\nBuilding tokenizer: {args.tokenizer}')
    if args.tokenizer == 'tiktoken':
        tokenizer = tok_module.TiktokenTokenizer(args.tiktoken_encoding)
        print(f'  Encoding: {args.tiktoken_encoding}  |  vocab: {tokenizer.n_vocab:,}')
    else:
        tokenizer = tok_module.CharTokenizer().build_from_text(text)
        print(f'  Vocab: {tokenizer.n_vocab} unique characters')

    if args.save_tokenizer:
        tokenizer.save(args.save_tokenizer)
        print(f'  Tokenizer saved → {args.save_tokenizer}')

    print('\nTokenising...')
    ids = tokenizer.encode(text)
    print(f'  {len(ids):,} tokens')
    print(f'  Compression ratio: {len(text) / max(1, len(ids)):.2f} chars/token')

    arr = np.array(ids, dtype=np.uint32)

    output = args.output
    if not output.endswith('.npy'):
        output += '.npy'
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    np.save(output, arr)

    size_mb = arr.nbytes / 1e6
    print(f'\nSaved → {output}')
    print(f'  {len(arr):,} tokens  |  {size_mb:.1f} MB  |  dtype=uint32')
    print(f'\nTo train from this file:')
    print(f'  uv run python src/train.py \\')
    print(f'      --dataset_bin {output} \\')
    print(f'      --n_vocab {tokenizer.n_vocab} \\')
    print(f'      --model small-cpu')


if __name__ == '__main__':
    main()
