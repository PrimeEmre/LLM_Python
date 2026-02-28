import torch
import torch.nn as nn
import torch.nn.functional as F

# reading the file
with open ("input.txt", "r",encoding="utf-8") as f:
    text = f.read()
    print(f"Total characters in dataset: {len(text)}")

    # getting the characters
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Unique characters in dataset: {vocab_size}")

    # converting the