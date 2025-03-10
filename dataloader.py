import torch
import os
import glob
from transformers import AutoTokenizer


def _load_data_shard(path):
    # only reads the header, returns header data
    # header is 256 int32
    header = torch.from_file(path, False, 256, dtype=torch.int32)
    assert header[0] == 20240520, 'magic number mismatch in the data .bin file'
    assert header[1] == 1, 'unsupported version'
    num_tokens = int(header[2])  # number of tokens (claimed)
    with open(path, 'rb', buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)  # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())  # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, 'number of tokens read does not match header'
    return tokens


class DistributedDataLoader:

    def __init__(self, filename_pattern):
        self.files = sorted(glob.glob(filename_pattern))
        self.reset()

    def reset(self):
        self.current_shard = -1
        self.advance()

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = 0
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self, batch_size):
        # load next shard if necessary
        if self.current_position + batch_size + 1 >= len(self.tokens):
            print(f'Advancing')
            self.advance()
        pos = self.current_position + batch_size
        device_batch_tokens = self.tokens[self.current_position:pos + batch_size + 1]
        # advance current position
        self.current_position += batch_size
        inputs = device_batch_tokens[:-1].to(device='cuda', dtype=torch.int32, non_blocking=True)
        targets = device_batch_tokens[1:].to(device='cuda', dtype=torch.int64, non_blocking=True)
        return inputs, targets


def main():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    loader = DistributedDataLoader('data/fineweb10B/fineweb_val_*.bin')
    for i in range(1):
        inputs, targets = loader.next_batch(128)
        decoded = tokenizer.decode(inputs)
        print(decoded)

if __name__ == '__main__':
    main()
