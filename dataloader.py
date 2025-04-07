import torch
import os
import glob
import math
from transformers import Qwen2ForCausalLM, AutoTokenizer
from matplotlib import pyplot as plt


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
        self.gpt2_tokeniser = AutoTokenizer.from_pretrained("gpt2")

        self.model_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B")

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
        pos = self.current_position #+ batch_size
        device_batch_tokens = self.tokens[pos:pos + batch_size + 1]
        # advance current position
        self.current_position += batch_size
        inputs = device_batch_tokens[:-1].to(device='cuda', dtype=torch.int32, non_blocking=True)
        targets = device_batch_tokens[1:].to(device='cuda', dtype=torch.int64, non_blocking=True)

        (new_tokens, text) = self.retokenise(inputs)
        return (new_tokens, text) #inputs, targets

    def split_tensor_on_token(self, tokens: torch.Tensor, split_token: int = 50256):
        # Ensure the tokens tensor is 1D
        if tokens.dim() != 1:
            raise ValueError("This function only works on 1D token tensors.")

        # Find indices where the split_token occurs
        split_indices = (tokens == split_token).nonzero(as_tuple=True)[0]

        segments = []
        prev_index = 0
        for idx in split_indices:
            # Slice from the previous index up to the current split token index
            segment = tokens[prev_index:idx]
            if segment.numel() > 0:
                segments.append(segment)
            # Update previous index to start after the split token
            prev_index = idx.item() + 1

        # Append any remaining tokens after the last split token
        if prev_index < tokens.size(0):
            segments.append(tokens[prev_index:])

        return segments

    def retokenise(self, old_tokens):

        old_tokens = self.split_tensor_on_token(old_tokens)[-1:]
        new_tokens = []
        texts = []
        for old_tok in old_tokens:
            text = self.gpt2_tokeniser.decode(old_tok, skip_special_tokens=True)
            new_tok = self.model_tokenizer.encode(text, add_special_tokens=True)
            new_tok = torch.tensor(new_tok, device="cuda")
            new_tokens.append(new_tok), texts.append(text)
        return new_tokens, texts


def plot_all_attentions(attentions):
    """
    Plots attention maps from all layers in a grid.

    Args:
        attentions: A sequence of attention tensors.
            Each tensor should have shape (batch_size, num_heads, seq_len, seq_len).
            Typically obtained from the output of a forward pass with output_attentions=True.
    """
    if attentions.ndim == 2:
        num_plots = 1
        attentions = attentions.unsqueeze(0)
    else:
        num_plots = len(attentions)
    # Compute grid dimensions (rows and columns)
    num_cols = math.ceil(math.sqrt(num_plots))
    num_rows = math.ceil(num_plots / num_cols)

    # Create a grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 6, num_rows * 6))
    # Flatten axes array for easier indexing (handles the single subplot case too)
    if num_plots > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, attn in enumerate(attentions):
        # attn: shape (batch_size, num_heads, seq_len, seq_len)
        # For this example, take the first example in the batch and average over heads
        attn = attn.detach().cpu().numpy()  # shape: (seq_len, seq_len)

        im = axes[i].imshow(attn, cmap="viridis")
        axes[i].set_title(f"Layer {i + 1}")
        axes[i].set_xlabel("Key Position")
        axes[i].set_ylabel("Query Position")
        fig.colorbar(im, ax=axes[i])

    # If there are any empty subplots, remove them
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()



def main():

    loader = DistributedDataLoader('data/fineweb10B/fineweb_val_*.bin')
    inputs, targets = loader.next_batch(175)

    for i, tok in enumerate(inputs[0]):
        print(i, loader.model_tokenizer.decode(tok))

    # # Replace 'llama-3.1-8b' with the actual model identifier
    model_id =  "Qwen/Qwen2-1.5B"
    model = Qwen2ForCausalLM.from_pretrained(model_id).cuda()

    print(targets[0])

    with torch.no_grad():
        outputs = model(inputs[0].unsqueeze(0), output_attentions=True)

    # print(outputs)
    #[print(s.shape) for s in outputs.attentions]
    atn0 = outputs.attentions
    plot_all_attentions(atn0[28][0, :])

if __name__ == '__main__':
    main()
