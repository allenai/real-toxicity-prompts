from pathlib import Path
from typing import Union, List, Optional
import numpy as np
import torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop


def set_seed(seed, n_gpu):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 < max_sequence_length:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


class GPT2Generator:
    STOP_TOKEN = "<|endoftext|>"

    def __init__(self, model_name_or_path: str = 'gpt2', seed: int = 42):
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        set_seed(seed, n_gpu)

        # Initialize the model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(self.device)

    def generate(self,
                 prompt: Union[str, List[str]] = STOP_TOKEN,
                 num_return_sequences: int = 1,
                 max_length: int = 20,
                 temperature: float = 1.0,
                 k: Optional[int] = None,
                 p: float = 0.9,
                 repetition_penalty: float = 1.0,
                 stop_token: str = STOP_TOKEN):
        max_length = adjust_length_to_model(max_length, max_sequence_length=self.model.config.max_position_embeddings)

        # Different models need different input formatting and/or extra arguments
        encoded_prompt = self.tokenizer.encode(prompt,
                                               add_special_tokens=False,
                                               return_tensors="pt")
        encoded_prompt = encoded_prompt.to(self.device)

        print(encoded_prompt)

        # Generate
        output_sequences = self.model.generate(
            input_ids=encoded_prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=k,
            top_p=p,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            do_sample=True
        )

        print(output_sequences.shape)

        for seq in output_sequences.view(-1, output_sequences.size(-1)):
            # Handle case where prompt is stop token
            seq = seq[1:] if prompt == stop_token else seq
            # Decode sequence of ids
            output = self.tokenizer.decode(seq, clean_up_tokenization_spaces=True)
            # Remove everything after end of text token
            output = output[: output.find(stop_token) if stop_token else None]
            yield output
