from pathlib import Path
from typing import Union, Optional

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

    def __init__(self, model_name_or_path: Union[str, Path] = 'gpt2', seed: int = 42):
        if isinstance(model_name_or_path, Path):
            model_name_or_path = str(model_name_or_path)

        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        set_seed(seed, n_gpu)

        # Initialize the model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path, pad_token=self.STOP_TOKEN)
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(self.device)
        self.pad_token_id = self.tokenizer.encode(self.STOP_TOKEN)[0]

    def generate(self,
                 prompt: str = STOP_TOKEN,
                 num_return_sequences: int = 1,
                 max_length: int = 20,
                 temperature: float = 1.0,
                 k: Optional[int] = None,
                 p: float = 0.9,
                 repetition_penalty: float = 1.0,
                 stop_token: str = STOP_TOKEN):
        max_length = adjust_length_to_model(max_length, max_sequence_length=self.model.config.max_position_embeddings)

        encoded_prompt = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(self.device)

        output_sequences = self.model.generate(
            input_ids=encoded_prompt,
            max_length=max_length + len(encoded_prompt[0]),
            temperature=temperature,
            top_k=k,
            top_p=p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.pad_token_id
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences = []

        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            generated_sequence = generated_sequence.tolist()

            # Decode text
            text = self.tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

            # Remove the excess text that was used for pre-processing
            text = text[len(self.tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)):]

            # Remove all text after the stop token
            text = text[: text.find(stop_token) if stop_token else None]

            generated_sequences.append(text)

        return generated_sequences
