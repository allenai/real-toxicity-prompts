from pathlib import Path
from typing import Union, Optional, Sequence

import torch
import numpy as np
from transformers import XLNetLMHeadModel, XLNetTokenizer

# from utils import utils

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


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


def prepare_xlnet_input(args, prompt_text):
    prompt_text = (args.padding_text if args.padding_text else PADDING_TEXT) + prompt_text
    return prompt_text


class XLNetGenerator:
    def __init__(self, model: Union[str, Path, XLNetLMHeadModel] = 'xlnet-base-cased',
                 tokenizer: str = 'xlnet-base-cased', seed: int = 42):
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        set_seed(seed, n_gpu)

        # Set up model
        if isinstance(model, Path) or isinstance(model, str):
            model = XLNetLMHeadModel.from_pretrained(str(model))
        self.model = model.to(self.device)

        self.tokenizer = XLNetTokenizer.from_pretrained(tokenizer)

    def __repr__(self):
        return f'<XLNetGenerator model_name_or_path="{self.model}">'

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def generate(self,
                 prompts: Union[Sequence[str], str],
                 max_len: int = 20,
                 temperature: float = 1.0,
                 k: int = 0,
                 p: float = 0.9,
                 num_return_sequences: int = 1,
                 sample: bool = True,
                 repetition_penalty: float = 1.0,
                 padding_text: Optional[str] = None,
                 clean_up_tokenization_spaces: bool = True):
        if isinstance(prompts, str):
            prompts = [prompts]

        # Prepend rasputin-themed padding text to prompts
        if padding_text is None:
            padding_text = PADDING_TEXT
        prompts = [padding_text + prompt for prompt in prompts]

        # Tokenize prompts and pad them
        encodings_dict = self.tokenizer.batch_encode_plus(
            prompts, add_special_tokens=False, pad_to_max_length=True, return_tensors='pt'
        ).to(self.device)

        input_ids = encodings_dict['input_ids']
        batch_size, prompt_len = input_ids.shape

        # Generate up to max_len tokens
        max_len = adjust_length_to_model(max_len, max_sequence_length=self.model.config.max_position_embeddings)
        output_sequences = self.model.generate(
            input_ids=input_ids,
            max_length=max_len + prompt_len,
            temperature=temperature,
            top_k=k,
            top_p=p,
            repetition_penalty=repetition_penalty,
            do_sample=sample,
            num_return_sequences=num_return_sequences,
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        # Decode outputs, trimming everything after EOS
        decoded_outputs = []
        for output in output_sequences:
            output = output[prompt_len:]
            try:
                stop_index = [i for i, x in enumerate(output) if x == self.tokenizer.eos_token_id][0]
            except IndexError:
                stop_index = None
            output = output[:stop_index]
            decoded_outputs.append(
                self.tokenizer.decode(output, clean_up_tokenization_spaces=clean_up_tokenization_spaces)
            )

        return decoded_outputs
