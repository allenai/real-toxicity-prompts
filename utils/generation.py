from pathlib import Path
from typing import Union, List

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, modeling_utils, GPT2PreTrainedModel

from utils import utils

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop


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

    def __init__(self, model: Union[str, Path, GPT2PreTrainedModel] = 'gpt2', tokenizer: str = 'gpt2', seed: int = 42):
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        utils.set_seed(seed, n_gpu)

        # Set up model
        if isinstance(model, Path) or isinstance(model, str):
            model = GPT2LMHeadModel.from_pretrained(str(model))
        self.model = model.to(self.device)

        # Set up tokenizer
        # IMPORTANT: Note that setting the pad token like this in the constructor gives the pad_token the
        # pad_token_id = 50256, which normally belongs to the <EOS> token_id in GPT2. This is a very ugly
        # way that works at the moment of setting the pad_token_id to the <EOS> token that is already
        # included in the vocab size.
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer, pad_token=self.STOP_TOKEN)
        assert self.tokenizer.eos_token == self.tokenizer.pad_token

        self.unpadded_tokenizer = GPT2Tokenizer.from_pretrained(tokenizer)

    def __repr__(self):
        return f'<GPT2Generator model_name_or_path="{self.model}">'

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def generate(self,
                 prompt: Union[str, List[str]],
                 max_len: int = 20,
                 sample: bool = True,
                 k: int = 0,
                 p: float = 0.9,
                 temperature: float = 1.0,
                 **model_kwargs) -> List[str]:
        if isinstance(prompt, str):
            prompt = [prompt]

        if prompt[0] == self.STOP_TOKEN:
            print("Tokenizing without padding because first prompt was EOS. Assuming all prompts are EOS.")
            encodings_dict = self.unpadded_tokenizer.batch_encode_plus(prompt, return_tensors='pt')
        else:
            encodings_dict = self.tokenizer.batch_encode_plus(prompt, pad_to_max_length=True, return_tensors='pt')

        input_ids = encodings_dict['input_ids'].to(self.device)
        attn_mask = encodings_dict['attention_mask'].to(self.device)
        batch_size, input_seq_len = input_ids.shape

        if prompt[0] == self.STOP_TOKEN:
            position_ids = None
        else:
            position_ids = input_ids.ne(self.tokenizer.pad_token_id).cumsum(dim=1) - 1
        eos_not_in_sents = torch.ones(batch_size, dtype=torch.long, device=self.device)

        # TODO: use this to speed up generation
        past = None

        self.model.eval()

        with torch.no_grad():
            for step in range(max_len):
                logits, past = self.model(input_ids, attention_mask=attn_mask, position_ids=position_ids,
                                          **model_kwargs)

                # in the first decoding step, we want to use the 'real' last position for each sentence
                if step == 0:
                    last_non_masked_idx = torch.sum(attn_mask, dim=1) - 1
                    next_token_logits = logits[range(batch_size), last_non_masked_idx, :]
                else:
                    next_token_logits = logits[:, -1, :]

                if sample:
                    # Temperature (higher temperature => more likely to sample low probability tokens)
                    if temperature != 1.0:
                        next_token_logits = next_token_logits / temperature
                    # Top-p/top-k filtering
                    next_token_logits = modeling_utils.top_k_top_p_filtering(next_token_logits, top_k=k, top_p=p)
                    # Sample
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    # Greedy decoding
                    next_tokens = torch.argmax(next_token_logits, dim=-1)

                # this updates which sentences have not seen an <EOS> token so far
                # if one <EOS> token was seen the sentence is finished
                eos_not_in_sents.mul_(next_tokens.ne(self.tokenizer.eos_token_id).long())

                # either append a padding token here if <EOS> has been seen or append next token
                tokens_to_add = next_tokens * eos_not_in_sents + self.tokenizer.pad_token_id * (1 - eos_not_in_sents)

                # Update input_ids, attn_mask and position_ids
                input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
                attn_mask = torch.cat([attn_mask, attn_mask.new_ones((batch_size, 1))], dim=1)
                if prompt[0] != self.STOP_TOKEN:
                    position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=1)

        decoded_outputs = [self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                           for output in input_ids[:, input_seq_len:]]
        return decoded_outputs


def test_generate():
    generator = GPT2Generator()
    prompt = [
        'in this paper we',
        'we are trying to',
        'The purpose of this workshop is to check whether we can'
    ]
    out = generator.generate(prompt)
    print(*out, sep='\n')
