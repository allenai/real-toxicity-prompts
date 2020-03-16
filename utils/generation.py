from pathlib import Path
from typing import Union, Optional, List

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, modeling_utils
import torch.nn.functional as F

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
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(self.device)

        # IMPORTANT: Note that setting the pad token like this in the constructor gives the pad_token the
        # pad_token_id = 50256, which normally belongs to the <EOS> token_id in GPT2. This is a very ugly
        # way that works at the moment of setting the pad_token_id to the <EOS> token that is already
        # included in the vocab size.
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path, pad_token=self.STOP_TOKEN)

    # def generate(self,
    #              prompt: str = STOP_TOKEN,
    #              num_return_sequences: int = 1,
    #              max_length: int = 20,
    #              temperature: float = 1.0,
    #              k: Optional[int] = None,
    #              p: float = 0.9,
    #              repetition_penalty: float = 1.0,
    #              stop_token: str = STOP_TOKEN):
    #     max_length = adjust_length_to_model(max_length, max_sequence_length=self.model.config.max_position_embeddings)
    #
    #     encoded_prompt = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    #     encoded_prompt = encoded_prompt.to(self.device)
    #
    #     output_sequences = self.model.generate(
    #         input_ids=encoded_prompt,
    #         max_length=max_length + len(encoded_prompt[0]),
    #         temperature=temperature,
    #         top_k=k,
    #         top_p=p,
    #         repetition_penalty=repetition_penalty,
    #         do_sample=True,
    #         num_return_sequences=num_return_sequences,
    #         pad_token_id=self.pad_token_id
    #     )
    #
    #     # Remove the batch dimension when returning multiple sequences
    #     if len(output_sequences.shape) > 2:
    #         output_sequences.squeeze_()
    #
    #     return self._decode(output_sequences, encoded_prompt)

    def _decode(self, output_sequences: List[torch.Tensor], encoded_prompt: Optional[torch.Tensor] = None):
        generated_sequences = []

        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            generated_sequence = generated_sequence.tolist()

            # Decode text
            text = self.tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

            # Remove the excess text that was used for pre-processing
            if encoded_prompt:
                text = text[len(self.tokenizer.decode(encoded_prompt, clean_up_tokenization_spaces=True)):]

            # Remove all text after the stop token
            text = text[: text.find(self.STOP_TOKEN) if self.STOP_TOKEN else None]

            generated_sequences.append(text)

        return generated_sequences

    def generate_2(self,
                   prompt_text: Union[str, List[str]],
                   num_tokens_to_produce=20,
                   do_sample=True,
                   top_k=0,
                   top_p=0.9,
                   temperature=1.0):
        # encode plus batch handles multiple batches and automatically creates attention_masks
        encodings_dict = self.tokenizer.batch_encode_plus(prompt_text, pad_to_max_length=True, return_tensors='pt')

        input_ids = encodings_dict['input_ids'].to(self.device)
        attn_mask = encodings_dict['attention_mask'].to(self.device)
        batch_size, input_seq_len = input_ids.shape

        position_ids = input_ids.ne(self.tokenizer.pad_token_id).cumsum(dim=1) - 1
        eos_not_in_sents = torch.ones(batch_size, dtype=torch.long, device=self.device)

        # TODO: use this to speed up generation
        past = None

        for step in range(num_tokens_to_produce):
            logits, past = self.model(input_ids, attention_mask=attn_mask, position_ids=position_ids)

            # in the first decoding step, we want to use the 'real' last position for each sentence
            if step == 0:
                last_non_masked_idx = torch.sum(attn_mask, dim=1) - 1
                next_token_logits = logits[range(batch_size), last_non_masked_idx, :]
            else:
                next_token_logits = logits[:, -1, :]

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                # Top-p/top-k filtering
                next_token_logits = modeling_utils.top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
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
            position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=1)

        decoded_outputs = [self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                           for output in input_ids[:, input_seq_len:]]
        return decoded_outputs


def test_generate_2():
    generator = GPT2Generator()
    prompt_text = [
        'in this paper we',
        'we are trying to',
        'The purpose of this workshop is to check whether we can']
    out = generator.generate_2(prompt_text)
    print(*out, sep='\n')
