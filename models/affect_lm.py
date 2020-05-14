from typing import Optional

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GPT2LMHeadModel

NUM_AFFECTS = 2


class Affect(nn.Module):
    """
    See section 4.3 of Ghosh:
        Affect-LM can also be used as a language model where the next predicted word is
        estimated from the words in the context, along with an affect category extracted
        from the context words themselves (instead of being encoded externally as in generation).

    beta in [1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]
    """

    IGNORE_IDS = [50256, 247, 251, 250]

    def __init__(self,
                 affect_dim: int,
                 vocab_size: int,
                 beta: float = 1.0,
                 ignore_special_tokens=False):
        super().__init__()
        self.beta = beta
        self.ignore_special_tokens = ignore_special_tokens
        self.affect2vocab = nn.Linear(affect_dim, vocab_size, bias=True)
        self.init_weights()

    def forward(self, affect_labels: torch.Tensor) -> torch.Tensor:
        out = self.affect2vocab(affect_labels)
        if self.ignore_special_tokens:
            out[:, :, self.IGNORE_IDS] = 0
        return self.beta * out

    def init_weights(self, initializer_range=0.02):
        # initializer_range taken from GPT2Config
        self.affect2vocab.weight.data.normal_(mean=0.0, std=initializer_range)
        self.affect2vocab.bias.data.zero_()


class AffectGPT2LMHeadModel(GPT2LMHeadModel):
    affect_labels: Optional[torch.Tensor]

    def __init__(self, config):
        super().__init__(config)
        self.affect = Affect(NUM_AFFECTS, config.vocab_size)
        self.affect_labels = None

    def freeze_transformer(self):
        for p in self.transformer.parameters():
            p.requires_grad = False

    def freeze_lm_head(self):
        for p in self.lm_head.parameters():
            p.requires_grad = False

    def set_affect_labels(self, affect_labels: torch.Tensor):
        print("Using static affect labels:", affect_labels)
        self.affect_labels = affect_labels

    def forward(
            self,
            input_ids=None,
            past=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            affect_labels=None,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        if affect_labels is None and self.affect_labels is not None:
            affect_labels = self.affect_labels.repeat(len(input_ids), 1, 1)

        if affect_labels is not None and self.affect is not None:
            # Add affect logits to lm logits
            affect_logits = self.affect(affect_labels)
            lm_logits += affect_logits

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)
