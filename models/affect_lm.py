import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GPT2LMHeadModel

from utils.datasets import NUM_AFFECTS


class Affect(nn.Module):
    """
    See section 4.3 of Ghosh:
        Affect-LM can also be used as a language model where the next predicted word is
        estimated from the words in the context, along with an affect category extracted
        from the context words themselves (instead of being encoded externally as in generation).

    beta in [1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]
    """

    def __init__(self,
                 affect_dim: int,
                 vocab_size: int,
                 beta: float = 1.0):
        super().__init__()
        self.beta = beta
        self.affect2vocab = nn.Linear(affect_dim, vocab_size)

    def forward(self, affect_labels: torch.Tensor) -> torch.Tensor:
        return self.beta * self.affect2vocab(affect_labels)


class AffectGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.affect = Affect(NUM_AFFECTS, config.vocab_size)

    def freeze_transformer(self):
        for p in self.transformer.parameters():
            p.requires_grad = False

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

        affect_logits = self.affect(affect_labels)
        lm_logits = self.lm_head(hidden_states)

        logits = lm_logits + affect_logits

        outputs = (logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, presents, (all hidden_states), (attentions)
