import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Model
from transformers.modeling_outputs import CausalLMOutput


class Wav2Vec2ForFrameClassificationSAT(Wav2Vec2ForCTC):
    def __init__(self, config, satvector_size=512):
        super().__init__(config)

        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: "
                "`Wav2Vec2ForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        output_hidden_size = (
            config.output_hidden_size
            if hasattr(config, "add_adapter") and config.add_adapter
            else config.hidden_size
        )
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)
        self.lm_head_ivec = nn.Linear(output_hidden_size + satvector_size, config.vocab_size)

        self.post_init()

    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        ixvector=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        hidden_states = self.dropout(hidden_states)

        # framewise append i/xvector
        # satvector = ixvector.unsqueeze(dim=1).repeat(1, hidden_states.shape[1], 1)
        satvector = ixvector.unsqueeze(dim=1).repeat(1, hidden_states.shape[1], 1)
        hidden_states = torch.cat([hidden_states, satvector], dim=-1)

        logits = self.lm_head_ivec(hidden_states)
        timesteps = logits.shape[1]
        loss = None
        if labels is not None:
            labels = labels[:, :timesteps]
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask
                if attention_mask is not None
                else torch.ones_like(input_values, dtype=torch.long)
            )
            # Never used
            # input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(
            #     torch.long
            # )

            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(2)), labels.flatten(), reduction="mean"
            )

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def _init_ivec_head_from_lm_head(self):
    with torch.no_grad():
        self.lm_head_ivec.weight.zero_()
        self.lm_head_ivec.bias.zero_()

        W = self.lm_head.weight  # [vocab, hidden]
        b = self.lm_head.bias  # [vocab]

        self.lm_head_ivec.weight[: W.size(0), : W.size(1)].copy_(W)
        self.lm_head_ivec.bias[: b.size(0)].copy_(b)


def from_pretrained(cls, *args, **kwargs):
    model = super().from_pretrained(*args, **kwargs)
    model._init_ivec_head_from_lm_head()
    return model


class Wav2Vec2ForFrameClassification(Wav2Vec2ForCTC):
    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask
                if attention_mask is not None
                else torch.ones_like(input_values, dtype=torch.long)
            )

            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(2)), labels.flatten(), reduction="mean"
            )

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


#
# class Wav2Vec2ForFrameClassificationSAT(Wav2Vec2ForCTC):
#     pass # Removed (not used in code) --- IGNORE ---
