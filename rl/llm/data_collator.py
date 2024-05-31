# type: ignore
"""Source: https://medium.com/@newhardwarefound/qlora-with-llama-2-ca1b4bcf26f0"""

import copy
from dataclasses import dataclass
from typing import Sequence

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence

IGNORE_INDEX = -100


@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    input_max_len: int
    output_max_len: int
    train_on_input: bool
    predict_with_generate: bool

    def __call__(
        self,
        instances: dict[str, Sequence[str]] | Sequence[dict[str, str]],
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        assert (
            self.tokenizer.pad_token_id is not None
        ), "The tokenizer must have a pad token set"

        if isinstance(instances, dict):
            if isinstance(instances["input"], str):
                instances = [instances]
            else:
                instances = [
                    {"input": input_, "output": output}
                    for input_, output in zip(instances["input"], instances["output"])
                ]
        sources = [
            f"{self.tokenizer.bos_token}{example['input']}" for example in instances
        ]
        tokenized_inputs = self.tokenizer(
            sources,
            max_length=self.input_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        if not self.predict_with_generate:
            targets = [
                f"{example['output']}{self.tokenizer.eos_token}"
                for example in instances
            ]
            tokenized_outputs = self.tokenizer(
                targets,
                max_length=self.output_max_len,
                truncation=True,
                add_special_tokens=False,
            )
        else:
            tokenized_outputs = tokenized_inputs
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_inputs["input_ids"], tokenized_outputs["input_ids"]
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_input:
                    labels.append(
                        torch.tensor(
                            [IGNORE_INDEX for _ in range(len(tokenized_source))]
                            + copy.deepcopy(tokenized_target)
                        )
                    )
                else:
                    labels.append(
                        torch.tensor(copy.deepcopy(tokenized_source + tokenized_target))
                    )
            else:
                input_ids.append(torch.tensor(tokenized_source))
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = (
            pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
            if not self.predict_with_generate
            else None
        )
        data_dict = {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict["labels"] = labels
        if device is not None:
            data_dict = {k: v.to(device) for k, v in data_dict.items()}
        return data_dict
