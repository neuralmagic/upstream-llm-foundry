# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Datasets for converting to MDS Shards."""
import os
import warnings
from typing import Dict, Iterable, Union, Optional

import datasets as hf_datasets
import numpy as np
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase


def open_platypus(sample):
    return sample['instruction'] + " " + sample['output']

def open_orca(sample):
    return sample['question'] + " " + sample['response']

def dolphin(sample):
    return sample['input'] + " " + sample['output']

def open_hermes_2_5(sample):
    txt = ""
    if sample['conversations'][0]['from'] == 'system':
        txt += sample['conversations'][0]['value']
        sample['conversations'].pop(0)

    for i in range(0, len(sample['conversations']), 2):
        txt += " Question: " + sample['conversations'][i]['value'] + " Answer: " + sample['conversations'][i+1]['value']

    return txt

def bagel_v03(sample):
    assert len(sample['conversations']) <= 3, "Fix the preprocessing code to account for multi-turn conversations"

    ans_from_gpt = sample['conversations'][-1]
    assert ans_from_gpt["from"] == "gpt", f"Sample is: {sample}"

    q_from_human = sample['conversations'][-2]
    assert q_from_human["from"] == "human", f"Sample is: {sample}"

    if len(sample['conversations']) == 3:
        system_prompt = sample['conversations'][0]
        assert system_prompt["from"] == "system", f"Sample is: {sample}"
    else:
        system_prompt = ""

    return system_prompt + " Question: " + q_from_human["value"] + " Answer: " + ans_from_gpt["value"]

def ultrachat(sample):
    assert len(sample['data']) % 2 == 0, "Some conversations dont have Q+A pairs"
    txt = ""
    for i in range(0, len(sample['data']), 2):
        txt += "Question: " + sample['data'][i] + " Answer: " + sample['data'][i+1] + " "

    return txt

def mmlu_aux_train(sample):
    if 'train' in sample.keys():
        sample = sample['train']
    return sample['question'] + " " + sample['choices'][sample['answer']]

def open_math_instruct_1(sample):
    if sample['is_correct'] == 'true' or sample['is_correct'] == True:
        return sample['question'] + " " + sample['generated_solution']
    else:
        return None

def gsm8k(sample):
    return sample['question'] + " " + sample['answer']

def alpaca_cleaned(sample):
    return sample['instruction'] + " " + sample['input'] + " " + sample['output']

def dolly_hhrlhf(sample):
    return sample['prompt'] + " " + sample['response']

def flan2021_submix_original(sample):
    return sample['inputs'] + " " + sample['targets']

def arc_corpus(sample):
    return sample

def flanv2(sample):
    return sample['inputs'] + " " + sample['targets']

def flan(sample):
    return sample['inputs'] + " " + sample['targets']

CONVERT_TO_PRETRAINING = {
    "garage-bAInd/Open-Platypus": open_platypus,
    "Open-Orca/OpenOrca": open_orca,
    "cognitivecomputations/dolphin": dolphin,
    "teknium/OpenHermes-2.5": open_hermes_2_5,
    "jondurbin/bagel-v0.3": bagel_v03,
    "stingning/ultrachat": ultrachat,
    "cais/mmlu": mmlu_aux_train,
    "nvidia/OpenMathInstruct-1": open_math_instruct_1,
    "gsm8k": gsm8k,
    "yahma/alpaca-cleaned": alpaca_cleaned,
    "mosaicml/dolly_hhrlhf": dolly_hhrlhf,
    "prigoyal/flan2021_submix_original": flan2021_submix_original,
    "arc_corpus": arc_corpus,
    "philschmid/flanv2": flanv2,
    "Open-Orca/FLAN": flan,
}


class NoConcatDataset(IterableDataset):
    """An IterableDataset that returns text samples for MDSWriter.

    Returns dicts of {'text': bytes}
    """

    def __init__(self, hf_dataset: Union[hf_datasets.IterableDataset,
                                         hf_datasets.Dataset]):
        self.hf_dataset = hf_dataset

    def __iter__(self) -> Iterable[Dict[str, bytes]]:
        for sample in self.hf_dataset:
            # convert to bytes to store in MDS binary format
            yield {'text': sample['text'].encode('utf-8')}


class ConcatTokensDataset(IterableDataset):
    """An IterableDataset that returns token samples for MDSWriter.

    Returns dicts of {'tokens': bytes}

    To use data created by this class and written to MDS format:

    ```python
        import torch
        from streaming.base import StreamingDataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained('your/tokenizer')
        ds = StreamingDataset(local='mds-data-folder', split='val')

        # note, you need to copy the numpy array because the original is non-writeable
        # and torch does not support non-writeable tensors, so you get a scary warning and
        # if you do try to write to the tensor you get undefined behavior
        tokens = torch.from_numpy(np.frombuffer(ds[0]['tokens'], dtype=np.int64).copy())
        print(tokenizer.decode(tokens))
    ```
    """

    def __init__(
        self,
        hf_dataset: Union[hf_datasets.IterableDataset, hf_datasets.Dataset],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        bos_text: str,
        eos_text: str,
        no_wrap: bool,
        hf_id: Optional[str] = None,
    ):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.max_length = max_length
        self.bos_text = bos_text
        self.eos_text = eos_text
        self.should_wrap = not no_wrap
        self.hf_id = hf_id
        self.debug_counter = 100

        self.bos_tokens = self.tokenizer(self.bos_text,
                                         truncation=False,
                                         padding=False,
                                         add_special_tokens=False)['input_ids']
        if len(self.bos_tokens) > 1:
            warnings.warn(
                f'You specified --concat_tokens with --bos_text, but your BOS text is not tokenizing to one token\
                , instead we got {self.bos_tokens}. Quit if this was in error.')

        self.eos_tokens = self.tokenizer(self.eos_text,
                                         truncation=False,
                                         padding=False,
                                         add_special_tokens=False)['input_ids']
        if len(self.eos_tokens) > 1:
            warnings.warn(
                f'You specified --concat_tokens with --eos_text, but your EOS text is not tokenizing to one token\
                , instead we got {self.eos_tokens}. Quit if this was in error.')

        eos_text_provided = self.eos_text != ''
        bos_text_provided = self.bos_text != ''
        test_text = self.tokenizer('')
        if len(test_text['input_ids']) > 0 and (eos_text_provided or
                                                bos_text_provided):
            message = 'both eos and bos' if eos_text_provided and bos_text_provided else (
                'eos_text' if eos_text_provided else 'bos_text')
            warnings.warn(
                f'The provided tokenizer adds special tokens, but you also specified {message}. This may result '
                +
                'in duplicated special tokens. Please be sure this is what you intend.'
            )

    def __iter__(self) -> Iterable[Dict[str, bytes]]:

        buffer = []
        for sample in self.hf_dataset:
            if self.hf_id in CONVERT_TO_PRETRAINING:
                sample['text'] = CONVERT_TO_PRETRAINING[self.hf_id](sample)
                if sample['text'] is None:
                    continue

            encoded = self.tokenizer(sample['text'],
                                     truncation=False,
                                     padding=False)

            if "ELDAR_DEBUG" in os.environ and os.environ["ELDAR_DEBUG"] == "1":
                print(f"\n [a sample] ---> {self.tokenizer.decode(encoded['input_ids'])}")
                self.debug_counter -= 1
                if self.debug_counter <= 0:
                    os.environ["ELDAR_DEBUG"] = "0"

            iids = encoded['input_ids']
            buffer = buffer + self.bos_tokens + iids + self.eos_tokens
            while len(buffer) >= self.max_length:
                concat_sample = buffer[:self.max_length]
                buffer = buffer[self.max_length:] if self.should_wrap else []
                yield {
                    # convert to bytes to store in MDS binary format
                    'tokens': np.asarray(concat_sample).tobytes()
                }

                if "ELDAR_DEBUG" in os.environ and os.environ["ELDAR_DEBUG"] == "1":
                    print(f"\n [concatenated samples] ---> {self.tokenizer.decode(concat_sample)}")
                    self.debug_counter -= 1
                    if self.debug_counter <= 0:
                        os.environ["ELDAR_DEBUG"] = "0"
