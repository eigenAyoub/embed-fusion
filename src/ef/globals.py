import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from datasets import Dataset
from functools import partial
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast, BatchEncoding, DataCollatorWithPadding
from transformers.modeling_outputs import BaseModelOutput
from torch import Tensor
from typing import *


BATCH_SIZE = 4


def transform_func(tokenizer: PreTrainedTokenizerFast,
            max_length: int,
            examples: Dict[str, List]) -> BatchEncoding:
    return tokenizer(examples['contents'],
        max_length=max_length,
        padding=True,
        return_token_type_ids=False,
        truncation=True)


def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda(non_blocking=True)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return tuple([_move_to_cuda(x) for x in maybe_tensor])
        elif isinstance(maybe_tensor, Mapping):
            return type(maybe_tensor)({k: _move_to_cuda(v) for k, v in maybe_tensor.items()})
        else:
            return maybe_tensor

    return _move_to_cuda(sample)


def pool(last_hidden_states: Tensor,
         attention_mask: Tensor,
         pool_type: str) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

    if pool_type == "avg":
        emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        # input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float() # gte-tiny
        # return torch.sum(last_hidden_states * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    elif pool_type == "cls":
        emb = last_hidden[:, 0]
    elif pool_type == "last":
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            emb = last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            emb = last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    else:
        raise ValueError(f"pool_type {pool_type} not supported")

    return emb


# Code from: https://github.com/Snowflake-Labs/arctic-embed/blob/main/src/mteb_retrieval.py
class RetrievalModel():
    # Refer to the code of DRESModel for the methods to overwrite
    def __init__(self, pretrained_model_name: str, task: str, **kwargs):
        self.pretrained_model_name = pretrained_model_name
        self.encoder = AutoModel.from_pretrained(pretrained_model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, trust_remote_code=True)
        self.gpu_count = torch.cuda.device_count()
        self.batch_size = BATCH_SIZE

        if pretrained_model_name.startswith('Alibaba-NLP/gte-Qwen2'):
            if 'msmarco' in task.lower():
                self.query_instruction = 'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {}'
            elif 'scifact' in task.lower():
                self.query_instruction = 'Instruct: Given a scientific claim, retrieve relevant scientific evidence\nClaim: {}'
            elif 'quora' in task.lower():
                self.query_instruction = 'Instruct: Given a question, retrieve questions that are semantically equivalent to the given question\nQuestion: {}'
            else:
                self.query_instruction = 'Instruct: Given a query, retrieve relevant passages\nQuery: {}'
            self.pool_type = 'last'
            self.max_length = 32768
        elif pretrained_model_name == 'dunzhang/stella_en_1.5B_v5':
            if 'msmarco' in task.lower():
                self.query_instruction = 'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {}'
            elif 'scifact' in task.lower():
                self.query_instruction = 'Instruct: Given a scientific claim, retrieve relevant scientific evidence\nClaim: {}'
            elif 'quora' in task.lower():
                self.query_instruction = 'Instruct: Given a question, retrieve questions that are semantically equivalent to the given question\nQuestion: {}'
            else:
                self.query_instruction = 'Instruct: Given a query, retrieve relevant passages\nQuery: {}'
            self.pool_type = 'avg'
            self.max_length = 512
        else:
            self.query_instruction = 'Represent this sentence for searching relevant passages: {}'
            self.pool_type = 'cls'
            self.max_length = 512

        self.encoder.cuda()
        self.encoder.eval()

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        input_texts = [self.query_instruction.format(q) for q in queries]
        return self._do_encode(input_texts)

    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs) -> np.ndarray:
        input_texts = ['{} {}'.format(d.get('title', ''), d['text']).strip() for d in corpus]
        return self._do_encode(input_texts)

    @torch.no_grad()
    def _do_encode(self, input_texts: List[str]) -> np.ndarray:
        dataset: Dataset = Dataset.from_dict({'contents': input_texts})
        dataset.set_transform(partial(transform_func, self.tokenizer, self.max_length))

        data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size * self.gpu_count,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            collate_fn=data_collator,
            pin_memory=True)

        encoded_embeds = []
        for batch_dict in tqdm(data_loader, desc='encoding', mininterval=10):
            batch_dict = move_to_cuda(batch_dict)

            with torch.amp.autocast('cuda'):
                outputs: BaseModelOutput = self.encoder(**batch_dict)
                embeds = pool(outputs.last_hidden_state, batch_dict['attention_mask'], pool_type=self.pool_type)
                if self.pool_type == 'last':
                    embeds = F.normalize(embeds, p=2, dim=-1)
                encoded_embeds.append(embeds.cpu().numpy())

        return np.concatenate(encoded_embeds, axis=0)