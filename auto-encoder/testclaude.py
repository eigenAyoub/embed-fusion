from sentence_transformers import SentenceTransformer
import mteb
from typing import List, Union, Dict, Any
import logging

from mteb.encoder_interface import Encoder, PromptType

import torch
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdaptiveSentenceTransformer(Encoder):
    MODEL_CATALOGUE: Dict[str, str] = {
        "mxbai": "mixedbread-ai/mxbai-embed-large-v1",
        "bge": "BAAI/bge-large-en-v1.5",
        "e5": "intfloat/e5-large-v2",
        "snowflake-l": "Snowflake/snowflake-arctic-embed-l",
        "gte-base": "thenlper/gte-base",
        "gte-large": "thenlper/gte-large",
        "gte-small": "thenlper/gte-small",
        "e5-small": "intfloat/e5-small-v2",
        "bge-small": "BAAI/bge-small-en-v1.5",
        "snowflake-m": "Snowflake/snowflake-arctic-embed-m-v1.5",
        "jina-v3": "jinaai/jina-embeddings-v3"
    }

    def __init__(self, models: List[Union[str, SentenceTransformer]], device: str = 'cuda'):
        self.device = device
        self.models: List[SentenceTransformer] = []
        self.model_keys: List[str] = []
        self.call_count = 0
        self.first_call_done = False


        for model_info in models:
            if isinstance(model_info, str):
                model_key = model_info
                model_path = self.MODEL_CATALOGUE.get(model_key)
                if model_path is None:
                    raise ValueError(f"Model key '{model_key}' not found in catalogue")
                model = SentenceTransformer(model_path, trust_remote_code=True).to(device)
            elif isinstance(model_info, SentenceTransformer):
                model_key = "custom_model"
                model = model_info.to(device)
            else:
                raise TypeError("Each item in 'models' must be either a string (model key) or a SentenceTransformer instance")

            self.models.append(model)
            self.model_keys.append(model_key)

        self.single_model = (len(self.models) == 1)

    def encode(self, sentences: Union[str, List[str]], **kwargs):
        self.call_count += 1
        print(f"encode called with kwargs: {kwargs}")
        embeddings_list = []

        # Remove NFCorpus prompt if present
        local_kwargs = kwargs.copy()
        if 'prompt_name' in local_kwargs and local_kwargs['prompt_name'] == 'NFCorpus':
            del local_kwargs['prompt_name']
        
        for model, model_key in zip(self.models, self.model_keys):
            model_kwargs = local_kwargs.copy()
            print(f"\n>> MF where are we? {model_key}\n")

            if "snowflake" in model_key:
                print("MF where are we?")
                if self.call_count == 1:
                    print(f"Setting prompt_name=PromptType.query for arctic model: {model_key}")
                    model_kwargs['prompt_name'] = PromptType.query
                    print(f"Updated kwargs for arctic: {model_kwargs}")
                elif 'prompt_name' in model_kwargs:
                    del model_kwargs['prompt_name']
                    
            print(f"Call #{self.call_count} for {model_key} with kwargs: {model_kwargs}")
            embeddings = model.encode(sentences, **model_kwargs)
            
            if isinstance(embeddings, torch.Tensor):
                print(f"Model {model_key} output tensor shape: {embeddings.shape}")
            elif isinstance(embeddings, np.ndarray):
                print(f"Model {model_key} output array shape: {embeddings.shape}")
                
            embeddings_list.append(embeddings)

        if self.single_model:
            return embeddings_list[0]
        else:
            if all(isinstance(e, torch.Tensor) for e in embeddings_list):
                concat = torch.cat(embeddings_list, dim=1)
                print(f"Concatenated tensor shape: {concat.shape}")
                return concat
            elif all(isinstance(e, np.ndarray) for e in embeddings_list):
                concat = np.concatenate(embeddings_list, axis=1)
                print(f"Concatenated array shape: {concat.shape}")
                return concat
            else:
                raise ValueError(f"Cannot concatenate embeddings of mixed types: {[type(e) for e in embeddings_list]}")




def evaluate_model(
    model_key: str,
    tasks: List[str] = ["NFCorpus"],
    output_folder: str = None,
    batch_size: int = 256,
) -> dict:
    if model_key not in AdaptiveSentenceTransformer.MODEL_CATALOGUE:
        raise ValueError(f"Model {model_key} not found in catalogue")

    model = AdaptiveSentenceTransformer(models=[model_key])

    if output_folder is None:
        output_folder = f"results/{model_key}"

    tasks = mteb.get_tasks(tasks=tasks)

    evaluation = mteb.MTEB(
        tasks=tasks,
        eval_splits=["test"],
        metrics=["ndcg@10"],
    )

    return evaluation.run(
        model,
        output_folder=output_folder,
        batch_size=batch_size,
    )

def main():
    model_keys = ["bge-small", "snowflake-m"]
    
    model = AdaptiveSentenceTransformer(models=model_keys)
    
    tasks = mteb.get_tasks(tasks=["NFCorpus"])
    evaluation = mteb.MTEB(
        tasks=tasks,
        eval_splits=["test"],
        metrics=["ndcg@10"]
    )
    
    results = evaluation.run(
        model,
        output_folder=f"results/combined_bge_snow_3",
        batch_size=128
    )

    print(results)
    return results

if __name__ == "__main__":
    main()