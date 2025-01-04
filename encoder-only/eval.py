from sentence_transformers import SentenceTransformer
import mteb
from typing import List, Union, Dict, Any, Optional

from mteb.encoder_interface import Encoder, PromptType
from encoder_residual import EncoderOnly, EncoderConfig 

import torch
import numpy as np

import sys

model_config = EncoderConfig.DEFAULT

inDim = model_config["input_dim"]
outDim = model_config["max_compressed_dim"]

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

    def __init__(self, 
                 models: List[Union[str, SentenceTransformer]], 
                 device: str = 'cuda',
                 checkpoint_path: Optional[str] = None,
                 input_dim:       Optional[int] = None, 
                 compressed_dim:  Optional[int] = None,
                 truncate:        Optional[int] = None,
                ):

        self.device = device
        self.models: List[SentenceTransformer] = []
        self.model_keys: List[str] = []
        self.call_count = 0
        self.first_call_done = False
        self.truncate = truncate

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

        if checkpoint_path and input_dim and compressed_dim:
            print("Initializing encoder with provided parameters")
            self.encoder = EncoderOnly(EncoderConfig.DEFAULT).to(device)
            self.encoder.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
        else:
            print(f"No encoder initialization - will use {'single model' if self.single_model else 'raw concatenation'}")

    def encode(self, sentences: Union[str, List[str]], **kwargs):
        self.call_count += 1
        print(f"> Embedding model called with kwargs: {kwargs}")
        embeddings_list = []

        # Remove NFCorpus prompt if present
        local_kwargs = kwargs.copy()
        #if 'prompt_name' in local_kwargs and local_kwargs['prompt_name'] == 'NFCorpus':
        if 'prompt_name' in local_kwargs:
            del local_kwargs['prompt_name']
        
        for model, model_key in zip(self.models, self.model_keys):
            model_kwargs = local_kwargs.copy()
            if "snowflake" in model_key:
                if self.call_count == 1:
                    print(f"## Updating {model_key} `model_kwargs` ## Setting `prompt_name=PromptType.query`")
                    model_kwargs['prompt_name'] = PromptType.query
                    print(f"## Updated kwargs for {model_key}: {model_kwargs}")
                elif 'prompt_name' in model_kwargs:
                    del model_kwargs['prompt_name']
                    
            print(f">> model.encode(.) #Number {self.call_count} for {model_key} with kwargs: {model_kwargs}")
            embeddings = model.encode(sentences, **model_kwargs)
            
            if isinstance(embeddings, torch.Tensor):
                print(f"Model {model_key} output tensor shape: {embeddings.shape}")
            elif isinstance(embeddings, np.ndarray):
                print(f"Model {model_key} output array shape: {embeddings.shape}")
                
            embeddings_list.append(embeddings)

        if self.single_model:
            print(f"Operating on a single model")
            return embeddings_list[0]
        else:
            if all(isinstance(e, torch.Tensor) for e in embeddings_list):
                concat = torch.cat(embeddings_list, dim=1)
                print(f">> Concatenated tensor shape: {concat.shape}")
                if hasattr(self, 'encoder'):
                    encoded = self.encoder(concat.to(self.device), 
                                        compressed_dim=self.truncate)
                    print(f">> Returning the encoder tensor shape: {encoded.shape}")
                    return encoded
                else:
                    print(f"Concat with No encoder")
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
    if len(sys.argv) < 4:
        print("Usage:   python eval.py <checkpoint> <truncate>  <model_type>")
        print("Example: python eval.py <checkpoint> <truncate>  <model_type>")
        print("model_type: bge-small, snowflake-m, or combined")
        sys.exit(1)
        
    ckpt = sys.argv[1] 
    trunc = int(sys.argv[2]) 
    model_type = sys.argv[3] # `combined` oder ... 
    tsk  = sys.argv[4]
    use_encoder = bool(int(sys.argv[5])) if len(sys.argv) > 5 else False

    # Configure model based on type
    if model_type == "combined":
        model_keys = ["bge-small", "snowflake-m"]
        model = AdaptiveSentenceTransformer(
            models=model_keys,
            device="cuda",
            checkpoint_path=f"models_pth/{inDim}_{outDim}/{ckpt}.pth" if use_encoder else None,
            input_dim=1152 if use_encoder else None,
            compressed_dim=768 if use_encoder else None,
            truncate=trunc if use_encoder else None
        )
        output_folder = f"results/residual"
    else:
        model = AdaptiveSentenceTransformer(
            models=[model_type],
            device="cuda"
        )
        output_folder = f"results/{tsk}_{model_type}"
    
    # Setup evaluation
    tasks = mteb.get_tasks(tasks=[tsk])
    evaluation = mteb.MTEB(
        tasks=tasks,
        eval_splits=["test"],
        metric="ndcg@10"
    )
    
    results = evaluation.run(
        model,
        output_folder=output_folder,
        eval_splits=["test"],
        batch_size=128
    )
    
    return results

if __name__ == "__main__":
    main()
