from sentence_transformers import SentenceTransformer
import mteb
import datasets
from typing import List, Union, Dict, Optional

from mteb.encoder_interface import Encoder, PromptType

from model import EncoderOnly  # Your custom encoder
from offline_quant import PerColumnQuantizer  # Your quantizer

import torch
import torch.nn.functional as F
import numpy as np

import sys
import random

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel  # Hugging Face

# Set seeds
seed = 37
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# Global model cache (to avoid reloading models unnecessarily)
MODEL_CACHE = {}

class AdaptiveSentenceTransformer(Encoder):
    MODEL_CATALOGUE: Dict[str, str] = {
        "mxbai": "mixedbread-ai/mxbai-embed-large-v1",
        "bge": "BAAI/bge-large-en-v1.5",
        "e5": "intfloat/e5-large-v2",
        "snowflake-l": "Snowflake/snowflake-arctic-embed-l",
        "gte-base": "thenlper/gte-base",
        "gte-large": "thenlper/gte-large",
        "gte-small": "thenlper/gte-small",
        "snowflake-m": "Snowflake/snowflake-arctic-embed-m-v1.5",
        "jina-v3": "jinaai/jina-embeddings-v3",
        "e5-small": "intfloat/e5-small-v2",
        "bge-small": "BAAI/bge-small-en-v1.5",
        "gist": "avsolatorio/GIST-small-Embedding-v0",
        "linq": "Linq-AI-Research/Linq-Embed-Mistral",
        "no-ins": "avsolatorio/NoInstruct-small-Embedding-v0"
    }

    def __init__(self,
                 models: List[str],
                 device: str = 'cuda',
                 checkpoint_path: Optional[str] = None,
                 quantizer_path: Optional[str] = None,
                 input_dim: Optional[int] = None,
                 compressed_dim: Optional[int] = None,
                 truncate: Optional[int] = None,
                 ):

        self.device = device
        self.models: List[Union[SentenceTransformer, AutoModel]] = []
        self.tokenizers: List[Union[None, AutoTokenizer]] = []
        self.model_keys: List[str] = []
        self.truncate = truncate

        for model_key in models:
            model_path = self.MODEL_CATALOGUE.get(model_key)
            if model_path is None:
                raise ValueError(f"Model '{model_key}' not found.")

            # Use the global model cache
            if model_path not in MODEL_CACHE:
                if model_key.startswith("e5"):
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    model = AutoModel.from_pretrained(model_path).to(device)
                    MODEL_CACHE[model_path] = (model, tokenizer)
                elif model_key == "no-ins":
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    model = AutoModel.from_pretrained(model_path).to(device)
                    MODEL_CACHE[model_path] = (model, tokenizer)
                else:
                    model = SentenceTransformer(model_path, trust_remote_code=True).to(device)
                    MODEL_CACHE[model_path] = (model, None)

            model.eval()
            model, tokenizer = MODEL_CACHE[model_path]
            self.models.append(model)
            self.tokenizers.append(tokenizer)
            self.model_keys.append(model_key)

        self.single_model = (len(self.models) == 1)

        if checkpoint_path and input_dim and compressed_dim:
            model_config = {"input_dim": input_dim, "output_dim": compressed_dim}
            self.encoder = EncoderOnly(model_config).to(device)
            self.encoder.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
            self.encoder.eval()
        else:
            print("No encoder initialization.")

        if quantizer_path:
            quant_p = torch.load(quantizer_path)
            self.quantizer = PerColumnQuantizer(num_bits=quant_p["num_bits"], device=device)
            self.quantizer.min_vals = quant_p["min_vals"].to(device)
            self.quantizer.scales = quant_p["scales"].to(device)
            self.quantizer.lut = quant_p["lut"].to(device)


    def _hf_encode(self, model: AutoModel, 
                   tokenizer: AutoTokenizer, 
                   sentences: Union[str, List[str]],
                   **kwargs) -> torch.Tensor:
        
        if isinstance(sentences, str):
            sentences = [sentences]
        
        prompt_name = kwargs.get("prompt_type", None)

        if prompt_name == PromptType.query:
            prompted_sentences = [f"query: {s}" for s in sentences]
        elif prompt_name == PromptType.passage:
            prompted_sentences = [f"passage: {s}" for s in sentences]
        else:  # Default to "query: " if no prompt is provided
            prompted_sentences = [f"passage: {s}" for s in sentences]

        batch_dict = tokenizer(prompted_sentences, max_length=512, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():  # Disable gradient calculation
            outputs = model(**batch_dict)
        embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def average_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
   
    def _noins_encode(self,
                    model: AutoModel,
                    tokenizer: AutoTokenizer,
                    sentences: Union[str, List[str]],
                    **kwargs) -> torch.Tensor:
        """
        Custom encoding for the no-ins model.
        For queries, this function applies mean pooling (weighted by the attention mask).
        For sentences/documents, it uses the [CLS] token representation.
        """
        # Ensure sentences is a list
        if isinstance(sentences, str):
            sentences = [sentences]
        
        prompt_name = kwargs.get("prompt_type", None)

        mode = "query" if prompt_name == PromptType.query else "sentence"
        
        inp = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        with torch.no_grad():
            output = model(**inp)
        
        if mode == "query":
            vectors = output.last_hidden_state * inp["attention_mask"].unsqueeze(2)
            vectors = vectors.sum(dim=1) / inp["attention_mask"].sum(dim=-1).view(-1, 1)
        else:
            vectors = output.last_hidden_state[:, 0, :]
        
        vectors = F.normalize(vectors, p=2, dim=1)
        return vectors

    def encode(self, sentences: Union[str, List[str]], batch_size: int = 32, prompt_name: Optional[str] = None, **kwargs) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(sentences, str):
            sentences = [sentences]

        embeddings_list = []

        for model, tokenizer, model_key in zip(self.models, self.tokenizers, self.model_keys):
            model_batches = []
            # Wrap the batch iteration in tqdm for a progress bar.
            for start in tqdm(range(0, len(sentences), batch_size), desc=f"Processing {model_key} batches", leave=False):
                batch = sentences[start:start + batch_size]
                if model_key.startswith("e5"):
                    batch_emb = self._hf_encode(model, tokenizer, batch, **kwargs)
                elif model_key == "no-ins":
                    batch_emb = self._noins_encode(model, tokenizer, batch, **kwargs)
                else:  # dealing with SentenceTransformer stuff
                    prompt_name = kwargs.get("prompt_type", None)
                    if prompt_name == PromptType.query:
                        if model_key == "mxbai" or "snowflake" in model_key:
                            batch_emb = model.encode(
                                batch,
                                batch_size=batch_size,
                                prompt_name="query",
                                show_progress_bar=False,
                                convert_to_tensor=True
                            ).to(self.device)
                        elif model_key == "linq":
                            task = 'Given a query, retrieve passages that answer the question/query'
                            prompt = f"Instruct: {task}\nQuery: "
                            batch_emb = model.encode(
                                batch,
                                batch_size=32,
                                prompt=prompt,
                                show_progress_bar=False,
                                convert_to_tensor=True
                            ).to(self.device)
                    else:
                        batch_emb = model.encode(
                            batch,
                            batch_size=32,
                            show_progress_bar=False,
                            convert_to_tensor=True
                        ).to(self.device)
                model_batches.append(batch_emb)
            model_embeddings = torch.cat(model_batches, dim=0)
            embeddings_list.append(model_embeddings)

        # If only one model is used, optionally pass through the encoder.
        if self.single_model:
            if hasattr(self, 'encoder'):
                print("> Decoder in")
                encoded = self.encoder(embeddings_list[0])
                if self.truncate:
                    print(f"> Truncating up to {self.truncate}")
                    return F.normalize(encoded[:,:self.truncate], p=2, dim=1)
                return encoded
            return embeddings_list[0]

        concat = torch.cat(embeddings_list, dim=1)
        concat = F.normalize(concat, p=2, dim=1)

        if hasattr(self, 'encoder'):
            encoded = self.encoder(concat)
            if self.truncate:
                print(f"> Encoder + Truncating up to {self.truncate}")
                return F.normalize(encoded[:,:self.truncate], p=2, dim=1)
            if hasattr(self, 'quantizer'):
                return self.quantizer.quantize(encoded)
            print("> Concat + Encoder (no truncation) > returning:", encoded.shape)
            return encoded
        else:
            print(f"Concat with No encoder {concat.shape}")
            return concat

def main():
    if len(sys.argv) < 4:
        print("Usage: python eval.py <checkpoint> <truncate> <model_type> ...")
        sys.exit(1)

    ckpt = sys.argv[1]
    trunc = int(sys.argv[2])
    model_type = sys.argv[3]
    tsk = sys.argv[4]
    use_encoder = bool(int(sys.argv[5])) if len(sys.argv) > 5 else False
    random_tag = sys.argv[6]
    my_quant = sys.argv[7]
    use_quant = len(sys.argv[7]) > 1
    batch_size = int(sys.argv[8]) if len(sys.argv) > 8 else 512

    run_id = random_tag.split("-")[0]
    print("Id of the run ", run_id)
    inDim, outDim = 0, 0

    if use_encoder:
        print("we here")
        log_file = "logs.txt"
        with open(log_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts[1] == run_id:
                    inDim, outDim = int(parts[2]), int(parts[3])
                    print("we here", inDim, outDim)
                    break

    # Determine model keys based on model_type
    if model_type == "combined":
        model_keys = ["e5-small", "no-ins"]
    elif model_type == "all-33":
        model_keys = ["bge-small", "e5-small", "gist"]
    elif model_type == "all-4":
        model_keys = ["bge-small", "e5-small", "gist", "snowflake-m"]
    else:
        model_keys = [model_type]  # Single model

    # Create the AdaptiveSentenceTransformer *once*
    model = AdaptiveSentenceTransformer(
        models=model_keys,
        device="cuda",
        checkpoint_path=f"models_pth/{inDim}_{outDim}/{ckpt}.pth" if use_encoder else None,
        input_dim=inDim if use_encoder else None,
        compressed_dim=outDim if use_encoder else None,
        truncate=trunc if use_encoder else None,
        quantizer_path=my_quant if use_quant else None
    )

    output_folder = f"results/{random_tag}"

    tasks = mteb.get_tasks(tasks=[tsk])

    print(f"Eval on {tsk} Starting")
    
    evaluation = mteb.MTEB(
        tasks=tasks,
        eval_splits=["test"],
    )
    
    results = evaluation.run(
        model,
        output_folder=output_folder,
        eval_splits=["test"],
        verbosity = 1,
        encode_kwargs={
            "batch_size" : 512,
        }
    )
    
    return results


if __name__ == "__main__":
    main()
