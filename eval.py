from sentence_transformers import SentenceTransformer
import mteb
import datasets
from typing import List, Union, Dict, Optional

from mteb.encoder_interface import Encoder, PromptType

from model import EncoderOnly  
from LSH import  TheSimpleNet, TheSigNet, TheTanNet

from offline_quant import PerColumnQuantizer  

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


class AdaptiveSentenceTransformer(Encoder):
    MODEL_CATALOGUE: Dict[str, str] = {
        "mxbai": "mixedbread-ai/mxbai-embed-large-v1",
        "e5": "intfloat/e5-large-v2",
        "bge": "BAAI/bge-large-en-v1.5",
        "snowflake-l": "Snowflake/snowflake-arctic-embed-l",
        "snowflake-m": "Snowflake/snowflake-arctic-embed-m-v1.5",
        "gte-base": "thenlper/gte-base",
        "gte-large": "thenlper/gte-large",
        "gte-small": "thenlper/gte-small",
        "jina-v3": "jinaai/jina-embeddings-v3",
        "e5-small": "intfloat/e5-small-v2",
        "bge-small": "BAAI/bge-small-en-v1.5",
        "gist": "avsolatorio/GIST-small-Embedding-v0",
        "linq": "Linq-AI-Research/Linq-Embed-Mistral",
        "no-ins": "avsolatorio/NoInstruct-small-Embedding-v0",
        "infly":"infly/inf-retriever-v1-1.5b"
    }

    def __init__(self,
                 models: List[str],
                 device: str = 'cuda',
                 checkpoint_path: Optional[str] = None,
                 quantizer_path: Optional[str] = None,
                 input_dim: Optional[int] = None,
                 compressed_dim: Optional[int] = None,
                 truncate: Optional[int] = None,
                 use_lsh: Optional[int] = 0,
                 lsh_info: Optional[tuple] = [None, None]
                 
                 ):

        self.device = device
        self.models: List[Union[SentenceTransformer, AutoModel]] = []
        self.tokenizers: List[Union[None, AutoTokenizer]] = []
        self.model_keys: List[str] = []
        self.truncate = truncate
        self.use_LSH = use_lsh

        for model_key in models:
            model_path = self.MODEL_CATALOGUE.get(model_key)

            if model_path is None:
                raise ValueError(f"Model '{model_key}' not found.")

            if model_key.startswith("e5"):
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModel.from_pretrained(model_path).to(device)
                #MODEL_CACHE[model_path] = (model, tokenizer)
            elif model_key == "no-ins":
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModel.from_pretrained(model_path).to(device)
                #MODEL_CACHE[model_path] = (model, tokenizer)
            else:
                if model_key == "mxbai":
                    ddd = 512
                    print(f"We truncating {model_key} to {ddd} dimensions")
                    model = SentenceTransformer(model_path, trust_remote_code=True, truncate_dim=ddd).to(device)
                else:
                    model = SentenceTransformer(model_path, trust_remote_code=True).to(device)
                tokenizer = None
                if model_key == "infly":
                    model.max_seq_length = 512 

            model.eval()
            #model, tokenizer = MODEL_CACHE[model_path]
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
       
        if self.use_LSH:
            print("Oupsie, guess we're here, LSH the fuck")
            #self.LSH = TheSigNet(768, 4096).to(device)
            self.LST_t = torch.load("thresh.pth", map_location=device)
            print(f"here is the goat threash {self.LST_t}")
            self.LSH_epoch  = lsh_info[0]
            self.LSH_run_id = lsh_info[1] 
            #self.LSH_ckpt = f"checks/checkpoint_epoch_{self.LSH_epoch}_{self.LSH_run_id}.pth"
            self.LSH = TheSimpleNet(768, 4096).to(device)
            self.LSH_ckpt = f"chillDude.pth"
            self.LSH.load_state_dict(torch.load(self.LSH_ckpt, map_location=device))


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
            for start in tqdm(range(0, len(sentences), batch_size), desc=f"Processing {model_key} batches", leave=False):
                batch = sentences[start:start + batch_size]
                if model_key.startswith("e5"):
                    batch_emb = self._hf_encode(model, tokenizer, batch, **kwargs)
                elif model_key == "no-ins":
                    batch_emb = self._noins_encode(model, tokenizer, batch, **kwargs)
                else:  # dealing with SentenceTransformer stuff
                    prompt_name = kwargs.get("prompt_type", None)
                    if prompt_name == PromptType.query:
                        if model_key in ["mxbai", "infly"] or "snowflake" in model_key:
                            print(f"> Encoding: {model_key}")
                            if model_key == "infly":
                                batch_size = 4
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
                        else:  # other sentenceTransformer models that do not rely on a prompt.
                            print(f"> batch_size {batch_size}")
                            batch_emb = model.encode(
                                batch,
                                batch_size=batch_size,
                                show_progress_bar=False,
                                convert_to_tensor=True
                            ).to(self.device)
                    else:  # for the documents encoding.
                        batch_emb = model.encode(
                            batch,
                            batch_size=batch_size,
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
            embeddings = embeddings_list[0]
            print(f"This is a single model, with no encoder, returned shape {embeddings.shape}")
            return embeddings 

        concat = torch.cat(embeddings_list, dim=1)
        concat = F.normalize(concat, p=2, dim=1)
        
        if self.use_LSH: 
            print(f"The LSH game is on! with fixed threshold {self.LST_t}")
            activations = self.LSH.encode(concat)
            what = (activations < self.LST_t).float()
            print(f"Returninig the activations {activations}")
            print(f"After applying {what}, shape {what.shape}")
            return what
            #concat = self.LSH(concat)
            #print(f"The sig output of lsh is \n{concat}")
            #concat = (concat < self.LST_t).float()
            return concat
        
        if hasattr(self, 'encoder'):
            encoded = self.encoder(concat)
            print(f"Encoder output shape: {encoded.shape}")
            if self.truncate:
                enc_trunc = F.normalize(encoded[:,:self.truncate], p=2, dim=1)
                print(f"> Encoder + Truncating up to {self.truncate} of encoded of shape {enc_trunc.shape}")
                return enc_trunc 
            if hasattr(self, 'quantizer'):
                return self.quantizer.quantize(encoded)
            print("> Concat + Encoder (no truncation) > returning:", encoded.shape)
            return encoded
        else:
            print(f"Concat with No encoder {concat.shape}")
            return concat

def main():
        
    
    print("Usage: python eval.py <model> <task> <use_enc> <ckpt> <trunc>  <use_quant> <quant> <tag>")
    # To eval a single model: 
    # python eval.py e5-small NFCorpus 0 x 0 0 x e5-small-metrics")

    # model_type + task
    model_type = sys.argv[1]
    tsk = sys.argv[2]

    # decoder # set to `0 x` if not needed.
    
    # mode:  cuz i suck at coding.
    # 0 nothing, just model itself.
    # 1 model + encoder, just model itself.b
    # 2 model + encoder + quantizer, just model itself.b
    # 3 model + LSH.

    use_encoder = bool(int(sys.argv[3])) 
    ckpt = sys.argv[4] if use_encoder else None

    # trunc
    trunc = int(sys.argv[5])   # no far it is only accounted if the encoder is in.

    # quant
    use_quant = bool(int(sys.argv[6]))
    my_quant = sys.argv[7] if use_quant else None

    # name tage for resutls
    random_tag = sys.argv[8]
   
    use_lsh = 0 
    if len(sys.argv) > 9:
        use_lsh = 1
        lsh_epoch = sys.argv[9]
        lsh_ckpt  = sys.argv[10]
        print(f"Using lsh with run-Id {lsh_ckpt}, and epoch {lsh_epoch}.")
        # please no more of this fucking interface.
    
    print(f"Encoder ? {use_encoder}")
    print(f"Quant? {use_quant}")
    print(f"LSH? {use_lsh}")

    inDim, outDim = 0, 0

    if use_encoder:
        run_id = random_tag.split("-")[0]
        print("Id of the run ", run_id)
        print("we here")
        log_file = "logs.txt"
        with open(log_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts[1] == run_id:
                    inDim, outDim = int(parts[2]), int(parts[3])
                    print("we here", inDim, outDim)
                    break

    if model_type == "combined-s":
        model_keys = ["no-ins", "gte-small"]
    elif model_type == "combined-l":
        #model_keys = ["e5", "mxbai"]
        model_keys =  ["mxbai", "no-ins"]
    elif model_type == "three-33M":
        model_keys = ["e5-small", "no-ins", "gte-small"]
    elif model_type == "all-4":
        model_keys = ["bge-small", "e5-small", "gist", "snowflake-m"]
    else:
        model_keys = [model_type]  # Single model

    print(f"Model keys > {model_keys}")

    # Create the AdaptiveSentenceTransformer *once*
    model = AdaptiveSentenceTransformer(
        models=model_keys,
        device="cuda",
        checkpoint_path=f"models_pth/{inDim}_{outDim}/{ckpt}.pth" if use_encoder else None,
        input_dim=inDim if use_encoder else None,
        compressed_dim=outDim if use_encoder else None,
        truncate=trunc if use_encoder else None,
        quantizer_path=my_quant if use_quant else None,
        use_lsh=use_lsh,
        lsh_info = [lsh_epoch,lsh_ckpt] if use_lsh else None
    )

    output_folder = f"results/{random_tag}"
    
    print(random_tag)

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
