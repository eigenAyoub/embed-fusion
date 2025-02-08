from sentence_transformers import SentenceTransformer
import mteb
from typing import List, Union, Dict, Any, Optional

from mteb.encoder_interface import Encoder, PromptType

from encoder_simple import EncoderOnly, EncoderConfig
from offline_quant import PerColumnQuantizer

import torch
import torch.nn.functional as F
import numpy as np

import sys

model_config = EncoderConfig.DEFAULT

inDim = model_config["input_dim"]
outDim = model_config["output_dim"]

#rand_indices = torch.sort(torch.randperm(1920)[:384]).values
#print("We are going to take these indices ", rand_indices)
#print("Main, we are going to take these indices ", len(rand_indices))

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
        "gist":"avsolatorio/GIST-small-Embedding-v0"
    }

    def __init__(self, 
                 models: List[Union[str, SentenceTransformer]], 
                 device: str = 'cuda',
                 checkpoint_path: Optional[str] = None,
                 quantizer_path: Optional[str] = None,
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
            print(">> Training mode > ", self.encoder.training)
            self.encoder.eval()
            print(">> Training mode > ", self.encoder.training)
        else:
            print(f"No encoder initialization - will use {'single model' if self.single_model else 'raw concatenation'}")

        if quantizer_path is not None:
            quant_p = torch.load(quantizer_path)
            self.quantizer = PerColumnQuantizer(num_bits=quant_p["num_bits"], device=device)
            self.quantizer.min_vals = quant_p["min_vals"].to(device)
            self.quantizer.scales   = quant_p["scales"].to(device)
            self.quantizer.lut      = quant_p["lut"].to(device)

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
            model_kwargs["normalize_embeddings"]=False
            print(">> model_kwargs[ normalize_embeddings ] = ", model_kwargs["normalize_embeddings"])

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

        if self.single_model :
            print(f">>> Operating on a single model, len is {len(embeddings_list)}")
            if hasattr(self, 'encoder'):
                encoded = self.encoder(embeddings_list[0].to(self.device), 
                                    dim=self.truncate)
                print(f">>> Returning the encoder tensor shape: {encoded.shape}")
                return encoded
            else:
                print(">>> Are we truncating this simple model? to what degree", self.truncate)
                #print(">>> with these rands ", rand_indices)
                #encoded = embeddings_list[0][:,rand_indices]
                encoded = embeddings_list[0][:,:self.truncate]
                print(">>> Returning tensor with size > ", encoded.shape)
                return encoded
        else:
            if all(isinstance(e, torch.Tensor) for e in embeddings_list):
                concat = torch.cat(embeddings_list, dim=1)
                concat = F.normalize(concat, p=2, dim=1)
                # TODO: check this ASAP
                # do you normalize the embeddings? when you concat them?
                print(f">>> Concatenated tensor shape, it is normalized per column as well {concat.shape}")
                if hasattr(self, 'encoder'):
#                    encoded = self.encoder(concat.to(self.device), 
#                                        dim=self.truncate)
                    encoded = self.encoder(concat.to(self.device), 
                                        dim=self.truncate)
                    print(f">>> Returning the encoder tensor shape: {encoded.shape}")
                    if hasattr(self, 'quantizer'):
                        print(">> we are using the quant")
                        print(">> orig > ", encoded[0][:100])
                        q_encoded = self.quantizer.quantize(encoded)
                        # Dequantize using the lookup table.
                        print(">> quant > ", q_encoded[0][:100])
                        deq_encoded = self.quantizer.dequantize(q_encoded)
                        print(">> de-quant > ", deq_encoded[0][:100])
                        #deq_trunc = deq_encoded[:,:384]
                        print("returning this size ", deq_encoded.shape)
                        return deq_encoded 
                    return encoded
                else:
                    print(f"Concat with No encoder, input shape now: ", concat.shape, type(concat))
                    #print(f"About to randomly sample {outDim} features, {rand_indices}") 
                    #concat = concat[:,rand_indices] 
                    #print(f"returned shape after concat then random sampling ", concat.shape)
                    return concat 

            elif all(isinstance(e, np.ndarray) for e in embeddings_list):
                # here you don't concat // but up there you do
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
    model_type = sys.argv[3] # `combined` oder `model_name` (single model)
    tsk  = sys.argv[4]
    use_encoder = bool(int(sys.argv[5])) if len(sys.argv) > 5 else False
    random_tag = sys.argv[6] 
    my_quant = sys.argv[7] 
    use_quant = len(sys.argv[7])>1
    
    print("Are you using a quant > ", use_quant) 
    if use_encoder:
        print(f"Reading checkpoint from: models_pth/{inDim}_{outDim}/{ckpt}.pth")

    # Configure model based on type
    if model_type == "combined":
        #model_keys = ["bge-small", "snowflake-m"]
        #model_keys = ["gist", "snowflake-m"]
        model_keys = ["e5-small", "snowflake-m"]
        print(f"We are concatenating {model_keys[0]} and {model_keys[1]}")
        print(f"Are we using an encoder > {use_encoder}")

        model = AdaptiveSentenceTransformer(
            models=model_keys,
            device="cuda",
            checkpoint_path=f"models_pth/{inDim}_{outDim}/{ckpt}.pth" if use_encoder else None,
            input_dim=inDim if use_encoder else None,
            compressed_dim=outDim if use_encoder else None,
            truncate=trunc if use_encoder else None
        )
        output_folder = f"results/{random_tag}"
    elif model_type == "all-33":
        model_keys = ["bge-small", "e5-small", "gist"]
        print("Are we using an encoder:", use_encoder)
        model = AdaptiveSentenceTransformer(
            models=model_keys,
            device="cuda",
            checkpoint_path=f"models_pth/{inDim}_{outDim}/{ckpt}.pth" if use_encoder else None,
            input_dim=1152 if use_encoder else None,
            compressed_dim=768 if use_encoder else None,
            truncate=trunc if use_encoder else None
        )
        output_folder = f"results/{random_tag}"
    elif model_type == "all-4":
        model_keys = ["bge-small", "e5-small", "gist", "snowflake-m"]
        print("Are we using an encoder with 4 models", use_encoder)
        model = AdaptiveSentenceTransformer(
            models=model_keys,
            device="cuda",
            checkpoint_path=f"models_pth/{inDim}_{outDim}/{ckpt}.pth" if use_encoder else None,
            input_dim=1152 if use_encoder else None,
            compressed_dim=33 if use_encoder else None,
            truncate=trunc if use_encoder else None,
            quantizer_path=my_quant if use_quant else None
        )
        output_folder = f"results/{random_tag}"
    else:
        print("Are we using a single model with Encoder?", use_encoder)
        if use_encoder:
            model = AdaptiveSentenceTransformer(
                models=[model_type],
                device="cuda",
                checkpoint_path=f"models_pth/{inDim}_{outDim}/{ckpt}.pth" if use_encoder else None,
                input_dim=inDim if use_encoder else None,
                compressed_dim=outDim if use_encoder else None,
                truncate=trunc if use_encoder else None
            )
            output_folder = f"results/{random_tag}"
        else:
            model = AdaptiveSentenceTransformer(
                models=[model_type],
                device="cuda",
                truncate=trunc if trunc != 0 else None
            )
            print("there we go")
            output_folder = f"results/{random_tag}"
    
    # Setup evaluation
    #tasks = mteb.get_tasks(tasks=["NFCorpus", "SciFact", "ArguAna"])

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
        batch_size=128
    )
    
    return results

if __name__ == "__main__":
    main()

## to just test bge-small:
# python eval.py 1152_784_ep_020 784 combined xx 1 e5-arctic-norm-20
# python eval.py x 384 bge-small NFCorpus 0 bge-small-only-nfc
