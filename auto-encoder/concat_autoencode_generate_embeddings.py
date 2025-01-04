from sentence_transformers import SentenceTransformer
from huggingface_hub import PyTorchModelHubMixin, HfApi

import numpy as np
import torch
import sys
import os

from model import AutoEncoder 

models = {
            "mxbai"     : "mixedbread-ai/mxbai-embed-large-v1",
            "bge"       : "BAAI/bge-large-en-v1.5",
            "e5"        : "intfloat/e5-large-v2"  ,
            "snowflake-l" : "Snowflake/snowflake-arctic-embed-l",
            "gte-base"        : "thenlper/gte-base",
            "gte-large"       : "thenlper/gte-large",
            "gte-small"       : "thenlper/gte-small",
            "e5-small"        : "intfloat/e5-small-v2", # (33M)
            "snowflake-m" : "Snowflake/snowflake-arctic-embed-m-v1.5",
            "bge-small"       : "BAAI/bge-small-en-v1.5" # (33M)
}

big   = ("e5", "mxbai")
small = ("e5-small", "bge-small")

nm1, nm2 = big

model_names = [models[nm1], models[nm2]]
main_models = [SentenceTransformer(nm).to("cuda") for nm in model_names]


INPUT_DIM       = int(sys.argv[1])
COMPRESSED_DIM  = int(sys.argv[2])

CHECKPOINT_PATH = None 

if len(sys.argv) > 3:
    CHECKPOINT_PATH = sys.argv[3]

import torch.nn as nn

class CombinedSentenceTransformer(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self, 
        models, 
        autoencoder_path: str = None,
        input_dim: int = 2048,  
        compressed_dim: int = 1024,
        device: str = "cuda",

        ## ? remove later
        repo_url: str = "your-repo-url",  
        pipeline_tag: str = "text-embedding",
        license: str = "mit",
        
    ):
        super().__init__()
        PyTorchModelHubMixin.__init__(self)
        ## ?  remove later

        self.models = models
        self.device = device
        for model in self.models:
            model.to(self.device)
            model.eval()
        
        if autoencoder_path:
            print(autoencoder_path)
            self.use_autoencoder = True
            self.autoencoder = AutoEncoder(input_dim=input_dim, compressed_dim=compressed_dim)
            self.autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=self.device))
            self.autoencoder.to(self.device)
            self.autoencoder.eval()  # Set to evaluation mode
        else:
            self.use_autoencoder = False
            self.autoencoder = None

    def forward(self, x):
        pass


    def encode(
        self,
        sentences,
        batch_size=512,
        show_progress_bar=False,
        device="cuda",
        convert_to_numpy=True,
        convert_to_tensor=False,
        normalize_embeddings=False,
        use_autoencoder: bool = None,
        **kwargs  # Capture all additional keyword arguments
    ):
        """
        Encode sentences by concatenating embeddings from all models.

        Parameters:
        - sentences (List[str]): List of sentences to encode. # I take pre-embedded sentences.
        - **kwargs: Additional keyword arguments. # this prevenets some errors, thanks o1

        Returns:
        - Combined embeddings as NumPy array or PyTorch tensor.
        """
        if device:
            self.to(device)

        # Filter out unexpected keyword arguments
        # Define allowed kwargs based on SentenceTransformer.encode's signature
        allowed_kwargs = {
            'convert_to_numpy',
            'convert_to_tensor',
            'normalize_embeddings',
            'output_value',  
			# Added based on SentenceTransformer.encode
            # Add other allowed kwargs if necessary
        }

        # Extract only allowed kwargs
        # Optionally, log or handle unexpected kwargs
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_kwargs}
        unexpected_kwargs = set(kwargs.keys()) - allowed_kwargs
        if unexpected_kwargs:
            print(f"Warning: Ignoring unexpected keyword arguments: {unexpected_kwargs}")

        # Encode with each model
        embeddings = [
            model.encode(
                sentences,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                device=device,
                convert_to_numpy=True,  # Always get NumPy for concatenation
                normalize_embeddings=normalize_embeddings,
                **filtered_kwargs  # Pass only allowed kwargs
            )
            for model in self.models
        ]

        combined_embeddings = np.concatenate(embeddings, axis=1)
        combined_tensor = torch.tensor(combined_embeddings, dtype=torch.float32).to(device)

        apply_autoencoder = use_autoencoder if use_autoencoder is not None else self.use_autoencoder
        if apply_autoencoder and self.autoencoder:
            with torch.no_grad():
                compressed_embeddings = self.autoencoder.encoder(combined_tensor)
            embeddings_to_return = compressed_embeddings
        else:
            embeddings_to_return = combined_tensor 

        # Convert to desired format
        if convert_to_numpy:
            embeddings_to_return = embeddings_to_return.cpu().numpy()
        elif convert_to_tensor:
            embeddings_to_return = embeddings_to_return
        else:
            embeddings_to_return = embeddings_to_return

        return embeddings_to_return

    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        
        for idx, model in enumerate(self.models):
            model_save_path = os.path.join(save_directory, f"sentence_transformer_{idx}")
            model.save(model_save_path)
        
        if self.use_autoencoder and self.autoencoder:
            torch.save(self.autoencoder.state_dict(), os.path.join(save_directory, "autoencoder.pth"))

    @classmethod
    def from_pretrained(cls, repo_id, **kwargs):
        repo_local_dir = Repository(local_dir=f"./{repo_id.split('/')[-1]}", clone_from=repo_id).local_dir
        state_dict = torch.load(os.path.join(repo_local_dir, "pytorch_model.bin"), map_location='cpu')
        model = cls(models=[], device='cpu')  # Initialize with empty models
        
        model.load_state_dict(state_dict)
        
        # Load SentenceTransformer models
        idx = 0
        model.models = []
        while True:
            model_path = os.path.join(repo_local_dir, f"sentence_transformer_{idx}")
            if os.path.exists(model_path):
                st_model = SentenceTransformer(model_path)
                model.models.append(st_model)
                idx += 1
            else:
                break
        
        # Load AutoEncoder if exists
        autoencoder_path = os.path.join(repo_local_dir, "autoencoder.pth")
        if os.path.exists(autoencoder_path):
            model.use_autoencoder = True
            model.autoencoder = AutoEncoder()
            model.autoencoder.load_state_dict(torch.load(autoencoder_path, map_location='cpu'))
        else:
            model.use_autoencoder = False
            model.autoencoder = None
        
        model.to(model.device)
        return model

    def to(self, device):
        self.device = device
        for model in self.models:
            model.to(device)
        if self.use_autoencoder and self.autoencoder:
            self.autoencoder.to(device)

autoencoder_path_ = None
if CHECKPOINT_PATH is not None:
    autoencoder_path_ = f'models_pth/{INPUT_DIM}_{COMPRESSED_DIM}/{CHECKPOINT_PATH}'


repo_url = "https://huggingface.co/benayad7/concat-e5-small-bge-small-01"

combined_model = CombinedSentenceTransformer(
    models=main_models,
    autoencoder_path=autoencoder_path_,  
    input_dim= INPUT_DIM,  
    compressed_dim= COMPRESSED_DIM,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    #repo_url=repo_url,
    #pipeline_tag="text-embedding",
    #license="mit",
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
combined_model.to(device)

sentence = ["convert_to_numpy (bool): Whether to convert embeddings to NumPy arrays."]


#save_directory = "test-01"
#combined_model.save_pretrained(save_directory)



