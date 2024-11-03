from sentence_transformers import SentenceTransformer
import mteb 

import numpy as np
import torch

import sys 

models = {
            "mxbai"     : "mixedbread-ai/mxbai-embed-large-v1",
            "bge"       : "BAAI/bge-large-en-v1.5"                 ,
            "e5"        : "intfloat/e5-large-v2"              ,
            "snowflake" : "Snowflake/snowflake-arctic-embed-m",
            "snowflake-l" : "Snowflake/snowflake-arctic-embed-l",
            "gte-base"        : "thenlper/gte-base",
            "gte-large"       : "thenlper/gte-large",
            "gte-small"       : "thenlper/gte-small",
            "jina-v3"         : "jinaai/jina-embeddings-v3"
}

import torch.nn as nn


n0 = 768
n1 = 512
n2 = 384

class AutoEncoder(nn.Module):
    def __init__(self, input_dim=1, compressed_dim=1):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(1024, n0),
            nn.BatchNorm1d(n0),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(n0, n1),
            nn.BatchNorm1d(n1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(n1, n2),
            nn.BatchNorm1d(n2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(n2, 128),

        )

        # Decoder
        self.decoder = nn.Sequential(

            nn.Linear(128, n2),
            nn.BatchNorm1d(n2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(n2, n1),
            nn.BatchNorm1d(n1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(n1, n0),
            nn.BatchNorm1d(n0),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(n0, 1024),
        )

    def forward(self, x):
        compressed = self.encoder(x)
        reconstructed = self.decoder(compressed)
        return reconstructed, compressed

from typing import List, Union

class EmbedEncode:
    def __init__(
        self, 
        model: SentenceTransformer, 
        autoencoder_path: str = None,
        input_dim: int = 2048,  # Adjust based on your model's embedding size
        compressed_dim: int = 1024,
        device: str = "cuda"
    ):

        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        
        if autoencoder_path:
            self.use_autoencoder = True
            self.autoencoder = AutoEncoder(input_dim=input_dim, compressed_dim=compressed_dim)
            self.autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=self.device))
            self.autoencoder.to(self.device)
            self.autoencoder.eval()  # Set to evaluation mode
        else:
            self.use_autoencoder = False
            self.autoencoder = None

    def encode(
        self,
        sentences: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        device: str = None,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        normalize_embeddings: bool = False,
        use_autoencoder: bool = None,
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:

        """
        Encode sentences using the SentenceTransformer model and optionally pass through the autoencoder.

        Parameters:
        - sentences (List[str]): List of sentences to encode.
        - batch_size (int): Batch size for encoding.
        - show_progress_bar (bool): Whether to show a progress bar.
        - device (str, optional): Device to run encoding on. If None, uses the instance's device.
        - convert_to_numpy (bool): Whether to convert embeddings to NumPy arrays.
        - convert_to_tensor (bool): Whether to convert embeddings to PyTorch tensors.
        - normalize_embeddings (bool): Whether to normalize embeddings to unit length.
        - use_autoencoder (bool, optional): Whether to use the autoencoder. 
          If None, defaults to the instance's `use_autoencoder` flag.
        - **kwargs: Additional keyword arguments for the `encode` method of SentenceTransformer.

        Returns:
        - Embeddings as a NumPy array or PyTorch tensor.
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
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_kwargs}

        # Optionally, log or handle unexpected kwargs
        unexpected_kwargs = set(kwargs.keys()) - allowed_kwargs
        if unexpected_kwargs:
            print(f"Warning: Ignoring unexpected keyword arguments: {unexpected_kwargs}")

        
        current_device = device if device else self.device

        # Encode sentences using the SentenceTransformer model
        embeddings = self.model.encode(
            sentences, 
            batch_size=batch_size, 
            show_progress_bar=show_progress_bar,
            device=current_device,
            convert_to_numpy  = True,  
            normalize_embeddings=normalize_embeddings,
            **filtered_kwargs  
        )
        
        # Ensure embeddings are on the correct device ## not sure how is this helpful.
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings, dtype=torch.float32).to(current_device)
        else:
            embeddings = embeddings.to(current_device)

        # is this useful, why not just if use_autoenc is not None:
        apply_autoencoder = use_autoencoder if use_autoencoder is not None else self.use_autoencoder
        if apply_autoencoder and self.autoencoder:
            with torch.no_grad():
                compressed_embeddings = self.autoencoder.encoder(embeddings)
            embeddings_to_return = compressed_embeddings
        else:
            embeddings_to_return = embeddings

        if embeddings_to_return.dtype == torch.bfloat16:
            embeddings_to_return = embeddings_to_return.to(torch.float32)

        # Convert to desired format
        if convert_to_numpy:
            embeddings_to_return = embeddings_to_return.cpu().numpy()
        elif convert_to_tensor:
            embeddings_to_return = embeddings_to_return
        else:
            # Default: keep as tensor
            embeddings_to_return = embeddings_to_return

        return embeddings_to_return

    def to(self, device: str):
        self.device = device
        self.model.to(device)
        if self.use_autoencoder and self.autoencoder:
            self.autoencoder.to(device)



from transformers import AutoModel

model = SentenceTransformer(models["jina-v3"],
                                  trust_remote_code=True,
                                  ).to("cuda")


if len(sys.argv) > 3:
    INPUT_DIM       = int(sys.argv[1])
    COMPRESSED_DIM  = int(sys.argv[2])
    CHECKPOINT_PATH = sys.argv[3]

autoencoder_path_ = "models_pth/2048_1024/000504.pth"

combined_model = EmbedEncode(
    model=model,
    autoencoder_path=autoencoder_path_,  
    input_dim=1024,  
    compressed_dim=128,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

eval_ = True 
if eval_:
    tasks = mteb.get_tasks(tasks=["NFCorpus"]) 
    evaluation = mteb.MTEB(tasks=tasks, eval_splits=["test"], metric="ndcg@10")
    results = evaluation.run(combined_model, 
                             output_folder = f"results/jina_dim_128_504",
                             batch_size = 16
                             )
