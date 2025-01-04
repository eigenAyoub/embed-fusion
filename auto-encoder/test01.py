from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import torch.nn as nn

from config import MODEL_CATALOGUE, INPUT_DIM, COMPRESSED_DIM

class CombinedSentenceTransformer(nn.Module):
    def __init__(
        self, 
        models, 
        device: str = "cuda",
        
    ):
        super().__init__()

        self.models = models
        self.device = device

        for model in self.models:
            model.to(self.device)
            model.eval()
    
    def forward(self, x):
        pass

    def encode(
        self,
        sentences,
        batch_size=512,
        show_progress_bar=False,
        device="cuda",
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


        embeddings = []
        for idx, model in enumerate(self.models):
            emb = model.encode(
                sentences,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                **filtered_kwargs  # Pass only allowed kwargs
            )
            print(f"shiiiit {type(emb)}")
            
            embeddings.append(emb) 

        combined_embeddings = torch.cat(embeddings, dim=1)
        #combined_embeddings = np.concatenate(embeddings, axis=1)

        #embeddings_to_return = torch.tensor(combined_embeddings, dtype=torch.float32).to(device)

        #print(">>>>>>> size check before ", embeddings_to_return.shape)
        print(">>>>>>> size check before ", combined_embeddings.shape)
        
        return combined_embeddings 

    def to(self, device):
        self.device = device
        for model in self.models:
            model.to(device)


if __name__ == "__main__":

    model_names = [MODEL_CATALOGUE["snowflake-m"]]

    print(f"Models > {model_names}")

    main_models = [SentenceTransformer(nm, trust_remote_code=True).to("cuda") for nm in model_names]
    
    combined_model = CombinedSentenceTransformer(
        models=main_models,  
        device='cuda',
    )
   
    import mteb

    tasks = mteb.get_tasks(tasks=["NFCorpus"]) 

    evaluation = mteb.MTEB(tasks=tasks, eval_splits=["test"], metric="ndcg@10")
    results = evaluation.run(combined_model, 
                             output_folder = f"results/snow13",
                             batch_size = 128
                             )
    
