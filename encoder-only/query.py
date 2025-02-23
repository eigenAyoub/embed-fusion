from eval import AdaptiveSentenceTransformer
import numpy as np
import torch
from tqdm import tqdm
import orjson


if __name__ == "__main__":
    
    model_type = "all-4" 
    model_keys = ["bge-small", "e5-small", "gist", "snowflake-m"]
    use_encoder = 1 
    use_quant   = 1 
    random_tag = "quant8" 
    ckpt = "1920_1024_8_ep_030"
    inDim  = 1920
    outDim = 1024
    checkpoint_path=f"models_pth/{inDim}_{outDim}/{ckpt}.pth" 
    quantizer_path = "quantizer_1920_1024_8_bits.pth" 
    device = "cuda" 
    trunc  = 1024

    model = AdaptiveSentenceTransformer(
        models=model_keys,
        device="cuda",
        checkpoint_path=f"models_pth/{inDim}_{outDim}/{ckpt}.pth" if use_encoder else None,
        input_dim=1152 if use_encoder else None,
        compressed_dim=33 if use_encoder else None,
        truncate=trunc if use_encoder else None,
        quantizer_path= quantizer_path if use_quant else None
    )


    jsonl_file = '../../ML-prod/document_chunks.jsonl'

    #num_test_lines= 26_325_566
    num_test_lines= 6_780_000
    embedding_dim = 1024 
    batch_size = 8192 
    dtype = "uint8"
    memmap_file = 'emb/embeddings_test.npy'

    loaded_embeddings = np.memmap("emb/embeddings_test.npy", 
                                  dtype='uint8', mode='r', 
                                  shape=(num_test_lines, embedding_dim))
    
    
    query = "Barcelona Messi, greatest football player in the world!" 
    print(query)
    print(query[0])

    query = model.encode([query]).cpu().detach().numpy()[0]
    query_norm_sq = np.dot(query, query)  

    # Option 1: Using np.sum
    print("we here 1")
    embedding_norms_sq = np.sum(loaded_embeddings ** 2, axis=1)

    # Option 2: Using np.einsum (which can sometimes be more efficient)
    # embedding_norms_sq = np.einsum('ij,ij->i', embeddings, embeddings)

    print("we here 2")
    dot_products = loaded_embeddings.dot(query)  # Shape (N,)
    print("we here 3")
    squared_distances = embedding_norms_sq - 2 * dot_products + query_norm_sq
    print("we here 4")
    top_10_indices = np.argpartition(squared_distances, 5)[:5]
    print(top_10_indices)
    
    
    #query = model.encode([query]).cpu().detach().numpy()[0]
    #query_norm_sq = np.dot(query, query)  
    #print("we here 2")
    #dot_products = loaded_embeddings.dot(query)  # Shape (N,)
    #print("we here 3")
    #squared_distances = embedding_norms_sq - 2 * dot_products + query_norm_sq
    #print("we here 4")
    #top_10_indices = np.argpartition(squared_distances, 5)[:5]
    #print(top_10_indices)
    
        