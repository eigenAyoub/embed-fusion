import numpy as np

tag = "val_embeddings.npy"

data1 = np.load(f"bge-small/{tag}")  # shape: (N, n_features1)
data2 = np.load(f"e5-small-v2_wiki_500k/{tag}")  # shape: (N, n_features2)
data3 = np.load(f"GIST-small-Embedding-v0_wiki_500k/{tag}")  # shape: (N, n_features3)

print(data1.shape)
print(data2.shape)
print(data3.shape)

concatenated = np.concatenate((data1, data2, data3), axis=1)  

print(f"shape after concat >> {concatenated.shape}")

row_norms = np.linalg.norm(concatenated, ord=2, axis=1, keepdims=True)
row_norms[row_norms == 0] = 1
normalized = concatenated / row_norms

np.save(f"all_three_val.npy", normalized)
