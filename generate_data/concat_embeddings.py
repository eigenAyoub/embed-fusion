#!/usr/bin/env python3
import numpy as np
import sys
import os

# Check if enough arguments are provided
if len(sys.argv) != 4:
    print("Usage: python script.py file1 file2 output_folder")
    sys.exit(1)

file1  = sys.argv[1]
file2  = sys.argv[2]
output = sys.argv[3]

# Load the training and validation embeddings
array1 = np.load(f"embeddings_data/{file1}/train_embeddings.npy")
array2 = np.load(f"embeddings_data/{file2}/train_embeddings.npy")
array3 = np.load(f"embeddings_data/{file1}/val_embeddings.npy")
array4 = np.load(f"embeddings_data/{file2}/val_embeddings.npy")

result_train = np.concatenate((array1, array2), axis=1)
result_val   = np.concatenate((array3, array4), axis=1)

def normalize_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1  
    return matrix / norms

result_train = normalize_rows(result_train)
result_val   = normalize_rows(result_val)

os.makedirs(output, exist_ok=True)

output_path_train = os.path.join(output, "train_embeddings.npy")
output_path_val   = os.path.join(output, "val_embeddings.npy")

np.save(output_path_train, result_train)
np.save(output_path_val, result_val)

print(f"Concatenated and normalized train embeddings saved to {output_path_train}")
print(f"Concatenated and normalized validation embeddings saved to {output_path_val}")
