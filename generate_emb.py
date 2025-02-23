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

    num_test_lines= 26_325_566
    embedding_dim = 1024 
    batch_size = 8192 
    dtype = "uint8"
    memmap_file = 'emb/embeddings_test.npy'
    embeddings = np.memmap(memmap_file, dtype=dtype, mode='w+', shape=(num_test_lines, embedding_dim))

    line_counter = 0  # Tracks how many embeddings have been written
    batch_texts = []  # Buffer for the current batch

    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, total=num_test_lines, desc="Processing lines")):

            data = orjson.loads(line)
            text = data.get("text", "")
            batch_texts.append(text)

            if len(batch_texts) >= batch_size:
                with torch.no_grad():
                    emb_batch = model.encode(batch_texts)
                embeddings[line_counter:line_counter + emb_batch.shape[0]] = emb_batch.cpu().detach().numpy()
                line_counter += emb_batch.shape[0]
                batch_texts = []  # Reset the batch

    if batch_texts:
        emb_batch = model.encode(batch_texts)
        embeddings[line_counter:line_counter + emb_batch.shape[0]] = emb_batch
        line_counter += emb_batch.shape[0]

    embeddings.flush()

print(f"Finished writing {line_counter} embeddings to {memmap_file}.")

print("Loaded embeddings shape:", embeddings.shape) 
    
    
    
    
    
    
    
    
        
        #tsk  = "NFCorpus"


        #data_path = "../generate_data/embeddings_data/all_4_val.npy"
        #data = np.load("../generate_data/embeddings_data/all_4_val.npy")
        #data = torch.tensor(data).to("cuda")
        #print(data.shape)
        #encoder = EncoderOnly(EncoderConfig.DEFAULT).to("cuda")
        #encoder.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
        #output_data = encoder(data, 1024).to("cuda")
        #print("About to fit") 
        #quantizer.fit(output_data)
        #torch.save({
        #    "num_bits": quantizer.num_bits,
        #    "min_vals": quantizer.min_vals,
        #    "scales": quantizer.scales,
        #    "lut": quantizer.lut,
        #}, "quantizer_1920_1024_4_bits.pth")
        #if quantizer_path is not None:
        #    quant_p = torch.load(quantizer_path)
        #    quantizer = PerColumnQuantizer(num_bits=quant_p["num_bits"], device=device)
        #    quantizer.min_vals = quant_p["min_vals"].to(device)
        #    quantizer.scales   = quant_p["scales"].to(device)
        #    quantizer.lut      = quant_p["lut"].to(device)
        #bnb = quantizer.quantize(output_data)
        #bnb_back = quantizer.dequantize(bnb)

        #output_folder = f"results/{random_tag}"


