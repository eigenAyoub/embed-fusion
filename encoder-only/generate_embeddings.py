from eval import AdaptiveSentenceTransformer
from encoder_simple import EncoderOnly, EncoderConfig
import numpy as np
import torch
import sys


from offline_quant import PerColumnQuantizer


if __name__ == "__main__":
    
    ckpt = "1920_1024_8_ep_030"
    model_type = "all-4" 
    #tsk  = "NFCorpus"

    random_tag = "quant8" 
    
    inDim  = 1920
    outDim = 1024
    trunc = 1024 

    data_path = "../generate_data/embeddings_data/all_4_val.npy"
    data = np.load("../generate_data/embeddings_data/all_4_val.npy")

    data = torch.tensor(data).to("cuda")

    print(data.shape)
    
    
    model_keys = ["bge-small", "e5-small", "gist", "snowflake-m"]

    checkpoint_path=f"models_pth/{inDim}_{outDim}/{ckpt}.pth" 

    encoder = EncoderOnly(EncoderConfig.DEFAULT).to("cuda")
    encoder.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
    
    output_data = encoder(data, 1024).to("cuda")
   
    
    print("About to fit") 
    quantizer =PerColumnQuantizer(num_bits=4, device = "cuda")
    quantizer.fit(output_data)
    
    torch.save({
        "num_bits": quantizer.num_bits,
        "min_vals": quantizer.min_vals,
        "scales": quantizer.scales,
        "lut": quantizer.lut,
    }, "quantizer_1920_1024_4_bits.pth")
    
   
    bnb = quantizer.quantize(output_data)
    bnb_back = quantizer.dequantize(bnb)

    #output_folder = f"results/{random_tag}"
