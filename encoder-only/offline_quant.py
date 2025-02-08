import torch
import torch.nn.functional as F


# o3-mini-high, you're such a legend!
# please don't replace me though!

class PerColumnQuantizer:
    def __init__(self, num_bits: int = 8, device: str = "cpu"):
        """
        Initializes the quantizer.
        Args:
          num_bits: Number of bits for quantization (e.g., 8 or 4).
          device: The device on which to store quantizer parameters ("cpu" or "cuda").
        """
        self.num_bits = num_bits
        self.device = device
        self.qmin = 0.0
        self.qmax = 2 ** num_bits - 1.0
        self.min_vals = None   # Will be a (1, D) tensor.
        self.scales = None     # Will be a (1, D) tensor.
        self.lut = None        # Lookup table, shape: (D, 2**num_bits).

    def fit(self, calibration_data: torch.Tensor):
        """
        Offline calibration: Compute per-column min and scale values and create the LUT.
        Args:
          calibration_data: A tensor of shape (N, D) where N is the number of samples
                            and D is the embedding dimension.
        """
        # If calibration_data is not already on the target device, move it.
        if calibration_data.device != torch.device(self.device):
            calibration_data = calibration_data.to(self.device)
        
        # Compute per-column (feature) min and max.
        self.min_vals = calibration_data.min(dim=0, keepdim=True)[0]  # Shape: (1, D)
        max_vals = calibration_data.max(dim=0, keepdim=True)[0]         # Shape: (1, D)
        
        # Compute per-column scales (avoid division by zero).
        self.scales = (max_vals - self.min_vals) / (self.qmax - self.qmin)
        self.scales[self.scales == 0] = 1.0
        
        # Create the lookup table for de-quantization.
        self.lut = self.create_lookup_table()

    def create_lookup_table(self) -> torch.Tensor:
        """
        Create a lookup table (LUT) for de-quantization.
        Returns:
          A tensor of shape (D, 2**num_bits) where each row contains the de-quantized
          floating-point values corresponding to quantized integers [0, 2**num_bits - 1].
        """
        # Determine the number of features D.
        D = self.min_vals.shape[1]
        # Create a tensor with all possible quantized values.
        q_values = torch.arange(self.qmin, self.qmax + 1, dtype=torch.float32, device=self.device).unsqueeze(0)  # Shape: (1, 2**num_bits)
        # Reshape min_vals and scales for broadcasting.
        min_vals = self.min_vals.squeeze(0).unsqueeze(1)   # Shape: (D, 1)
        scales = self.scales.squeeze(0).unsqueeze(1)         # Shape: (D, 1)
        # Compute the LUT: for each feature d, LUT[d, q] = q * scale_d + min_d.
        lut = q_values * scales + min_vals  # Shape: (D, 2**num_bits)
        return lut

    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Quantize a tensor using the pre-computed per-column parameters.
        Args:
          tensor: A tensor of shape (N, D) to quantize.
        Returns:
          A torch.uint8 tensor of the same shape with quantized values.
        """
        # Ensure tensor is on the same device as the quantizer.
        tensor = tensor.to(self.device)
        q_tensor = ((tensor - self.min_vals) / self.scales).round().clamp(self.qmin, self.qmax).to(torch.uint8)
        return q_tensor

    def dequantize(self, q_tensor: torch.Tensor) -> torch.Tensor:
        """
        Dequantize a quantized tensor using the pre-computed lookup table.
        Args:
          q_tensor: A quantized tensor of shape (N, D) with integer values.
        Returns:
          A de-quantized tensor (float) of shape (N, D).
        """
        # Ensure q_tensor is on the correct device.
        q_tensor = q_tensor.to(self.device)
        N, D = q_tensor.shape
        # Use vectorized gathering to apply the LUT:
        # 1. Expand LUT to shape (N, D, 2**num_bits).
        lut_expanded = self.lut.unsqueeze(0).expand(N, -1, -1)
        # 2. Reshape q_tensor to shape (N, D, 1) for indexing.
        q_indices = q_tensor.unsqueeze(2).long()
        # 3. Gather the corresponding dequantized values and squeeze the last dimension.
        deq_tensor = torch.gather(lut_expanded, 2, q_indices).squeeze(2)
        return deq_tensor


class AdaptiveSentenceTransformer:
    def __init__(self, model, device: str = "cuda", quantizer: PerColumnQuantizer = None):
        """
        Initializes the encoder wrapper.
        Args:
          model: The underlying embedding model (e.g., a SentenceTransformer or custom model).
          device: The device to run the model on.
          quantizer: A pre-computed PerColumnQuantizer (with LUT) to use during encoding.
        """
        self.model = model
        self.device = device
        self.quantizer = quantizer

    def encode(self, sentences, **kwargs):
        """
        Encode sentences into embeddings, then quantize and dequantize them using the
        pre-computed GPU-friendly quantizer (and its LUT).
        Args:
          sentences: A string or list of strings to encode.
          kwargs: Additional keyword arguments passed to the underlying model.
        Returns:
          A floating-point tensor of embeddings (de-quantized) ready for downstream tasks.
        """
        # Get embeddings from the underlying model.
        embeddings = self.model.encode(sentences, **kwargs)
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings)
        embeddings = embeddings.to(self.device)

        if self.quantizer is None:
            raise ValueError("Quantizer not provided; please supply a fitted PerColumnQuantizer.")
        
        # Quantize the embeddings using the pre-computed per-column parameters.
        q_embeddings = self.quantizer.quantize(embeddings)
        # Dequantize using the lookup table.
        deq_embeddings = self.quantizer.dequantize(q_embeddings)
        return deq_embeddings

# ==============================================================================
# Dummy Model and Example Usage
# ==============================================================================

class DummyModel:
    """A dummy model that simulates an encode() method returning random embeddings."""
    def encode(self, sentences, **kwargs):
        if isinstance(sentences, str):
            sentences = [sentences]
        # For example, return random embeddings with dimension 1024.
        return torch.randn(len(sentences), 1024)

if __name__ == "__main__":
    # -------------------------------
    # Offline Calibration
    # -------------------------------
    # Assume you have a calibration dataset of embeddings with shape (N, D).
    N, D = 10000, 1024  # e.g., 10,000 samples, each 1024-dimensional.
    calibration_data = torch.randn(N, D)
    
    # Choose the device for quantizer operations (e.g., "cuda" for GPU).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Instantiate and fit the quantizer on the chosen device.
    quantizer = PerColumnQuantizer(num_bits=8, device=device)
    quantizer.fit(calibration_data)
    
    # Optionally, save the quantizer parameters for later use.
    torch.save({
        "num_bits": quantizer.num_bits,
        "min_vals": quantizer.min_vals,
        "scales": quantizer.scales,
        "lut": quantizer.lut,
    }, "quantizer_params.pth")
    print("Offline calibration complete on device:", device)
    
    # -------------------------------
    # Inference / Encoding
    # -------------------------------
    # Later, load the quantizer parameters (this happens during inference).
    params = torch.load("quantizer_params.pth")
    quantizer_loaded = PerColumnQuantizer(num_bits=params["num_bits"], device=device)
    quantizer_loaded.min_vals = params["min_vals"].to(device)
    quantizer_loaded.scales = params["scales"].to(device)
    quantizer_loaded.lut = params["lut"].to(device)
    
    # Create your underlying model (or load a pretrained one).
    dummy_model = DummyModel()
    
    # Instantiate the encoder with the pre-computed, GPU-friendly quantizer.
    encoder = AdaptiveSentenceTransformer(model=dummy_model, device=device, quantizer=quantizer_loaded)
    
    # Encode some example sentences.
    sentences = ["This is a test sentence.", "Another example sentence."]
    deq_embeddings = encoder.encode(sentences)
    print("Dequantized embeddings shape:", deq_embeddings.shape)


