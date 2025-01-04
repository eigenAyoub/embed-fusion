## Options for AE; Padding and Mask
## I stopped using them?

import torch.nn as nn
import torch

from config import INPUT_DIM, COMPRESSED_DIM

input_dim = INPUT_DIM 
compressed_dim = COMPRESSED_DIM 



class SharedDecoderAutoEncoder(nn.Module):
    def __init__(self, input_dim=2048, encoder_hidden1=1024, encoder_hidden2=512, 
                 decoder_hidden1=1024, decoder_hidden2=2048, max_compressed_dim=512):
        super(SharedDecoderAutoEncoder, self).__init__()
        
        self.max_compressed_dim = max_compressed_dim
        self.encoder_output_dim = encoder_hidden2  # 512
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoder_hidden1),
            nn.BatchNorm1d(encoder_hidden1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(encoder_hidden1, encoder_hidden2),
            nn.BatchNorm1d(encoder_hidden2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(encoder_hidden2, max_compressed_dim),
        )
        
        # Shared Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.max_compressed_dim, encoder_hidden2),
            nn.BatchNorm1d(encoder_hidden2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(encoder_hidden2, encoder_hidden1),
            nn.BatchNorm1d(encoder_hidden1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(decoder_hidden1, input_dim),
        )
        
    def forward(self, x, compressed_dim):
        """
        Forward pass with padding.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            compressed_dim (int): The target compressed dimension (64, 128, 256, 384)

        Returns:
            torch.Tensor: Reconstructed output of shape (batch_size, input_dim)
        """
        encoded = self.encoder(x)  # Shape: (batch_size, 512)
        
        # Extract the first 'compressed_dim' dimensions
        subset = encoded[:, :compressed_dim]  # Shape: (batch_size, compressed_dim)
        
        # Calculate padding size
        padding_size = self.max_compressed_dim - compressed_dim
        
        if padding_size > 0:
            padding = torch.zeros((subset.size(0), padding_size), device=subset.device)
            subset_padded = torch.cat([subset, padding], dim=1)  # Shape: (batch_size, 512)
        else:
            subset_padded = subset  # No padding needed if compressed_dim == max_compressed_dim
        
        # Decode the padded subset
        decoded = self.decoder(subset_padded)  # Shape: (batch_size, 2048)
        return decoded

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)



class SharedDecoderAutoEncoderWithMask(nn.Module):
    def __init__(
        self, 
        input_dim=2048, 
        encoder_hidden1=1024, 
        encoder_hidden2=768, 
        encoder_hidden3=512, 
        decoder_hidden1=768, 
        decoder_hidden2=1024, 
        decoder_hidden3=2048, 
        max_compressed_dim=512
    ):
        """
        Initializes the Shared Decoder AutoEncoder with Masking.

        Args:
            input_dim (int): Dimensionality of the input data.
            encoder_hidden1 (int): Number of neurons in the first encoder layer.
            encoder_hidden2 (int): Number of neurons in the second encoder layer.
            encoder_hidden3 (int): Number of neurons in the third encoder layer.
            decoder_hidden1 (int): Number of neurons in the first decoder layer.
            decoder_hidden2 (int): Number of neurons in the second decoder layer.
            decoder_hidden3 (int): Number of neurons in the third decoder layer.
            max_compressed_dim (int): Maximum size for compressed representations.
        """
        super(SharedDecoderAutoEncoderWithMask, self).__init__()
        
        self.max_compressed_dim = max_compressed_dim
        self.encoder_output_dim = encoder_hidden3  # 512
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoder_hidden1),
            nn.BatchNorm1d(encoder_hidden1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(encoder_hidden1, encoder_hidden2),
            nn.BatchNorm1d(encoder_hidden2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(encoder_hidden2, encoder_hidden3),
            nn.BatchNorm1d(encoder_hidden3),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(encoder_hidden3, max_compressed_dim),
        )
        
        # Decoder with Masking
        # Input size is doubled (512 compressed + 512 mask = 1024)
        self.decoder = nn.Sequential(
            nn.Linear(self.max_compressed_dim * 2, decoder_hidden1),
            nn.BatchNorm1d(decoder_hidden1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(decoder_hidden1, decoder_hidden2),
            nn.BatchNorm1d(decoder_hidden2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(decoder_hidden2, decoder_hidden3),
            nn.BatchNorm1d(decoder_hidden3),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(decoder_hidden3, input_dim),
        )
        
    def forward(self, x, compressed_dim):
        """
        Forward pass with masking.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            compressed_dim (int): The target compressed dimension (e.g., 64, 128, 256, 384, 512)

        Returns:
            torch.Tensor: Reconstructed output of shape (batch_size, input_dim)
        """
        # Encode the input
        encoded = self.encoder(x)  # Shape: (batch_size, max_compressed_dim)
        
        # Extract the first 'compressed_dim' dimensions
        subset = encoded[:, :compressed_dim]  # Shape: (batch_size, compressed_dim)
        
        # Calculate padding size
        padding_size = self.max_compressed_dim - compressed_dim
        
        if padding_size > 0:
            # Create a padding tensor of zeros
            padding = torch.zeros((subset.size(0), padding_size), device=subset.device)
            # Concatenate the subset with padding
            subset_padded = torch.cat([subset, padding], dim=1)  # Shape: (batch_size, max_compressed_dim)
            # Create the mask: 1s for actual data, 0s for padding
            mask = torch.cat([
                torch.ones((subset.size(0), compressed_dim), device=subset.device),
                torch.zeros((subset.size(0), padding_size), device=subset.device)
            ], dim=1)  # Shape: (batch_size, max_compressed_dim)
        else:
            subset_padded = subset  # Shape: (batch_size, max_compressed_dim)
            mask = torch.ones((subset.size(0), self.max_compressed_dim), device=subset.device)  # Shape: (batch_size, max_compressed_dim)
        
        # Concatenate the masked encoded vector to provide masking information
        # This results in a tensor of shape (batch_size, 2 * max_compressed_dim)
        decoder_input = torch.cat([subset_padded, mask], dim=1)
        
        # Decode the concatenated tensor
        decoded = self.decoder(decoder_input)  # Shape: (batch_size, input_dim)
        
        return decoded
