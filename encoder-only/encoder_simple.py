import torch.nn as nn
import torch.nn.functional as F


class EncoderConfig:
    DEFAULT = {
        'input_dim':  1920,
        'output_dim': 1024,
        'dropout': 0.1
    }

outDim = EncoderConfig.DEFAULT["output_dim"]

#COMPRESSED_DIMENSIONS = [64, outDim]   # low MRL  for 384

#COMPRESSED_DIMENSIONS = [64, 128, 256, 384, outDim]   

#COMPRESSED_DIMENSIONS = [64, 154, 256, 384, outDim]   # low MRL  for 384

#COMPRESSED_DIMENSIONS = [32, 64, 128, 205, outDim] # high MRL

COMPRESSED_DIMENSIONS = [32, 64, 80, 100, 120, 150, 160, 175, 180, 205, 256, 384, outDim]   # low MRL  for 384

#COMPRESSED_DIMENSIONS = [205, 256, 512, outDim]  # mid MRL

class EncoderOnly(nn.Module):
    def __init__(self, config: dict = None):
        super().__init__()
        cfg = config or EncoderConfig.DEFAULT
        
        self.encoder = nn.Sequential(
            nn.LayerNorm(cfg['input_dim']),
            nn.Linear(cfg['input_dim'], cfg['output_dim']),
            nn.BatchNorm1d(cfg['output_dim']),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.LayerNorm(cfg['output_dim']),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, dim):
        out = self.encoder(x)
        # L2 normalize output embeddings
        return F.normalize(out[:,:dim], p=2, dim=1)
    
    
