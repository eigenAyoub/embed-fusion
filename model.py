import torch.nn as nn
import torch.nn.functional as F

class EncoderOnly(nn.Module):
    def __init__(self, cfg: dict = None):
        super().__init__()

        self.input_norm = nn.LayerNorm(cfg['input_dim'])

        self.hidden_layer = nn.Sequential(
            nn.Linear(cfg['input_dim'], 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.output_layer = nn.Sequential(
            nn.Linear(1024, cfg['output_dim']),
            nn.LayerNorm(cfg['output_dim']),
            #nn.LeakyReLU(0.2, inplace=True),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x_norm = self.input_norm(x)
        hidden = self.hidden_layer(x_norm)
        #output = self.output_layer(hidden) + x_norm
        output = self.output_layer(hidden) 
        return output #, hidden


class SimpleEncoder(nn.Module):
    def __init__(self, cfg: dict = None):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.LayerNorm(cfg['input_dim']),
            nn.Linear(cfg['input_dim'], cfg['output_dim']),
            nn.BatchNorm1d(cfg['output_dim']),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.encoder(x)
        return out 
    