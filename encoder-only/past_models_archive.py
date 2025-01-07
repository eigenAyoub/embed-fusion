import torch.nn as nn
import torch.nn.functional as F

class EncoderConfigOne:
    DEFAULT = {
        'input_dim': 1152,
        'hidden_dim': 768,
        'output_dim': 512,
        'dropout': 0.1
    }

class EncoderOnlyOne(nn.Module):
    def __init__(self, config: dict = None):
        super().__init__()
        cfg = config or EncoderConfigOne.DEFAULT
        
        self.encoder = nn.Sequential(
            nn.Linear(cfg['input_dim'], cfg['hidden_dim']),
            nn.BatchNorm1d(cfg['hidden_dim']),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(cfg['dropout']),
            
            nn.Linear(cfg['hidden_dim'], cfg['output_dim']),
            nn.BatchNorm1d(cfg['output_dim']),
            nn.LeakyReLU(0.2, inplace=True)
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
        return self.encoder(x)[:,:dim]

class EncoderConfigSimple:
    DEFAULT = {
        'input_dim': 1152,
        'output_dim': 512,
        'dropout': 0.1
    }

class EncoderOnlySimple(nn.Module):
    def __init__(self, config: dict = None):
        super().__init__()
        cfg = config or EncoderConfigSimple.DEFAULT
        
        self.encoder = nn.Sequential(
            nn.Linear(cfg['input_dim'], cfg['output_dim']),
            nn.BatchNorm1d(cfg['output_dim']),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(cfg['dropout'])
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
        return self.encoder(x)[:,:dim]
 

class EncoderConfigDeep:
    DEFAULT = {
        'input_dim': 1152,
        'mid_dim': 768,      # Added middle dimension
        'output_dim': 512,
        'dropout': 0.1
    }

class EncoderOnlyDeep(nn.Module):
    def __init__(self, config: dict = None):
        super().__init__()
        cfg = config or EncoderConfig.DEFAULT
        
        self.encoder = nn.Sequential(
            nn.LayerNorm(cfg['input_dim']),
            nn.Linear(cfg['input_dim'], cfg['mid_dim']),
            nn.BatchNorm1d(cfg['mid_dim']),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(cfg['dropout']),
            
            nn.LayerNorm(cfg['mid_dim']),
            nn.Linear(cfg['mid_dim'], cfg['output_dim']),
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
        return F.normalize(out[:, :dim], p=2, dim=1)