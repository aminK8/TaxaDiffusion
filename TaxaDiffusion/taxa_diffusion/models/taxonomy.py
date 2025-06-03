import torch
import torch.nn as nn


class TaxonomyModel(nn.Module):
    def __init__(self, word_dim=768, nhead=8, level_number=7, num_layers_per_level=1):
        super(TaxonomyModel, self).__init__()
        self.level_number = level_number
        
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=word_dim, nhead=nhead, batch_first=True),
                num_layers=num_layers_per_level
            ) for _ in range(self.level_number)
        ])

    
    def forward(self, inputs, cut_off=0):
        query_word = self.encoder_layers[0](inputs[:, 0])

        for i in range(1, min(cut_off + 1, self.level_number)):
            query_word += self.encoder_layers[i](inputs[:, i])
        
        return query_word
