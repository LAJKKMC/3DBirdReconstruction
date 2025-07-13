import torch
import torch.nn as nn

class PositionalEncoding:
    def __init__(self, num_freqs=10):
        self.num_freqs = num_freqs
        self.freq_bands = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)

    def __call__(self, x):  # x: [N, 3]
        out = [x]
        for freq in self.freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, dim=-1)  # [N, 3 + 2 * 3 * num_freqs]

class TinyNeRF(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_enc = PositionalEncoding(10)
        self.mlp = nn.Sequential(
            nn.Linear(3 + 2 * 3 * 10, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 4)  # [density, R, G, B]
        )

    def forward(self, x):  # x: [N, 3]
        x = self.pos_enc(x)
        return self.mlp(x)