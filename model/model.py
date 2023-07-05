import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .infogcn import InfoGCN
from .generator import Generator

class Predictor_Corrector(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        # Predictor parameters
        self.num_class = args.num_class
        self.num_point = args.num_point
        self.num_person = args.num_person
        self.graph = args.graph
        self.in_channels = args.in_channels
        self.drop_out = 0
        self.num_head = args.num_head
        self.noise_ratio = args.noise_ratio
        self.k= args.k
        self.gain= args.gain

        # Corrector parameters
        self.latent_dim= args.latent_dim
        self.out_channels= args.out_channels
        self.t_size= args.t_size

        self.predictor = InfoGCN(num_class=self.num_class, num_point=self.num_point, num_person=self.num_person, graph=self.graph, 
                                 in_channels=self.in_channels, drop_out=self.drop_out, num_head=self.num_head, noise_ratio=self.noise_ratio,
                                 k=self.k, gain=self.gain)
        
        self.corrector = Generator(in_channels=self.latent_dim, out_channels=self.out_channels, n_classes=self.num_class, t_size=self.t_size)

    def load_predictor(self, path):
        self.predictor.load_state_dict(torch.load(path))
        for param in self.predictor.parameters():
          param.requires_grad = False

    def load_corrector(self, path):
        self.corrector.load_state_dict(torch.load(path))
        for param in self.corrector.parameters():
          param.requires_grad = False
    def forward(self, x):
        y_hat, z = self.predictor(x)

        x_cor = self.corrector(z)
        x_cor = x_cor.unsqueeze(-1)
        # print(f'x_cor :{x_cor.shape}')
        y_hat_cor,_ = self.predictor(x_cor)

        return y_hat, y_hat_cor, x_cor

if __name__ == '__main__':
    
    model = Predictor_Corrector()