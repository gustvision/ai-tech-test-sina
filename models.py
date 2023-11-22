import torch
from torch import nn

class ModelLin(nn.Module):
    def __init__(self, 
                 feat_size,
                 hidden_size=64, 
                 output_size=1, 
                 ):
        super().__init__()
        self.feat_size = feat_size
        self.lin1 = nn.Linear(feat_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        output = self.lin1(x)
        output = self.lin2(output)
        return output

def test_model_lin():
    feat_size = 80
    inp = torch.rand(5, 100, feat_size)
    model = ModelLin(feat_size)
    out = model(inp)
    assert list(out.size())==[5,100,1]
