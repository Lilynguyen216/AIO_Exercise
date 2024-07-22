import torch
import torch.nn as nn

class softmax(nn.Module):
  def __init__(self):
    super.__init__()

  def softmax(self, x):
    return torch.exp(x) / torch.sum(torch.exp(x), dim=0)

  def forward(self, x):
    res = self.softmax(x)
    return res

class softmax_stable(nn.Module):
  def __init__(self):
    super().__init__()

  def softmax_stable(self, x):
    return torch.exp(x - torch.max(x))/torch.sum(torch.exp(x - torch.max(x)), dim=0)

  def forward(self, x):
    res = self.stable_softmax(x)
    return res

if __name__ == '__main__':
  x = torch.tensor([3, 5, 7])
  softmax = softmax()
  softmax_stable = softmax_stable()
  print(softmax(x))
  print(softmax_stable(x))
