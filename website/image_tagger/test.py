import torch

model = torch.jit.load("../../Models/finetuned_model.pt")
print(model)