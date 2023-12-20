import torch
from termcolor import colored

# tril = torch.tril(torch.ones(2,2,3,3))

# print(f"tril {tril}")

# mask = tril.masked_fill(tril == 0, float('-inf')).masked_fill(tril == 1, float(0.0))

# print(f"mask {mask}")

random_tensor = torch.randn(2, 2, 3, 3)

print(f"random_tensor {random_tensor}")

# print(f"add {mask + random_tensor}")

tril = torch.tril(random_tensor)
mask = tril.masked_fill(tril == 0, float('-inf'))
print(f"mask {mask}")