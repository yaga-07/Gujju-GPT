import torch
from termcolor import colored
from model import config , TransformerDecoderModel

transformer = TransformerDecoderModel(config['vocab_size'],
                                      config['d_model'],
                                      config['num_heads'],
                                      config['num_layers'],
                                      config['d_ff'],
                                      config['max_len'],
                                      dropout=0.2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(colored(f'Device :{device}','blue'))
transformer.to(device)
print(colored('transformer initialized','yellow'))
# Print total trainable and non-trainable parameters
total_params = sum(p.numel() for p in transformer.parameters())
trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)

print(colored(f"Total parameters: {total_params}",'light_green'))
print(colored(f"Trainable parameters: {trainable_params}",'light_green'))
print(colored(f"Non-trainable parameters: {total_params - trainable_params}",'light_green'))