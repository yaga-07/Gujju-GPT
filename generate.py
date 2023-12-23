import argparse
from termcolor import colored
import numpy as np
from transformers import AutoTokenizer
import torch
from torch.nn import functional as F
from model import config , TransformerDecoderModel


@torch.no_grad
def generate(text,device='cpu'):
    print(f"Device : {colored(device,'green')}")
    tokenizer = AutoTokenizer.from_pretrained("tokenizer/Gujju-GPT")
    # print(colored("Gujju-GPT tokenizer loaded",'light_blue'))
    state = torch.load('checkpoint\gujju-gpt_1000.pth',map_location=torch.device('cpu'))
    model = TransformerDecoderModel(config['vocab_size'],
                                      config['d_model'],
                                      config['num_heads'],
                                      config['num_layers'],
                                      config['d_ff'],
                                      config['max_len'],
                                      dropout=0.2).to(device)
    model.load_state_dict(state)
    print(colored("Gujju-GPT model loaded",'yellow'))
    try:
        print(colored("Generating . . .","green"))
        idxs = tokenizer.encode(text)
        if len(idxs) <= 128:
            raise ValueError("Please give longer sentences for now")
        
        out_idxs = []
        for _ in range(100):
            idxs = idxs[-128:]
            # print(f"input len {len(idxs)}")
            t_idx = torch.tensor(idxs)[None]
            o = torch.argmax(F.softmax(model(t_idx)[:,-1,:], dim=-1))
            out_idxs.append(o)
            idxs.append(o)
        print(f"output 🤖 : {colored(tokenizer.decode(out_idxs),'light_green')}")
            # print("\033c", end="")
        
        return idxs
    except Exception as e:
        print(colored(e,'red'))
        
        
if __name__ == "__main__":
    
    ascii = """  _____       _  _          _____ _____ _______ 
 / ____|     (_)(_)        / ____|  __ \__   __|
| |  __ _   _ _  _ _   _  | |  __| |__) | | |   
| | |_ | | | | || | | | | | | |_ |  ___/  | |   
| |__| | |_| | || | |_| | | |__| | |      | |   
 \_____|\__,_| || |\__,_|  \_____|_|      |_|   
            _/ |/ |                             
           |__/__/                              """
           
    print(colored(ascii,'cyan'))
    text = input("Enter long Gujju text : ")
    generate(text=text)
        