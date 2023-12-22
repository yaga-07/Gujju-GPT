import argparse
from termcolor import colored
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import pickle
# from torch.utils.tensorboard import SummaryWriter
from model import config , TransformerDecoderModel



def train_gpt(train_data, val_data, batch_size, learning_rate, epochs, device, model_path):
    
    print(f"Device in use {colored(device,'light_green')}")
    # Enable anomaly detection
    torch.autograd.set_detect_anomaly(True)
    
    vocab_size = config['vocab_size']
    block_size = config['max_len']
    model = TransformerDecoderModel(config['vocab_size'],
                                      config['d_model'],
                                      config['num_heads'],
                                      config['num_layers'],
                                      config['d_ff'],
                                      config['max_len'],
                                      dropout=0.2).to(device)
    
    print(colored("Gujju GPT initialized","yellow"))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Parameters: {colored(total_params,'yellow')}")    

    # Initialize model, loss, and optimizer
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 5)

    # Set up TensorBoard
    # writer = SummaryWriter()
    
    print(f"Strating traing for {colored(epochs,'yellow')}")
    # Training loop
    train_loss_hist = []
    val_loss_hist = []
    for epoch in range(1,epochs):
        # Set the model in training mode
        model.train()

        # Randomly sample indices for training data
        ix = torch.randint(len(train_data) - block_size, (batch_size,))

        # Prepare input and target tensors
        x = torch.stack([torch.from_numpy((train_data[i:i + block_size]).astype(np.int64)) for i in ix]).to(device)
        y = torch.stack([torch.from_numpy((train_data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix]).to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(x)
        # print(colored(f"logits shape {logits.shape}"))
        # print(colored(f"logits shape {logits.view(-1, vocab_size).shape}"))


        # Compute the loss
        # loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()
        
        
        
        print(f"| Epoch : {colored(epoch,'cyan')}")
        print(f"| Train Loss : {colored(loss.item(), 'green')} |")
        print(f"| Current LR : {colored(optimizer.param_groups[0]['lr'], 'magenta')} |")
        print()
        train_loss_hist.append(loss.item())
        # Log training loss to TensorBoard
        # writer.add_scalar('Loss/Train', loss.item(), epoch)

        # Validation
        if epoch % 10 == 0:
            # Set the model in evaluation mode
            model.eval()

            # Randomly sample indices for validation data
            ix_val = torch.randint(len(val_data) - block_size, (batch_size,))

            # Prepare input and target tensors for validation
            x_val = torch.stack([torch.from_numpy((val_data[i:i + block_size]).astype(np.int64)) for i in ix_val]).to(device)
            y_val = torch.stack([torch.from_numpy((val_data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix_val]).to(device)

            # Forward pass for validation
            val_logits = model(x_val)

            # Compute validation loss
            val_loss = criterion(val_logits.view(-1, vocab_size), y_val.view(-1))
            scheduler.step(val_loss)
            
            print(f"| Epoch : {colored(epoch,'cyan')} | Val Loss : {colored(val_loss.item(), 'green')} |")
            print()
            val_loss_hist.append(val_loss.item())
            # Log validation loss to TensorBoard
            # writer.add_scalar('Loss/Validation', val_loss.item(), epoch)

            # Set the model back to training mode
            model.train()
            
        if epoch % 100 == 0:

            # Save the trained model
            torch.save(model.state_dict(), model_path + f'/gujju-gpt_{epoch}.pth')
            
    with open(model_path + f'/gujju-gpt_{epoch}_train_loss.pkl', 'wb') as file:
        pickle.dump(train_loss_hist, file)
        
    with open(model_path + f'/gujju-gpt_{epoch}_val_loss.pkl', 'wb') as file:
        pickle.dump(val_loss_hist, file)

    # Close the TensorBoard writer
    # writer.close()

def main():
    parser = argparse.ArgumentParser(description='Train GPT model')
    parser.add_argument('--train_data', required=True, help='Path to training data (train.npy)')
    parser.add_argument('--val_data', required=True, help='Path to validation data (valid.npy)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device for training (cuda or cpu)')
    parser.add_argument('--model_save', required=True, help='Path to training data (train.npy)')
    args = parser.parse_args()

    # Load data
    train_data = np.load(args.train_data)
    val_data = np.load(args.val_data)

    train_gpt(train_data, val_data, args.batch_size, 
              args.learning_rate, args.epochs, args.device, args.model_save)

if __name__ == "__main__":
    main()
