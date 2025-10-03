import torch
import torch.nn as nn
import torch.nn.functional as F
# from VGG import vgg16
from torchvision.models import mobilenet_v2
from dataloader import get_cifar10
from utils import *

# For seed, log & plots
import random
import numpy as np
import os
import matplotlib.pyplot as plt
import time

# Function to set seed for reproducibility (default seed=47)
def set_seed(seed=47):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Function to plot and save loss/accuracy metrics
def plot_and_save_metrics(epoch_list, train_losses, train_accs, test_accs, loss_save_path, acc_save_path):
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- Plot 1: Loss ---
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_list, train_losses, label='Train Loss', color='blue')
    plt.title('Training Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_save_path)
    plt.close() # Close the figure to free memory
    print(f"Training loss plot saved to {loss_save_path}")

    # --- Plot 2: Accuracy ---
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_list, train_accs, label='Train Accuracy', color='green')
    plt.plot(epoch_list, test_accs, label='Test Accuracy', color='orange')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_save_path)
    plt.close() # Close the figure
    print(f"Accuracy plot saved to {acc_save_path}")

#===============
# START
#---------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":

    print("Training MobileNetV2 on CIFAR-10")

    # Setting seed
    set_seed(47)

    # Log and checkpoint init
    print("1. Setting up logging")
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    # Using tee for logging
    # log_file = open("./logs/mobilenetv2_training_log.txt", "w")
    epoch_list, train_losses, train_accs, test_accs, epoch_durations = [], [], [], [], []

    # Load data
    print("2. Setting up data")
    train_loader,test_loader = get_cifar10(batchsize=128)

    # MobileNetV2 model with modifications
    print("3. Loading MobileNetV2 and updating for CIFAR-10")
    model = mobilenet_v2(weights=None) 

    # Changing strides in first few layers as per
    # https://github.com/chenhang98/mobileNet-v2_cifar10?tab=readme-ov-file
    # Reason: Mobilenet is designed for Imagenet with 224x224 size
    #         so it quickly downsamples elements with stride = 2
    #         But CIFAR images are 32x32, this aggressive downsampling would affect
    #         spatial resolution
    model.features[0][0].stride = (1, 1)      # First conv2d layer
    # model.features[1].conv[1].stride = (1, 1) # First InvertedResidual layer (if stride=2 originally)
    # model.features[2].conv[1].stride = (1, 1) # Second InvertedResidual layer with stride 2
    # model.features[3].conv[1].stride = (1, 1) # Third InvertedResidual layer with stride 2
  
    # Updated after checking print(model) to identify initial layers with stride = 2
    # TODO: Commented because training is too slow
    model.features[2].conv[1][0].stride = (1, 1)
    model.features[4].conv[1][0].stride = (1, 1)
    # model.features[7].conv[1][0].stride = (1, 1)

    # Modifying classifier for CIFAR-10's 10 classes (MobileNetV2 by default is trained on ImageNet with 1000 classifiers)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 10)
    
    model.to(device)

    print("4. Setting Training hyperparameters")
    epochs = 300
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.07,momentum=0.9,weight_decay=5e-4,nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("5. Start Training")
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            train_loss += loss.item()
            loss.mean().backward()
            optimizer.step()
            _, predicted = outputs.max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
        
        # Record time
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        train_acc = 100. * correct / total
        test_acc = evaluate(model, test_loader, device)

        # Logging
        # print(f"Epoch {epoch+1:2d}: Loss={train_loss:.4f} | Train Acc={train_acc:.2f}% | Test Acc={test_acc:.2f}%")
        log_input = f"Epoch {epoch+1:2d}: Duration = {epoch_duration:.2f}s | Loss={train_loss:.4f} | Train Acc={train_acc:.2f}% | Test Acc={test_acc:.2f}%"
        print(log_input)
        # log_file.write(log_input + "\n")
        epoch_list.append(epoch + 1)
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        epoch_durations.append(epoch_duration)
        
        scheduler.step()
   
    # Save the model
    torch.save(model.state_dict(),'./checkpoints/mobilenetv2_cifar10.pth')

    # Plot and save metrics
    plot_and_save_metrics(
        epoch_list, 
        train_losses, 
        train_accs, 
        test_accs, 
        loss_save_path='./plots/training_loss_plot.png',
        acc_save_path='./plots/accuracy_plot.png'
    )
    
    # Logging end
    # log_file.close()
    print("6. Training completed; Model & plots saved")


