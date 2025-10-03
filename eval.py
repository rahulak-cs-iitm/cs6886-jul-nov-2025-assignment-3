# eval.py
import torch
from dataloader import get_cifar10
from utils import *

# For seed, log & plots
import random
import numpy as np
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    set_seed(47)
    train_loader,test_loader = get_cifar10(batchsize=2)

    # model = vgg16(num_classes=10, dropout=0.5)
    model = get_model_architecture() 

    model.load_state_dict(torch.load('./checkpoints/mobilenetv2_cifar10.pth',weights_only=True))
    model.to(device)
    model.eval()
    inference_start_time = time.time()
    test_acc = evaluate(model, test_loader, device)
    inference_end_time = time.time()
    inference_duration = inference_end_time - inference_start_time

    print(f"Inference Time = {inference_duration:.2f}s | Test Acc={test_acc:.2f}%")
