import torch
import torch.nn as nn
from train_utils import get_device

def evaluate_model(model: nn.Module, dataloader):
    """
    Evaluates the model on test data, typically used by server or client sequentially.
    """
    device = get_device()
    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    
    print(f"\n[Evaluation] Starting evaluation on {device}...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    
    print(f"[Evaluation] Results - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%\n")
    model.cpu()  
    
    return avg_loss, accuracy
