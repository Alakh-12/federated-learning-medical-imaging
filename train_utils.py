import torch
import torch.nn as nn
import torch.optim as optim

def get_device():
    # Automatically detect GPU if available
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def train_local_model(model: nn.Module, dataloader, epochs: int, lr: float = 0.001):
    """
    Performs local training on a client device.
    """
    device = get_device()
    model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"[Client] Starting local training on {device} for {epochs} epochs...")
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        print(f"[Client] Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        
    print("[Client] Training complete.")
    # Return the model state on CPU to serialize
    return model.cpu().state_dict()
