import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(data_dir: str, batch_size: int = 32, is_train: bool = True):
    """
    Loads data from a directory structured as <data_dir>/class_name/...
    Dynamically extracts the number of classes based on the folder structure.
    """
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory missing: {data_dir}")

    # Standard image transforms
    transform_list = [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    
    transform = transforms.Compose(transform_list)

    # Use PyTorch ImageFolder generic dataset handler
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    num_classes = len(dataset.classes)
    
    print(f"Loaded dataset from {data_dir} with {num_classes} classes and {len(dataset)} samples.")

    # DataLoader creation
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=is_train, 
        num_workers=0,  # Set to 0 to avoid multiprocessing issues on windows without if __name__ == '__main__'
        pin_memory=True
    )
    
    return dataloader, num_classes, len(dataset)
