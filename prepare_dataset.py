import os
import shutil
import random

def prepare_federated_dataset():
    """
    Splits the chest_xray dataset into a federated learning format:
    - client_1 (50% of training data)
    - client_2 (50% of training data)
    - test_data (100% of testing data)
    """
    # Source Directories
    src_train = os.path.join("chest_xray", "train")
    src_test = os.path.join("chest_xray", "test")
    
    # Target Directories
    dest_base = "dataset"
    client1_dir = os.path.join(dest_base, "client_1")
    client2_dir = os.path.join(dest_base, "client_2")
    test_dir = os.path.join(dest_base, "test_data")
    
    classes = ["NORMAL", "PNEUMONIA"]
    
    print("====================================")
    print(" Federation Dataset Preparation Tool")
    print("====================================\n")

    # 1. Process Test Data -> dataset/test_data/
    print("--- 1. Processing Test Data ---")
    for cls in classes:
        src_path = os.path.join(src_test, cls)
        target_path = os.path.join(test_dir, cls)
        
        # Create target directories
        os.makedirs(target_path, exist_ok=True)
        
        if not os.path.exists(src_path):
            print(f"  Warning: Source directory '{src_path}' not found. Skipping.")
            continue
            
        files = [f for f in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, f))]
        copied_count = 0
        
        for f in files:
            src_file = os.path.join(src_path, f)
            target_file = os.path.join(target_path, f)
            
            # Safe copy (overwrite protection)
            if not os.path.exists(target_file):
                shutil.copy2(src_file, target_file)
                copied_count += 1
                
        print(f"  [{cls}] Found {len(files)} files. Copied {copied_count} new files to {target_path}.")

    # 2. Process Training Data -> client_1 and client_2
    print("\n--- 2. Processing Training Data (50/50 Split) ---")
    
    # We use a fixed seed so running this multiple times produces the exact same layout
    random.seed(42)  

    for cls in classes:
        src_path = os.path.join(src_train, cls)
        c1_target = os.path.join(client1_dir, cls)
        c2_target = os.path.join(client2_dir, cls)
        
        os.makedirs(c1_target, exist_ok=True)
        os.makedirs(c2_target, exist_ok=True)
        
        if not os.path.exists(src_path):
            print(f"  Warning: Source directory '{src_path}' not found. Skipping.")
            continue
            
        files = [f for f in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, f))]
        files.sort()  # Sort to guarantee determinism prior to shuffling
        random.shuffle(files)
        
        # 50/50 Split Logic
        midpoint = len(files) // 2
        client1_files = files[:midpoint]
        client2_files = files[midpoint:]
        
        # Copy to Client 1
        c1_copied = 0
        for f in client1_files:
            src_file = os.path.join(src_path, f)
            target_file = os.path.join(c1_target, f)
            if not os.path.exists(target_file):
                shutil.copy2(src_file, target_file)
                c1_copied += 1
                
        # Copy to Client 2
        c2_copied = 0
        for f in client2_files:
            src_file = os.path.join(src_path, f)
            target_file = os.path.join(c2_target, f)
            if not os.path.exists(target_file):
                shutil.copy2(src_file, target_file)
                c2_copied += 1
                
        print(f"  [{cls}] Total original training files: {len(files)}")
        print(f"    -> Client 1 allocated total: {len(client1_files)} | Copied {c1_copied} new items.")
        print(f"    -> Client 2 allocated total: {len(client2_files)} | Copied {c2_copied} new items.")

    print("\n[SUCCESS] Federation dataset restructuring is complete!")

if __name__ == "__main__":
    prepare_federated_dataset()
