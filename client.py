import os
import io
import time
import requests
import torch
import argparse
from model import SimpleCNN
from data_utils import get_dataloader
from train_utils import train_local_model

def get_server_status(server_url):
    """
    Polls the server for its current status (round number, target clients).
    """
    try:
        response = requests.get(f"{server_url}/status")
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.ConnectionError:
        return None

def fetch_global_model(server_url, num_classes):
    """
    Downloads the current global model state from the parameter server.
    """
    print("[Client] Fetching global model from server...")
    response = requests.get(f"{server_url}/model?client_num_classes={num_classes}")
    if response.status_code != 200:
        raise Exception(f"Failed to fetch model: {response.text}")
    
    # Load model from bytes into memory securely
    buffer = io.BytesIO(response.content)
    state_dict = torch.load(buffer, map_location="cpu", weights_only=True)
    
    model = SimpleCNN(num_classes)
    model.load_state_dict(state_dict)
    return model

def send_model_update(server_url, client_id, model, sample_size):
    """
    Uploads the locally trained model delta back to the server using HTTP POST multipart.
    """
    print("[Client] Sending updated model vector to server...")
    buffer = io.BytesIO()
    torch.save(model.cpu().state_dict(), buffer)
    buffer.seek(0)
    
    response = requests.post(
        f"{server_url}/update",
        data={"client_id": client_id, "sample_size": sample_size},
        files={"model_file": ("model.pt", buffer, "application/octet-stream")}
    )
    if response.status_code == 200:
        print(f"[Client] Update accepted. Ready for next round.\n")
    else:
        print(f"[Client] Server rejected update: {response.text}\n")
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, required=True, help="Unique identifier for this client (e.g., 'client_1')")
    parser.add_argument("--data", type=str, required=True, help="Path to local partition of dataset")
    parser.add_argument("--epochs", type=int, default=1, help="Local training epochs per communication round")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8000", help="HTTP URL of the FedAvg server")
    args = parser.parse_args()
    
    print(f"--- Federated Learning Edge Client Node Started ---")
    print(f"ID: {args.id} | Server: {args.server}")
    
    # 1. Initialize local data
    print("[Client] Extracting local dataset...")
    dataloader, num_classes, sample_size = get_dataloader(args.data, batch_size=32, is_train=True)
    
    last_round_trained = 0
    poll_interval = 3  # Seconds
    
    # 2. Main Federated Learning Loop
    while True:
        status = get_server_status(args.server)
        if not status:
            print("[Client] Connection refused by server. Retrying and waiting for server...")
            time.sleep(5)
            continue
            
        current_round = status["round"]
        
        if current_round > last_round_trained:
            print(f"\n=====================================")
            print(f"  [Client] Commencing Round {current_round}")
            print(f"=====================================")
            
            try:
                # Stage A: Retrieve Global Parameters
                global_model = fetch_global_model(args.server, num_classes)
                
                # Stage B: Train on Local Data
                updated_state_dict = train_local_model(global_model, dataloader, epochs=args.epochs)
                global_model.load_state_dict(updated_state_dict)
                
                # Stage C: Upload Model Weight Updates
                send_model_update(args.server, args.id, global_model, sample_size)
                
                # Advance local pointer
                last_round_trained = current_round
                print("[Client] Task complete. Pending server aggregation synchronization...")
            except Exception as e:
                print(f"[Client] Error encountered during Round {current_round}: {e}")
                print("[Client] Will retry round...")
                time.sleep(poll_interval)
        else:
            # Idling phase, waiting for other synchronous clients to fulfill round requirement
            time.sleep(poll_interval)

if __name__ == "__main__":
    main()
