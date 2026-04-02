import io
import copy
import torch
import threading
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
import uvicorn

from model import SimpleCNN
from data_utils import get_dataloader
from eval_utils import evaluate_model

app = FastAPI(title="Federated Learning Server")

# Server State 
N_TARGET_CLIENTS = 2  
TEST_DATALOADER = None  
current_round = 1
global_model = None
num_classes = None

client_updates = []
received_client_ids = set()
lock = threading.Lock()

@app.get("/status")
def get_status():
    with lock:
        return {
            "round": current_round,
            "target_clients": N_TARGET_CLIENTS,
            "received_updates": len(client_updates),
            "model_initialized": global_model is not None,
            "num_classes": num_classes
        }

@app.get("/model")
def get_model(client_num_classes: int = None):
    global global_model, num_classes, TEST_DATALOADER
    with lock:
        if global_model is None:
            if client_num_classes is None:
                raise HTTPException(status_code=400, detail="Server model uninitialized. First client must provide 'client_num_classes'.")
            
            print(f"[Server] Initializing global model for {client_num_classes} classes...")
            num_classes = client_num_classes
            global_model = SimpleCNN(num_classes)
            
            # Prepare test dataloader if test data path was provided via args
            import sys
            import argparse
            # Since FastAPI doesn't easily let us pass args natively inside the router, 
            # we rely on a global variable populated by the main block.
            if hasattr(app, "test_data_path") and app.test_data_path:
                print(f"[Server] Loading test dataset from {app.test_data_path}...")
                TEST_DATALOADER, test_classes, _ = get_dataloader(app.test_data_path, batch_size=32, is_train=False)
                if test_classes != num_classes:
                    print(f"Warning: Test dataset classes ({test_classes}) don't match client classes ({num_classes})!")

        elif client_num_classes is not None and client_num_classes != num_classes:
             raise HTTPException(status_code=400, detail=f"Class count mismatch. Server expects {num_classes} classes.")

        # Serialize model safely using io.BytesIO
        buffer = io.BytesIO()
        torch.save(global_model.state_dict(), buffer)
        return Response(content=buffer.getvalue(), media_type="application/octet-stream")

@app.post("/update")
async def receive_update(
    client_id: str = Form(...),
    sample_size: int = Form(...),
    model_file: UploadFile = File(...)
):
    global current_round
    
    with lock:
        if client_id in received_client_ids:
            raise HTTPException(status_code=400, detail=f"Client '{client_id}' already sent an update for round {current_round}.")
        
        # Read the file directly into memory and load the state dictionary
        buffer = io.BytesIO(await model_file.read())
        try:
            state_dict = torch.load(buffer, map_location="cpu", weights_only=True)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load weights: {e}")
            
        client_updates.append({
            "state_dict": state_dict, 
            "sample_size": sample_size, 
            "client_id": client_id
        })
        received_client_ids.add(client_id)
        
        print(f"[Server] Received update from {client_id}. Total updates for Round {current_round}: {len(client_updates)}/{N_TARGET_CLIENTS}")
        
        # Check if we hit the synchronous threshold
        if len(client_updates) == N_TARGET_CLIENTS:
            aggregate_models()
        
    return {"message": "Update successfully received", "round": current_round}

def aggregate_models():
    global global_model, current_round, client_updates, received_client_ids, TEST_DATALOADER
    print(f"\n[Server] === Aggregating updates for Round {current_round} ===")
    
    total_samples = sum([u["sample_size"] for u in client_updates])
    
    # Initialize the aggregated state_dict with the first client's keys (zeroed out)
    agg_state_dict = copy.deepcopy(client_updates[0]["state_dict"])
    for key in agg_state_dict:
        agg_state_dict[key] = torch.zeros_like(agg_state_dict[key], dtype=torch.float32)
        
    # Perform strict weighted FedAvg
    for update in client_updates:
        weight = update["sample_size"] / total_samples
        for key in agg_state_dict:
            agg_state_dict[key] += update["state_dict"][key] * weight
            
    # Load the averaged weights back into the global model
    global_model.load_state_dict(agg_state_dict)
    print(f"[Server] FedAvg complete. Aggregated over {total_samples} total samples.")
    
    # Evaluation phase
    if TEST_DATALOADER is not None:
        evaluate_model(global_model, TEST_DATALOADER)
    else:
        print("[Server] Skipping evaluation (No test data provided).")

    # Reset states for the next round
    client_updates.clear()
    received_client_ids.clear()
    current_round += 1
    print(f"[Server] === Starting Round {current_round} ===\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=2, help="Number of clients required per round")
    parser.add_argument("--test-data", type=str, default=None, help="Optional path to a test dataset for global model evaluation")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP address to bind the server")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server")
    args = parser.parse_args()
    
    N_TARGET_CLIENTS = args.clients
    app.test_data_path = args.test_data
    
    print(f"[Server] Starting on {args.host}:{args.port}")
    print(f"[Server] Target clients per round: {N_TARGET_CLIENTS}")
    print(f"[Server] Test data path: {args.test_data}")
    
    uvicorn.run(app, host=args.host, port=args.port, reload=False)
