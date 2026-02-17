
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import gc

def find_max_batch_size(device, transform=None, min_bs=8, max_bs=4096, model_name='resnet50'):
    """
    Attempts to find the maximum batch size that fits in GPU memory for a given model.
    Uses binary search to find an OOM-free batch size.
    """
    print(f"\nSearching for maximum Batch Size in range [{min_bs}, {max_bs}] for model '{model_name}'...")
    
    best_bs = min_bs
    low = min_bs
    high = max_bs
    
    # Check if min_bs works first? No, logic assumes low is valid or part of search.
    # The snippet logic is binary search: `bs = (low + high) // 2`
    
    while low <= high:
        bs = (low + high) // 2
        if bs == 0: break
        
        print(f"\nTesting Batch Size: {bs}...")
        
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()   # 🔥 PEAK reset
            
            # Create a dummy model
            dummy_model = timm.create_model(model_name, pretrained=True).to(device)
            dummy_optimizer = optim.Adam(dummy_model.parameters())
            dummy_criterion = nn.CrossEntropyLoss()
            
            # Create dummy input. Assuming standard ImageNet size 224x224
            # If transform implies a different size, this might be inaccurate, 
            # but usually for batch sizing we care about tensor size.
            # We can try to infer size from transform if possible, but 224 is standard for resnet50.
            input_size = (3, 224, 224) 
            x = torch.randn(bs, *input_size).to(device)
            y = torch.randint(0, 2, (bs,)).to(device)
            
            dummy_optimizer.zero_grad()
            output = dummy_model(x)
            loss = dummy_criterion(output, y)
            loss.backward()
            dummy_optimizer.step()
            
            # 🔥 VRAM measurement
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                peak = torch.cuda.max_memory_allocated() / 1024**2
                print(f"SUCCESS | Allocated: {allocated:.2f} MB | "
                      f"Reserved: {reserved:.2f} MB | "
                      f"Peak: {peak:.2f} MB")
            
            best_bs = bs
            low = bs + 1
            
        except torch.cuda.OutOfMemoryError:
            print("OOM ❌")
            high = bs - 1
            
        except Exception as e:
            print(f"ERROR: {e}")
            # If error is not OOM (e.g. CUDA error or other), might still be related to size or not.
            # Assuming it fails for this BS, try smaller.
            high = bs - 1
            
        finally:
            if 'dummy_model' in locals(): del dummy_model
            if 'x' in locals(): del x
            if 'y' in locals(): del y
            if 'output' in locals(): del output
            if 'loss' in locals(): del loss
            if 'dummy_optimizer' in locals(): del dummy_optimizer
            if 'dummy_criterion' in locals(): del dummy_criterion
            
            gc.collect()
            torch.cuda.empty_cache()
                
    print(f"\n>>> Maximum working Batch Size found: {best_bs} <<<")
    
    safe_bs = int(best_bs * 0.8)
    print(f">>> Recommended Safe Batch Size: {safe_bs} <<<")
    
    return safe_bs

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("Warning: Running on CPU, OOM checks are less relevant.")
        
    find_max_batch_size(device)
