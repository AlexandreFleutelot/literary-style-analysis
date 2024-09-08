import torch
import sys

def check_cuda():
    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version}")
    
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available.")
        print("Checking why CUDA might not be available:")
        
        if not hasattr(torch, 'cuda'):
            print("- Your PyTorch installation was not built with CUDA support.")
        elif not torch.cuda.is_available():
            print("- CUDA is not available on your system, or there's a problem with your CUDA installation.")
        
        print("\nTo use CUDA, make sure you have:")
        print("1. A CUDA-capable GPU")
        print("2. CUDA Toolkit installed (https://developer.nvidia.com/cuda-toolkit)")
        print("3. PyTorch installed with CUDA support")
        print("\nIf you installed PyTorch via pip, try reinstalling with CUDA support:")
        print("pip uninstall torch")
        print("pip install torch --index-url https://download.pytorch.org/whl/cu118")

if __name__ == "__main__":
    check_cuda()