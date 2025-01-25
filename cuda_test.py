import torch

print(f"Is CUDA available? ", torch.cuda.is_available())
print(f"CUDA device name: ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
