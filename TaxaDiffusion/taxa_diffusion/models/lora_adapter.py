import torch
import torch.nn as nn

class LoRAAdapter(nn.Module):
    def __init__(self, input_dim, rank=4):
        super(LoRAAdapter, self).__init__()
        self.rank = rank
        self.W_down = nn.Linear(input_dim, rank, bias=False)
        self.W_up = nn.Linear(rank, input_dim, bias=False)
    
    def forward(self, x):
        return self.W_up(self.W_down(x))


def inject_lora_into_attention(unet, rank=4):
    for name, module in unet.named_modules():
        if "attentions" in name:
            input_dim = 0
            for _, submodule in module.named_children():
                # Check if the submodule has a `proj_out` attribute
                if hasattr(submodule, 'proj_out'):
                    # Handle both Linear and Conv2d cases
                    if isinstance(submodule.proj_out, nn.Conv2d):
                        input_dim = submodule.proj_out.out_channels
                    elif isinstance(submodule.proj_out, nn.Linear):
                        input_dim = submodule.proj_out.out_features
                    break
            # print(f"name: {name} , input_dim: {input_dim}")
            if input_dim > 0:
                # Create a LoRA adapter and attach it to the attention module
                module.lora_adapter = LoRAAdapter(input_dim, rank)

                # Wrap the original forward method to include LoRA's effect
                original_forward = module.forward

                def lora_forward(x, *args, lora_adapter=module.lora_adapter, original_forward=original_forward, **kwargs):
                    # Compute the original output
                    output = original_forward(x, *args, **kwargs)
                    # Apply the LoRA adapter to the output
                    lora_output = lora_adapter(x)
                    return output + lora_output

                # Replace the forward method with the wrapped version
                module.forward = lora_forward

    print("LoRA adapters have been injected into attention layers.")
