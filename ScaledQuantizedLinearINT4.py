import torch
import torch.nn as nn

class ScaledQuantizedLinearINT4(nn.Module):
    def __init__(self, in_features, out_features, bits=4, alpha=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.alpha = nn.Parameter(torch.tensor(alpha))  # Learnable scaling factor

        # Define trainable weight matrix (FP32 initially)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

        # Quantization scale factor (computed dynamically)
        self.register_buffer("scale_factor", torch.tensor(1.0))

    def quantize_weights(self):
        """Quantize weights to INT4 using uniform quantization."""
        W_scaled = torch.sign(self.weight) * torch.pow(torch.abs(self.weight), self.alpha)  # Scaling before quantization

        # Compute quantization scale
        W_min, W_max = W_scaled.min(), W_scaled.max()
        self.scale_factor = (W_max - W_min) / ((2 ** self.bits) - 1)  # Scale for 4-bit quantization

        # Uniform 4-bit quantization
        W_q = torch.round((W_scaled - W_min) / self.scale_factor).clamp(0, (2 ** self.bits) - 1)
        return W_q.to(torch.int8)  # Store as INT4 (emulated in INT8)

    def forward(self, x):
        # Step 1: Scale input before quantization
        x_scaled = torch.sign(x) * torch.pow(torch.abs(x), self.alpha)

        # Step 2: Quantize weights
        W_q = self.quantize_weights()

        # Step 3: Perform Integer Matrix Multiplication (Emulating INT4)
        output = torch.matmul(x_scaled.to(torch.int8), W_q.T.to(torch.int8))  # INT4 GEMM

        # Step 4: Reverse Scaling After Multiplication
        output = torch.sign(output) * torch.pow(torch.abs(output), 1 / self.alpha)

        return output

# Example Usage
batch_size, input_dim, output_dim = 32, 512, 512
x = torch.randn(batch_size, input_dim).cuda()  # Input matrix
quantized_linear = ScaledQuantizedLinearINT4(input_dim, output_dim).cuda()

output = quantized_linear(x)
print("Output Shape:", output.shape)  # Should be (32, 512)
