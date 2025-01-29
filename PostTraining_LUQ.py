import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import copy

class LogarithmicUnbiasedQuantization(nn.Module):
    """ Logarithmic Unbiased Quantization (LUQ) for Integer-Based Computation """
    def __init__(self, bit_width=4):
        super(LogarithmicUnbiasedQuantization, self).__init__()
        self.bit_width = bit_width
        self.scale_factor = (2 ** bit_width) - 1  # Number of quantization levels

    def quantize(self, tensor):
        """ Convert tensor to logarithmic quantized INT8 representation """
        epsilon = 1e-8
        sign = torch.sign(tensor)  # Preserve sign
        log_tensor = torch.log(torch.abs(tensor) + epsilon)  # Log transformation

        log_min, log_max = log_tensor.min(), log_tensor.max()
        log_tensor = (log_tensor - log_min) / (log_max - log_min)  # Normalize

        quantized_log_tensor = torch.round(log_tensor * self.scale_factor)  # Quantization in log-space
        int_representation = quantized_log_tensor.to(torch.int8)  # Convert to INT8

        return int_representation, log_min, log_max, sign

    def dequantize(self, int_tensor, log_min, log_max, sign):
        """ Convert INT8 tensor back to floating point """
        log_tensor = (int_tensor.float() / self.scale_factor) * (log_max - log_min) + log_min
        return sign * torch.exp(log_tensor)

# Function to Apply LUQ to Model Weights Without Dequantizing Back
def apply_luq_quantization(model, bit_width=4):
    quantizer = LogarithmicUnbiasedQuantization(bit_width)

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            with torch.no_grad():
                int_weights, log_min, log_max, sign = quantizer.quantize(module.weight)
                module.weight.data = int_weights  # Store INT8 weights
                module.register_buffer("log_min", log_min)
                module.register_buffer("log_max", log_max)
                module.register_buffer("sign", sign)

    return model

# LUQ-Based Linear Layer with INT8 Computation
class LUQLinear(nn.Module):
    def __init__(self, in_features, out_features, bit_width=4):
        super(LUQLinear, self).__init__()
        self.fc = nn.Linear(in_features, out_features, bias=True)
        self.luq = LogarithmicUnbiasedQuantization(bit_width)

    def forward(self, x):
        # Quantize Activations to INT8
        int_input, log_min_x, log_max_x, sign_x = self.luq.quantize(x)

        # Perform Integer-based Linear Transformation
        int_output = F.linear(int_input.float(), self.fc.weight.float())  # INT8 computation

        # Dequantize Output (Only If Needed)
        output = self.luq.dequantize(int_output, self.fc.log_min + log_min_x, self.fc.log_max + log_max_x, self.fc.sign * sign_x)

        # Add bias
        output += self.fc.bias
        return output

# LUQ-Based Conv2D Layer with INT8 Computation
class LUQConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bit_width=4):
        super(LUQConv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.luq = LogarithmicUnbiasedQuantization(bit_width)

    def forward(self, x):
        # Quantize Input Activations to INT8
        int_input, log_min_x, log_max_x, sign_x = self.luq.quantize(x)

        # Perform Integer-based Convolution
        int_output = F.conv2d(int_input.float(), self.conv.weight.float(), stride=self.conv.stride, padding=self.conv.padding)

        # Dequantize Output (Only If Needed)
        output = self.luq.dequantize(int_output, self.conv.log_min + log_min_x, self.conv.log_max + log_max_x, self.conv.sign * sign_x)
        return output

# Load Pretrained Model
fp_model = models.resnet18(pretrained=True)
quantized_model = copy.deepcopy(fp_model)

# Apply LUQ Post-Training Quantization
quantized_model = apply_luq_quantization(quantized_model, bit_width=4)

# Move Model to CUDA (If Available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
quantized_model = quantized_model.to(device)

def evaluate_model(model, test_loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

fp32_accuracy = evaluate_model(fp_model, test_loader)
luq_accuracy = evaluate_model(quantized_model, test_loader)

print(f"FP32 Accuracy: {fp32_accuracy:.2f}%")
print(f"LUQ INT8 Accuracy: {luq_accuracy:.2f}%")


import time

def measure_inference_time(model, test_loader, num_batches=10):
    model.eval()
    start_time = time.time()

    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            if i >= num_batches:
                break
            images = images.to(device)
            _ = model(images)

    return (time.time() - start_time) / num_batches

fp32_time = measure_inference_time(fp_model, test_loader)
luq_time = measure_inference_time(quantized_model, test_loader)

print(f"FP32 Inference Time: {fp32_time:.4f} sec")
print(f"LUQ INT8 Inference Time: {luq_time:.4f} sec (~{fp32_time / luq_time:.2f}x speedup)")


import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define Data Transformations for CIFAR-10
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet requires 224x224 images
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load CIFAR-10 Test Dataset
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

import timm
model = timm.create_model('resnet18_cifar10', pretrained=True)

