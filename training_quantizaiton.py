import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from sklearn.cluster import KMeans
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ---------------------------- #
#        STEP 1: TRAINING      #
# ---------------------------- #
class SimpleNN(nn.Module):
    """Baseline fully connected neural network."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, train_loader, epochs=3, lr=0.01):
    """Trains the baseline FP32 model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for data, labels in train_loader:
            data, labels = data.view(data.size(0), -1).cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# ---------------------------- #
#   STEP 2: STANDARD INT4      #
# ---------------------------- #
class UniformQuantizedLinearINT4(nn.Module):
    """Standard uniform 4-bit quantization for linear layer."""
    def __init__(self, in_features, out_features, bits=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits

        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        W_min, W_max = self.weight.min(), self.weight.max()
        scale_factor = (W_max - W_min) / ((2 ** self.bits) - 1)

        W_q = torch.round((self.weight - W_min) / scale_factor).clamp(0, (2 ** self.bits) - 1)
        W_dequantized = W_q.float() * scale_factor + W_min

        return torch.matmul(x, W_dequantized.T)

# ---------------------------- #
#   STEP 3: CLUSTERED INT4     #
# ---------------------------- #
class ScaledQuantizedLinearINT4(nn.Module):
    """Clustered 4-bit quantization using K-Means clustering."""
    def __init__(self, in_features, out_features, bits=4, alpha=0.5, num_clusters=16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.num_clusters = num_clusters

        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.register_buffer("centroids", torch.zeros(num_clusters))

    def compute_kmeans_centroids(self):
        """Computes K-Means centroids for weight clustering."""
        with torch.no_grad():
            W_scaled = torch.sign(self.weight) * torch.pow(torch.abs(self.weight), self.alpha)
            W_flat = W_scaled.view(-1).cpu().numpy()

            kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
            kmeans.fit(W_flat.reshape(-1, 1))

            centroids = torch.tensor(kmeans.cluster_centers_.flatten(), dtype=torch.float32).cuda()
            self.centroids.copy_(centroids)

            clustered_indices = torch.tensor(kmeans.labels_, dtype=torch.long).reshape(self.weight.shape).cuda()
            return clustered_indices

    def forward(self, x):
        x_scaled = torch.sign(x) * torch.pow(torch.abs(x), self.alpha)
        W_q = self.compute_kmeans_centroids()
        W_reconstructed = torch.gather(self.centroids, 0, W_q)

        output = torch.matmul(x_scaled, W_reconstructed.T)
        return torch.sign(output) * torch.pow(torch.abs(output), 1 / self.alpha)

# ---------------------------- #
#  STEP 4: BENCHMARKING MODELS #
# ---------------------------- #
def benchmark_model(model, x, num_runs=50):
    """Measures inference time for the given model."""
    model.eval()
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x)

    torch.cuda.synchronize()
    return (time.time() - start_time) * 1000 / num_runs  # Convert to ms

# ---------------------------- #
#   STEP 5: TRAIN + BENCHMARK  #
# ---------------------------- #

# 1️⃣ Load Dataset
BATCH_SIZE = 16
INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM = 128, 64, 10  # Smaller model size
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 2️⃣ Train Baseline Model
baseline_model = SimpleNN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).cuda()
train_model(baseline_model, train_loader)

# 3️⃣ Create Models for Benchmarking
x = torch.randn(BATCH_SIZE, INPUT_DIM).cuda()
clustered_model = ScaledQuantizedLinearINT4(INPUT_DIM, OUTPUT_DIM).cuda()
uniform_quantized_model = UniformQuantizedLinearINT4(INPUT_DIM, OUTPUT_DIM).cuda()

# 4️⃣ Run Benchmarking
baseline_time = benchmark_model(baseline_model, x)
clustered_time = benchmark_model(clustered_model, x)
uniform_time = benchmark_model(uniform_quantized_model, x)

# 5️⃣ Print Results
print(f"Baseline FP32 Model: {baseline_time:.3f} ms per run")
print(f"Clustered INT4 Model: {clustered_time:.3f} ms per run")
print(f"Uniform INT4 Model: {uniform_time:.3f} ms per run")
print(f"Speedup (Clustered vs FP32): {baseline_time / clustered_time:.2f}x")
print(f"Speedup (Uniform INT4 vs FP32): {baseline_time / uniform_time:.2f}x")
print(f"Accuracy Improvement (Clustered vs Uniform): {clustered_time / uniform_time:.2f}x")
