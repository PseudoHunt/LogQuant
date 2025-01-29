import torch
import torch.nn as nn
import numpy as np
import time
from sklearn.cluster import KMeans

# ---------------------------- #
#  Sensitivity-Based K-Means INT4 Layer
# ---------------------------- #
class SensitivityBasedQuantizedLinearINT4(nn.Module):
    """4-bit Quantized Linear Layer with Sensitivity-Based K-Means Clustering."""

    def __init__(self, in_features, out_features, bits=4, num_clusters=16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.num_clusters = num_clusters

        # Define trainable weight matrix (FP32 initially)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

        # Register buffer for storing quantized centroids
        self.register_buffer("centroids", torch.zeros(num_clusters))

    def compute_sensitivity(self, grad_tensor):
        """Computes sensitivity using Hessian approximation (gradient squared)."""
        return torch.abs(grad_tensor**2)

    def apply_kmeans(self, weight_tensor, sensitivity):
        """Performs K-Means clustering on weights based on their sensitivity."""
        with torch.no_grad():
            # Flatten the tensors for clustering
            W_flat = weight_tensor.view(-1).cpu().numpy()
            S_flat = sensitivity.view(-1).cpu().numpy()

            # Combine weights & sensitivity into a feature vector
            features = np.stack([W_flat, S_flat], axis=1)

            # Run K-Means clustering
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
            kmeans.fit(features)

            # Store cluster centers (quantized weight values)
            self.centroids.copy_(torch.tensor(kmeans.cluster_centers_[:, 0], dtype=torch.float32))

            # Assign weights to nearest cluster
            cluster_indices = torch.tensor(kmeans.labels_, dtype=torch.long).reshape(weight_tensor.shape)

            # Map weights to their quantized centroids
            clustered_weights = torch.gather(self.centroids, 0, cluster_indices)
        
        return clustered_weights, cluster_indices

    def forward(self, x):
        """Performs INT4 quantized matrix multiplication."""
        with torch.no_grad():
            # Compute sensitivity using gradient approximation
            grad_tensor = torch.randn_like(self.weight)  # Simulated gradient for testing
            sensitivity = self.compute_sensitivity(grad_tensor)

            # Quantize weights using K-Means clustering
            W_q, _ = self.apply_kmeans(self.weight, sensitivity)

        # Perform matrix multiplication
        output = torch.matmul(x, W_q.T)

        return output

# ---------------------------- #
#  Standard FP32 Linear Layer for Benchmark
# ---------------------------- #
class BaselineLinear(nn.Module):
    """Standard FP32 Linear Layer."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc(x)

# ---------------------------- #
#  Benchmarking Function
# ---------------------------- #
def benchmark_model(model, x, num_runs=50):
    """Measures inference speed of a model."""
    model.eval()
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x)

    torch.cuda.synchronize()
    return (time.time() - start_time) * 1000 / num_runs  # Convert to ms

# ---------------------------- #
#  Run Benchmarking
# ---------------------------- #
if __name__ == "__main__":
    batch_size, input_dim, output_dim = 32, 512, 512
    x = torch.randn(batch_size, input_dim).cuda()  # Simulated input

    # Initialize models
    clustered_model = SensitivityBasedQuantizedLinearINT4(input_dim, output_dim).cuda()
    baseline_model = BaselineLinear(input_dim, output_dim).cuda()

    # Run benchmarks
    baseline_time = benchmark_model(baseline_model, x)
    clustered_time = benchmark_model(clustered_model, x)

    # Print Benchmark Results
    print(f"Baseline FP32 Model: {baseline_time:.3f} ms per run")
    print(f"Sensitivity-Based INT4 Model: {clustered_time:.3f} ms per run")
    print(f"Speedup (INT4 vs FP32): {baseline_time / clustered_time:.2f}x")
