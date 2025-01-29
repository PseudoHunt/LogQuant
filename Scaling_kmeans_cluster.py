import torch
import torch.nn as nn
from sklearn.cluster import KMeans

class ScaledQuantizedLinearINT4(nn.Module):
    def __init__(self, in_features, out_features, bits=4, alpha=0.5, num_clusters=16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.alpha = nn.Parameter(torch.tensor(alpha))  # Learnable scaling factor
        self.num_clusters = num_clusters  # Number of unique centroids

        # Define trainable weight matrix (FP32 initially)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

        # Register buffers for quantization
        self.register_buffer("scale_factor", torch.tensor(1.0))  # For uniform quantization
        self.register_buffer("centroids", torch.zeros(num_clusters))  # Store k-means centroids

    def compute_kmeans_centroids(self):
        """Computes k-means centroids for the weight matrix."""
        with torch.no_grad():
            W_scaled = torch.sign(self.weight) * torch.pow(torch.abs(self.weight), self.alpha)  # Apply scaling
            W_flat = W_scaled.view(-1).cpu().numpy()  # Flatten for k-means clustering

            # Run k-means clustering
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
            kmeans.fit(W_flat.reshape(-1, 1))

            # Convert centroids to a PyTorch tensor and store
            centroids = torch.tensor(kmeans.cluster_centers_.flatten(), dtype=torch.float32).cuda()
            self.centroids.copy_(centroids)  # Store in buffer

            # Store weight indices (closest centroid for each weight)
            clustered_indices = torch.tensor(kmeans.labels_, dtype=torch.long).reshape(self.weight.shape).cuda()
            return clustered_indices

    def quantize_weights(self):
        """Quantizes weights using k-means centroids and uniform quantization."""
        clustered_indices = self.compute_kmeans_centroids()  # Get nearest centroid indices

        # Compute quantization scale
        W_min, W_max = self.centroids.min(), self.centroids.max()
        self.scale_factor = (W_max - W_min) / ((2 ** self.bits) - 1)  # Scale for 4-bit quantization

        # Quantize indices (already in [0, num_clusters-1])
        W_q = clustered_indices.clamp(0, self.num_clusters - 1)  # Ensure it's within range
        return W_q.to(torch.int8)  # Store as INT4 (emulated in INT8)

    def forward(self, x):
        # Step 1: Scale input before quantization
        x_scaled = torch.sign(x) * torch.pow(torch.abs(x), self.alpha)

        # Step 2: Quantize weights (retrieve k-means indices)
        W_q = self.quantize_weights()

        # Step 3: Convert quantized indices back to values using centroids
        W_reconstructed = torch.gather(self.centroids, 0, W_q)  # Retrieve centroid values

        # Step 4: Perform Integer Matrix Multiplication (Emulating INT4)
        output = torch.matmul(x_scaled, W_reconstructed.T)  # INT4 GEMM

        # Step 5: Reverse Scaling After Multiplication
        output = torch.sign(output) * torch.pow(torch.abs(output), 1 / self.alpha)

        return output

# Example Usage
batch_size, input_dim, output_dim = 32, 512, 512
x = torch.randn(batch_size, input_dim).cuda()  # Input matrix
quantized_linear = ScaledQuantizedLinearINT4(input_dim, output_dim).cuda()

output = quantized_linear(x)
print("Output Shape:", output.shape)  # Should be (32, 512)
