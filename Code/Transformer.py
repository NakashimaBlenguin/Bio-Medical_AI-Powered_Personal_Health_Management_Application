import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from torch.utils.data import DataLoader

# Scaled Dot-Product Attention computes attention scores and weights
class ScaledDotProductAttention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)  # Dimensionality of keys
        # Compute the attention scores using scaled dot product
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        if mask is not None:
            # Apply mask to ignore specific positions
            scores = scores.masked_fill(mask == 0, -1e9)
        # Normalize scores to probabilities
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        # Compute the weighted sum of values
        return torch.matmul(p_attn, value), p_attn

# Multi-Head Attention combines scaled dot-product attention across multiple heads
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads  # Dimensionality per head
        self.num_heads = num_heads

        # Linear layers for projecting queries, keys, and values
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model, bias=False)

        # Scaled dot-product attention mechanism
        self.attention = ScaledDotProductAttention()
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)  # Normalize outputs for stability

    def forward(self, query, key, value, mask=None, register_hook=False):
        batch_size = query.size(0)
        # Project input to multiple heads and reshape
        query, key, value = [
            l(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            for l, x in zip(self.linear_layers, (query, key, value))
        ]
        if mask is not None:
            mask = mask.unsqueeze(1)
        # Apply attention and combine heads
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        # Add and normalize the output
        x = self.layer_norm(self.output_linear(x) + x)
        return x

# Position-wise Feed-Forward Network applies transformations to each position independently
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        # Two linear layers with dropout and activation in between
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        residual = x  # Save the input for residual connection
        x = self.w_1(x)  # Expand dimensionality
        x = self.activation(x)  # Apply non-linearity
        x = self.dropout(x)  # Dropout for regularization
        x = self.w_2(x)  # Compress back to original dimensionality
        return self.layer_norm(x + residual)  # Add residual and normalize

# Transformer Block combining Multi-Head Attention and Feed-Forward Network
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(num_heads, d_model, dropout)  # Multi-head attention module
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)  # Feed-forward network

    def forward(self, x, mask=None):
        x = self.attention(x, x, x, mask)  # Apply self-attention
        x = self.feed_forward(x)  # Apply position-wise feed-forward network
        return x

# Model for predicting chronic diseases based on input features
class ChronicDiseasePredictor(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, d_ff, num_diseases, dropout=0.1):
        super(ChronicDiseasePredictor, self).__init__()
        self.feature_mapping = nn.Linear(input_dim, d_model)  # Map input features to model dimensionality
        self.transformer_block = TransformerBlock(d_model, num_heads, d_ff, dropout)  # Transformer encoder block
        self.output_layer = nn.Linear(d_model, num_diseases)  # Output layer for disease predictions

    def forward(self, x, mask=None):
        x = self.feature_mapping(x)  # Project input features
        x = self.transformer_block(x, mask)  # Pass through transformer block
        x = x.mean(dim=1)  # Aggregate across sequence dimension
        x = self.output_layer(x)  # Generate predictions
        return x

if __name__ == "__main__":
    # Load synthetic EHR dataset
    with open("/content/synthetic_ehr_dataset_10k_downloadable.json", "r") as f:
        synthetic_data = json.load(f)

    # Dataset for processing synthetic EHR data
    class SyntheticEHRDataset(torch.utils.data.Dataset):
        def __init__(self, data, num_diseases):
            self.data = data
            self.num_diseases = num_diseases

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            sample = self.data[idx]
            # Compute average feature vector for the sample
            features = torch.tensor(sample["list_vectors"], dtype=torch.float32).mean(dim=0)
            # One-hot encode the disease label
            label = torch.zeros(self.num_diseases, dtype=torch.float32)
            label[sample["label"]] = 1
            return features, label

    num_diseases = 5
    dataset = SyntheticEHRDataset(synthetic_data, num_diseases=num_diseases)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize the model, optimizer, and loss function
    model = ChronicDiseasePredictor(input_dim=10, d_model=512, num_heads=8, d_ff=2048, num_diseases=num_diseases, dropout=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    for epoch in range(100):
        for data_batch in train_loader:
            features, labels = data_batch  # Extract batch data
            optimizer.zero_grad()  # Clear gradients
            outputs = model(features)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters
        print(f"Epoch {epoch + 1}: Loss = {loss.item()}")
