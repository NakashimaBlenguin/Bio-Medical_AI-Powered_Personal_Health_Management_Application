import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_model, d_model, bias=False) for _ in range(3)]
        )
        self.output_linear = nn.Linear(d_model, d_model, bias=False)
        self.attention = ScaledDotProductAttention()

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)  # Added layer normalization

        self.attn_gradients = None
        self.attn_map = None

    # helper functions for interpretability
    def get_attn_map(self):
        return self.attn_map 
    
    def get_attn_grad(self):
        return self.attn_gradients

    def save_attn_grad(self, attn_grad):
        self.attn_gradients = attn_grad 

    # register_hook option allows us to save the gradients in backwarding
    def forward(self, query, key, value, mask=None, register_hook=False):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => num_heads x d_k
        query, key, value = [
            l(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            for l, x in zip(self.linear_layers, (query, key, value))
        ]
        
        # 2) Apply attention on all the projected vectors in batch.
        if mask is not None:
            mask = mask.unsqueeze(1)
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        
        self.attn_map = attn # save the attention map
        if register_hook:
            attn.register_hook(self.save_attn_grad)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
  
        # Apply layer normalization for better performance
        x = self.layer_norm(self.output_linear(x) + x)
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(d_model)  # Added layer normalization

    def forward(self, x, mask=None):
        residual = x  # Save the input for residual connection
        x = self.w_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.w_2(x)
        # Apply layer normalization with residual connection
        return self.layer_norm(x + residual)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(num_heads, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        x = self.attention(x, x, x, mask)
        x = self.feed_forward(x)
        return x


class ChronicDiseasePredictor(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_diseases, dropout=0.1):
        super(ChronicDiseasePredictor, self).__init__()
        self.transformer_block = TransformerBlock(d_model, num_heads, d_ff, dropout)
        self.output_layer = nn.Linear(d_model, num_diseases)  # Output layer for multiple diseases

    def forward(self, x, mask=None):
        x = self.transformer_block(x, mask)
        x = x.mean(dim=1)  # Pooling across the sequence dimension
        x = self.output_layer(x)
        return x

if __name__ == "__main__":
    from pyhealth.datasets import SampleEHRDataset

    samples = [
        {
            "patient_id": "patient-0",
            "visit_id": "visit-0",
            "single_vector": [1, 2, 3],
            "list_codes": ["505800458", "50580045810", "50580045811"],  # NDC
            "list_vectors": [[1.0, 2.55, 3.4], [4.1, 5.5, 6.0]],
            "list_list_codes": [["A05B", "A05C", "A06A"], ["A11D", "A11E"]],  # ATC-4
            "list_list_vectors": [
                [[1.8, 2.25, 3.41], [4.50, 5.9, 6.0]],
                [[7.7, 8.5, 9.4]],
            ],
            "label": 1,
        },
        {
            "patient_id": "patient-0",
            "visit_id": "visit-1",
            "single_vector": [1, 5, 8],
            "list_codes": [
                "55154191800",
                "551541928",
                "55154192800",
                "705182798",
                "70518279800",
            ],
            "list_vectors": [[1.4, 3.2, 3.5], [4.1, 5.9, 1.7], [4.5, 5.9, 1.7]],
            "list_list_codes": [["A04A", "B035", "C129"]],
            "list_list_vectors": [
                [[1.0, 2.8, 3.3], [4.9, 5.0, 6.6], [7.7, 8.4, 1.3], [7.7, 8.4, 1.3]],
            ],
            "label": 0,
        },
    ]

    # dataset
    dataset = SampleEHRDataset(samples=samples, dataset_name="test")

    # data loader
    from pyhealth.datasets import get_dataloader

    train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)

    # model
    num_diseases = 5  # Assume we are predicting 5 common chronic diseases
    model = ChronicDiseasePredictor(d_model=512, num_heads=8, d_ff=2048, num_diseases=num_diseases, dropout=0.1)

    # data batch
    data_batch = next(iter(train_loader))

    # try the model
    query = torch.randn(2, 10, 512)  # Example input tensor for query
    ret = model(query)
    print(ret)

    # try loss backward
    target = torch.randint(0, 2, (2, num_diseases)).float()  # Example target for multiple diseases (binary classification)
    criterion = nn.BCEWithLogitsLoss()  # Use binary cross-entropy loss for multi-label classification
    loss = criterion(ret, target)
    loss.backward()
