import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    """
    h: Height of the patch.
    w: Width of the patch.
    dim: The dimension of the model embeddings.
    """

    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"

    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


def triplet_loss(queries, keys, margin=1.0):
    """
    Inputs:
    queries (b x D): A batch of training examples.
    keys (b x D): A batch of training examples. The ith example in keys is a positive
                  example for the ith example in queries.
    margin: The margin, m, in the equation above.

    Outputs:
    The triplet loss, calculated as described above.
    """
    b = queries.shape[0]  # batch size
    device = queries.device
    n = b * 2  # total number of examples

    # TODO1: Implement triplet loss
    # Hint: Whenever you create a new tensor, make sure to send it to the same
    #       location (device) your model and data are on.
    # Hint: How might you use matrices/matrix operations to keep track of distances between
    #       positive and negative pairs? (looking ahead to the instructions in part 1.2 maybe be useful)
    #################

    queries = F.normalize(queries, p=2, dim=1)
    keys = F.normalize(keys, p=2, dim=1)

    sims = torch.matmul(queries, keys.T) 

    pos_sims = torch.diag(sims).unsqueeze(1) 

    losses = F.relu(sims - pos_sims + margin)

    mask = ~torch.eye(b, dtype=torch.bool, device=device)

    loss = losses[mask].mean()

    return loss


def nt_xent_loss(queries, keys, temperature=0.1):
    """
    Inputs:
    queries (b x D): A batch of training examples.
    keys (b x D): A batch of training examples. The ith example in keys is a
                  differently-augmented view of the ith example in queries.
    temperature: The temperature, tau, in the equation above.

    Outputs:
    The SimCLR loss, calculated as described above.
    """
    b, device = queries.shape[0], queries.device
    n = b * 2

    # TODO2: Implement the SimCLR loss
    # Hint: Whenever you create a new tensor, make sure to send it to the same
    #       location (device) your model and data are on.
    # Hint: Which loss function does the first equation in step 3 remind you of?
    #################
    
    queries = F.normalize(queries, p=2, dim=1)
    keys = F.normalize(keys, p=2, dim=1)

    reps = torch.cat([queries, keys], dim=0)

    sim = reps @ reps.T 

    logits = sim / temperature
    mask = torch.eye(n, dtype=torch.bool, device=device)
    logits = logits.masked_fill(mask, float('-inf'))

    targets = torch.cat([
        torch.arange(b, 2 * b, device=device),
        torch.arange(0, b, device=device)
    ], dim=0)

    loss = F.cross_entropy(logits, targets)
    return loss
    
class ViT(nn.Module):
    def __init__(self, d_model, num_layers, patch_size=4, img_side_length=32, p=0.05):
        """
        Inputs:
        d_model: The dimension of the encoder embeddings.
        num_layers: Number of encoder layers.
        patch_size: Side length of the square image patches.
        img_side_length: The height and width of the images.
        p: Dropout probability.
        """
        super(ViT, self).__init__()

        d_ff = 4 * d_model
        num_heads = d_model // 32

        # TODO3: define the ViT
        #################
        
        channels = 3 * patch_size * patch_size
        patch_h = img_side_length // patch_size
        patch_w = img_side_length // patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_size,
                p2=patch_size,
            ),
            nn.LayerNorm(channels),
            nn.Linear(channels, d_model),
            nn.LayerNorm(d_model),
        )

        self.register_buffer(
            "pos_embedding",
            posemb_sincos_2d(patch_h, patch_w, d_model),
        )

        self.dropout = nn.Dropout(p)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=p,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_ln = nn.LayerNorm(d_model)

        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),   # if this still fails, try nn.ReLU()
            nn.Linear(d_model, d_model),
        )
        ################

    def forward(self, x, return_embedding=False):

        ## TODO4: Write the forward pass for the ViT
        #################
        x = self.to_patch_embedding(x)
        x = x + self.pos_embedding.unsqueeze(0)
        x = self.dropout(x)

        x = x.transpose(0, 1)
        x = self.encoder(x)
        x = x.transpose(0, 1)

        x = x.mean(dim=1)
        x = self.output_ln(x)

        if return_embedding:
            return x
        return self.projection_head(x)
        #################
