import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

# Model Definition
class GraphSAGE(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_layers=None,
                 out_channels=3,
                 dropout=0.2,
                 activation="relu",
                 aggr="mean"):               # <-- Neu: aggr-Parameter
        super().__init__()

        hidden_layers = hidden_layers or [128]
        dims = [in_channels, *hidden_layers, out_channels]

        self.convs = torch.nn.ModuleList(
            SAGEConv(dims[i], dims[i + 1], aggr=aggr)   # <-- aggr einfügen
            for i in range(len(dims) - 1)
        )
        self.dropout = dropout
        self.act = getattr(F, activation)

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = self.act(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


# Train function
def train_sage(
    model,
    loader,
    optimiser,
    loss_fn,
    device,
    epochs=200,
    log_every=50,
):
    """Trains the GraphSAGE model and returns the loss of the last epoch."""
    model.to(device).train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            batch = batch.to(device)

            optimiser.zero_grad()
            preds = model(batch.x, batch.edge_index)

            mask = ~torch.isnan(batch.y).any(dim=1)
            loss = loss_fn(preds[mask], batch.y[mask])

            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()

        if (epoch + 1) % log_every == 0:
            print(
                f'Epoch {epoch + 1:3d}/{epochs} | loss {epoch_loss / len(loader):.4f}'
            )

    return epoch_loss / len(loader)