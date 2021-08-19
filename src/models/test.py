import torch


def test_model_inputs(G, D, DEVICE="cpu"):
    with torch.no_grad():
        G.eval().to(DEVICE)
        D.eval().to(DEVICE)

        tensor = torch.rand(2, 3, 128, 128).to(DEVICE)

        G_out = G(tensor)
        D_out = D(G_out)
    return G_out, D_out
