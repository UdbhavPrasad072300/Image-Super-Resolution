import torch


def Test_Model_Inputs(G, D, DEVICE="cpu"):
    with torch.no_grad():
        G.eval().to(DEVICE)
        D.eval().to(DEVICE)

        tensor = torch.rand(2, 3, 256, 256).to(DEVICE)

        G_out = G(tensor)
        D_out = D(G_out)
    return G_out, D_out
