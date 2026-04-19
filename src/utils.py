import torch
import torch.nn as nn
import torch.nn.functional as F
from config import MARGIN, TRIPLET_MINING


class TripletLoss(nn.Module):
    def __init__(self, margin=MARGIN, mining=TRIPLET_MINING):
        super().__init__()
        self.margin = margin
        self.mining = mining

    def pairwise_dist(self, emb):
        dot   = emb @ emb.T
        sq    = dot.diagonal().unsqueeze(1)
        dist2 = (sq + sq.T - 2 * dot).clamp(min=1e-12)
        return dist2.sqrt()

    def masks(self, labels):
        eq  = labels.unsqueeze(0) == labels.unsqueeze(1)
        eye = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
        return eq & ~eye, ~eq

    def all(self, dist, pos_mask, neg_mask):
        d_ap   = dist.unsqueeze(2)
        d_an   = dist.unsqueeze(1)
        losses = F.relu(d_ap - d_an + self.margin)
        valid  = pos_mask.unsqueeze(2) & neg_mask.unsqueeze(1)
        losses = losses * valid.float()
        n_act  = int((losses > 0).sum().item())
        return losses.sum() / (n_act + 1e-8), n_act

    def hard(self, dist, pos_mask, neg_mask):
        INF  = 1e9
        d_ap = (dist * pos_mask.float()).max(dim=1).values
        d_an = (dist + (~neg_mask).float() * INF).min(dim=1).values
        losses = F.relu(d_ap - d_an + self.margin)
        valid  = pos_mask.any(dim=1) & neg_mask.any(dim=1)
        losses = losses * valid.float()
        n_act  = int((losses > 0).sum().item())
        return losses.mean(), n_act

    def semi(self, dist, pos_mask, neg_mask):
        INF      = 1e9
        d_ap_col = dist.unsqueeze(2)
        d_an_row = dist.unsqueeze(1)

        sh_mask = (
            neg_mask.unsqueeze(1)
            & (d_an_row > d_ap_col)
            & (d_an_row < d_ap_col + self.margin)
        )
        d_an_sh = torch.where(
            sh_mask,
            d_an_row.expand_as(sh_mask),
            torch.full_like(d_an_row.expand_as(sh_mask), INF),
        ).min(dim=2).values

        d_an_hard = (dist + (~neg_mask).float() * INF).min(dim=1, keepdim=True).values
        d_an_used = torch.where(d_an_sh < INF / 2, d_an_sh, d_an_hard.expand_as(d_an_sh))

        losses = F.relu(dist - d_an_used + self.margin) * pos_mask.float()
        n_act  = int((losses > 0).sum().item())
        return losses.sum() / (n_act + 1e-8), n_act

    def forward(self, embeddings, labels):
        dist = self.pairwise_dist(embeddings)
        pos_mask, neg_mask = self.masks(labels)
        if self.mining == "all":
            return self.all(dist, pos_mask, neg_mask)
        elif self.mining == "hard":
            return self.hard(dist, pos_mask, neg_mask)
        elif self.mining == "semi":
            return self.semi(dist, pos_mask, neg_mask)
        else:
            raise ValueError(f"Unknown mining: {self.mining!r}")


def accuracy(dist, label, threshold):
    pred    = (dist >= threshold).long()
    correct = (pred == label).sum().item()
    return correct / max(label.size(0), 1)