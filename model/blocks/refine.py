import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableSoftMorph(nn.Module):
    def __init__(self, k_open=3, k_close=5, tau=0.05):
        super().__init__()
        assert k_open >= 1 and k_close >= 1
        self.k_open, self.k_close = k_open, k_close
        self.log_tau = nn.Parameter(torch.log(torch.tensor(float(tau))))

        def make_kernel(k):
            n = k * k
            return nn.Parameter(torch.zeros(1, n))

        self.we_open_erode   = make_kernel(k_open)
        self.wd_open_dilate  = make_kernel(k_open)
        self.wd_close_dilate = make_kernel(k_close)
        self.we_close_erode  = make_kernel(k_close)

        # after sigomid: α ∈ [0,1]
        self.alpha_raw = nn.Parameter(torch.tensor(-5.0))  

    @staticmethod
    def _logsumexp_pool(x_cols, w, tau):
        # x_cols: [B, 1, K, HW]; w: [1, K]; tau: scalar
        # soft max: tau * logsumexp( (x + w)/tau, dim=2 )
        z = torch.logsumexp((x_cols + w.unsqueeze(-1)) / tau, dim=2) * tau
        return z  # [B, 1, HW]

    def _soft_dilate(self, x, k, w, tau):
        if k <= 1: return x
        B, _, H, W = x.shape
        pad = k // 2
        cols = F.unfold(x, k, padding=pad)           # [B, 1*K*K, HW]
        cols = cols.view(B, 1, k*k, H*W)             # [B, 1, K*K, HW]
        z = self._logsumexp_pool(cols, w, tau)       # [B, 1, HW]
        return z.view(B, 1, H, W)

    def _soft_erode(self, x, k, w, tau):
        # min(x+s) = -softmax( -x + (-s) )
        return -self._soft_dilate(-x, k, -w, tau)

    def forward(self, logit_2ch):
        _, C, _, _ = logit_2ch.shape
        assert C == 2, "Expect 2-channel logits for binary segmentation."

        p_fg = F.softmax(logit_2ch, dim=1)[:, 1:2]  # [B,1,H,W]
        tau = torch.exp(self.log_tau).clamp_min(1e-4)

        # Open: erode -> dilate
        p = self._soft_erode (p_fg, self.k_open,  self.we_open_erode,   tau)
        p = self._soft_dilate(p,    self.k_open,  self.wd_open_dilate,  tau)
        # Close: dilate -> erode
        p = self._soft_dilate(p,    self.k_close, self.wd_close_dilate, tau)
        p = self._soft_erode (p,    self.k_close, self.we_close_erode,  tau)

        fg_logit_refined = torch.logit(p.clamp(1e-6, 1-1e-6), eps=1e-6)

        alpha = torch.sigmoid(self.alpha_raw)  # [0,1]
        out = logit_2ch.clone()
        out[:, 1:2] = out[:, 1:2] + alpha * (fg_logit_refined - out[:, 1:2])

        return out