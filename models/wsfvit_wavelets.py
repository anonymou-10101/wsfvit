import pywt
import torch
import torch.nn as nn
from torch.autograd import Function

class DWT_Function_FFT_L2(Function):
    @staticmethod
    def forward(ctx, x, w_l, w_h):

        B, C, H, W = x.shape
        w_ls = w_l.shape[1]
        w_hs = w_h.shape[1]

        H2n = int(2 ** torch.ceil(torch.log2(torch.as_tensor(H))))
        W2n = int(2 ** torch.ceil(torch.log2(torch.as_tensor(W))))

        if H2n > W2n:
            W2n = H2n
        else:
            H2n = W2n

        n_padding_w = W2n-W
        n_padding_h = H2n-H
        padding = (0, n_padding_w, 0, n_padding_h, 0, 0, 0, 0)
        x = torch.nn.functional.pad(x, padding)
        
        padding = (0, H2n-w_ls)
        w_l = torch.nn.functional.pad(w_l, padding)
        w_l = w_l.repeat(B,C,W2n,1)

        padding = (0, W2n-w_hs)
        w_h = torch.nn.functional.pad(w_h, padding)
        w_h = w_h.repeat(B,C,H2n,1)

        x_fft = torch.fft.fft(x, dim = 3)
        w_l_fft = torch.fft.fft(w_l, dim = 3)
        w_h_fft = torch.fft.fft(w_h, dim = 3)
        x_l_fft = x_fft * w_l_fft
        x_h_fft = x_fft * w_h_fft
        x_l = torch.fft.ifft(x_l_fft, dim = 3)
        x_h = torch.fft.ifft(x_h_fft, dim = 3)

        x_l = x_l.real
        x_h = x_h.real
        
        shift = int(w_ls/2)
        x_l_s = torch.roll(x_l, shifts=-shift, dims=-1)
        shift = int(w_hs/2)
        x_h_s = torch.roll(x_h, shifts=-shift, dims=-1)
        x_l_s_t = x_l_s.transpose(-2, -1)
        x_h_s_t = x_h_s.transpose(-2, -1)

        x_l_fft = torch.fft.fft(x_l_s_t, dim = 3)
        x_h_fft = torch.fft.fft(x_h_s_t, dim = 3)

        x_ll_fft = x_l_fft * w_l_fft
        x_lh_fft = x_l_fft * w_h_fft
        x_hl_fft = x_h_fft * w_l_fft
        x_hh_fft = x_h_fft * w_h_fft

        x_ll = torch.fft.ifft(x_ll_fft, dim = 3)
        x_lh = torch.fft.ifft(x_lh_fft, dim = 3)
        x_hl = torch.fft.ifft(x_hl_fft, dim = 3)
        x_hh = torch.fft.ifft(x_hh_fft, dim = 3)

        x_ll = x_ll.real
        x_lh = x_lh.real
        x_hl = x_hl.real
        x_hh = x_hh.real

        shift = int(w_ls/2)
        x_ll_s = torch.roll(x_ll, shifts=-shift, dims=-1)
        x_lh_s = torch.roll(x_lh, shifts=-shift, dims=-1)
        shift = int(w_hs/2)
        x_hl_s = torch.roll(x_hl, shifts=-shift, dims=-1)
        x_hh_s = torch.roll(x_hh, shifts=-shift, dims=-1)

        x_ll_t = x_ll_s.transpose(-2, -1)
        x_lh_t = x_lh_s.transpose(-2, -1)
        x_hl_t = x_hl_s.transpose(-2, -1)
        x_hh_t = x_hh_s.transpose(-2, -1)

        x_ll = x_ll_t[..., ::1, ::1]
        x_lh = x_lh_t[..., ::1, ::1]
        x_hl = x_hl_t[..., ::1, ::1]
        x_hh = x_hh_t[..., ::1, ::1]

        n_padding_h = int(n_padding_h)
        n_padding_w = int(n_padding_w)

        if H2n-W == 0 and H2n-H == 0:
            x_ll = x_ll
            x_lh = x_lh
            x_hl = x_hl
            x_hh = x_hh
        else:
            x_ll = x_ll[..., :-n_padding_h, :-n_padding_w]
            x_lh = x_lh[..., :-n_padding_h, :-n_padding_w]
            x_hl = x_hl[..., :-n_padding_h, :-n_padding_w]
            x_hh = x_hh[..., :-n_padding_h, :-n_padding_w]
    
        return x_ll, x_lh, x_hl, x_hh


class DWT_2D_FFT_L2(nn.Module):
    def __init__(self, wave):
        super(DWT_2D_FFT_L2, self).__init__()
        w = pywt.Wavelet(wave)

        dec_lo = torch.Tensor(w.dec_lo[::-1])
        dec_hi = torch.Tensor(w.dec_hi[::-1]) 

        w_l = dec_lo.unsqueeze(0)
        w_h = dec_hi.unsqueeze(0)

        self.register_buffer('w_l', w_l)
        self.register_buffer('w_h', w_h)


        self.w_l = self.w_l.to(dtype=torch.float32)
        self.w_h = self.w_h.to(dtype=torch.float32)

    def forward(self, x):
        return DWT_Function_FFT_L2.apply(x, self.w_l, self.w_h)
