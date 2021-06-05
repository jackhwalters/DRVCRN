import torch


def l2_norm(s1, s2):
    norm = torch.sum(s1*s2, -1, keepdim=True)
    return norm

class SiSnr(object):
    def __call__(self, estimate, clean, eps=1e-8):
        dot = l2_norm(estimate, clean)
        norm = l2_norm(clean, clean)

        s_target =  (dot * clean)/(norm+eps)
        e_nosie = estimate - s_target

        target_norm = l2_norm(s_target, s_target)
        noise_norm = l2_norm(e_nosie, e_nosie)
        snr = 10*torch.log10((target_norm)/(noise_norm+eps)+eps)

        return torch.mean(snr)
