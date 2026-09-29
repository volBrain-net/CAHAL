import torch
import torch.nn.functional as F


def correlation(y_pred, y_true):
    x, y = y_true, y_pred
    xm, ym = x - x.mean(), y - y.mean()
    r_num = torch.sum(xm * ym)
    r_den = torch.sqrt(torch.sum(xm**2) * torch.sum(ym**2))
    return r_num / (r_den + 1e-10)


def get_laplacian_kernel_3d(device):
    kernel = torch.tensor([[
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    ]]).float().to(device)
    return kernel.unsqueeze(0)  # [1, 1, 3, 3, 3]


def WMAE(y_pred, y_true, eps=1e-6, gamma=1.0):
    """MAE weighted by the (log-compressed) Laplacian response of y_true, emphasizing edges."""
    kernel = get_laplacian_kernel_3d(y_true.device)
    lap_mag = torch.abs(F.conv3d(y_true, kernel, padding=1))
    lap_log = torch.log1p(gamma * lap_mag)
    weights = lap_log / (lap_log.max() + eps)
    mae = torch.abs(y_pred - y_true)
    return (mae * (1 + weights)).mean()


def fft_loss(y_pred, y_true):
    """Frequency-domain loss that acts as perceptual loss."""
    y_pred, y_true = y_pred.float(), y_true.float()
    pred_fft = torch.fft.fftn(y_pred, dim=(2, 3, 4))
    true_fft = torch.fft.fftn(y_true, dim=(2, 3, 4))
    diff = torch.abs(pred_fft - true_fft) ** 2
    return diff.mean()


def mixloss(y_pred, y_true):
    l_wmae = WMAE(y_pred, y_true)
    with torch.cuda.amp.autocast(enabled=False):
        l_ft = 0.000002 * fft_loss(y_pred, y_true)
    return l_wmae + l_ft


def mdice(y_true, y_pred):
    num_classes = y_pred.size(1)
    dice = 0
    for c in range(num_classes):
        a, b = y_true[:, c, ...], y_pred[:, c, ...]
        dice += 2 * torch.sum(a * b) / (torch.sum(a) + torch.sum(b))
    return dice / num_classes


def mdice_loss(y_true, y_pred):
    return 1 - mdice(y_true, y_pred)


def calculate_psnr(original, reconstructed):
    max_value = torch.max(original)
    mse = F.mse_loss(original, reconstructed)
    mse = mse if mse != 0 else 1e-10
    return 10 * torch.log10((max_value**2) / mse)
