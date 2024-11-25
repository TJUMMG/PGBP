import torch

def generate_gaussian_tensor(inter_label, key_frames, hp_sigma):
    """
    Generate a tensor with each batch as a Gaussian sequence.

    :param B: Batch size.
    :param L: Length of each sequence.
    :param key_frames: Tensor of shape (B,) containing key frames.
    :param variances: Tensor of shape (B,) containing variances.
    :return: Tensor with shape (B, L) containing Gaussian sequences.
    """
    # Generate a range of values from 0 to L-1
    B,L = inter_label.shape
    variances = hp_sigma * torch.sum(inter_label,dim =1)
    x_values = torch.arange(0, L, 1).float().cuda()

    # Repeat key_frames and variances for each batch
    key_frames = key_frames.view(-1, 1).repeat(1, L)
    variances = variances.view(-1, 1).repeat(1, L)

    # Calculate Gaussian values using the norm.pdf function
    gaussian_values = torch.exp(-(x_values - key_frames)**2 / (2 * variances**2))
    return gaussian_values


