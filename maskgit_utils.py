import math
import torch

# ❤️ https://github.com/lucidrains/muse-maskgit-pytorch
def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t, generator=None):
    noise = torch.zeros_like(t).uniform_(0, 1, generator=generator)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1, generator=None):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t, generator=generator)).argmax(dim=dim)


def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)


def linear_schedule(t):
    mask_ratio = 1 - t
    mask_ratio = mask_ratio.clamp(min=1e-6, max=1.0)
    return mask_ratio


def mask_inputs(x, mask_schedule, mask_id):
    batch_size, seq_len = x.shape
    # Sample a random timestep for each image
    timesteps = torch.rand(batch_size, device=x.device)
    # Sample a random mask probability for each image using timestep and cosine schedule
    mask_prob = mask_schedule(timesteps)

    # creat a random mask for each sample
    num_token_masked = (seq_len * mask_prob).round().clamp(min=1)
    batch_randperm = torch.rand(batch_size, seq_len, device=x.device).argsort(dim=-1)
    mask = batch_randperm < num_token_masked.unsqueeze(-1)
    mask = mask.reshape(batch_size, seq_len)

    input_ids = torch.where(mask, mask_id, x)
    labels = torch.where(mask, x, -100) # -100 is ignored
    return input_ids, labels, mask

if __name__ == "__main__":
    # Test masking
    import matplotlib.pyplot as plt
    t = torch.arange(0, 1, 0.1)
    plt.plot(t, cosine_schedule(t))
    plt.show()
    x = torch.arange(12).reshape(3, 4)
    print(x)
    input_ids, labels, mask = mask_inputs(x, cosine_schedule, 100)
    print(input_ids)
    print(labels)
    print(mask)