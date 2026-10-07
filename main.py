import sys
from numba import cuda, float32

import numpy as np
import matplotlib.image as image

# Config
R = 21
K = 7
H_PARAM = 10.0

RS = R // 2
KS = K // 2
PAD = RS + KS
H2 = H_PARAM * H_PARAM
NORM = float(K * K * 3) 

device=cuda.select_device(0)
print(f"Select cuda: {device}")


def show_image(path):
    img = image.imread(path)

    cuda.to_device(img)

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]

    if img.dtype == np.uint8:
        img = img.astype(np.float32)
    else:
        img = img.astype(np.float32) * 255.0

    h, w, c = img.shape
    print(f"{h}, {w}, {c}, {img.dtype}")
    print(f'Value range: [{img.min():.1f}, {img.max():.1f}]')
    print(f'Pixels     : {w * h:,}')

    return img


# Method
@cuda.jit(device=True, inline=True)
def weight(d2):
    return np.exp(-d2 / H2)


@cuda.jit(device=True, inline=True)
def patch_distance2(src, iy, ix, jy, jx):
    acc = float32(0.0)
    for py in range(-KS, KS + 1):
        for px in range(-KS, KS + 1):
            for c in range(3):
                diff = src[iy + py, ix + px, c] - src[jy + py, jx + px, c]
                acc += diff * diff
    return acc / NORM


@cuda.jit
def nlm_kernel(src, dst):
    x, y = cuda.grid(2)
    if y >= dst.shape[0] or x >= dst.shape[1]:
        return

    iy = y + PAD
    ix = x + PAD

    acc_r = float32(0.0)
    acc_g = float32(0.0)
    acc_b = float32(0.0)
    acc_w = float32(0.0)

    for dy in range(-RS, RS + 1):
        for dx in range(-RS, RS + 1):
            jy = iy + dy
            jx = ix + dx

            d2 = patch_distance2(src, iy, ix, jy, jx)
            w = weight(d2)

            acc_r += w * src[jy, jx, 0]
            acc_g += w * src[jy, jx, 1]
            acc_b += w * src[jy, jx, 2]
            acc_w += w

    dst[y, x, 0] = acc_r / acc_w
    dst[y, x, 1] = acc_g / acc_w
    dst[y, x, 2] = acc_b / acc_w

