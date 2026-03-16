import numpy as np
import cv2
from ConvolutionLayer import ConvolutionLayer as CONV
from RectifiedLinearUnitLayer import RectifiedLinearUnitLayer as RELU
from PoolingLayer import PoolingLayer as POOL

SEP = "=" * 50
N, M, K = 3, 6, 1
# N=nombre de blocs CONV+RELU, M=nombre de blocs avec POOL, K=nombre de blocs FC+RELU
# INPUT -> [[CONV -> RELU]*N -> POOL]*M -> [FC -> RELU]*K -> FC

def save_image(data, path):
    norm = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(path, norm)
    return path

def build_convolved_output(conv_layer, image):
    kernel_h, kernel_w = conv_layer.kernel.shape
    img_h, img_w = image.shape
    out_h = (img_h - kernel_h) // conv_layer.stride + 1
    out_w = (img_w - kernel_w) // conv_layer.stride + 1
    output = np.zeros((out_h, out_w), dtype=np.float64)
    row, col = 0, 0
    for patch in conv_layer.patch_generator(image):
        output[row, col] = conv_layer.kernel_convolution(patch)
        col += 1
        if col >= out_w:
            col = 0
            row += 1
    return output

def main():
    path   = "../data/3_image_generates/outputs/bacteria-5012.jpg"
    kernel = np.array([[0, 1, 1], [0, 0, 1], [-1, -1, 0]])
    stride = 1

    print(SEP)
    print("  TruePneumoniaAI — Pipeline CNN")
    print(f"  Architecture : [[CONV -> RELU]*{N} -> POOL]*{M}")
    print(SEP)

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"[ERREUR] Impossible de charger '{path}'")
        return

    print(f"[OK] Image    : {image.shape[1]}x{image.shape[0]} px")
    print(f"[OK] Kernel   : {kernel.shape[0]}x{kernel.shape[1]}")

    relu = RELU()
    pool = POOL(pool_size=2, stride=2)
    data = image.astype(np.float64)

    for m in range(M):
        print(f"\n{SEP}")
        print(f"  Bloc {m+1}/{M}")
        print(SEP)

        for n in range(N):
            # --- CONV ---
            conv_layer = CONV(kernel, stride)
            data = build_convolved_output(conv_layer, data)
            tag = f"bloc{m+1}_conv{n+1}"
            save_image(data, f"outputs/{tag}.jpg")
            print(f"[CONV {m+1}.{n+1}] Sortie : {data.shape[1]}x{data.shape[0]} px"
                  f"  min={data.min():.1f}  max={data.max():.1f}  -> {tag}.jpg")

            # --- RELU ---
            data = relu.forward(data)
            tag = f"bloc{m+1}_relu{n+1}"
            save_image(data, f"outputs/{tag}.jpg")
            print(f"[RELU {m+1}.{n+1}] Sortie : {data.shape[1]}x{data.shape[0]} px"
                  f"  min={data.min():.1f}  max={data.max():.1f}  -> {tag}.jpg")

        # --- POOL ---
        data = pool.forward(data[np.newaxis, np.newaxis, :, :])[0, 0]
        tag = f"bloc{m+1}_pool"
        save_image(data, f"outputs/{tag}.jpg")
        print(f"[POOL {m+1}  ] Sortie : {data.shape[1]}x{data.shape[0]} px"
              f"  min={data.min():.1f}  max={data.max():.1f}  -> {tag}.jpg")

    print(f"\n{SEP}")
    print("  Pipeline terminé")
    print(SEP)

if __name__ == "__main__":
    main()
