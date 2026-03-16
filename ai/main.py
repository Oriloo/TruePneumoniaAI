import numpy as np
import cv2
from ConvolutionLayer import ConvolutionLayer as CONV
from RectifiedLinearUnitLayer import RectifiedLinearUnitLayer as RELU
from PoolingLayer import PoolingLayer as POOL
from GlobalAveragePoolingLayer import GlobalAveragePoolingLayer as GAP
from ClassActivationMapLayer import ClassActivationMapLayer as CAM
from FullyConnected import FullyConnectedLayer as FC

SEP = "=" * 50
N, M, K = 3, 5, 1
# N=nombre de blocs CONV+RELU, M=nombre de blocs avec POOL, K=nombre de blocs FC+RELU
# INPUT -> [[CONV -> RELU]*N -> POOL]*M -> [FC -> RELU]*K -> FC

def save_image(data, path):
    d = np.mean(data, axis=2) if data.ndim == 3 else data
    norm = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(path, norm)
    return path

def main():
    path   = "../data/3_image_generates/outputs/bacteria-8000.jpg"
    kernel = np.array([
        [[[ 0], [ 1], [ 1]], [[ 0], [ 0], [ 1]], [[-1], [-1], [ 0]]],
        [[[ 0], [ 0], [ 1]], [[ 0], [ 1], [ 1]], [[-1], [ 0], [ 0]]],
        [[[ 0], [ 1], [ 0]], [[ 0], [ 1], [ 0]], [[ 0], [-1], [-1]]],
    ])
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
            conv = CONV(kernel, stride)
            data = conv.forward(data)
            tag = f"bloc{m+1}_conv{n+1}"
            save_image(data, f"outputs/{tag}.jpg")
            print(f"[CONV {m+1}.{n+1}] Sortie : {data.shape[1]}x{data.shape[0]}x{data.shape[2]}"
                  f"  min={data.min():.1f}  max={data.max():.1f}  -> {tag}.jpg")

            # --- RELU ---
            data = relu.forward(data)
            tag = f"bloc{m+1}_relu{n+1}"
            save_image(data, f"outputs/{tag}.jpg")
            print(f"[RELU {m+1}.{n+1}] Sortie : {data.shape[1]}x{data.shape[0]}x{data.shape[2]}"
                  f"  min={data.min():.1f}  max={data.max():.1f}  -> {tag}.jpg")

        # --- POOL ---
        D = data.shape[2]
        pooled = []
        for d in range(D):
            fm = data[:, :, d]
            p = pool.forward(fm[np.newaxis, np.newaxis, :, :])[0, 0]
            pooled.append(p)
        data = np.stack(pooled, axis=2)
        tag = f"bloc{m+1}_pool"
        save_image(data, f"outputs/{tag}.jpg")
        print(f"[POOL {m+1}  ] Sortie : {data.shape[1]}x{data.shape[0]}x{data.shape[2]}"
              f"  min={data.min():.1f}  max={data.max():.1f}  -> {tag}.jpg")

    print(f"\n{SEP}")
    print("  Pipeline terminé")
    print(SEP)

if __name__ == "__main__":
    main()
