import numpy as np
import cv2
from ConvolutionLayer import ConvolutionLayer as CONV
from RectifiedLinearUnitLayer import RectifiedLinearUnitLayer as RELU
from PoolingLayer import PoolingLayer as POOL
from GlobalAveragePoolingLayer import GlobalAveragePoolingLayer as GAP
from ClassActivationMapLayer import ClassActivationMapLayer as CAM
from FullyConnected import FullyConnectedLayer as FC

SEP = "=" * 50
N, M = 3, 5 # N=nombre de blocs CONV+RELU, M=nombre de blocs avec POOL
FC_HIDDEN = 128 # Nombre de neurones dans la couche entièrement connectée

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
    print(SEP)

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"[ERREUR] Impossible de charger '{path}'")
        return

    print(f"[OK] Image  : {image.shape[1]}x{image.shape[0]} px")
    print(f"[OK] Kernel : {kernel.shape[0]}x{kernel.shape[1]}x{kernel.shape[2]}")

    relu = RELU()
    pool = POOL(pool_size=2, stride=2)
    data = image.astype(np.float64)

    print(f"\n{SEP}")
    print("  CONV / RELU / POOL")
    print(SEP)

    for m in range(M):
        print(f"[OK] Bloc {m+1}/{M}")

        for n in range(N):
            # --- CONV ---
            conv = CONV(kernel, stride)
            data = conv.forward(data)

            # --- RELU ---
            data = relu.forward(data)

        # --- POOL ---
        D = data.shape[2]
        pooled = []
        for d in range(D):
            fm = data[:, :, d]
            p = pool.forward(fm[np.newaxis, np.newaxis, :, :])[0, 0]
            pooled.append(p)
        data = np.stack(pooled, axis=2)

    # --- GAP ---
    last_feature_maps = data
    gap = GAP()
    data = gap.forward(data)
    print(f"\n{SEP}")
    print("  GAP")
    print(SEP)
    print(f"[OK] Vecteur GAP : {data.shape[0]} valeurs")

    # --- CAM ---
    cam_weights = np.ones(data.shape[0])
    cam_layer = CAM()
    cam_map = cam_layer.forward(last_feature_maps, cam_weights)
    cam_path = "outputs/cam_output.jpg"
    cv2.imwrite(cam_path, cam_map)
    print(f"\n{SEP}")
    print("  CAM")
    print(SEP)
    print(f"[OK] CAM : {cam_map.shape[1]}x{cam_map.shape[0]} px -> {cam_path}")

    print(f"\n{SEP}")
    print("  FC")
    print(SEP)

    # --- FC ---
    fc1 = FC(data.shape[0], FC_HIDDEN)
    data = relu.forward(fc1.forward(data))
    fc2 = FC(FC_HIDDEN, 3)
    output = fc2.forward(data)
    print(
        f"[OK] Normal    = {output[0]:.4f}\n"
        f"[OK] Bactérien = {output[1]:.4f}\n"
        f"[OK] Viral     = {output[2]:.4f}"
    )

    print(f"\n{SEP}")
    print("  Pipeline terminé")
    print(SEP)

if __name__ == "__main__":
    main()
