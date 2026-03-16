import numpy as np
import ConvolutionLayer
import cv2

kernel = np.array([[0, 1, 0], [0, 0, 1], [-1, -1, 0]])
conv_layer = ConvolutionLayer.ConvolutionLayer(kernel)

path = "../data/3_image_generates/outputs/bacteria-5012.jpg"
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print(f"Erreur : impossible de charger l'image '{path}'")
else:
    print(f"Image chargee : {image.shape} (hauteur x largeur)")
    print(f"Kernel : {kernel.shape}")

    expected_patches = (image.shape[0] - kernel.shape[0] + 1) * (image.shape[1] - kernel.shape[1] + 1)
    print(f"Nombre de patches attendus : {expected_patches}")

    count = 0
    for patch in conv_layer.patch_generator(image):
        if count < 3:
            print(f"\nPatch {count} (shape={patch.shape}) :")
            print(patch)
        count += 1

    print(f"\nNombre total de patches generes : {count}")
    print(f"Test patch_generator {'OK' if count == expected_patches else 'ECHEC'} (attendu={expected_patches}, obtenu={count})")

    # --- Test kernel_convolution ---
    print("\n--- Test kernel_convolution ---")

    output_height = image.shape[0] - kernel.shape[0] + 1
    output_width = image.shape[1] - kernel.shape[1] + 1
    output = np.zeros((output_height, output_width), dtype=np.float64)

    row, col = 0, 0
    for patch in conv_layer.patch_generator(image):
        output[row, col] = conv_layer.kernel_convolution(patch)
        col += 1
        if col >= output_width:
            col = 0
            row += 1

    print(f"Image de sortie : {output.shape}")
    print(f"Valeurs min={output.min():.1f}, max={output.max():.1f}")

    # Normaliser en 0-255 pour sauvegarder
    output_normalized = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX)
    output_image = output_normalized.astype(np.uint8)

    output_path = "convolution_output.jpg"
    cv2.imwrite(output_path, output_image)
    print(f"Image resultat sauvegardee : {output_path}")
    print(f"Test kernel_convolution OK")
