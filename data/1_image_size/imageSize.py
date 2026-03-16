import os
from PIL import Image
import matplotlib.pyplot as plt

# Récupérer la taille de toutes les images
roots = ["../../chest_Xray/train/NORMAL", "../../chest_Xray/train/PNEUMONIA"]
allSize = []
for root in roots:
    for path, subdirs, files in os.walk(root):
        for name in files:
            if name.endswith(".jpeg"):
                im = Image.open(root + '/' + name)
                w, h = im.size
                allSize.append([w, h])

# Calculer la moyenne
moyW = 0
moyH = 0
for size in allSize:
    moyW += size[0]
    moyH += size[1]

moyW = moyW / len(allSize)
moyH = moyH / len(allSize)

# Calculer la median
px = []
for size in allSize:
    px.append([size, size[0] * size[1]])
px.sort(key=lambda x: x[1])

# Afficher les résultats
os.system('cls' if os.name == 'nt' else 'clear')
print("Moyenne : ", [moyW, moyH])
print("Médiane : ", px[len(px) // 2])

widths = [s[0] for s in allSize]
heights = [s[1] for s in allSize]

plt.figure('Scatter')
plt.scatter(widths, heights, s=5, alpha=0.3)
plt.xlabel('Largeur (px)')
plt.ylabel('Hauteur (px)')
plt.title('Image Size Distribution - Nuage de points')

plt.figure('Heatmap')
plt.hist2d(widths, heights, bins=50, cmap='hot')
plt.colorbar(label='Nombre d\'images')
plt.xlabel('Largeur (px)')
plt.ylabel('Hauteur (px)')
plt.title('Image Size Distribution - Carte de densité')

plt.show()
