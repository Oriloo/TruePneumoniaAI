import os
import sys
import time
import base64

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from ConvolutionLayer import ConvolutionLayer as CONV
from RectifiedLinearUnitLayer import RectifiedLinearUnitLayer as RELU
from PoolingLayer import PoolingLayer as POOL
from GlobalAveragePoolingLayer import GlobalAveragePoolingLayer as GAP
from ClassActivationMapLayer import ClassActivationMapLayer as CAM
from FullyConnected import FullyConnectedLayer as FC
from SoftmaxLayer import SoftmaxLayer as SOFTMAX
from CrossEntropyLoss import CrossEntropyLoss
from SGDOptimizer import SGDOptimizer
from DatasetLoader import DatasetLoader
import dashboard_server as dashboard

# ─────────────────────────────────────────────
#  Hyperparamètres
# ─────────────────────────────────────────────
NB_EPOCHS     = 20
LEARNING_RATE = 0.001
MOMENTUM      = 0.9
NB_FILTRES    = 8     # 32 → 208h/run | 8 → ~25-40h/run (CONV2 : 16x moins d'ops)
NB_BLOCS      = 5
NB_CONV_BLOC  = 3
FC_HIDDEN     = 128
KERNEL_SIZE   = 3
STRIDE_CONV   = 1
POOL_SIZE     = 2
STRIDE_POOL   = 2
GRAD_CLIP     = 1.0   # clip élément par élément des gradients (anti-explosion)
LOG_INTERVAL  = 10    # log + broadcast dashboard toutes les N images

# Dataset régénéré à 484×660 px (÷2, ratio identique à l'original 968×1320)
IMAGE_TARGET_SIZE = None  # images déjà à la bonne taille

# Mode debug : limite le nombre d'images chargées (None = tout le dataset)
DEBUG_MAX_IMAGES = None  # ex: 10 pour vérifier le pipeline rapidement

# Chemins depuis la racine du projet (D:/Docker/TruePneumoniaAI/)
_ROOT = os.path.join(os.path.dirname(__file__), "..")
TRAIN_DIR = os.path.join(_ROOT, "data", "dataset", "train")
VAL_DIR   = os.path.join(_ROOT, "data", "dataset", "val")

CLASS_NAMES = ["Normal", "Bactérien", "Viral"]
SEP = "=" * 55


# ─────────────────────────────────────────────
#  Construction du réseau
# ─────────────────────────────────────────────
def build_network():
    """Crée tous les blocs CONV+RELU+POOL avec initialisation He."""
    blocs = []
    in_ch = 1  # image grayscale = 1 canal d'entrée

    for _ in range(NB_BLOCS):
        convs = []
        relus = []
        for _ in range(NB_CONV_BLOC):
            convs.append(CONV.create(NB_FILTRES, KERNEL_SIZE, KERNEL_SIZE, in_ch, STRIDE_CONV))
            relus.append(RELU())
            in_ch = NB_FILTRES  # après le 1er CONV, tous les blocs ont NB_FILTRES canaux
        blocs.append({
            "convs": convs,
            "relus": relus,
            "pool": POOL(POOL_SIZE, STRIDE_POOL),
        })

    gap      = GAP()
    relu_fc  = RELU()
    softmax  = SOFTMAX()
    # FC1 et FC2 seront initialisés après le 1er forward pass (on ne connaît pas D a priori)
    return blocs, gap, relu_fc, softmax



# ─────────────────────────────────────────────
#  Forward pass
# ─────────────────────────────────────────────
def forward(image, blocs, gap, relu_fc, fc1, fc2, softmax):
    data = image.astype(np.float64)
    if data.ndim == 2:
        data = data[:, :, np.newaxis]

    for bloc in blocs:
        for conv, relu in zip(bloc["convs"], bloc["relus"]):
            data = relu.forward(conv.forward(data))
        data = bloc["pool"].forward(data)

    last_fmaps = data          # [H_final, W_final, NB_FILTRES]
    data = gap.forward(data)   # [NB_FILTRES]
    data = relu_fc.forward(fc1.forward(data))  # [FC_HIDDEN]
    output = softmax.forward(fc2.forward(data))  # [3]

    return output, last_fmaps


# ─────────────────────────────────────────────
#  Backward pass
# ─────────────────────────────────────────────
def backward(grad, blocs, gap, relu_fc, fc1, fc2):
    grad = fc2.backward(grad)
    grad = fc1.backward(relu_fc.backward(grad))
    grad = gap.backward(grad)

    for bloc in reversed(blocs):
        grad = bloc["pool"].backward(grad)
        for conv, relu in zip(reversed(bloc["convs"]), reversed(bloc["relus"])):
            grad = conv.backward(relu.backward(grad))

    return grad


# ─────────────────────────────────────────────
#  Utilitaires
# ─────────────────────────────────────────────
def compute_cam_b64(last_fmaps, fc1, fc2, predicted_class):
    """Génère une CAM colorisée encodée en base64 (JPEG)."""
    W1 = np.array([n.weights for n in fc1.neurons])          # [FC_HIDDEN, D]
    w2 = fc2.neurons[predicted_class].weights                 # [FC_HIDDEN]
    cam_weights = W1.T @ w2                                   # [D]

    cam_map = np.dot(last_fmaps, cam_weights)
    cam_map = np.maximum(cam_map, 0)
    if cam_map.max() > 0:
        cam_map = (cam_map / cam_map.max() * 255).astype(np.uint8)
    else:
        cam_map = cam_map.astype(np.uint8)

    cam_resized = cv2.resize(cam_map, (224, 224))
    cam_colored = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    _, buf = cv2.imencode(".jpg", cam_colored, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")


def evaluate(val_data, blocs, gap, relu_fc, fc1, fc2, softmax):
    """Calcule loss et accuracy sur le jeu de validation."""
    loss_fn = CrossEntropyLoss()
    total_loss = 0.0
    correct = 0

    for image, label in val_data:
        output, _ = forward(image, blocs, gap, relu_fc, fc1, fc2, softmax)
        total_loss += loss_fn.forward(output, label)
        if np.argmax(output) == label:
            correct += 1

    n = len(val_data)
    return total_loss / n, correct / n


def zero_all_grads(learnable_layers):
    for layer in learnable_layers:
        layer.zero_grads()


# ─────────────────────────────────────────────
#  Boucle principale
# ─────────────────────────────────────────────
def main():
    print(SEP)
    print("  TruePneumoniaAI — Entraînement")
    print(SEP)

    # Démarrage du dashboard
    dashboard.start_background()
    time.sleep(1)  # laisse le serveur s'initialiser

    # Indexation du dataset (chemins uniquement — images chargées à la volée)
    print("\n[1] Indexation du dataset…")
    loader = DatasetLoader(
        TRAIN_DIR, VAL_DIR,
        target_size=IMAGE_TARGET_SIZE,
        max_images=DEBUG_MAX_IMAGES,
    )
    loader.load()
    train_data = loader.get_train()
    val_data   = loader.get_val()

    if len(train_data) == 0:
        print("[ERREUR] Aucune image trouvée dans le répertoire d'entraînement.")
        return

    # Construction du réseau
    print("\n[2] Construction du réseau…")
    blocs, gap, relu_fc, softmax = build_network()

    # D = NB_FILTRES : le GAP fait la moyenne spatiale sans changer le nombre de canaux
    D = NB_FILTRES
    fc1 = FC(D, FC_HIDDEN)
    fc2 = FC(FC_HIDDEN, 3)
    print(f"     GAP → {D} canaux  |  FC({D}→{FC_HIDDEN})  |  FC({FC_HIDDEN}→3)")

    # Optimiseur
    learnable_layers = []
    for bloc in blocs:
        learnable_layers.extend(bloc["convs"])
    learnable_layers.extend([fc1, fc2])
    optimizer = SGDOptimizer(learnable_layers, learning_rate=LEARNING_RATE, momentum=MOMENTUM)

    loss_fn = CrossEntropyLoss()
    cam_layer = CAM()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\n[3] Entraînement : {NB_EPOCHS} epochs — {len(train_data)} images/epoch")
    print(f"     LR={LEARNING_RATE}  momentum={MOMENTUM}  filtres={NB_FILTRES}")
    print(f"     Taille images : 484×660 px (ratio 968×1320 ÷ 2)")
    print(SEP)

    for epoch in range(1, NB_EPOCHS + 1):
        t_start = time.time()
        epoch_loss = 0.0
        correct = 0
        pred_counts = [0, 0, 0]
        batch_loss_history = []   # dernières pertes par batch (rolling 100)

        # Mélange du dataset à chaque epoch
        train_data.shuffle()

        last_output = None
        last_fmaps  = None

        for idx, (image, label) in enumerate(train_data):
            # 1. Mise à zéro des gradients
            zero_all_grads(learnable_layers)

            # 2. Forward
            output, fmaps = forward(image, blocs, gap, relu_fc, fc1, fc2, softmax)

            # 3. Loss
            loss_val = loss_fn.forward(output, label)
            epoch_loss += loss_val
            batch_loss_history.append(float(loss_val))
            if len(batch_loss_history) > 100:
                batch_loss_history.pop(0)

            predicted = int(np.argmax(output))
            if predicted == label:
                correct += 1
            pred_counts[predicted] += 1

            # 4. Backward
            grad = loss_fn.backward()
            backward(grad, blocs, gap, relu_fc, fc1, fc2)

            # 5. Clip des gradients (anti-explosion)
            for layer in learnable_layers:
                for _, g in layer.get_params_and_grads():
                    np.clip(g, -GRAD_CLIP, GRAD_CLIP, out=g)

            # 6. Mise à jour des paramètres
            optimizer.step()

            last_output = output
            last_fmaps  = fmaps

            # Log + dashboard toutes les LOG_INTERVAL images
            if (idx + 1) % LOG_INTERVAL == 0:
                avg_loss = epoch_loss / (idx + 1)
                acc = correct / (idx + 1)
                global_step = (epoch - 1) * len(train_data) + idx + 1

                # Timing
                elapsed = time.time() - t_start
                speed = (idx + 1) / elapsed                       # img/s
                remaining_epoch = (len(train_data) - idx - 1) / speed
                eta_total = remaining_epoch + (NB_EPOCHS - epoch) * len(train_data) / speed

                print(f"  Epoch {epoch:02d} | {idx+1:5d}/{len(train_data)} | "
                      f"Loss={avg_loss:.4f} | Acc={acc:.3f} | "
                      f"{speed:.2f}img/s | ETA epoch {remaining_epoch/60:.0f}min", end='\r')

                # CAM sur la dernière image traitée
                cam_b64 = None
                if last_fmaps is not None:
                    try:
                        cam_b64 = compute_cam_b64(last_fmaps, fc1, fc2, predicted)
                    except Exception:
                        pass

                dashboard.broadcast({
                    "type": "batch_update",
                    "epoch": epoch,
                    "total_epochs": NB_EPOCHS,
                    "batch": idx + 1,
                    "total_batches": len(train_data),
                    "global_step": global_step,
                    "current_loss": float(avg_loss),
                    "current_accuracy": float(acc),
                    "prediction_distribution": list(pred_counts),
                    "batch_loss_history": batch_loss_history,
                    "images_per_sec": round(float(speed), 2),
                    "elapsed_epoch": round(elapsed),
                    "eta_epoch": round(remaining_epoch),
                    "eta_total": round(eta_total),
                    "cam_image": cam_b64,
                })

        epoch_time = time.time() - t_start
        train_loss = epoch_loss / len(train_data)
        train_acc  = correct / len(train_data)

        # Validation
        print(f"\n  Epoch {epoch:02d} — validation…", end='\r')
        val_loss, val_acc = evaluate(val_data, blocs, gap, relu_fc, fc1, fc2, softmax)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))

        print(f"  Epoch {epoch:02d}/{NB_EPOCHS} "
              f"| Loss train={train_loss:.4f} val={val_loss:.4f} "
              f"| Acc train={train_acc:.3f} val={val_acc:.3f} "
              f"| {epoch_time:.1f}s")

        # Génération de la CAM pour la dernière image traitée
        cam_b64 = None
        if last_fmaps is not None:
            try:
                predicted_class = int(np.argmax(last_output))
                cam_b64 = compute_cam_b64(last_fmaps, fc1, fc2, predicted_class)
            except Exception as e:
                print(f"  [AVERT] CAM non générée : {e}")

        # Envoi des métriques au dashboard
        dashboard.broadcast({
            "type": "epoch_update",
            "epoch": epoch,
            "total_epochs": NB_EPOCHS,
            "global_step": epoch * len(train_data),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_accuracy": float(train_acc),
            "val_accuracy": float(val_acc),
            "epoch_time": round(epoch_time, 1),
            "prediction_distribution": pred_counts,
            "cam_image": cam_b64,
            "history": history,
        })

    print(f"\n{SEP}")
    print("  Entraînement terminé")
    print(SEP)


if __name__ == "__main__":
    main()
