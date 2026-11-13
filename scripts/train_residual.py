"""Train the residual network and save its weights.

    python scripts/train_residual.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batterybrain.datagen import build_dataset
from batterybrain.residual_net import MLP, OUT_SCALE, WEIGHTS_PATH


def main():
    print("simulating training campaign ...")
    X, y = build_dataset(temps=(0, 10, 20, 30, 40), seeds=(1, 2))
    print("simulating validation campaign (unseen temperatures and seeds) ...")
    Xv, yv = build_dataset(temps=(5, 25, 35), seeds=(50,))
    print(f"train samples {len(y)}, validation samples {len(yv)}")
    print(f"nominal model voltage error: train {np.sqrt(np.mean(y**2)) * OUT_SCALE * 1e3:.1f} mV rms,"
          f" val {np.sqrt(np.mean(yv**2)) * OUT_SCALE * 1e3:.1f} mV rms")

    net = MLP([X.shape[1], 24, 24, 1], seed=0)
    history = net.fit(X, y, Xv, yv, epochs=40, lr=3e-3, batch=256)

    WEIGHTS_PATH.parent.mkdir(exist_ok=True)
    with open(WEIGHTS_PATH, "w") as fh:
        json.dump(net.to_dict(), fh)
    out = Path(__file__).resolve().parents[1] / "docs" / "training_history.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"rmse_mv": [[math.sqrt(a) * OUT_SCALE * 1e3, math.sqrt(b) * OUT_SCALE * 1e3]
                               for a, b in history],
                   "nominal_val_rmse_mv": float(np.sqrt(np.mean(yv**2)) * OUT_SCALE * 1e3)}, fh)
    print(f"saved {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
