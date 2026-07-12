"""train.py — train the RGCN on the synthetic seller-fraud graph.

Run:
    python -m models.gnn.train          (from the ml-service/ directory)
or
    python models/gnn/train.py
Produces artifacts/gnn/rgcn_seller_fraud.pth and artifacts/gnn/metrics.json.
"""
from __future__ import annotations
import json
from pathlib import Path

try:
    import torch  # type: ignore
except ImportError as e:
    raise ImportError("torch is required. Install it with: pip install torch") from e

from .model import RGCN
from . import preprocessing

ART_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "gnn"


def _load_graph():
    graph_path = ART_DIR / "graph.pt"
    if not graph_path.exists():
        print("[train] processed graph not found -> running preprocessing...")
        preprocessing.main()
    return torch.load(graph_path, weights_only=False)


def _metrics(logits, labels, mask):
    pred = logits[mask].argmax(dim=1)
    y = labels[mask]
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    acc = (tp + tn) / max(1, (tp + fp + fn + tn))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"acc": round(acc, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main(epochs: int = 200, hidden: int = 32, lr: float = 0.01, weight_decay: float = 5e-4):
    torch.manual_seed(42)
    g = _load_graph()
    x = g["x"]
    edge_index = g["edge_index"]
    edge_type = g["edge_type"]
    labels = g["labels"]
    train_mask, val_mask, test_mask = g["train_mask"], g["val_mask"], g["test_mask"]

    model = RGCN(in_feats=x.shape[1], hidden=hidden,
                 num_rels=g["num_relations"], num_classes=g["num_classes"])

    # class weights to counter imbalance on the training sellers
    n_pos = int(labels[train_mask].sum())
    n_neg = int((labels[train_mask] == 0).sum())
    w = torch.tensor([1.0, max(1.0, n_neg / max(1, n_pos))], dtype=torch.float)
    criterion = torch.nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_f1, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index, edge_type)
        loss = criterion(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                logits = model(x, edge_index, edge_type)
            tr = _metrics(logits, labels, train_mask)
            va = _metrics(logits, labels, val_mask)
            print(f"[train] epoch {epoch:3d} loss {loss.item():.4f} "
                  f"| train f1 {tr['f1']:.3f} acc {tr['acc']:.3f} "
                  f"| val f1 {va['f1']:.3f} acc {va['acc']:.3f}")
            if va["f1"] > best_val_f1:
                best_val_f1 = va["f1"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index, edge_type)
    test_m = _metrics(logits, labels, test_mask)
    print(f"[train] TEST: {test_m}")

    ART_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": {"in_feats": x.shape[1], "hidden": hidden,
                   "num_rels": g["num_relations"], "num_classes": g["num_classes"]},
    }, ART_DIR / "rgcn_seller_fraud.pth")
    with open(ART_DIR / "metrics.json", "w") as f:
        json.dump({"val_f1": best_val_f1, "test": test_m}, f, indent=2)
    print(f"[train] saved model -> {ART_DIR / 'rgcn_seller_fraud.pth'}")
    return test_m


if __name__ == "__main__":
    main()
