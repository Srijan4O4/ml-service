"""
preprocessing.py — Build a heterogeneous seller-fraud graph for the RGCN.

No public dataset ships with this project (data/HOPE_WE_FOUND.csv is empty), so we
*synthesize* a realistic e-commerce graph with embedded collusion / fraud patterns:

  Users  --review-->  Products  --sold_by-->  Sellers
  Users  --bought-->  Sellers (inferred)

Fraud is injected through a generative process (NOT random labels) so the RGCN can
actually learn from both node features AND graph structure:

  * A "fraud ring" of colluding users posts bursty 5-star reviews on the products of
    fraudulent sellers (review-spike / collusion signal).
  * Fraudulent sellers have high return ratios and high review burstiness.
  * Honest sellers get organic, low-burst reviews and low returns.

Node features (3 dims each, matching the API contract):
  Seller : [return_ratio, avg_rating, burstiness]
  User   : [num_reviews, mean_rating_given, product_diversity]
  Product: [num_reviews, avg_rating, rating_variance]

Outputs (written to data/processed/):
  graph.pt          -> dict with x, edge_index, edge_type, labels, masks, offsets
  mappings.json     -> name<->index maps for users/products/sellers
  seller_features.json -> human-readable seller feature dict (for the API)
"""

from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parents[2]   # ml-service/
OUT_DIR = BASE_DIR / "artifacts" / "gnn"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------- configuration ------------------------------------
SEED = 42

N_SELLERS = 260
FRAUD_SELLER_RATE = 0.12          # ~31 fraudulent sellers
N_USERS = 3000
N_FRAUD_USERS = 180               # colluding "ring" users
PRODUCTS_PER_SELLER = (3, 14)     # inclusive range
TIME_SPAN_DAYS = 365

RNG = np.random.default_rng(SEED)


# ----------------------------- data generation ----------------------------------
def _generate_raw():
    """Generate raw sellers, products, users and reviews with fraud structure."""
    # --- sellers ---
    n_fraud_sellers = int(round(N_SELLERS * FRAUD_SELLER_RATE))
    seller_is_fraud = np.zeros(N_SELLERS, dtype=int)
    fraud_seller_ids = RNG.choice(N_SELLERS, size=n_fraud_sellers, replace=False)
    seller_is_fraud[fraud_seller_ids] = 1
    seller_names = [f"Seller_{i:04d}" for i in range(N_SELLERS)]

    # --- products (each belongs to one seller) ---
    products = []            # list of dicts: {pid, seller}
    seller_products = {s: [] for s in range(N_SELLERS)}
    pid = 0
    for s in range(N_SELLERS):
        k = RNG.integers(PRODUCTS_PER_SELLER[0], PRODUCTS_PER_SELLER[1] + 1)
        for _ in range(k):
            products.append({"pid": pid, "seller": s})
            seller_products[s].append(pid)
            pid += 1
    n_products = pid

    # --- users ---
    user_is_fraud = np.zeros(N_USERS, dtype=int)
    fraud_user_ids = RNG.choice(N_USERS, size=N_FRAUD_USERS, replace=False)
    user_is_fraud[fraud_user_ids] = 1

    # --- reviews ---
    # review record: (user, product, seller, rating, day)
    reviews = []
    # returns counter per seller (orders, returns)
    seller_orders = np.zeros(N_SELLERS, dtype=int)
    seller_returns = np.zeros(N_SELLERS, dtype=int)

    for s in range(N_SELLERS):
        prods = seller_products[s]
        fraud = seller_is_fraud[s] == 1

        if fraud:
            # Bursty 5-star reviews from a subset of ring users over a few days.
            ring = RNG.choice(fraud_user_ids, size=RNG.integers(15, 45), replace=False)
            burst_start = RNG.integers(0, TIME_SPAN_DAYS - 5)
            n_reviews = RNG.integers(60, 160)
            for _ in range(n_reviews):
                u = int(RNG.choice(ring))
                p = int(RNG.choice(prods))
                # mostly 5-star, occasional 4 to look "natural"
                rating = 5 if RNG.random() < 0.85 else 4
                day = int(np.clip(burst_start + RNG.integers(0, 4), 0, TIME_SPAN_DAYS - 1))
                reviews.append((u, p, s, rating, day))
            # high return ratio
            orders = int(n_reviews * RNG.uniform(1.2, 2.0))
            ret_ratio = RNG.uniform(0.35, 0.75)
            seller_orders[s] = orders
            seller_returns[s] = int(orders * ret_ratio)
        else:
            # Organic reviews from random honest users, spread across the year.
            n_reviews = RNG.integers(20, 90)
            for _ in range(n_reviews):
                u = int(RNG.integers(0, N_USERS))
                p = int(RNG.choice(prods))
                # realistic spread of ratings centred ~4
                rating = int(np.clip(round(RNG.normal(4.0, 1.0)), 1, 5))
                day = int(RNG.integers(0, TIME_SPAN_DAYS))
                reviews.append((u, p, s, rating, day))
            orders = int(n_reviews * RNG.uniform(1.1, 1.8))
            ret_ratio = RNG.uniform(0.02, 0.22)
            seller_orders[s] = orders
            seller_returns[s] = int(orders * ret_ratio)

    return {
        "seller_names": seller_names,
        "seller_is_fraud": seller_is_fraud,
        "seller_products": seller_products,
        "products": products,
        "n_products": n_products,
        "user_is_fraud": user_is_fraud,
        "reviews": reviews,
        "seller_orders": seller_orders,
        "seller_returns": seller_returns,
    }


# ----------------------------- feature computation -------------------------------
def _compute_features(raw):
    reviews = raw["reviews"]
    n_products = raw["n_products"]

    # group reviews
    user_reviews: dict[int, list] = {}
    product_reviews: dict[int, list] = {}
    seller_reviews: dict[int, list] = {}
    for (u, p, s, rating, day) in reviews:
        user_reviews.setdefault(u, []).append((p, s, rating, day))
        product_reviews.setdefault(p, []).append((u, s, rating, day))
        seller_reviews.setdefault(s, []).append((u, p, rating, day))

    # ---- user features: [num_reviews, mean_rating_given, product_diversity] ----
    user_feat = np.zeros((N_USERS, 3), dtype=np.float32)
    for u in range(N_USERS):
        revs = user_reviews.get(u, [])
        if revs:
            ratings = [r[2] for r in revs]
            prods = {r[0] for r in revs}
            user_feat[u, 0] = len(revs)
            user_feat[u, 1] = float(np.mean(ratings))
            user_feat[u, 2] = len(prods) / len(revs)  # 1.0 = all distinct, low = repetitive
        else:
            user_feat[u] = [0.0, 0.0, 1.0]

    # ---- product features: [num_reviews, avg_rating, rating_variance] ----
    product_feat = np.zeros((n_products, 3), dtype=np.float32)
    for p in range(n_products):
        revs = product_reviews.get(p, [])
        if revs:
            ratings = [r[2] for r in revs]
            product_feat[p, 0] = len(revs)
            product_feat[p, 1] = float(np.mean(ratings))
            product_feat[p, 2] = float(np.var(ratings))
        else:
            product_feat[p] = [0.0, 0.0, 0.0]

    # ---- seller features: [return_ratio, avg_rating, burstiness] ----
    seller_feat = np.zeros((N_SELLERS, 3), dtype=np.float32)
    seller_feat_readable = {}
    for s in range(N_SELLERS):
        revs = seller_reviews.get(s, [])
        orders = max(int(raw["seller_orders"][s]), 1)
        returns = int(raw["seller_returns"][s])
        return_ratio = returns / orders
        if revs:
            ratings = [r[2] for r in revs]
            days = [r[3] for r in revs]
            avg_rating = float(np.mean(ratings))
            # burstiness = max reviews in a single day / total reviews (0..1, high = spiky)
            counts = np.bincount(days, minlength=TIME_SPAN_DAYS)
            burstiness = float(counts.max() / len(revs))
        else:
            avg_rating, burstiness = 0.0, 0.0
        seller_feat[s] = [return_ratio, avg_rating, burstiness]
        seller_feat_readable[raw["seller_names"][s]] = {
            "return_ratio": round(return_ratio, 4),
            "avg_rating": round(avg_rating, 3),
            "burstiness": round(burstiness, 4),
            "num_reviews": len(revs),
            "num_products": len(raw["seller_products"][s]),
        }

    return user_feat, product_feat, seller_feat, seller_feat_readable


# ----------------------------- graph assembly ------------------------------------
def _build_graph(raw, user_feat, product_feat, seller_feat):
    U, P, S = N_USERS, raw["n_products"], N_SELLERS
    # Unified index space: users [0..U), products [U..U+P), sellers [U+P..U+P+S)
    u_off, p_off, s_off = 0, U, U + P
    N = U + P + S

    # stack features into a single [N, 3] matrix (per-type meaning, shared dims)
    x = np.vstack([user_feat, product_feat, seller_feat]).astype(np.float32)

    # standardize each column (helps GNN training)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-6
    x = (x - mean) / std

    edge_src, edge_dst, edge_type = [], [], []

    def add_edge(a, b, t):
        # add both directions so message passing flows both ways
        edge_src.append(a); edge_dst.append(b); edge_type.append(t)
        edge_src.append(b); edge_dst.append(a); edge_type.append(t)

    # relation 0: user -> product (review)
    seen_user_seller = set()
    for (u, p, s, _rating, _day) in raw["reviews"]:
        add_edge(u_off + u, p_off + p, 0)
        # relation 2: user -> seller (inferred purchase)
        key = (u, s)
        if key not in seen_user_seller:
            add_edge(u_off + u, s_off + s, 2)
            seen_user_seller.add(key)

    # relation 1: product -> seller (sold_by)
    for prod in raw["products"]:
        add_edge(p_off + prod["pid"], s_off + prod["seller"], 1)

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)

    # labels: defined on seller nodes only (0 normal, 1 fraud)
    labels = torch.zeros(N, dtype=torch.long)
    seller_labels = torch.tensor(raw["seller_is_fraud"], dtype=torch.long)
    labels[s_off:s_off + S] = seller_labels

    # stratified train/val/test split over seller nodes
    seller_global_idx = np.arange(s_off, s_off + S)
    fraud = raw["seller_is_fraud"].astype(bool)
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)

    def split_indices(idx):
        idx = idx.copy()
        RNG.shuffle(idx)
        n = len(idx)
        n_tr, n_va = int(0.6 * n), int(0.2 * n)
        return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]

    for cls in (False, True):
        cls_idx = seller_global_idx[fraud == cls]
        tr, va, te = split_indices(cls_idx)
        train_mask[tr] = True
        val_mask[va] = True
        test_mask[te] = True

    return {
        "x": torch.tensor(x),
        "edge_index": edge_index,
        "edge_type": edge_type,
        "labels": labels,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "num_relations": 3,
        "num_classes": 2,
        "offsets": {"user": u_off, "product": p_off, "seller": s_off,
                    "num_users": U, "num_products": P, "num_sellers": S, "num_nodes": N},
    }


def main():
    print("[preprocessing] generating synthetic fraud graph (seed=%d)..." % SEED)
    raw = _generate_raw()
    print(f"[preprocessing] sellers={N_SELLERS}  products={raw['n_products']}  "
          f"users={N_USERS}  reviews={len(raw['reviews'])}  "
          f"fraud_sellers={int(raw['seller_is_fraud'].sum())}")

    user_feat, product_feat, seller_feat, seller_readable = _compute_features(raw)
    graph = _build_graph(raw, user_feat, product_feat, seller_feat)

    torch.save(graph, OUT_DIR / "graph.pt")

    # name <-> index mappings
    s_off = graph["offsets"]["seller"]
    seller_to_index = {raw["seller_names"][s]: int(s_off + s) for s in range(N_SELLERS)}
    mappings = {
        "seller_to_index": seller_to_index,
        "index_to_seller": {str(v): k for k, v in seller_to_index.items()},
        "offsets": graph["offsets"],
    }
    with open(OUT_DIR / "mappings.json", "w") as f:
        json.dump(mappings, f, indent=2)
    with open(OUT_DIR / "seller_features.json", "w") as f:
        json.dump(seller_readable, f, indent=2)

    print(f"[preprocessing] edges={graph['edge_index'].shape[1]}  "
          f"nodes={graph['offsets']['num_nodes']}")
    print(f"[preprocessing] saved -> {OUT_DIR}")
    print("[preprocessing] done.")


if __name__ == "__main__":
    main()
