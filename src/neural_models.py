#!/usr/bin/env python3
"""Fit the frozen neural models on a required CPU or explicitly allocated GPU."""
import json
import random
import time
import numpy as np
from rdkit import Chem, RDLogger
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, GATv2Conv, global_mean_pool

MAX_EPOCHS = 100
PATIENCE = 15
SEED = 20261103
RDLogger.DisableLog("rdApp.*")
torch.set_num_threads(1)
DEVICE = torch.device("cpu")


def onehot(value, levels):
    return [float(value == v) for v in levels] + [float(value not in levels)]


def graph(smiles, bits, desc, idx):
    m = Chem.MolFromSmiles(str(smiles))
    if m is None or len(Chem.GetMolFrags(m)) != 1:
        raise ValueError("Invalid graph structure")
    atoms = []
    for a in m.GetAtoms():
        atoms.append(
            onehot(a.GetAtomicNum(), [1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53])
            + onehot(a.GetDegree(), [0, 1, 2, 3, 4, 5])
            + onehot(a.GetFormalCharge(), [-2, -1, 0, 1, 2])
            + onehot(a.GetTotalNumHs(), [0, 1, 2, 3, 4])
            + onehot(str(a.GetHybridization()), ["SP", "SP2", "SP3", "SP3D"])
            + [float(a.GetIsAromatic()), float(a.IsInRing()), a.GetMass() / 100.0]
        )
    edge_idx = []
    edges = []
    for b in m.GetBonds():
        feat = (
            onehot(str(b.GetBondType()), ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"])
            + [float(b.GetIsConjugated()), float(b.IsInRing())]
            + onehot(str(b.GetStereo()), ["STEREONONE", "STEREOZ", "STEREOE"])
        )
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edge_idx.extend([(i, j), (j, i)])
        edges.extend([feat, feat])
    ei = (
        torch.tensor(edge_idx, dtype=torch.long).T.contiguous()
        if edge_idx
        else torch.empty((2, 0), dtype=torch.long)
    )
    ea = (
        torch.tensor(edges, dtype=torch.float32)
        if edges
        else torch.empty((0, 11), dtype=torch.float32)
    )
    return Data(
        x=torch.tensor(atoms, dtype=torch.float32),
        edge_index=ei,
        edge_attr=ea,
        fingerprint=torch.tensor(bits, dtype=torch.float32).reshape(1, -1),
        descriptors=torch.tensor(desc, dtype=torch.float32).reshape(1, -1),
        idx=torch.tensor([idx]),
    )


class Regressor(nn.Module):
    def __init__(self, method, atom_dim, edge_dim, desc_mean, desc_std):
        super().__init__()
        self.method = method
        self.register_buffer(
            "desc_mean", torch.tensor(desc_mean, dtype=torch.float32).reshape(1, -1)
        )
        self.register_buffer(
            "desc_std", torch.tensor(desc_std, dtype=torch.float32).reshape(1, -1)
        )
        if method == "mlp":
            self.network = nn.Sequential(
                nn.Linear(2055, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
            )
        else:
            hidden = 96
            self.atom = nn.Linear(atom_dim, hidden)
            self.layers = nn.ModuleList()
            for _ in range(3):
                mlp = nn.Sequential(
                    nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
                )
                conv = (
                    GINEConv(mlp, train_eps=True, edge_dim=edge_dim)
                    if method == "gine"
                    else GATv2Conv(
                        hidden, hidden // 4, heads=4, dropout=0.1, edge_dim=edge_dim
                    )
                )
                self.layers.append(conv)
            self.network = nn.Sequential(
                nn.Linear(hidden + 7, 96), nn.ReLU(), nn.Dropout(0.1), nn.Linear(96, 1)
            )

    def forward(self, b):
        d = (b.descriptors - self.desc_mean) / self.desc_std
        if self.method == "mlp":
            return self.network(torch.cat([b.fingerprint, d], dim=1)).view(-1)
        h = self.atom(b.x)
        for layer in self.layers:
            h = torch.relu(layer(h, b.edge_index, b.edge_attr))
        return self.network(torch.cat([global_mean_pool(h, b.batch), d], dim=1)).view(
            -1
        )


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def labelled(graphs, indices, values):
    out = []
    for i in indices:
        g = graphs[int(i)].clone()
        g.y = torch.tensor([float(values[int(i)])], dtype=torch.float32)
        out.append(g)
    return out


def create_model(method, graphs, indices, desc, seed):
    seed_all(seed)
    mean = desc[indices].mean(axis=0)
    std = desc[indices].std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    return Regressor(
        method, graphs[0].x.shape[1], graphs[0].edge_attr.shape[1], mean, std
    ).to(DEVICE)


def optimize_epoch(model, loader, optimizer, y_mean, y_std):
    model.train()
    for b in loader:
        b = b.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(b)
        target = (b.y - y_mean) / y_std
        loss = torch.nn.functional.smooth_l1_loss(prediction, target, beta=1.0)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite neural training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()


def validation_mae(model, loader, y_mean, y_std):
    model.eval()
    errors = []
    with torch.no_grad():
        for b in loader:
            b = b.to(DEVICE)
            errors.extend((model(b) * y_std + y_mean - b.y).abs().cpu().tolist())
    if not errors or not np.isfinite(errors).all():
        raise ValueError("Invalid validation errors")
    return float(np.mean(errors))


def model_fit(method, graphs, train, test, values, desc, seed):
    rng = np.random.default_rng(seed)
    inner = np.array(train, copy=True)
    rng.shuffle(inner)
    nval = max(1, int(round(0.2 * len(inner))))
    val, fit = inner[:nval], inner[nval:]
    if set(fit) & set(val) or set(train) & set(test):
        raise ValueError("Non-disjoint train/validation/test")
    ymean = float(values[fit].mean())
    ystd = float(values[fit].std())
    if ystd < 1e-8:
        raise ValueError("Zero target variation")
    model = create_model(method, graphs, fit, desc, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    gen = torch.Generator().manual_seed(seed)
    tr = DataLoader(
        labelled(graphs, fit, values), batch_size=64, shuffle=True, generator=gen
    )
    va = DataLoader(labelled(graphs, val, values), batch_size=128, shuffle=False)
    best = float("inf")
    best_epoch = 0
    patience = 0
    started = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        optimize_epoch(model, tr, optimizer, ymean, ystd)
        score = validation_mae(model, va, ymean, ystd)
        if score < best - 1e-5:
            best = score
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break
        if epoch % 20 == 0:
            print(
                json.dumps(
                    {
                        "method": method,
                        "epoch": epoch,
                        "inner_val_mae": round(score, 4),
                        "elapsed_seconds": round(time.time() - started, 1),
                    }
                ),
                flush=True,
            )
    if best_epoch < 1:
        raise ValueError("No valid neural checkpoint")
    # Refit from the same seed on all outer-training chemicals for the selected epoch count.
    ymean = float(values[train].mean())
    ystd = float(values[train].std())
    model = create_model(method, graphs, train, desc, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    tr = DataLoader(
        labelled(graphs, train, values),
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    for _ in range(best_epoch):
        optimize_epoch(model, tr, optimizer, ymean, ystd)
    te = DataLoader([graphs[int(i)] for i in test], batch_size=128, shuffle=False)
    model.eval()
    indices = []
    predictions = []
    with torch.no_grad():
        for b in te:
            if hasattr(b, "y") and b.y is not None:
                raise ValueError("Test batch must not contain toxicity labels")
            indices.extend(b.idx.tolist())
            b = b.to(DEVICE)
            predictions.extend((model(b) * ystd + ymean).cpu().tolist())
    if indices != list(test) or not np.isfinite(predictions).all():
        raise ValueError("Incomplete neural predictions")
    return predictions, {
        "model": method,
        "seed": seed,
        "inner_training_n": len(fit),
        "inner_validation_n": len(val),
        "outer_training_n": len(train),
        "testing_n": len(test),
        "best_epoch": best_epoch,
        "epochs_ran": epoch,
        "inner_best_mae_p": best,
        "refit_on_all_outer_training": True,
        "test_label_used": False,
        "seconds": round(time.time() - started, 1),
        "parameters": sum(x.numel() for x in model.parameters()),
        "device": str(DEVICE),
        "cuda_runtime": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(DEVICE) if DEVICE.type == "cuda" else None
        ),
    }
