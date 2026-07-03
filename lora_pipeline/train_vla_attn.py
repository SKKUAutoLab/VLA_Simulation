#!/usr/bin/env python3
"""
순수 VLA WP 학습 (attention-pool) — Qwen 비전 70토큰(동결)을 mean-pool 대신
'학습형 쿼리 attention'으로 풀링해 공간정보(차선 위치) 보존. 비전 동결→토큰 캐싱.
저장: vla_attn_head.pt
"""
import os, csv, glob, random
import cv2, numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from train_wp import WP_N, WP_SCALE
from vla_vision import load_vision, extract_tokens, FEAT_DIM

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "dataset")
CACHE = os.path.join(DATA, "vla_tok_cache.pt")
OUT = os.path.join(HERE, "vla_attn_head.pt")
NTOK = 70
SEED = 0


class AttnHead(nn.Module):
    def __init__(self, nout, dim=FEAT_DIM, nq=4, heads=8):
        super().__init__()
        self.q = nn.Parameter(torch.randn(nq, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln = nn.LayerNorm(dim)
        self.emb = nn.Embedding(2, 16)
        self.net = nn.Sequential(
            nn.Linear(dim*nq + dim + 16, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, nout))

    def forward(self, tokens, lane):           # tokens (B,T,dim)
        B = tokens.shape[0]
        t = self.ln(tokens)
        q = self.q.unsqueeze(0).expand(B, -1, -1)
        pooled, _ = self.attn(q, t, t)         # (B,nq,dim)
        feat = torch.cat([pooled.flatten(1), t.mean(1), self.emb(lane)], dim=1)
        return self.net(feat)


def fix_tok(f):
    if f.shape[0] >= NTOK:
        return f[:NTOK]
    return torch.cat([f, f[-1:].expand(NTOK-f.shape[0], -1)], 0)


def load_rows():
    rows = []
    for cf in sorted(glob.glob(os.path.join(DATA, "labels_wpL*.csv"))):
        for r in csv.DictReader(open(cf)):
            ip = os.path.join(DATA, "images", r["fname"])
            if not os.path.exists(ip):
                continue
            wps = [float(r[f"{c}{k}"]) for k in range(WP_N) for c in ("ex", "ey")]
            rows.append((ip, wps, int(r["lane"])))
    return rows


def build_cache(paths):
    cache = torch.load(CACHE) if os.path.exists(CACHE) else {}
    miss = [p for p in paths if p not in cache]
    if miss:
        print(f"토큰특징 추출 {len(miss)}개 (캐시 {len(cache)})...")
        vis, proc = load_vision()
        for i, p in enumerate(miss):
            bgr = cv2.imread(p)
            if bgr is None:
                continue
            cache[p] = fix_tok(extract_tokens(vis, proc, bgr).cpu()).half()
            if (i+1) % 500 == 0:
                print(f"  {i+1}/{len(miss)}")
        torch.save(cache, CACHE); del vis; torch.cuda.empty_cache()
        print(f"캐시 저장 {len(cache)}")
    else:
        print(f"캐시 사용 {len(cache)}")
    return cache


class TokSet(Dataset):
    def __init__(self, rows, cache): self.rows = rows; self.cache = cache
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        p, wps, lane = self.rows[i]
        return (self.cache[p].float(), torch.tensor(lane, dtype=torch.long),
                torch.tensor([v/WP_SCALE for v in wps], dtype=torch.float32))


def main():
    torch.manual_seed(SEED)
    rows = load_rows()
    cache = build_cache(sorted({p for p, _, _ in rows}))
    rows = [r for r in rows if r[0] in cache]
    random.Random(SEED).shuffle(rows)
    nv = max(1, int(len(rows)*0.1)); val, tr = rows[:nv], rows[nv:]
    print(f"train {len(tr)} / val {len(val)}")
    dev = "cuda:0"; nout = 2*WP_N
    tl = DataLoader(TokSet(tr, cache), batch_size=64, shuffle=True, num_workers=2)
    vl = DataLoader(TokSet(val, cache), batch_size=64, num_workers=2)
    net = AttnHead(nout).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=7e-4, weight_decay=1e-5)
    lossf = nn.MSELoss(); best = 1e9
    for ep in range(80):
        net.train(); tre = 0
        for f, lane, y in tl:
            f, lane, y = f.to(dev), lane.to(dev), y.to(dev)
            opt.zero_grad(); l = lossf(net(f, lane), y); l.backward(); opt.step(); tre += l.item()*len(f)
        net.eval(); mae = {0: 0, 1: 0}; cnt = {0: 0, 1: 0}
        with torch.no_grad():
            for f, lane, y in vl:
                f, lane, y = f.to(dev), lane.to(dev), y.to(dev); p = net(f, lane)
                for li in (0, 1):
                    m = lane == li
                    if m.any(): mae[li] += (p[m]-y[m]).abs().sum().item()*WP_SCALE; cnt[li] += m.sum().item()*nout
        m0 = mae[0]/max(1, cnt[0]); m1 = mae[1]/max(1, cnt[1]); mm = (m0+m1)/2
        if mm < best:
            best = mm
            torch.save({"state_dict": net.state_dict(), "wp_n": WP_N, "wp_scale": WP_SCALE, "nout": nout, "ntok": NTOK}, OUT)
        if ep % 5 == 0 or ep == 79:
            print(f"ep{ep} mse={tre/len(tr):.4f} val_wpMAE 1차선={m0:.2f} 2차선={m1:.2f} (best {best:.2f})")
    print(f"\n✓ best wpMAE {best:.2f}m → {OUT}")


if __name__ == "__main__":
    main()
