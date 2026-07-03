#!/usr/bin/env python3
"""
빠른 검증용: 비전 동결(LoRA 없음) → 토큰특징(70×2048) 1회 캐시 → 공간헤드만 학습.
목적 = "mean-pool 제거(공간보존 헤드)가 직선 비틀거림을 잡는가"를 ~15분 내 판정.
효과 확인되면 train_vla_lora.py(비전 LoRA 포함)로 길게 돌린다.
저장: vla_lora_head_fast.pt  (어댑터 없음 → 주행노드는 VLA_NO_ADAPTER=1 로 베이스 비전 사용)
"""
import os, time, random
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from train_vla_lora import load_rows, build_px, Head, NTOK
from train_wp import WP_N, WP_SCALE
from vla_vision import load_vision, FEAT_DIM

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "dataset")
FEATCACHE = os.path.join(DATA, "vla_tokfeat_cache.pt")   # 동결 베이스 비전 토큰특징(70,2048) 캐시
HEAD_OUT = os.path.join(HERE, "vla_lora_head_fast.pt")
SEED = 0


def build_feat(pxcache, paths, vis, dev):
    cache = torch.load(FEATCACHE) if os.path.exists(FEATCACHE) else {}
    miss = [p for p in paths if p not in cache]
    if miss:
        print(f"토큰특징 추출 {len(miss)}개 (동결 베이스 비전, 1회)...")
        grid = torch.tensor([[1, 14, 20]], device=dev)
        t0 = time.time()
        with torch.no_grad():
            for i, p in enumerate(miss):
                px = pxcache[p].to(dev).to(torch.bfloat16)        # (280,1536)
                out = vis(px, grid_thw=grid)
                f = out[0] if isinstance(out, (list, tuple)) else out  # (70,2048)
                cache[p] = f.float().cpu().half()
                if (i + 1) % 2000 == 0:
                    print(f"  {i+1}/{len(miss)}  ({(i+1)/(time.time()-t0):.0f}/s)")
        torch.save(cache, FEATCACHE)
        print(f"토큰특징 캐시 저장 {len(cache)} → {FEATCACHE}")
    else:
        print(f"토큰특징 캐시 사용 {len(cache)}")
    return cache


class FeatSet(Dataset):
    def __init__(self, rows, cache): self.rows = rows; self.cache = cache
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        p, wps, lane = self.rows[i]
        y = torch.tensor([v / WP_SCALE for v in wps], dtype=torch.float32)
        return self.cache[p].float(), lane, y


def collate(batch):
    feat = torch.stack([b[0] for b in batch])                 # (B,70,2048)
    lane = torch.tensor([b[1] for b in batch], dtype=torch.long)
    y = torch.stack([b[2] for b in batch])
    return feat, lane, y


def main():
    torch.manual_seed(SEED)
    dev = "cuda:0"; nout = 2 * WP_N
    vis, proc = load_vision()       # 베이스 비전(동결, bf16)
    vis.eval()
    rows = load_rows()
    pxcache = build_px(sorted({p for p, _, _ in rows}), proc)
    rows = [r for r in rows if r[0] in pxcache]
    fcache = build_feat(pxcache, sorted({p for p, _, _ in rows}), vis, dev)
    del vis; torch.cuda.empty_cache()                          # 비전 해제 → 헤드학습 가속
    rows = [r for r in rows if r[0] in fcache]
    random.Random(SEED).shuffle(rows)
    nv = max(1, int(len(rows) * 0.1)); val, tr = rows[:nv], rows[nv:]
    print(f"train {len(tr)} / val {len(val)}")

    head = Head(nout).float().to(dev)
    tl = DataLoader(FeatSet(tr, fcache), batch_size=256, shuffle=True, num_workers=4, collate_fn=collate)
    vl = DataLoader(FeatSet(val, fcache), batch_size=256, num_workers=2, collate_fn=collate)
    EPOCHS = int(os.environ.get("EPOCHS", "60"))
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lossf = nn.MSELoss(); best = 1e9
    for ep in range(EPOCHS):
        head.train(); tre = 0; t0 = time.time()
        for feat, lane, y in tl:
            feat, lane, y = feat.to(dev), lane.to(dev), y.to(dev)
            opt.zero_grad(); l = lossf(head(feat, lane), y); l.backward(); opt.step()
            tre += l.item() * len(y)
        head.eval(); mae = {0: 0, 1: 0}; cnt = {0: 0, 1: 0}
        with torch.no_grad():
            for feat, lane, y in vl:
                feat, lane, y = feat.to(dev), lane.to(dev), y.to(dev); p = head(feat, lane)
                for li in (0, 1):
                    m = lane == li
                    if m.any(): mae[li] += (p[m]-y[m]).abs().sum().item()*WP_SCALE; cnt[li] += m.sum().item()*nout
        m0 = mae[0]/max(1, cnt[0]); m1 = mae[1]/max(1, cnt[1]); mm = (m0+m1)/2
        if mm < best:
            best = mm
            torch.save({"state_dict": head.state_dict(), "wp_n": WP_N, "wp_scale": WP_SCALE, "nout": nout}, HEAD_OUT)
        sched.step()
        print(f"ep{ep} mse={tre/len(tr):.4f} val 1차선={m0:.3f} 2차선={m1:.3f} (best {best:.3f}) {time.time()-t0:.1f}s")
    print(f"\n✓ best wpMAE {best:.3f}m → {HEAD_OUT}")


if __name__ == "__main__":
    main()
