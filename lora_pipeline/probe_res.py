#!/usr/bin/env python3
"""
공정한 지각 재측정: 깨끗한(비복구) 커브 이미지에 대해 해상도별로
베이스 Qwen3-VL 이 도로 방향(좌/우/직진)을 맞히는지.
clean CSV(labels_inner, labels_outer_clean)만 사용 → gt부호=도로곡률 신뢰.
"""
import os, csv, collections
import cv2
from PIL import Image as PILImage
import torch
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

HERE = os.path.dirname(__file__)
BASE = "Qwen/Qwen3-VL-2B-Instruct"
IMG_DIR = os.path.join(HERE, "dataset", "images")
CLEAN_CSVS = ["labels_inner.csv", "labels_outer_clean.csv"]
PERCEPT = ("Look at the road ahead in this driving camera image. Does the road go "
    "STRAIGHT, curve LEFT, or curve RIGHT? Answer one word: STRAIGHT, LEFT, or RIGHT.")
RES = [(320,240),(640,480),(896,672)]   # 해상도 스윕


def sample_clean(per_class=6):
    rows=[]
    for cf in CLEAN_CSVS:
        p=os.path.join(HERE,"dataset",cf)
        if os.path.exists(p):
            for r in csv.DictReader(open(p)): rows.append(r)
    byc=collections.defaultdict(list)
    for r in rows:
        st=int(r["steering"])
        cls = "LEFT" if st<=-3 else ("RIGHT" if st>=3 else "STRAIGHT")
        img=os.path.join(IMG_DIR,r["fname"])
        if os.path.exists(img): byc[cls].append((cls,img))
    out=[]
    for cls in ("LEFT","STRAIGHT","RIGHT"):
        L=byc[cls]; step=max(1,len(L)//per_class)
        out+=L[::step][:per_class]
    return out


def main():
    samples=sample_clean(6)
    print("샘플:",collections.Counter(c for c,_ in samples))
    model=Qwen3VLForConditionalGeneration.from_pretrained(
        BASE,torch_dtype=torch.bfloat16,device_map="cuda:0",attn_implementation="sdpa").eval()
    for (W,H) in RES:
        px=W*H
        proc=Qwen3VLProcessor.from_pretrained(BASE,min_pixels=px,max_pixels=px)
        hit=0; conf=collections.Counter()
        for cls,p in samples:
            bgr=cv2.resize(cv2.imread(p),(W,H),interpolation=cv2.INTER_AREA)
            pil=PILImage.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
            msgs=[{"role":"user","content":[{"type":"image","image":pil},{"type":"text","text":PERCEPT}]}]
            text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
            inp=proc(text=[text],images=[pil],return_tensors="pt").to("cuda:0")
            with torch.inference_mode():
                out=model.generate(**inp,max_new_tokens=6,do_sample=False,use_cache=True)
            ans=proc.decode(out[0][inp.input_ids.shape[1]:],skip_special_tokens=True).strip().upper()
            pred = "LEFT" if "LEFT" in ans else ("RIGHT" if "RIGHT" in ans else ("STRAIGHT" if "STRAIGHT" in ans else "?"))
            hit += (pred==cls); conf[pred]+=1
        print(f"  {W}x{H} ({px//1000}k px): 지각정확도 {hit}/{len(samples)} | 답분포 {dict(conf)}")


if __name__=="__main__":
    main()
