"""
Kaggle GPU training kernel: DeBERTa-v3 fine-tune, 5-fold GroupKFold on prompt_id.

Outputs (in /kaggle/working):
  oof_deberta.parquet   out-of-fold probabilities (id + 18 class columns)
  fold{k}.pt            fp16 state_dict of the best epoch for each fold
  cv_summary.json
"""
import gc
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

MODEL_NAME = os.environ.get("MODEL_NAME", "microsoft/deberta-v3-large")
MAX_LEN = 512
N_FOLDS = 5
EPOCHS = 4
BS = 8
LR = 1e-5 if "large" in MODEL_NAME else 2e-5
WARMUP = 0.1
LABEL_SMOOTH = 0.05
SEED = 42
DATA = "/kaggle/input/competitions/name-that-llm"
OUT = "/kaggle/working"

torch.manual_seed(SEED)
np.random.seed(SEED)
dev = torch.device("cuda")
print("GPU:", torch.cuda.get_device_name(0), "| model:", MODEL_NAME, flush=True)

train = pd.read_csv(f"{DATA}/train.csv")
classes = np.array(sorted(train["model"].unique()))
cls2id = {c: i for i, c in enumerate(classes)}
y = train["model"].map(cls2id).values
K = len(classes)

tok = AutoTokenizer.from_pretrained(MODEL_NAME)


class DS(Dataset):
    def __init__(self, texts, labels=None):
        self.enc = tok(list(texts), truncation=True, max_length=MAX_LEN)
        self.labels = labels

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        item = {"input_ids": self.enc["input_ids"][i], "attention_mask": self.enc["attention_mask"][i]}
        if self.labels is not None:
            item["labels"] = int(self.labels[i])
        return item


def collate(batch):
    ml = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), ml), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(batch), ml), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        att[i, :n] = 1
    out = {"input_ids": ids, "attention_mask": att}
    if "labels" in batch[0]:
        out["labels"] = torch.tensor([b["labels"] for b in batch])
    return out


@torch.no_grad()
def predict(model, loader):
    model.eval()
    probs = []
    for b in loader:
        b = {k: v.to(dev) for k, v in b.items() if k != "labels"}
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(**b).logits
        probs.append(torch.softmax(logits.float(), -1).cpu().numpy())
    return np.concatenate(probs)


oof = np.zeros((len(train), K), dtype=np.float32)
summary = {}
gkf = GroupKFold(n_splits=N_FOLDS)
for fold, (tr_idx, va_idx) in enumerate(gkf.split(train, y, groups=train["prompt_id"])):
    t0 = time.time()
    tr_dl = DataLoader(DS(train.text.iloc[tr_idx], y[tr_idx]), batch_size=BS, shuffle=True, collate_fn=collate, num_workers=2)
    va_dl = DataLoader(DS(train.text.iloc[va_idx], y[va_idx]), batch_size=BS * 4, shuffle=False, collate_fn=collate, num_workers=2)

    # force fp32 master weights; autocast handles fp16 compute (newer transformers may load checkpoint dtype)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=K).float().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total = len(tr_dl) * EPOCHS
    sch = get_linear_schedule_with_warmup(opt, int(WARMUP * total), total)
    scaler = torch.cuda.amp.GradScaler()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    best_ba, best_probs, best_state = -1, None, None
    for ep in range(EPOCHS):
        model.train()
        run_loss, n_skip = 0.0, 0
        for b in tr_dl:
            b = {k: v.to(dev) for k, v in b.items()}
            labels = b.pop("labels")
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(**b).logits
            loss = loss_fn(logits.float(), labels)
            if not torch.isfinite(loss):
                n_skip += 1
                opt.zero_grad()
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
            opt.zero_grad()
            run_loss += loss.item()
        probs = predict(model, va_dl)
        ba = balanced_accuracy_score(y[va_idx], probs.argmax(1))
        print(f"fold {fold} ep {ep}: loss={run_loss/len(tr_dl):.4f} val_bal_acc={ba:.4f} skipped={n_skip} ({time.time()-t0:.0f}s)", flush=True)
        if ba > best_ba:
            best_ba, best_probs = ba, probs
            best_state = {k: v.detach().half().cpu() for k, v in model.state_dict().items()}

    oof[va_idx] = best_probs
    torch.save(best_state, f"{OUT}/fold{fold}.pt")
    summary[f"fold{fold}"] = float(best_ba)
    print(f"fold {fold} best={best_ba:.4f}", flush=True)
    del model, opt, sch, scaler, best_state
    gc.collect()
    torch.cuda.empty_cache()

ba = balanced_accuracy_score(y, oof.argmax(1))
summary["oof"] = float(ba)
summary["model"] = MODEL_NAME
print(f"\nOOF balanced accuracy: {ba:.4f}", flush=True)

df = pd.DataFrame(oof, columns=classes)
df.insert(0, "id", train["id"].values)
df.to_parquet(f"{OUT}/oof_deberta.parquet", index=False)
json.dump(summary, open(f"{OUT}/cv_summary.json", "w"), indent=2)
