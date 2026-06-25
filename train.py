"""
VeriPromiseESG 2026 - BERT 訓練程式碼
訓練四個子任務的分類模型並儲存為 .pth 格式

改進：
- 合併 train_augmented + val_1000 作為訓練資料
- 任務2/3/4 改為 cascade 架構：只訓練 Yes 樣本，N/A 由任務1決定
- 任務2/3/4 統一使用 data 欄位文本（不使用 promise_string），與測試集一致
- FocalLoss 處理嚴重類別不平衡
- 早期停止 (Early Stopping) 避免過擬合
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import warnings
import test
warnings.filterwarnings('ignore')

from config import (
    TRAIN_DATA_PATH, EXTRA_TRAIN_DATA_PATH, MODEL_DIR, BERT_MODEL_NAME,
    SEED, MAX_LENGTH, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    TASK_CONFIG
)


# ============================================
# 設定隨機種子
# ============================================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# ============================================
# Focal Loss
# ============================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, labels):
        ce = F.cross_entropy(logits, labels, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ============================================
# 自定義 Dataset
# ============================================
class ESGDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ============================================
# 資料載入
# ============================================
def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)


def prepare_task_data(df, task_name, label_map):
    """
    任務1：全部樣本
    任務2/3/4：只使用有效標籤的樣本（cascade），僅用 data 文本（不用 promise_string）
    """
    target_col = TASK_CONFIG[task_name]['target_column']

    if task_name == "task1":
        texts = df['data'].tolist()
        labels = [label_map[l] for l in df['promise_status']]
    else:
        # 只保留 target_col 在 label_map 中有對應的樣本
        valid_mask = df[target_col].isin(label_map.keys())
        filtered_df = df[valid_mask].reset_index(drop=True)
        texts = filtered_df['data'].tolist()
        labels = [label_map[v] for v in filtered_df[target_col]]

    return texts, labels


# ============================================
# 訓練函數
# ============================================
def train_epoch(model, dataloader, optimizer, scheduler, device, criterion):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        preds = torch.argmax(outputs.logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, f1


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return f1


def train_task(task_name, train_texts, train_labels,
               num_labels, tokenizer, device, output_dir):
    print(f"\n{'='*50}")
    print(f"開始訓練 {TASK_CONFIG[task_name]['name']}")
    print(f"{'='*50}")
    print(f"訓練樣本數: {len(train_texts)}, 類別數: {num_labels}")

    # 90% train / 10% val for early stopping
    t_texts, v_texts, t_labels, v_labels = train_test_split(
        train_texts, train_labels, test_size=0.1, random_state=SEED,
        stratify=train_labels if len(set(train_labels)) > 1 else None
    )

    # Focal Loss with class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(t_labels),
        y=t_labels
    )
    # Pad to num_labels if some classes missing from split
    padded_weights = np.ones(num_labels)
    for i, cls in enumerate(np.unique(t_labels)):
        padded_weights[cls] = class_weights[i]
    class_weights_tensor = torch.tensor(padded_weights, dtype=torch.float).to(device)
    criterion = FocalLoss(gamma=2.0, weight=class_weights_tensor)
    print(f"Class weights: {padded_weights}")

    train_dataset = ESGDataset(t_texts, t_labels, tokenizer, MAX_LENGTH)
    val_dataset = ESGDataset(v_texts, v_labels, tokenizer, MAX_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME, num_labels=num_labels
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    task_output_dir = os.path.join(output_dir, task_name)
    os.makedirs(task_output_dir, exist_ok=True)
    best_model_path = os.path.join(task_output_dir, "best_model.pth")

    best_val_f1 = -1
    patience = 3
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
        train_loss, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, device, criterion
        )
        val_f1 = evaluate(model, val_loader, device)
        print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'train_f1': train_f1,
                'num_labels': num_labels,
                'model_name': BERT_MODEL_NAME,
                'label_map': TASK_CONFIG[task_name]['labels']
            }, best_model_path)
            print(f"  ✓ 新最佳模型 (Val F1: {best_val_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"最佳 Val F1: {best_val_f1:.4f}，已儲存至 {best_model_path}")
    return best_val_f1


# ============================================
# 主程式
# ============================================
def main():
    print("=" * 60)
    print("VeriPromiseESG 2026 - BERT 訓練程式")
    print("=" * 60)

    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 載入並合併所有有標籤的資料
    print("\n[1] 載入並合併訓練資料...")
    df1 = load_data(TRAIN_DATA_PATH)
    df2 = load_data(EXTRA_TRAIN_DATA_PATH)
    df = pd.concat([df1, df2], ignore_index=True)
    print(f"  train_augmented: {len(df1)} 筆")
    print(f"  val_1000: {len(df2)} 筆")
    print(f"  合併後總計: {len(df)} 筆")

    # 資料統計
    print("\n[2] 資料分佈統計:")
    for task_key, config in TASK_CONFIG.items():
        col = config['target_column']
        if col in df.columns:
            print(f"\n{config['name']} ({col}):")
            print(df[col].value_counts())

    # 載入 Tokenizer
    print(f"\n[3] 載入 Tokenizer: {BERT_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # 訓練各任務
    print("\n[4] 開始訓練各子任務...")
    results = {}

    for task_key, config in TASK_CONFIG.items():
        train_texts, train_labels = prepare_task_data(df, task_key, config['labels'])
        num_labels = len(set(config['labels'].values()))

        best_f1 = train_task(
            task_key,
            train_texts, train_labels,
            num_labels=num_labels,
            tokenizer=tokenizer,
            device=device,
            output_dir=MODEL_DIR
        )
        results[task_key] = best_f1

    print("\n" + "=" * 60)
    print("訓練完成！各任務最佳 Val Macro-F1:")
    print("=" * 60)
    for task_key, config in TASK_CONFIG.items():
        print(f"{config['name']}: {results[task_key]:.4f}")

    avg_f1 = np.mean(list(results.values()))
    print(f"\n平均 Val Macro-F1: {avg_f1:.4f}")
    print(f"\n模型已儲存至 {MODEL_DIR}/ 目錄")

    # 訓練完成後自動執行推理生成 test.csv
    print("\n[5] 執行測試集推理...")
    import subprocess, sys
    subprocess.run([sys.executable, 'test.py'], check=True)
    test.main() # 確保 test.py 中有 main() 函數可直接呼叫

if __name__ == "__main__":
    main()
