"""
VeriPromiseESG 2026 - 純 BERT 推理程式碼
使用訓練好的 BERT 模型進行 ESG 承諾識別推理
"""

import json
import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from config import (
    BERT_MODEL_NAME, MAX_LENGTH, BATCH_SIZE,
    VAL_DATA_PATH, MODEL_DIR, OUTPUT_DIR,
    SEED,
    TASK_CONFIG, FIELD_WEIGHTS
)

# 設定中文字體 (Windows)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


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
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
            'text': text
        }


# ============================================
# 資料載入與預處理
# ============================================

def load_data(path: str) -> pd.DataFrame:
    """載入 JSON 資料"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)


def prepare_datasets(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """將資料分割為訓練集和驗證集"""
    train_df, val_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df['promise_status']
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def prepare_task_data(df: pd.DataFrame, task_name: str, label_map: Dict):
    """
    任務1：全部樣本，使用 data 文本
    任務2/3/4：全部樣本，統一使用 data 文本（cascade 在輸出時由 task1 結果決定 N/A）
    不再使用 promise_string，確保訓練與測試集文本格式一致
    """
    texts = df['data'].tolist()

    if task_name == "task1":
        promise_statuses = df['promise_status'].tolist() if 'promise_status' in df.columns else ['Yes'] * len(df)
        labels = [label_map.get(l, 0) for l in promise_statuses]
    else:
        target_col = TASK_CONFIG[task_name]['target_column']
        if target_col in df.columns:
            labels = [label_map.get(str(v), 0) for v in df[target_col]]
        else:
            labels = [0] * len(df)

    return texts, labels

def load_bert_model(model_path: str, device: torch.device):
    """載入 BERT 模型"""
    print(f"載入模型: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    model_name = checkpoint.get('model_name', BERT_MODEL_NAME)
    num_labels = checkpoint['num_labels']
    label_map = checkpoint['label_map']
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"  - 模型: {model_name}")
    print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
    if checkpoint.get('val_f1'):
        print(f"  - Val F1: {checkpoint.get('val_f1'):.4f}")
    
    return model, tokenizer, label_map


def bert_predict(model, dataloader, device) -> Tuple[List, List, List, List]:
    """
    BERT 推理
    
    Returns:
        predictions: 預測的類別索引
        confidences: 預測的信心度 (最大 softmax 機率)
        labels: 真實標籤
        texts: 原始文本
    """
    model.eval()
    all_preds = []
    all_confidences = []
    all_labels = []
    all_texts = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="BERT Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            texts = batch['text']
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # 計算 softmax 機率
            probs = F.softmax(outputs.logits, dim=-1)
            
            # 取得預測和信心度
            confidences, preds = torch.max(probs, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_texts.extend(texts)
    
    return all_preds, all_confidences, all_labels, all_texts

def inference_task(
    task_name: str,
    model_path: str,
    val_df: pd.DataFrame,
    device: torch.device
) -> Dict:
    """
    純 BERT 推理流程
    """
    config = TASK_CONFIG[task_name]
    label_map = config['labels']
    # 使用第一次出現的 label 名稱（避免 longer_than_5_years 被 more_than_5_years 覆蓋）
    id_to_label = {}
    for k, v in label_map.items():
        if v not in id_to_label:
            id_to_label[v] = k

    # 載入 BERT 模型
    model, tokenizer, _ = load_bert_model(model_path, device)
    
    # 準備資料
    texts, labels = prepare_task_data(val_df, task_name, label_map)
    dataset = ESGDataset(texts, labels, tokenizer, MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    
    # BERT 推理
    predictions, confidences, true_labels, all_texts = bert_predict(
        model, dataloader, device
    )
    
    # 計算指標 (若是純測試集 true_labels 可能是預設的0，這裡防呆計算不會報錯)
    macro_f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)
    micro_f1 = f1_score(true_labels, predictions, average='micro', zero_division=0)
    
    # 分類報告
    target_names = [id_to_label[i] for i in sorted(id_to_label.keys())]
    report = classification_report(
        true_labels, predictions,
        labels=list(range(len(label_map))),
        target_names=target_names,
        digits=4,
        zero_division=0
    )
    
    return {
        'task_name': task_name,
        'name': config['name'],
        'field': config.get('target_column', task_name),
        'weight': config.get('weight', 0.25),
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'report': report,
        'predictions': predictions,
        'labels': true_labels,
        'confidences': confidences,
        'avg_confidence': float(np.mean(confidences)),
        'id_to_label': id_to_label # 新增將映射表帶出供繳交檔案使用
    }

def print_evaluation_report(results: Dict):
    """印出評估報告"""
    print("\n" + "=" * 70)
    print("VeriPromiseESG 2026 - 純 BERT 評估報告")
    print("=" * 70)
    
    total_weighted_score = 0
    
    for task_name in ['task1', 'task2', 'task3', 'task4']:
        if task_name in results:
            r = results[task_name]
            weighted_score = r['macro_f1'] * r['weight']
            total_weighted_score += weighted_score
            
            print(f"\n【{r['name']}】")
            print(f"  欄位: {r['field']}")
            print(f"  權重: {r['weight']:.0%}")
            print(f"  Macro-F1: {r['macro_f1']:.4f}")
            print(f"  Micro-F1: {r['micro_f1']:.4f}")
            print(f"  平均信心度: {r['avg_confidence']:.2%}")
            print(f"  加權分數: {weighted_score:.4f}")
            print(f"\n  分類報告:\n{r['report']}")
    
    print("=" * 70)
    print(f"總加權分數: {total_weighted_score:.4f}")
    print("=" * 70)
    
    return total_weighted_score


def visualize_results(results: Dict, output_path: str):
    """視覺化評估結果"""
    tasks = []
    scores = []
    confidences = []
    weights = []
    
    for task_name in ['task1', 'task2', 'task3', 'task4']:
        if task_name in results:
            r = results[task_name]
            tasks.append(r['name'].replace('子任務', '').replace('：', '\n'))
            scores.append(r['macro_f1'])
            confidences.append(r['avg_confidence'])
            weights.append(r['weight'])
    
    if not tasks:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 圖1: 各任務 F1 分數
    x = np.arange(len(tasks))
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    
    bars = axes[0].bar(x, scores, color=colors[:len(tasks)], alpha=0.8)
    axes[0].set_ylabel('Macro-F1 Score')
    axes[0].set_title('各任務 Macro-F1 分數')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tasks)
    axes[0].set_ylim(0, 1)
    
    if np.sum(weights) > 0:
        avg_score = np.average(scores, weights=weights)
        axes[0].axhline(y=avg_score, color='red', 
                        linestyle='--', label=f'加權平均: {avg_score:.4f}')
        axes[0].legend()
    
    for bar, score in zip(bars, scores):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 圖2: 信心度分布
    bars2 = axes[1].bar(x, [c * 100 for c in confidences], color=colors[:len(tasks)], alpha=0.8)
    axes[1].set_ylabel('平均信心度 (%)')
    axes[1].set_title('各任務平均信心度')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tasks)
    axes[1].set_ylim(0, 100)
    
    for bar, conf in zip(bars2, confidences):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{conf:.1%}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"視覺化結果已儲存至: {output_path}")
    plt.close()


def save_results(results: Dict, evaluation_path: str):
    """儲存結果"""
    results_save = {}
    
    for k, v in results.items():
        if isinstance(v, dict):
            results_save[k] = {
                'task_name': v['task_name'],
                'name': v['name'],
                'field': v['field'],
                'weight': v['weight'],
                'macro_f1': v['macro_f1'],
                'micro_f1': v['micro_f1'],
                'avg_confidence': v['avg_confidence'],
                'predictions': [int(x) for x in v['predictions']]
            }
        else:
            results_save[k] = v
    
    with open(evaluation_path, 'w', encoding='utf-8') as f:
        json.dump(results_save, f, ensure_ascii=False, indent=2)
    print(f"評估報告已儲存至: {evaluation_path}")

# ============================================
# 新增: 生成比賽繳交格式 CSV
# ============================================

def generate_submission_csv(df: pd.DataFrame, results: Dict, output_path: str):
    """將預測結果整理成競賽繳交格式並儲存"""
    
    if 'id' not in df.columns:
        df['id'] = range(1, len(df) + 1)
        
    submission = pd.DataFrame({'id': df['id']})
    
    for task_name, r in results.items():
        field_name = r['field']
        id_to_label = r['id_to_label']
        
        preds_str = [id_to_label[p] for p in r['predictions']]
        submission[field_name] = preds_str

    expected_cols = ['id', 'promise_status', 'verification_timeline', 'evidence_status', 'evidence_quality']
    
    for col in expected_cols:
        if col not in submission.columns:
            submission[col] = 'N/A'

    if 'promise_status' in submission.columns:
        no_promise_mask = submission['promise_status'] == 'No'
        for col in ['verification_timeline', 'evidence_status', 'evidence_quality']:
            submission.loc[no_promise_mask, col] = 'N/A'
            
    if 'evidence_status' in submission.columns and 'evidence_quality' in submission.columns:
        no_evidence_mask = submission['evidence_status'].isin(['No', 'N/A'])
        submission.loc[no_evidence_mask, 'evidence_quality'] = 'N/A'

    submission = submission[expected_cols]
    
    # 儲存成 CSV，確保不包含 index，並使用 utf-8
    submission.to_csv(output_path, index=False, encoding='utf-8')
    print("\n" + "=" * 70)
    print(f"競賽繳交檔案已成功產生！儲存至: {output_path}")
    print("=" * 70)


# ============================================
# 主程式
# ============================================

def main():
    parser = argparse.ArgumentParser(description='VeriPromiseESG 2026 - 純 BERT 推理')
    parser.add_argument('--data', type=str, default=VAL_DATA_PATH, help='資料路徑')
    parser.add_argument('--model-dir', type=str, default=MODEL_DIR, help='模型目錄')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR, help='輸出目錄')
    args = parser.parse_args()
    
    # 建立輸出目錄
    os.makedirs(args.output, exist_ok=True)
    
    print("=" * 70)
    print("VeriPromiseESG 2026 - 純 BERT 推理系統")
    print("=" * 70)
    
    # 檢查 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ 使用裝置: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    df = load_data(args.data)
    print(f"測試集筆數: {len(df)}")
    val_df = df  

    # 執行各任務推理
    print(f"\n開始推理...")
    results = {}
    
    for task_name, config in TASK_CONFIG.items():
        model_path = os.path.join(args.model_dir, task_name, "best_model.pth")
        
        if not os.path.exists(model_path):
            print(f"警告: 找不到 {task_name} 的模型，跳過")
            continue
        
        print(f"\n{'='*50}")
        print(f"處理 {config['name']}")
        print(f"{'='*50}")
        
        result = inference_task(
            task_name=task_name,
            model_path=model_path,
            val_df=val_df,
            device=device
        )
        
        results[task_name] = result
    
    # 評估結果 (若為純測試集沒有Ground truth，此處列印的分數將不具參考意義，但不影響預測結果)
    total_score = print_evaluation_report(results)
    
    # 視覺化結果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_path = os.path.join(args.output, f"bert_results_{timestamp}.png")
    visualize_results(results, viz_path)
    
    # 儲存評估數據 (Json)
    eval_json_path = os.path.join(args.output, f"bert_evaluation_{timestamp}.json")
    save_results(results, eval_json_path)

    # ===== 新增: 產生並儲存最終繳交檔案 =====
    submission_path = os.path.join(args.output, "test.csv")
    generate_submission_csv(val_df, results, submission_path)
    
    print("\n完成！")


if __name__ == "__main__":
    main()