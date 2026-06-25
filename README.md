# VeriPromiseESG 2026 — 純 BERT 多任務分類系統

本專案為 **VeriPromiseESG 2026 競賽** 的解決方案，利用中文 RoBERTa 模型對企業 ESG（環境、社會、治理）報告中的承諾文句進行四項子任務分類，並輸出符合競賽格式的預測結果。

---

## 目錄

1. [專案概述](#專案概述)
2. [目錄結構](#目錄結構)
3. [環境安裝](#環境安裝)
4. [資料說明](#資料說明)
5. [資料前處理](#資料前處理)
6. [模型訓練](#模型訓練)
7. [推理與預測](#推理與預測)
8. [資料後處理（Cascade 邏輯）](#資料後處理cascade-邏輯)
9. [輸出結果](#輸出結果)
10. [設定檔說明](#設定檔說明)
11. [執行方式](#執行方式)
12. [各任務評估指標](#各任務評估指標)
13. [常見問題](#常見問題)

---

## 專案概述

### 任務描述

給定企業 ESG 報告中的一段文字（`data` 欄位），模型需同時預測以下四個子任務：

| 子任務 | 欄位名稱 | 說明 | 類別 | 評估權重 |
|--------|----------|------|------|----------|
| Task 1 | `promise_status` | 承諾識別：文句中是否包含 ESG 承諾 | Yes / No | 20% |
| Task 2 | `evidence_status` | 證據支持：承諾是否有具體證據支持 | Yes / No | 30% |
| Task 3 | `evidence_quality` | 清晰度評估：證據的清晰程度 | Clear / Not Clear / Misleading | 35% |
| Task 4 | `verification_timeline` | 驗證時機：預測可驗證的時間範圍 | already / within\_2\_years / between\_2\_and\_5\_years / more\_than\_5\_years | 15% |

### 評估指標

競賽採用**加權 Macro-F1**：

```
總分 = Σ (各任務 Macro-F1 × 任務權重)
     = F1(task1) × 0.20 + F1(task2) × 0.30 + F1(task3) × 0.35 + F1(task4) × 0.15
```

### 系統架構

```
原始 JSON 資料
     │
     ▼
┌─────────────────────────────────┐
│         資料前處理               │
│  合併 train_augmented + val_1000 │
│  過濾各任務有效樣本               │
│  Tokenize (中文 RoBERTa)         │
└──────────────┬──────────────────┘
               │
     ┌─────────┴──────────┐
     ▼                    ▼
  Task 1 模型          Task 2/3/4 模型
  (全部樣本)           (cascade 架構，僅 Yes 樣本訓練)
     │                    │
     └─────────┬──────────┘
               ▼
┌─────────────────────────────────┐
│           推理階段               │
│  四個模型各自對測試集預測          │
│  輸出預測標籤 + 信心度            │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│       資料後處理 (Cascade)        │
│  Task1=No → Task2/3/4 設為 N/A   │
│  Task2=No → Task3 設為 N/A       │
└──────────────┬──────────────────┘
               │
               ▼
        test.csv（競賽繳交）
```

---

## 目錄結構

```
BERT2/
├── config.py                          # 全域設定（模型名稱、路徑、超參數、任務設定）
├── train.py                           # 訓練腳本
├── test.py                            # 推理腳本（生成繳交 CSV）
├── dataset/
│   ├── vpesg4k_train_augmented.json   # 擴增訓練集（4,992 筆）
│   ├── vpesg4k_val_1000.json          # 驗證集（合併至訓練，1,000 筆）
│   └── vpesg4k_test_2000.json         # 測試集（2,000 筆，無標籤）
├── models/
│   ├── task1/best_model.pth           # Task 1 最佳模型（~1.2 GB）
│   ├── task2/best_model.pth           # Task 2 最佳模型（~1.2 GB）
│   ├── task3/best_model.pth           # Task 3 最佳模型（~1.2 GB）
│   └── task4/best_model.pth           # Task 4 最佳模型（~1.2 GB）
├── results/
│   ├── test.csv                       # 競賽繳交預測結果
│   ├── bert_evaluation_*.json         # 各任務詳細評估指標
│   └── bert_results_*.png             # 視覺化圖表
└── txt/
    └── requirements.txt               # Python 套件需求
```

---

## 環境安裝

### 系統需求

- Python 3.8+
- CUDA 11.8+（訓練時建議使用 GPU，VRAM ≥ 8 GB）
- 磁碟空間：至少 8 GB（模型檔案約 4.8 GB）

### 安裝套件

```bash
pip install -r txt/requirements.txt
```

主要套件版本：

| 套件 | 版本 | 用途 |
|------|------|------|
| `transformers` | ≥ 4.30.0 | BERT 模型與 Tokenizer |
| `torch` | ≥ 2.0.0 | 深度學習框架 |
| `pandas` | ≥ 2.0.0 | 資料處理 |
| `numpy` | ≥ 1.24.0 | 數值運算 |
| `scikit-learn` | ≥ 1.3.0 | 評估指標、類別權重 |
| `tqdm` | ≥ 4.65.0 | 進度條 |
| `matplotlib` | ≥ 3.7.0 | 視覺化圖表 |

---

## 資料說明

### JSON 欄位格式

每筆資料的完整欄位如下：

```json
{
  "id": 10001,
  "data": "企業 ESG 報告原始文字（中文），為模型輸入文本",
  "esg_type": "E",
  "promise_status": "Yes",
  "promise_string": "承諾的摘錄文字",
  "verification_timeline": "within_2_years",
  "evidence_status": "Yes",
  "evidence_string": "證據的摘錄文字",
  "evidence_quality": "Clear",
  "company": "mediatek",
  "ticker": 2454,
  "page_number": 48,
  "pdf_url": "https://...",
  "company_source": "https://..."
}
```

**模型輸入**：統一使用 `data` 欄位（原始完整段落），不使用 `promise_string` 或 `evidence_string`，確保訓練與測試集文本格式一致。

### 資料集統計

| 資料集 | 檔案 | 筆數 | 用途 |
|--------|------|------|------|
| 擴增訓練集 | `vpesg4k_train_augmented.json` | 4,992 | 訓練 |
| 驗證集 | `vpesg4k_val_1000.json` | 1,000 | 合併至訓練（擴大訓練資料） |
| 測試集 | `vpesg4k_test_2000.json` | 2,000 | 推理生成繳交檔 |
| **訓練合計** | | **5,992** | |

> **關鍵決策**：將原本的驗證集（`vpesg4k_val_1000.json`）也合併入訓練，擴大訓練資料量以提升模型泛化能力，是本版本（0.59）相較前一版的主要改動。

---

## 資料前處理

### 1. 資料載入與合併

```
train.py → load_data()
```

- 分別載入 `vpesg4k_train_augmented.json` 與 `vpesg4k_val_1000.json`
- 使用 `pd.concat()` 合併為 5,992 筆訓練資料
- 印出各任務欄位的類別分佈，便於確認資料平衡性

### 2. 各任務樣本過濾

```
train.py → prepare_task_data()
```

各任務的訓練樣本選取策略不同（Cascade 架構）：

| 任務 | 使用樣本 | 原因 |
|------|----------|------|
| Task 1 | 全部 5,992 筆 | 判斷是否有承諾，所有文字均可訓練 |
| Task 2 | 只保留 `evidence_status` 有效值的樣本 | 若 Task1=No，Task2 答案為 N/A，不參與訓練 |
| Task 3 | 只保留 `evidence_quality` 有效值的樣本 | 同上 |
| Task 4 | 只保留 `verification_timeline` 有效值的樣本 | 同上 |

### 3. 標籤映射

各任務的字串標籤轉換為整數索引：

```python
task1: {"No": 0, "Yes": 1}
task2: {"No": 0, "Yes": 1}
task3: {"Clear": 0, "Not Clear": 1, "Misleading": 2}
task4: {
    "already": 0,
    "within_2_years": 1,
    "between_2_and_5_years": 2,
    "more_than_5_years": 3,
    "longer_than_5_years": 3   # 視為同一類別
}
```

### 4. 文字 Tokenize

```
train.py → ESGDataset.__getitem__()
```

使用 `hfl/chinese-roberta-wwm-ext` 的 Tokenizer：

- `truncation=True`：超過 512 token 的文字截斷
- `padding='max_length'`：不足 512 token 的文字補齊
- `max_length=512`：最大序列長度
- 輸出：`input_ids`、`attention_mask`、`labels`

### 5. 訓練集 / 驗證集切分

```
train.py → train_task()
```

- 比例：90% 訓練 / 10% 驗證（用於早期停止）
- 採用 `stratify=train_labels` 保持類別分佈一致
- 亂數種子：`SEED=42`（可重現）

---

## 模型訓練

### 模型架構

- **基底模型**：`hfl/chinese-roberta-wwm-ext`
  - 12 層 Transformer
  - 768 維隱藏層
  - 約 1.1 億參數
  - 以全詞遮罩（Whole Word Masking）在大型中文語料預訓練

- **分類頭**：`AutoModelForSequenceClassification`
  - Task 1：線性層（768 → 2）
  - Task 2：線性層（768 → 2）
  - Task 3：線性層（768 → 3）
  - Task 4：線性層（768 → 4）

四個任務各自訓練獨立模型，不共享參數。

### 損失函數：Focal Loss

為處理 ESG 資料的嚴重**類別不平衡**問題，採用 Focal Loss：

```
FocalLoss(logits, labels) = mean((1 - p_t)^γ × CE_loss)
```

- `γ = 2.0`：聚焦因子，讓模型更專注於難分樣本
- 結合 `compute_class_weight('balanced')` 計算類別權重，進一步平衡訓練

### 優化器與學習率排程

```python
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps = 10% × total_steps,   # 線性 warmup
    num_training_steps = total_steps
)
```

- **Warmup**：前 10% 訓練步數線性升溫，避免初期更新過激
- **Decay**：warmup 後線性衰減至 0

### 梯度裁切

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

防止梯度爆炸，穩定訓練過程。

### 早期停止（Early Stopping）

- **監控指標**：驗證集 Macro-F1
- **Patience**：連續 3 個 Epoch 無改善即停止
- **最大 Epoch**：10
- 每次驗證 F1 創新高時儲存 checkpoint

### 訓練流程

```
for task in [task1, task2, task3, task4]:
    1. 過濾該任務有效樣本
    2. 計算類別權重 → 建立 FocalLoss
    3. 90/10 切分訓練/驗證集
    4. 初始化 AutoModelForSequenceClassification
    5. for epoch in range(1, 11):
         a. 訓練一個 Epoch（前向傳播、損失計算、反向傳播、梯度裁切、更新）
         b. 在驗證集評估 Macro-F1
         c. 若 F1 > 歷史最佳 → 儲存 best_model.pth
         d. 若連續 3 Epoch 未改善 → Early Stopping
    6. 印出最佳驗證 F1
```

### Checkpoint 格式

每個任務的 `best_model.pth` 包含：

```python
{
    'epoch': int,                  # 最佳 Epoch 編號
    'model_state_dict': ...,       # 模型權重
    'optimizer_state_dict': ...,   # 優化器狀態
    'val_f1': float,               # 最佳驗證 Macro-F1
    'train_f1': float,             # 同 Epoch 訓練 Macro-F1
    'num_labels': int,             # 類別數
    'model_name': str,             # 模型名稱
    'label_map': dict              # 標籤對應表（如 {"No": 0, "Yes": 1}）
}
```

---

## 推理與預測

```
test.py → main()
```

### 流程

1. **載入模型**：從 `models/task{1-4}/best_model.pth` 讀取 checkpoint，恢復模型權重與標籤對應表
2. **載入測試集**：讀取 `vpesg4k_test_2000.json`（2,000 筆）
3. **Tokenize**：與訓練時完全相同的 Tokenize 設定
4. **批次推理**：`batch_size=8`，`torch.no_grad()` 節省記憶體
5. **機率計算**：對 logits 做 Softmax，取最大值作為預測類別，最大機率作為信心度

```python
probs = F.softmax(outputs.logits, dim=-1)
confidences, predictions = torch.max(probs, dim=-1)
```

6. **索引轉標籤**：利用 checkpoint 內的 `label_map` 將整數預測轉回字串

---

## 資料後處理（Cascade 邏輯）

推理完成後，依 ESG 承諾的邏輯關聯性套用 Cascade 規則，確保預測的語義一致性：

```
如果 promise_status == "No"
    → evidence_status      = "N/A"
    → evidence_quality     = "N/A"
    → verification_timeline = "N/A"

如果 evidence_status ∈ {"No", "N/A"}
    → evidence_quality = "N/A"
```

**原因**：若一段文字中沒有 ESG 承諾（Task1=No），則後續的證據支持、清晰度、驗證時機均無意義；同理，若無證據支持（Task2=No），則清晰度（Task3）也不適用。

此邏輯在 `test.py → generate_submission_csv()` 中實作。

---

## 輸出結果

### 1. 競賽繳交檔案：`results/test.csv`

格式範例：

```csv
id,promise_status,verification_timeline,evidence_status,evidence_quality
12001,Yes,already,Yes,Clear
12002,Yes,within_2_years,No,N/A
12003,No,N/A,N/A,N/A
```

欄位順序固定為：`id`, `promise_status`, `verification_timeline`, `evidence_status`, `evidence_quality`

### 2. 評估報告：`results/bert_evaluation_YYYYMMDD_HHMMSS.json`

JSON 格式，記錄各任務的：
- `macro_f1`、`micro_f1`
- `avg_confidence`（平均信心度）
- `predictions`（所有預測索引）

### 3. 視覺化圖表：`results/bert_results_YYYYMMDD_HHMMSS.png`

兩張子圖：
- **左圖**：各任務 Macro-F1 分數長條圖（含加權平均參考線）
- **右圖**：各任務平均信心度長條圖

### 控制台評估報告範例

```
======================================================================
VeriPromiseESG 2026 - 純 BERT 評估報告
======================================================================

【子任務一：承諾識別】
  欄位: promise_status
  權重: 20%
  Macro-F1: 0.8500
  Micro-F1: 0.8800
  平均信心度: 73.40%
  加權分數: 0.1700

...

======================================================================
總加權分數: 0.5900
======================================================================
```

---

## 設定檔說明

所有超參數集中於 `config.py`，修改此檔即可調整訓練行為：

```python
# 模型設定
BERT_MODEL_NAME = "hfl/chinese-roberta-wwm-ext"   # 可替換其他 HuggingFace 模型
MAX_LENGTH      = 512                              # Token 最大長度
BATCH_SIZE      = 8                                # 批次大小（GPU 記憶體不足可調低）
EPOCHS          = 10                               # 最大訓練輪數
LEARNING_RATE   = 2e-5                             # 學習率

# 資料路徑
TRAIN_DATA_PATH       = "dataset/vpesg4k_train_augmented.json"
EXTRA_TRAIN_DATA_PATH = "dataset/vpesg4k_val_1000.json"
VAL_DATA_PATH         = "dataset/vpesg4k_test_2000.json"
MODEL_DIR             = "./models"
OUTPUT_DIR            = "./results"

# 評估設定
SEED      = 42     # 亂數種子（保證可重現）
TEST_SIZE = 0.2    # 早期停止用內部驗證比例（train_task 中覆蓋為 0.1）

# 評估欄位與權重
FIELD_WEIGHTS = {
    'promise_status':        0.20,
    'evidence_status':       0.30,
    'evidence_quality':      0.35,
    'verification_timeline': 0.15
}
```

---

## 執行方式

### 完整流程（訓練 + 推理）

```bash
python train.py
```

訓練完成後會**自動呼叫** `test.py` 執行推理，生成 `results/test.csv`。  
訓練時間依 GPU 而異，約 1～3 小時。

### 僅執行推理（模型已訓練完成）

```bash
python test.py
```

選用參數：

```bash
# 指定自訂測試集路徑
python test.py --data dataset/vpesg4k_test_2000.json

# 指定模型目錄
python test.py --model-dir ./models

# 指定輸出目錄
python test.py --output ./results
```

### 執行環境確認

```bash
# 確認 GPU 可用性
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 各任務評估指標

### Task 1：承諾識別（promise\_status）

- 二元分類：Yes / No
- 測試集中 Yes 比例遠高於 No（資料不平衡），Focal Loss 有助緩解
- 此任務為 Cascade 的起點，準確率對整體 pipeline 影響最大

### Task 2：證據支持（evidence\_status）

- 二元分類：Yes / No（N/A 由 Task1 決定，不參與訓練）
- 訓練樣本為 Task1=Yes 的子集
- 評估權重 30%，為各任務中佔比最高者之一

### Task 3：清晰度評估（evidence\_quality）

- 三元分類：Clear / Not Clear / Misleading
- 訓練樣本進一步過濾至有有效 `evidence_quality` 標籤的樣本
- **評估權重 35%**，為四任務中最高，對最終分數影響最大

### Task 4：驗證時機（verification\_timeline）

- 四元分類：already / within\_2\_years / between\_2\_and\_5\_years / more\_than\_5\_years
- `longer_than_5_years` 與 `more_than_5_years` 合併為同一類別（index 3）
- 評估權重 15%，為四任務中最低

---

## 常見問題

### Q: 訓練時出現 CUDA out of memory

調小 `config.py` 中的 `BATCH_SIZE`：

```python
BATCH_SIZE = 4   # 從 8 改為 4
```

### Q: 找不到模型檔案（推理時報錯）

請先執行訓練：

```bash
python train.py
```

確認 `models/task1/best_model.pth`、`models/task2/best_model.pth`、`models/task3/best_model.pth`、`models/task4/best_model.pth` 均存在。

### Q: 繳交的 test.csv 中出現非預期的 N/A

這是 Cascade 後處理的正常行為。若 Task1 預測為 No，Task2/3/4 的預測值會被強制設為 N/A，符合競賽規範。

### Q: 想更換 BERT 模型

修改 `config.py`：

```python
# 更換為其他中文預訓練模型
BERT_MODEL_NAME = "hfl/chinese-macbert-large"
# 或
BERT_MODEL_NAME = "bert-base-chinese"
```

### Q: 如何提升 Task 3 分數（清晰度評估）

Task 3 評估權重最高（35%），可優先嘗試：
1. 增加 Task 3 的訓練資料（資料擴增）
2. 對 Task 3 單獨調整 `LEARNING_RATE` 或 `EPOCHS`
3. 使用更大的預訓練模型（如 `hfl/chinese-macbert-large`）

---

## 版本紀錄

| 版本 | 主要改動 |
|------|----------|
| BERT2（本版） | 合併 val\_1000 至訓練資料；Task2/3/4 改用 cascade 架構，僅用 `data` 欄位；FocalLoss；Early Stopping |
| BERT1（前版） | 僅使用 train\_augmented；Task2/3/4 使用 promise\_string |
