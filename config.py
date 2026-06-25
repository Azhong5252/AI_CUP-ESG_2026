"""
VeriPromiseESG 2026 - BERT 設定檔
"""

# ============================================
# BERT 模型設定
# ============================================
# BERT_MODEL_NAME = "bert-base-chinese"
BERT_MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 2e-5

# ============================================
# 資料路徑
# ============================================
TRAIN_DATA_PATH = "dataset/vpesg4k_train_augmented.json"
EXTRA_TRAIN_DATA_PATH = "dataset/vpesg4k_val_1000.json"
VAL_DATA_PATH = "dataset/vpesg4k_test_2000.json"
MODEL_DIR = "./models"
OUTPUT_DIR = "./results"

# ============================================
# 評估設定
# ============================================
SEED = 42
TEST_SIZE = 0.2

# 評估欄位與權重 (與主辦方一致)
EVAL_FIELDS = ['promise_status', 'evidence_status', 'evidence_quality', 'verification_timeline']
FIELD_WEIGHTS = {
    'promise_status': 0.20,        # 承諾辨識
    'evidence_status': 0.30,       # 證據支持
    'evidence_quality': 0.35,      # 清晰度
    'verification_timeline': 0.15  # 時機預測
}

# ============================================
# 任務設定
# ============================================
TASK_CONFIG = {
    'task1': {
        'name': '子任務一：承諾識別',
        'labels': {"No": 0, "Yes": 1},
        'target_column': 'promise_status',
        'weight': 0.20,
        'description': '判斷文本中是否包含企業的 ESG 承諾'
    },
    'task2': {
        'name': '子任務二：證據支持判斷',
        'labels': {"No": 0, "Yes": 1},  # cascade: N/A handled externally by task1
        'target_column': 'evidence_status',
        'weight': 0.30,
        'description': '判斷承諾是否有證據支持'
    },
    'task3': {
        'name': '子任務三：清晰度評估',
        'labels': {"Clear": 0, "Not Clear": 1, "Misleading": 2},  # cascade: N/A handled externally
        'target_column': 'evidence_quality',
        'weight': 0.35,
        'description': '評估證據的清晰程度'
    },
    'task4': {
        'name': '子任務四：驗證時機預測',
        'labels': {
            "already": 0,
            "within_2_years": 1,
            "between_2_and_5_years": 2,
            "more_than_5_years": 3,       
            "longer_than_5_years": 3     
        },
        'target_column': 'verification_timeline',
        'weight': 0.15,
        'description': '預測承諾的驗證時機'
    }
}
