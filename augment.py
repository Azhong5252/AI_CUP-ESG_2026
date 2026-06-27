# augment.py — 使用 claude-sonnet-4-6 對 ESG 訓練資料進行語意改寫擴增
import json
import time
import copy
import anthropic

INPUT_FILE   = "dataset/vpesg4k_train_1000.json"
OUTPUT_FILE  = "dataset/vpesg4k_train_augmented.json"
AUGMENT_PER_SAMPLE = 4   # 每筆原始資料產生幾筆改寫
MODEL        = "claude-sonnet-4-6"
MAX_TOKENS   = 1024

client = anthropic.Anthropic()  # 讀取環境變數 ANTHROPIC_API_KEY


def build_prompt(text: str) -> str:
    return f"""你是一位專業的中文 ESG 報告撰寫者。
請將以下這段企業 ESG 報告文字，用**不同的用詞與句式**改寫，使語意與原文相同但表達方式不同。
要求：
1. 保留所有具體數字、年份、指標、公司名稱等關鍵事實
2. 保留原文中承諾、證據、時間範圍等語意資訊
3. 僅回傳改寫後的文字，不要加任何前綴或說明

原文：
{text}

改寫："""


def augment_one(record: dict, new_id: int, text: str) -> dict | None:
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": build_prompt(text)}],
        )
        rewritten = response.content[0].text.strip()
        if not rewritten:
            return None
        new_record = copy.deepcopy(record)
        new_record["id"]   = new_id
        new_record["data"] = rewritten
        return new_record
    except Exception as e:
        print(f"  [WARN] id={record['id']} 擴增失敗：{e}")
        return None


def main():
    with open(INPUT_FILE, encoding="utf-8") as f:
        original = json.load(f)

    augmented = []
    new_id = max(r["id"] for r in original) + 1  # 新 id 從原始最大值 +1 開始

    for i, record in enumerate(original):
        augmented.append(record)  # 保留原始資料
        text = record.get("data", "")
        if not text:
            continue

        print(f"[{i+1}/{len(original)}] id={record['id']} 擴增中...")
        for _ in range(AUGMENT_PER_SAMPLE):
            result = augment_one(record, new_id, text)
            if result:
                augmented.append(result)
                new_id += 1
            time.sleep(0.3)  # 避免觸發速率限制

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(augmented, f, ensure_ascii=False, indent=2)

    print(f"\n完成！原始 {len(original)} 筆 → 擴增後 {len(augmented)} 筆")
    print(f"輸出至：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
