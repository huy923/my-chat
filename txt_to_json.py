import json

with open("data_train.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

with open("data.jsonl", "w", encoding="utf-8") as f:
    for line in lines:
        json.dump({"text": line.strip()}, f, ensure_ascii=False)
        f.write("\n")
