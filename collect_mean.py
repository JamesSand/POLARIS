import os
import re

ROOT_DIR = "./szz_eval_outputs"          # 当前目录
OUTPUT_FILE = "paste.txt"

# 正则：匹配
# Grading file: xxx_processed.jsonl
# Mean@32:  0.078125
grading_re = re.compile(r"Grading file:\s*(\S+)")
mean_re = re.compile(r"Mean@\d+:\s*([0-9.eE+-]+)")

results = {}   # folder -> {dataset: mean}

for folder in sorted(os.listdir(ROOT_DIR)):
    folder_path = os.path.join(ROOT_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    grade_path = os.path.join(folder_path, "grade.txt")
    if not os.path.isfile(grade_path):
        continue

    with open(grade_path, "r", encoding="utf-8") as f:
        text = f.read()

    datasets = grading_re.findall(text)
    means = mean_re.findall(text)

    if len(datasets) != len(means):
        print(f"[WARN] mismatch in {folder}: {len(datasets)} datasets vs {len(means)} means")

    results[folder] = {}
    for d, m in zip(datasets, means):
        # e.g. aime24_processed.jsonl -> aime24
        name = d.replace("_processed.jsonl", "")
        results[folder][name] = m

# 收集所有 dataset 名字（列）
all_datasets = sorted(
    {ds for folder_data in results.values() for ds in folder_data}
)

# 写 TSV
with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    # header
    out.write("folder\t" + "\t".join(all_datasets) + "\n")

    for folder in sorted(results):
        row = [folder]
        for ds in all_datasets:
            row.append(results[folder].get(ds, ""))
        out.write("\t".join(row) + "\n")

print(f"[DONE] Wrote {OUTPUT_FILE}")




