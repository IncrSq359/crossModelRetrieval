from pathlib import Path
PROJECT_ROOT =Path(__file__).resolve().parents[1]
csv_path=PROJECT_ROOT/"data"/"annotations"/"captions_flickr30k.csv"

import csv

lengths=[]
with open(csv_path,encoding="utf-8",newline="") as f:
    reader =csv.DictReader(f)
    for row in reader:
        if row["split"]=="train":
            text =row["caption"]
            lengths.append(len(text.split()))

lengths.sort()
n=len(lengths)
mean_len=sum(lengths)/n
median_len=lengths[n//2]
p95_index=int(0.95*(n-1))
p95_len=lengths[p95_index]

print("train caption 条数：", n)
print("单词数 min / max：", lengths[0], lengths[-1])
print("平均单词数：", round(mean_len, 2))
print("中位数 P50：", median_len)
print("P95：", p95_len)