import os
os.environ["HF_HUB_OFFLINE"] = "1"   # 必须在 import transformers 之前
import torch
from transformers import BertModel, BertTokenizer
from text_feature import load_bert

# 准备几句长短不一的话，这样会出现 padding，顺便检验 mask 对不对
texts = [
    "a dog running on the grass",
    "two men are playing guitar on a stage",
    "hi",
]

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
encoded = tokenizer(texts, padding="max_length", truncation=True,
                    max_length=16, return_tensors="pt")

# 官方版
hf_model = BertModel.from_pretrained("bert-base-uncased")
hf_model.eval()
# 手写版
my_model = load_bert()

with torch.no_grad():
    hf_out = hf_model(input_ids=encoded["input_ids"],
                      attention_mask=encoded["attention_mask"]).last_hidden_state
    my_out = my_model(encoded["input_ids"], encoded["attention_mask"])

# 两边都取 CLS 位置
hf_cls = hf_out[:, 0, :]
my_cls = my_out[:, 0, :]

# 最大的逐元素差距，越接近 0 越好
print("最大绝对差:", (hf_cls - my_cls).abs().max().item())
# 逐句的余弦相似度，应该都接近 1
print("余弦相似度:", torch.nn.functional.cosine_similarity(hf_cls, my_cls, dim=-1))
# 是否在容许误差内完全一致
print("allclose:", torch.allclose(hf_cls, my_cls, atol=1e-4))