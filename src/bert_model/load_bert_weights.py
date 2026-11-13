import os

os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import BertModel

def load_hf_weights(my_bert):
    """把官方 bert-base-uncased 的权重搬进手写的 BERT

    Args:
        my_bert: 手写的 BERT 实例
    """
    hf_sd=BertModel.from_pretrained("bert-base-uncased").state_dict()
    new_sd={}
    
    new_sd["embedding.token.weight"]= hf_sd["embeddings.word_embeddings.weight"]
    new_sd["embedding.pos.pe.weight"]=hf_sd["embeddings.position_embeddings.weight"]
    new_sd["embedding.token_type.weight"] = hf_sd["embeddings.token_type_embeddings.weight"]
    new_sd["embedding.ln.weight"] = hf_sd["embeddings.LayerNorm.weight"]
    new_sd["embedding.ln.bias"] = hf_sd["embeddings.LayerNorm.bias"]
    
    pairs = [
        ("att.wq", "attention.self.query"),
        ("att.wk", "attention.self.key"),
        ("att.wv", "attention.self.value"),
        ("att.wo", "attention.output.dense"),
        ("ln1", "attention.output.LayerNorm"),
        ("ffn.w1", "intermediate.dense"),
        ("ffn.w2", "output.dense"),
        ("ln2", "output.LayerNorm"),
    ]
    
    for i in range(12):
        for mine,hf in pairs:
            for suffix in ["weight","bias"]:
                my_key=f"layers.{i}.{mine}.{suffix}"
                hf_key=f"encoder.layer.{i}.{hf}.{suffix}"
                new_sd[my_key]=hf_sd[hf_key]
    
    # 把整理好的字典装进手写 BERT
    # strict=True：只要有一个名字对不上就报错，并列出哪些对不上，这是内置的检查
    my_bert.load_state_dict(new_sd, strict=True)
    # 切到推理模式：关闭 dropout，否则每次算出来的结果都会有随机性
    my_bert.eval()
    return my_bert