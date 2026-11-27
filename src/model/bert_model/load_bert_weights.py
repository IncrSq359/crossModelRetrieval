from transformers import BertModel


def _load_hf_state(name="bert-base-uncased"):
    """优先用本地缓存（离线可用）；没有缓存时再联网下载一次。

    云端机器第一次跑需要联网，之后就走本地缓存。
    国内平台连不上 huggingface.co 时，先设镜像再重试：
        export HF_ENDPOINT=https://hf-mirror.com
    """
    try:
        return BertModel.from_pretrained(name, local_files_only=True).state_dict()
    except OSError:
        pass
    try:
        return BertModel.from_pretrained(name).state_dict()
    except Exception as exc:  # noqa: BLE001 - 网络/鉴权/磁盘各种原因都归到这里
        raise SystemExit(
            f"下载 {name} 失败：{type(exc).__name__}: {exc}\n"
            "国内云平台通常是连不上 huggingface.co，按下面任一方式处理：\n"
            "  1) 设镜像后重跑： export HF_ENDPOINT=https://hf-mirror.com\n"
            "  2) 在有网的机器上把 ~/.cache/huggingface/hub/models--bert-base-uncased 拷过来，\n"
            "     然后 export HF_HUB_OFFLINE=1 再跑"
        ) from exc


def load_hf_weights(my_bert, name="bert-base-uncased"):
    """把官方 bert-base-uncased 的权重搬进手写的 BERT

    Args:
        my_bert: 手写的 BERT 实例
        name: HuggingFace 上的模型名
    """
    hf_sd = _load_hf_state(name)
    new_sd = {}

    new_sd["embedding.token.weight"] = hf_sd["embeddings.word_embeddings.weight"]
    new_sd["embedding.pos.pe.weight"] = hf_sd["embeddings.position_embeddings.weight"]
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
        for mine, hf in pairs:
            for suffix in ["weight", "bias"]:
                my_key = f"layers.{i}.{mine}.{suffix}"
                hf_key = f"encoder.layer.{i}.{hf}.{suffix}"
                new_sd[my_key] = hf_sd[hf_key]

    # 把整理好的字典装进手写 BERT
    # strict=True：只要有一个名字对不上就报错，并列出哪些对不上，这是内置的检查
    my_bert.load_state_dict(new_sd, strict=True)
    # 切到推理模式：关闭 dropout，否则每次算出来的结果都会有随机性
    my_bert.eval()
    return my_bert