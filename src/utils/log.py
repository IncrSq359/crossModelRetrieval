import os
import datetime
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# data_store 与「挑战实验」同级，即项目根目录的上一级
DATA_STORE = os.path.join(os.path.dirname(PROJECT_ROOT), "data_store")


def write_log(name, basePath, content, mode="a"):
    os.makedirs(basePath, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(DATA_STORE, f"train_{stamp}_{name}.txt")
    with open(path, mode, encoding="utf-8") as f:
        f.write(content)
    return path


def write_train_log(
    name, hparams, n_params, loss_history, epoch_times, total_time, recalls
):
    """把一次训练的结果写入 data_store 下的独立 txt 文件。

    Args:
        name: 本次实验的名字，会放进文件名。
        hparams: 超参数字典。
        n_params: 可训练参数总数。
        loss_history: 每个 epoch 的平均 loss 列表。
        epoch_times: 每个 epoch 的耗时（秒）列表。
        total_time: 总用时（秒）。
        recalls: Recall 字典，键如 "val_T2I@10"、"train_T2I@10"。
    Returns:
        写入的文件路径。
    """
    content = "===== 超参数 =====\n"
    for key, value in hparams.items():
        content += f"{key}: {value}\n"
    content += f"可训练参数: {n_params}\n总用时: {total_time:.1f}s\n\n"
    content += f"总用时: {total_time:.1f}s\n\n"
    content += "===== 每个 epoch =====\n"
    for i, (loss, t) in enumerate(zip(loss_history, epoch_times), start=1):
        content += f"epoch {i}  loss {loss:.4f}  用时 {t:.1f}s\n"
    content += "\n===== Recall =====\n"
    for key, value in recalls.items():
        content += f"{key}: {value:.4f}\n"
    return write_log(name, DATA_STORE, content)
