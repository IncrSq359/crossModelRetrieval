"""数据加载模块：扫描图片目录，整理出"路径+标签"的记录列表。"""

from pathlib import Path

def get_image_data(image_dir):
    """递归扫描目录中的图片，生成图片记录列表。

    Args:
        image_dir: 图片根目录（如 "data/images"），其下按类别分子文件夹，
            每个子文件夹名即为该类别的标签。

    Returns:
        记录列表，每条是一个字典：
        {"path": 图片的Path对象, "label": 类别名(图片所在文件夹名)}。
    """
    folder = Path(image_dir)
    image_paths=[]
    patterns=("*.jpg","*.jpeg","*.png")
    for pattern in patterns:
        image_paths.extend(folder.rglob(pattern))
    records=[{"path":p,"label":p.parent.name} for p in image_paths]
    return records
