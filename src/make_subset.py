from pathlib import Path
import shutil
import random

def make_subset(src_dir,dst_dir,num_classes=10,per_class=10,seed=42):
    src=Path(src_dir)
    dst=Path(dst_dir)
    random.seed(seed)
    categories=[]
    for p in src.iterdir():
        if p.is_dir():
            categories.append(p)
    
    chosen=random.sample(categories,num_classes)
    for cat in chosen:
        target_cat_dir=dst/cat.name
        target_cat_dir.mkdir(parents=True,exist_ok=True)
        images_list=list(cat.glob("*.jpg"))
        chosen_images=random.sample(images_list,per_class)
        for img in chosen_images:
            dst_img=target_cat_dir/img.name
            shutil.copy(img,dst_img)
if __name__ == "__main__":
    make_subset(
        src_dir="data/raw/101_ObjectCategories",   # 全量数据
        dst_dir="data/images",                     # 子集输出位置
        num_classes=10,
        per_class=10,
        seed=42,
    )
    print("子集生成完成")