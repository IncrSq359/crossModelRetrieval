from pathlib import Path
def get_image_data(image_dir):
    folder = Path(image_dir)
    image_paths=[]
    patterns=("*.jpg","*.jpeg","*.png")
    for pattern in patterns:
        image_paths.extend(folder.rglob(pattern))
    records=[{"path":p,"label":p.parent.name} for p in image_paths]
    return records
