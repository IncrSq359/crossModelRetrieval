import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent/"src"))

from data_loader import get_image_data
from preprocess import get_image_transform
from image_feature import ImageDataset,extract_image_features
from image_feature import save_features as save_image_features
from image_feature import load_features as load_image_features
from text_feature import load_captions,extract_text_features    
from text_feature import load_features as load_text_features
from text_feature import save_features as save_text_features
from similarity import cosine_similarity_matrix,analyze_similarity


def main():
    #image
    records=get_image_data("data/images")
    transform=get_image_transform()
    dataset=ImageDataset(records,transform)

    img_features,img_labels,img_paths=extract_image_features(dataset)
    save_image_features(img_features,img_labels,img_paths,"output/features/image_features.pth")
    img_features2,img_labels2,_=load_image_features("output/features/image_features.pth")

    #text
    texts,txt_labels,txt_paths=load_captions("data/texts/captions.csv")
    txt_features=extract_text_features(texts)
    save_text_features(txt_features,txt_labels,txt_paths,"output/features/text_features.pth")
    txt_features2,txt_labels2,_=load_text_features("output/features/text_features.pth")

    #analyze
    img_sim=cosine_similarity_matrix(img_features2)
    img_same,img_diff,_,_=analyze_similarity(img_sim,img_labels2)
    print ( "【图像】同类平均:" , round (img_same, 4 ), " 不同类平均:" , round (img_diff, 4 ))

    txt_sim = cosine_similarity_matrix(txt_features2)
    txt_same, txt_diff, _, _ = analyze_similarity(txt_sim, txt_labels2) 
    print ( "【文本】同类平均:" , round (txt_same, 4 ), " 不同类平均:" , round (txt_diff, 4 ))
    
if __name__ == "__main__" :
    main()
 