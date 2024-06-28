import os
import cv2
import xml.etree.ElementTree as ET
from datasets import Dataset
from tqdm import tqdm


def convert_to_detection_string(bboxs, image_width, image_height):
    def format_location(value, max_value):
        return f"<loc{int(round(value * 1024 / max_value)):04}>"

    detection_strings = []
    for bbox in bboxs:
        x1, y1, x2, y2 = bbox
        name = "text"
        locs = [
            format_location(y1, image_height),
            format_location(x1, image_width),
            format_location(y2, image_height),
            format_location(x2, image_width),
        ]
        detection_string = "".join(locs) + f" {name}"
        detection_strings.append(detection_string)

    return " ; ".join(detection_strings)


def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    bboxs = []
    for obj in root.iter('object'):
        if obj.find('polygon') is not None:
            xmlbox = obj.find('polygon')
            bbox = [int(xmlbox.find('x1').text), int(xmlbox.find('y1').text),
                    int(xmlbox.find('x3').text), int(xmlbox.find('y3').text)]
            # else:
            #     xmlbox = obj.find('bndbox')
            #     bbox = [int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text),
            #             int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text)]
            bboxs.append(bbox)

    return bboxs


def format_objects(example):
    image_path = example['image_path']
    xml_path = example['xml_path']

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to open image {image_path}")
        return {"label_for_paligemma": ""}  # 返回一个包含默认值的字典

    height, width = image.shape[:2]

    bboxs = parse_xml(xml_path)
    formatted_objects = convert_to_detection_string(bboxs, width, height)
    return {
        "label_for_paligemma": formatted_objects}  # <loc0110><loc0124><loc0224><loc0389> plate ; <loc0244><loc0130><loc0281><loc0430> plate ; <loc0364><loc0820><loc0403><loc0951> plate ; <loc0470><loc0140><loc0521><loc0228> plate ; <loc0558><loc0953><loc0582><loc0988> plate ; <loc0570><loc0149><loc0619><loc0228> plate ; <loc0792><loc0062><loc0827><loc0315> plate ; <loc0829><loc0062><loc0865><loc0343> plate ; <loc0556><loc0906><loc0592><loc0940> plate ; <loc0690><loc0837><loc0715><loc0853> plate ; <loc0770><loc0792><loc0800><loc0808> plate ; <loc0767><loc0833><loc0798><loc0853> plate ; <loc0765><loc0879><loc0796><loc0900> plate ; <loc0124><loc0058><loc0212><loc0097> plate


if __name__ == "__main__":
    base_folder = "/home/image_team/image_team_docker_home/lgd/e_commerce_lmm/OCRPaliGemma/data/EcommerceGT2/"

    image_folder = os.path.join(base_folder, 'images')
    xml_folder = os.path.join(base_folder, 'labels')

    image_files = os.listdir(image_folder)
    xml_files = os.listdir(xml_folder)

    image_paths, xml_paths = [], []
    for image_file in tqdm(image_files):
        xml_file = image_file.replace('.jpg', '.xml')  # 假设XML文件和图像文件有相同的名字，只是扩展名不同
        if xml_file in xml_files:
            image_path = os.path.join(image_folder, image_file)
            xml_path = os.path.join(xml_folder, xml_file)
            image_paths.append(image_path)
            xml_paths.append(xml_path)

    data = {'image_path': image_paths, 'xml_path': xml_paths}

    dataset = Dataset.from_dict(data)
    dataset = dataset.map(format_objects, num_proc=24)
    dataset = dataset.filter(lambda x: x["label_for_paligemma"] != "")
    dataset.to_parquet(
        "/home/image_team/image_team_docker_home/lgd/e_commerce_lmm/OCRPaliGemma/data/EcommerceGT2/text_ecommerce_detection.parquet")

    with tqdm(total=3, desc="Saving to parquet files") as pbar:
        # 划分数据集
        splits = dataset.train_test_split(test_size=0.08)  # 91%的数据用于训练，9%的数据用于测试
        train_dataset, test_dataset = splits['train'], splits['test']

        # 再将训练集划分为训练集和验证集
        splits = train_dataset.train_test_split(test_size=0.01)  # 99%的数据用于训练，1%的数据用于验证
        train_dataset, val_dataset = splits['train'], splits['test']

        # 保存为Parquet文件
        train_dataset.to_parquet(
            "/home/image_team/image_team_docker_home/lgd/e_commerce_lmm/OCRPaliGemma/data/EcommerceGT2/text_ecommerce_detection_train.parquet")
        pbar.update()
        val_dataset.to_parquet(
            "/home/image_team/image_team_docker_home/lgd/e_commerce_lmm/OCRPaliGemma/data/EcommerceGT2/text_ecommerce_detection_val.parquet")
        pbar.update()
        test_dataset.to_parquet(
            "/home/image_team/image_team_docker_home/lgd/e_commerce_lmm/OCRPaliGemma/data/EcommerceGT2/text_ecommerce_detection_test.parquet")
        pbar.update()
