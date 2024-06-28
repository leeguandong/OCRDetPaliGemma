from datasets import load_dataset


def coco_to_xyxy(coco_bbox):
    x, y, width, height = coco_bbox
    x1, y1 = x, y
    x2, y2 = x + width, y + height
    return [x1, y1, x2, y2]


def convert_to_detection_string(bboxs, image_width, image_height):
    '''
    PaliGemma 可以使用detect [entity]提示来检测图像中的实体。它会以特殊的<loc[value]>令牌形式输出边界框坐标的位置，其中value是一个表示归一化坐标的数字。每次检测都由四个位置坐标代表——y_min, x_min, y_max, x_max_，后跟检测到的框中的标签。要将这些值转换为坐标，你需要首先将数字除以1024，然后将y乘以图像高度，x乘以宽度。这将给你提供相对于原始图像大小的边界框坐标。
    '''

    def format_location(value, max_value):
        return f"<loc{int(round(value * 1024 / max_value)):04}>"

    detection_strings = []
    for bbox in bboxs:
        x1, y1, x2, y2 = coco_to_xyxy(bbox) # [[471.0, 603.0, 42.2400016784668, 20.472957611083984], [1007.0, 519.0, 14.079999923706055, 12.795683860778809]]
        name = "plate"
        locs = [
            format_location(y1, image_height),
            format_location(x1, image_width),
            format_location(y2, image_height),
            format_location(x2, image_width),
        ]
        detection_string = "".join(locs) + f" {name}" # '<loc0905><loc0471><loc0936><loc0513> plate'
        detection_strings.append(detection_string)

    return " ; ".join(detection_strings)


def format_objects(example):
    height = example["height"]
    width = example["width"]
    bboxs = example["objects"]["bbox"]
    formatted_objects = convert_to_detection_string(bboxs, width, height)
    return {"label_for_paligemma": formatted_objects}  # '<loc0789><loc0184><loc0874><loc0316> plate'


if __name__ == "__main__":
    # load the dataset
    dataset_id = "keremberke/license-plate-object-detection"
    print(f"[INFO] loading {dataset_id} from hub...")
    dataset = load_dataset(dataset_id, "full",
                           cache_dir="E:/comprehensive_library/e_commerce_lmm/OCRPaliGemma/data/license-plate-object-detection")

    # modify the coco bbox format
    dataset["train"] = dataset["train"].map(format_objects)
    dataset["validation"] = dataset["validation"].map(format_objects)
    dataset["test"] = dataset["test"].map(format_objects)

    # push to hub
    dataset.push_to_hub("license-detection-paligemma")

    # https://huggingface.co/datasets/ariG23498/license-detection-paligemma?row=0
