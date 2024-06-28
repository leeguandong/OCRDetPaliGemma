import os

import PIL.Image
import torch
import numpy as np
from tqdm import tqdm
from utils.parse import extract_objs
from utils.utils import draw_bbox
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, AutoProcessor
from config import OCRSFTConfig


def infer_on_model(model, image_file, before_pt=True):
    index = 0

    prompt = "Detect text."
    image = PIL.Image.open(image_file)
    inputs = processor(prompt, image, return_tensors="pt")

    with torch.inference_mode():
        generated_outputs = model.generate(
            **inputs, max_new_tokens=100, do_sample=False
        )
        generated_outputs = processor.batch_decode(
            generated_outputs, skip_special_tokens=True
        )

    if before_pt:
        # generation of the pre trained model
        for element in generated_outputs:
            location = element.split("\n")[1]
            if location == "":
                print("No bbox found")
            else:
                print(location)
    else:
        # generation of the fine tuned model
        element = generated_outputs[index]
        detection_string = element.split("\n")[1]
        objects = extract_objs(detection_string, 224, 224, unique_labels=False)
        draw_bbox(image, objects)


if __name__ == "__main__":
    config = OCRSFTConfig()

    processor = AutoProcessor.from_pretrained(config.model_file)
    model = PaliGemmaForConditionalGeneration.from_pretrained(config.model_file, torch_dtype=torch.float16)
    model_sft = PaliGemmaForConditionalGeneration.from_pretrained(config.output_dir, torch_dtype=torch.float16)

    # before pt
    base_folder = "E:/comprehensive_library/e_commerce_lmm/OCRPaliGemma/data/test_detect_images/"
    image_folder = os.path.join(base_folder, 'images')
    image_files = os.listdir(image_folder)
    for image_file in tqdm(image_files):
        infer_on_model(model, image_file, before_pt=True)

        infer_on_model(model_sft, image_file, before_pt=False)
