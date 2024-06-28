import os
import random
import torch
import numpy as np
from PIL import Image
from datetime import datetime
from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    AutoProcessor
)
from matplotlib import pyplot as plt, patches
from datasets import load_dataset
from torch.utils.data import DataLoader
from config import OCRSFTConfig, OCRSFTTorchConfig, ModelType, PlateSFTConfig
from functools import partial
from utils.parse import extract_objs
from pathlib import Path

# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
#
# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_flash_sdp(False)

type = ModelType().Type


def freeze_layers(model, freeze):
    if freeze == "v1":
        not_to_freeze = "attn"
        for name, param in model.named_parameters():
            if not_to_freeze in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        # 冻结图像编码器和投影器，仅微调解码器，如果你的图像属于特定领域，这些领域可能不在模型预训练的数据集中，可能想跳过冻结图像编码器
        for param in model.vision_tower.parameters():
            param.requires_grad = False
        for param in model.multi_modal_projector.parameters():
            param.requires_grad = True

    # for name, param in model.named_parameters():
    #     print(f"{name:<70}: requires_grad={param.requires_grad}, dtype={param.dtype}")
    return model


def get_sft_dataset(config):
    if type == "plate":
        train_dataset = load_dataset(config.train_file, split="train")
        test_dataset = load_dataset(config.train_file, split="test")
    elif type == "ocr":
        train_dataset = load_dataset(path='parquet', data_files=config.train_file, split="train")
        test_dataset = load_dataset(path='parquet', data_files=config.test_file, split="train")
    else:
        raise TypeError("just support plate/ocr!!!!")
    print(f"[INFO] {len(train_dataset)=}")
    print(f"[INFO] {len(test_dataset)=}")
    return train_dataset, test_dataset


def collate_fn(examples, processor, device, train):
    if type == "ocr":
        images = [Image.open(example["image_path"]).convert("RGB") for example in examples]
        prompt = ["Detect text." for _ in examples]
        if train:
            suffix = [example["label_for_paligemma"] for example in examples]
        else:
            suffix = None
    elif type == "plate":
        images = [example["image"].convert("RGB") for example in examples]
        prompt = ["Detect license plate." for _ in examples]
        if train:
            suffix = [example["label_for_paligemma"] for example in examples]
        else:
            suffix = None
    else:
        raise TypeError("just support ocr/plate!!!")

    # Help from: https://github.com/huggingface/transformers/issues/30987
    inputs = processor(
        images=images,
        text=prompt,
        suffix=suffix,
        return_tensors="pt",
        padding="longest",
    )

    inputs = inputs.to(torch.bfloat16).to(device)
    return inputs


def get_sft_dataloader(config, processor, train_dataset, test_dataset, device):
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=partial(
            collate_fn,
            processor=processor,
            device=device,
            train=True,
        ),
        batch_size=config.batch_size,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=partial(
            collate_fn,
            processor=processor,
            device=device,
            train=False,
        ),
        batch_size=config.batch_size,
        shuffle=False,
    )
    return train_dataloader, test_dataloader


def get_sft_model(config, device, freeze="v1"):
    print(f"[INFO] loading {config.model_file} model...")
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        config.model_file, torch_dtype=config.model_dtype,
        device_map=device,
        revision=config.model_revision)

    print(f"[INFO] freezing the model weights...")
    model = freeze_layers(model, freeze)
    return model


def save_bbox(config, image, objects):
    # 获取当前时间的时分秒并格式化为字符串
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    # 生成一个随机数（例如，两位的随机整数）
    random_num = str(random.randint(10, 99))
    # 构建文件名
    output_dir = os.path.join(config.output_dir, "finetuning")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    output_path = f'{output_dir}/{timestamp}_{random_num}.png'

    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for obj in objects:
        bbox = obj["xyxy"]
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
        plt.text(
            bbox[0], bbox[1] - 10, "plate", color="red", fontsize=12, weight="bold"
        )
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # 保存图片
    plt.close(fig)  # 关闭图像窗口


def infer_on_model(config, model, processor, test_batch, before_pt=True):
    # hardcoding the index to get same before and after results
    index = 0

    # help from : https://discuss.huggingface.co/t/vitimageprocessor-output-visualization/76335/6
    mean = processor.image_processor.image_mean
    std = processor.image_processor.image_std

    pixel_value = test_batch["pixel_values"][index].cpu().to(torch.float32)

    unnormalized_image = (pixel_value.numpy() * np.array(std)[:, None, None]) + np.array(mean)[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)

    with torch.inference_mode():
        generated_outputs = model.generate(
            **test_batch, max_new_tokens=100, do_sample=False
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
        save_bbox(config, unnormalized_image, objects)


def sft_train():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # step 1. load config
    if type == "ocr":
        config = OCRSFTTorchConfig()
    elif type == "plate":
        config = PlateSFTConfig()
    else:
        raise TypeError("just support ocr/plate!!!")

    # step 2. load dataset
    print(f"[INFO] loading {type} data from hub...")
    train_dataset, test_dataset = get_sft_dataset(config)

    # step 3. load PaliGemmaProcessor
    processor = AutoProcessor.from_pretrained(config.model_file)

    # step 4. build dataloader
    train_dataloader, test_dataloader = get_sft_dataloader(config, processor, train_dataset, test_dataset, device)

    # step 5. load model
    model = get_sft_model(config, device)

    # step 6. run model generation before fine tuning
    test_batch = next(iter(test_dataloader))
    infer_on_model(config, model, processor, test_batch)

    # step 7. fine tuning model
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    for epoch in range(config.epoch):
        for idx, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            if idx % 50 == 0:
                print(f"Epoch: {epoch} Iter: {idx} Loss: {loss.item():.4f}")

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # step 8.run model generation after fine tuning
    infer_on_model(config, model, processor, test_batch, before_pt=False)


if __name__ == "__main__":
    sft_train()
