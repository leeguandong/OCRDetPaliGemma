import re
import os
import time
import numpy as np
import pandas as pd
from PIL import Image
from transformers import (
    PaliGemmaForConditionalGeneration,
    TrainingArguments,
    Trainer,
    PaliGemmaProcessor,
    AutoProcessor
)
from datasets import load_dataset

import torch
from config import OCRSFTConfig
from peft import get_peft_model, LoraConfig
from functools import partial

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)


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

    for name, param in model.named_parameters():
        print(f"{name:<70}: requires_grad={param.requires_grad}, dtype={param.dtype}")
    return model


def get_sft_dataset(config):
    train_dataset = load_dataset(path='parquet', data_files=config.train_file, split="train")
    test_dataset = load_dataset(path='parquet', data_files=config.test_file)
    return train_dataset, test_dataset


def get_sft_model(config, freeze="v1"):
    print(f"[INFO] loading {config.model_file} model...")
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        config.model_file, torch_dtype=config.model_dtype,
        revision=config.model_revision)

    print(f"[INFO] freezing the model weights...")
    model = freeze_layers(model, freeze)
    return model


def collate_fn(examples, processor):
    images = [Image.open(example["image_path"]).convert("RGB") for example in examples]

    prompt = ["Detect text." for _ in examples]
    suffix = [example['label_for_paligemma'] for example in examples]

    # Help from: https://github.com/huggingface/transformers/issues/30987
    inputs = processor(
        images=images,
        text=prompt,
        suffix=suffix,
        return_tensors="pt",
        padding="longest",
        tokenize_newline_separately=False,
    )

    inputs = inputs.to(torch.bfloat16)
    return inputs


def sft_train():
    # step 1. load config
    config = OCRSFTConfig()

    # step 2. load PaliGemmaProcessor
    processor = AutoProcessor.from_pretrained(config.model_file)
    collate_fn_trainer = partial(collate_fn, processor=processor)

    # step 3. load dataset
    train_dataset, test_dataset = get_sft_dataset(config)
    model = get_sft_model(config)

    # step 4. Define the training argument
    if config.lora:
        lora_config = LoraConfig(
            r=config.r,
            target_modules=config.target_modules,
            task_type=config.task_type,
        )

        model = get_peft_model(model, lora_config)
        print(model.print_trainable_parameters())

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        evaluation_strategy=config.evaluation_strategy,

        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,

        learning_rate=config.learning_rate,
        weight_decay=getattr(config, "weight_decay", 0),
        adam_beta2=config.adam_beta2,

        num_train_epochs=config.num_train_epochs,
        lr_scheduler_type=getattr(config, "lr_scheduler_type", "linear"),
        warmup_steps=getattr(config, "warmup_steps", 0),

        log_level=getattr(config, "log_level", "info"),
        logging_first_step=getattr(config, "logging_first_step", False),
        logging_steps=config.logging_steps,

        save_strategy=getattr(config, "save_strategy", "steps"),
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,

        optim=getattr(config, "optim", "adamw_torch"),
        bf16=config.bf16,
        fp16=config.fp16,
        eval_steps=getattr(config, "eval_steps", 2000),
        report_to=config.report_to,

        remove_unused_columns=config.remove_unused_columns,
        dataloader_pin_memory=config.dataloader_pin_memory,
        seed=config.seed,
    )

    # step 5. Define the trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collate_fn_trainer,
        args=training_args
    )

    # step 6. train
    trainer.train()

    # step 7. eval
    eval_results = trainer.evaluate()
    print(f"Perplexity: {np.exp(eval_results['eval_loss']):.2f}")

    # step 8: save log
    loss_log = pd.DataFrame(trainer.state.log_history)
    loss_log.to_csv(f"{config.output_dir}/pre_train_log_{time.strftime('%Y%m%d-%H%M')}.csv")

    # Step 9: Save the model
    trainer.save_model(config.output_dir)


if __name__ == "__main__":
    sft_train()

# accelerate==0.27.2
