import torch
from typing import List
from dataclasses import dataclass, field


def default_target_modules():
    return ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class ModelType:
    Type: str = "ocr"


@dataclass
class OCRSFTConfig:
    lora: bool = False
    r: int = 8
    target_modules: list = field(default_factory=default_target_modules)
    task_type: str = "CAUSAL_LM"

    output_dir: str = "/home/lgd/e_commerce_lmm/OCRPaliGemma/results/ocr_commerce_torch/"
    evaluation_strategy: str = "steps"  # no
    per_device_train_batch_size: int = 2  # 默认是8
    per_device_eval_batch_size: int = 2  # 默认是8
    # 注意：计算梯度时相当于batch_size * gradient_accumulation_steps，说人话就是梯度累积步数>1时，等于增大n倍的batch_size
    gradient_accumulation_steps: int = 2  # 累积梯度更新步数,默认1

    learning_rate: float = 5e-5  # 5e-5
    weight_decay: float = 1e-6  # 0
    adam_beta2: float = 0.999

    num_train_epochs: int = 1  # 3
    lr_scheduler_type: str = "cosine"  # linear
    warmup_steps: int = 2  # 0

    log_level: str = "info"  # passive
    logging_first_step: bool = True  # False
    logging_steps: int = 1  # 500

    save_strategy: str = "steps"  # steps
    save_steps: int = 20  # 500
    save_total_limit: int = 4  # 最多只保存4次

    bf16: bool = torch.cuda.is_bf16_supported()
    fp16: bool = not torch.cuda.is_bf16_supported()
    optim: str = "adamw_hf"  # adamw_torch
    eval_steps: int = 100  #
    report_to: str = "tensorboard"  # all
    seed: int = 42
    remove_unused_columns: bool = False
    dataloader_pin_memory: bool = False

    model_dtype = torch.bfloat16
    model_revision = "bfloat16"

    # ---------------------------------------
    model_file: str = "/home/lgd/e_commerce_lmm/OCRPaliGemma/weights/paligemma-3b-pt-224/"
    train_file: str = "/home/lgd/e_commerce_lmm/OCRPaliGemma/data/EcommerceGT2/text_ecommerce_detection_train.parquet"
    validation_file: str = "/home/lgd/e_commerce_lmm/OCRPaliGemma/data/EcommerceGT2/text_ecommerce_detection_val.parquet"
    test_file: str = "/home/lgd/e_commerce_lmm/OCRPaliGemma/data/EcommerceGT2/text_ecommerce_detection_test.parquet"


@dataclass
class PlateSFTConfig:
    output_dir: str = "/home/lgd/e_commerce_lmm/OCRPaliGemma/results/ocr_commerce_torch/"
    batch_size: int = 8
    learning_rate: float = 5e-5
    epoch: int = 1
    model_dtype = torch.bfloat16
    model_revision = "bfloat16"

    model_file: str = "/home/lgd/e_commerce_lmm/OCRPaliGemma/weights/paligemma-3b-pt-224/"
    train_file: str = "ariG23498/license-detection-paligemma"


@dataclass
class OCRSFTTorchConfig:
    output_dir: str = "/home/lgd/e_commerce_lmm/OCRPaliGemma/results/ocr_commerce_torch/"
    batch_size: int = 8
    learning_rate: float = 5e-5
    epoch: int = 1
    model_dtype = torch.bfloat16
    model_revision = "bfloat16"

    model_file: str = "/home/lgd/e_commerce_lmm/OCRPaliGemma/weights/paligemma-3b-pt-224/"
    train_file: str = "/home/lgd/e_commerce_lmm/OCRPaliGemma/data/EcommerceGT2/text_ecommerce_detection_train.parquet"
    validation_file: str = "/home/lgd/e_commerce_lmm/OCRPaliGemma/data/EcommerceGT2/text_ecommerce_detection_val.parquet"
    test_file: str = "/home/lgd/e_commerce_lmm/OCRPaliGemma/data/EcommerceGT2/text_ecommerce_detection_test.parquet"
