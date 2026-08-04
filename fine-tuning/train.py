import logging
import random
from dataclasses import dataclass, field
from typing import Optional

import fire
import torch
import yaml
from datasets import Dataset
from peft import LoraConfig as PeftLoraConfig
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.cache_utils import DynamicCache
from trl import SFTTrainer
from trl.trainer.sft_config import SFTConfig

from data.synthetic_dataset import generate_dataset

logger = logging.getLogger(__name__)

if not hasattr(DynamicCache, "seen_tokens"):
    DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    logger.debug("Patched DynamicCache.seen_tokens -> get_seq_length()")

if not hasattr(DynamicCache, "get_max_length"):
    DynamicCache.get_max_length = lambda self, *a, **kw: (
        self.get_max_cache_shape() if hasattr(self, "get_max_cache_shape") else 0
    )
    logger.debug("Patched DynamicCache.get_max_length -> get_max_cache_shape()")

if not hasattr(DynamicCache, "get_usable_length"):
    DynamicCache.get_usable_length = lambda self, new_seq_length, layer_idx=None: (
        self.get_seq_length() if layer_idx is None or not hasattr(self, "get_seq_length") else self.get_seq_length(layer_idx)
    )
    logger.debug("Patched DynamicCache.get_usable_length -> get_seq_length()")


def setup_logging(log_file: Optional[str] = None, verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


@dataclass
class ModelConfig:
    base_model: str = "microsoft/Phi-3-mini-4k-instruct"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class LoraConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    output_dir: str = "./output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    learning_rate: float = 2.0e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    bf16: bool = True
    fp16: bool = False
    max_length: int = 2048
    packing: bool = False
    dataset_text_field: str = "text"
    remove_unused_columns: bool = False
    report_to: str = "none"
    log_file: Optional[str] = None
    verbose: bool = False


@dataclass
class CheckpointConfig:
    resume_from: Optional[str] = None
    save_only_model: bool = False


@dataclass
class DataConfig:
    num_synthetic_samples: int = 500
    train_split: float = 0.9
    seed: int = 42


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    data: DataConfig = field(default_factory=DataConfig)


def load_config(config_path: str) -> Config:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(
        model=ModelConfig(**raw.get("model", {})),
        lora=LoraConfig(**raw.get("lora", {})),
        training=TrainingConfig(**raw.get("training", {})),
        checkpoint=CheckpointConfig(**raw.get("checkpoint", {})),
        data=DataConfig(**raw.get("data", {})),
    )


def prepare_dataset(config: DataConfig):
    random.seed(config.seed)
    logger.info("Generating synthetic dataset...")
    samples = generate_dataset(num_samples=config.num_synthetic_samples, seed=config.seed)

    logger.info("Shuffling and splitting dataset...")
    random.shuffle(samples)
    split_idx = int(len(samples) * config.train_split)
    train_data = samples[:split_idx]
    eval_data = samples[split_idx:]

    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data) if eval_data else None
    logger.info("Train samples: %d, Eval samples: %d", len(train_dataset), len(eval_dataset) if eval_dataset else 0)
    return train_dataset, eval_dataset


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    import os
    import re

    if not os.path.isdir(output_dir):
        return None
    checkpoints = [
        d for d in os.listdir(output_dir)
        if re.match(r"checkpoint-\d+$", d) and os.path.isdir(os.path.join(output_dir, d))
    ]
    if not checkpoints:
        return None
    latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
    return os.path.join(output_dir, latest)


def train(
    config_path: str = "config.yaml",
    resume_from_checkpoint: Optional[str] = None,
    verbose: bool = False,
):
    config = load_config(config_path)
    setup_logging(log_file=config.training.log_file, verbose=verbose or config.training.verbose)

    logger.info("Loading base model: %s", config.model.base_model)

    model_config = AutoConfig.from_pretrained(config.model.base_model, trust_remote_code=True)
    if hasattr(model_config, "rope_scaling") and model_config.rope_scaling is not None:
        if isinstance(model_config.rope_scaling, dict) and "type" not in model_config.rope_scaling:
            logger.warning("Invalid rope_scaling config detected — disabling")
            model_config.rope_scaling = None

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=config.model.load_in_4bit,
        bnb_4bit_quant_type=config.model.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=(
            torch.float16 if config.model.bnb_4bit_compute_dtype == "float16" else torch.bfloat16
        ),
        bnb_4bit_use_double_quant=config.model.bnb_4bit_use_double_quant,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model.base_model,
        config=model_config,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    model.config.pretraining_tp = 1
    logger.info("Model loaded successfully")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.base_model,
        trust_remote_code=True,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token else tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{{ messages | map(attribute='content') | join('\n') }}"

    train_dataset, eval_dataset = prepare_dataset(config.data)

    peft_config = PeftLoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        target_modules=config.lora.target_modules,
        bias=config.lora.bias,
        task_type=config.lora.task_type,
    )

    effective_resume = resume_from_checkpoint or config.checkpoint.resume_from
    if effective_resume is None:
        auto_checkpoint = find_latest_checkpoint(config.training.output_dir)
        if auto_checkpoint:
            logger.info("Found latest checkpoint: %s", auto_checkpoint)
            effective_resume = auto_checkpoint

    training_args = SFTConfig(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        gradient_checkpointing=config.training.gradient_checkpointing,
        learning_rate=config.training.learning_rate,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler_type,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        eval_steps=config.training.eval_steps,
        save_total_limit=config.training.save_total_limit,
        save_only_model=config.checkpoint.save_only_model,
        bf16=config.training.bf16,
        fp16=config.training.fp16,
        packing=config.training.packing,
        dataset_text_field=config.training.dataset_text_field,
        remove_unused_columns=config.training.remove_unused_columns,
        report_to=config.training.report_to,
        eval_strategy="steps" if eval_dataset else "no",
        eval_on_start=False,
        load_best_model_at_end=bool(eval_dataset),
        metric_for_best_model="eval_loss" if eval_dataset else None,
        max_length=config.training.max_length,
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    logger.info(
        "Starting training — %d epochs, %d train steps total",
        config.training.num_train_epochs,
        len(train_dataset) // (config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps) * config.training.num_train_epochs,
    )
    trainer.train(resume_from_checkpoint=effective_resume)

    logger.info("Saving final model to %s", config.training.output_dir)
    trainer.save_model(config.training.output_dir)
    tokenizer.save_pretrained(config.training.output_dir)

    if eval_dataset:
        eval_results = trainer.evaluate()
        logger.info("Final evaluation results: %s", eval_results)

    logger.info("Training complete!")


if __name__ == "__main__":
    fire.Fire(train)
