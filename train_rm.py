import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import imagesize
import numpy as np
import torch
import torch.nn as nn
import yaml
from datasets import load_dataset, load_from_disk
from PIL import Image
from qwen_vl_utils import process_vision_info, vision_process
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    HfArgumentParser,
    PreTrainedModel,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from trl import AutoModelForCausalLMWithValueHead, RewardConfig, RewardTrainer


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 28 * 28 * 4,
    max_pixels: int = 4096 * 28 * 28,
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > 20:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {20}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="/mnt/user-ssd/lilei35/oss_models/Qwen2.5-VL-7B-Instruct",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    attn_implementation: str = field(
        default="flash_attention_2",
        metadata={
            "help": "The attention implementation to use",
            "choices": ["sdpa", "flash_attention_2"],
        },
    )
    min_pixels: int = field(
        default=8 * 28 * 28, metadata={"help": "Minimum number of pixels in images"}
    )
    max_pixels: int = field(
        default=1024 * 28 * 28, metadata={"help": "Maximum number of pixels in images"}
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_path: str = field(
        default="/mnt/user-ssd/lilei35/trl/preference_data_250520_doubao_vs_mimo_10k",
        metadata={"help": "Path to the dataset"},
    )
    max_seq_length: int = field(
        default=2560,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated."
        },
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set."
        },
    )


@dataclass
class RMTrainingArguments(RewardConfig):
    """
    Arguments pertaining to training configuration.
    """

    output_dir: str = field(
        default="./vlm_reward_model_debug",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    freeze_vit: bool = field(
        default=False, metadata={"help": "Whether to freeze the ViT."}
    )
    num_train_epochs: int = field(
        default=1, metadata={"help": "Total number of training epochs to perform."}
    )
    per_device_train_batch_size: int = field(
        default=1, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=4, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={
            "help": "Number of updates steps to accumulate before performing a backward/update pass."
        },
    )
    learning_rate: float = field(
        default=1e-5,
        metadata={"help": "The initial learning rate for AdamW optimizer."},
    )
    eval_steps: int = field(
        default=100, metadata={"help": "Number of steps between evaluations."}
    )
    save_steps: int = field(
        default=500, metadata={"help": "Number of steps between model saves."}
    )
    warmup_steps: int = field(
        default=100,
        metadata={
            "help": "Number of steps used for a linear warmup from 0 to learning_rate."
        },
    )
    logging_steps: int = field(
        default=10, metadata={"help": "Number of steps between logging."}
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={
            "help": "Whether to use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    disable_dropout: bool = field(
        default=True, metadata={"help": "Whether to disable dropout during training"}
    )
    deepspeed: Optional[str] = field(
        default="zero2.json", metadata={"help": "Path to deepspeed config file."}
    )
    # Irrelevant query pair parameters
    use_irrelevant_queries: bool = field(
        default=False, metadata={"help": "Whether to use irrelevant query pairs for robustness training"}
    )
    num_irrelevant_pairs: int = field(
        default=1, metadata={"help": "Number of irrelevant query pairs per original pair"}
    )
    irrelevant_loss_weight: float = field(
        default=1, metadata={"help": "Weight for the irrelevant query pair loss"}
    )

    def __post_init__(self):
        self.remove_unused_columns = False
        self.eval_strategy = "steps"
        self.save_strategy = "steps"
        self.report_to = "none"
        super().__post_init__()


class VLMRewardTrainer(Trainer):

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        args: Optional[RewardConfig] = None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(
            None,
            None,
        ),
        preprocess_logits_for_metrics=None,
        peft_config=None,
    ):
        if args.disable_dropout:
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )


    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Compute loss for VLM reward model training.
        
        The batch structure is:
        - Original chosen responses (first N samples)
        - Original rejected responses (next N samples)
        - Irrelevant chosen responses (same responses, different multimodal queries)
        - Irrelevant rejected responses (same responses, different multimodal queries)
        
        Loss components:
        1. Original preference loss: chosen > rejected
        2. Irrelevant pair loss: chosen ≈ rejected (should be tied for irrelevant queries)
        """
        # Get the reward scores from the model
        values = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
            return_dict=True,
            use_cache=False,
        )[
            -1
        ]  # Get the value head outputs

        # Get the reward scores at the end of each sequence
        scores = values.gather(
            dim=-1, index=(inputs["attention_mask"].sum(dim=-1, keepdim=True) - 1)
        )

        # Get batch structure information
        original_batch_size = inputs.get("original_batch_size", scores.shape[0] // 2)
        irrelevant_batch_size = inputs.get("irrelevant_batch_size", 0)
        
        # Split scores based on batch structure
        # Original pairs: [chosen, rejected]
        original_chosen_scores = scores[:original_batch_size]
        original_rejected_scores = scores[original_batch_size:original_batch_size * 2]
        
        # Irrelevant pairs: [irrelevant_chosen, irrelevant_rejected]
        if irrelevant_batch_size > 0:
            irrelevant_chosen_scores = scores[original_batch_size * 2:original_batch_size * 2 + irrelevant_batch_size]
            irrelevant_rejected_scores = scores[original_batch_size * 2 + irrelevant_batch_size:]
            
            # Compute original preference loss
            original_loss = - torch.nn.functional.logsigmoid(original_chosen_scores - original_rejected_scores).mean()
            
            # Compute irrelevant pair loss (should be tied - difference should be close to 0)
            irrelevant_loss = - 0.5 * self.args.irrelevant_loss_weight * (torch.nn.functional.logsigmoid(irrelevant_chosen_scores - irrelevant_rejected_scores)
                                       + torch.nn.functional.logsigmoid(irrelevant_rejected_scores - irrelevant_chosen_scores)).mean() 

            # Combine losses using configurable weight
            loss = original_loss + irrelevant_loss
            
            # # Log losses for monitoring
            # if hasattr(self, 'log') and self.log is not None:
            #     self.log({
            #         "original_loss": original_loss.detach().item(),
            #         "irrelevant_loss": irrelevant_loss.detach().item(),
            #         "total_loss": loss.detach().item(),
            #         "irrelevant_margin": (irrelevant_chosen_scores - irrelevant_rejected_scores).mean().detach().item(),
            #     })
            
            if return_outputs:
                return loss, {
                    "original_chosen_scores": original_chosen_scores,
                    "original_rejected_scores": original_rejected_scores,
                    "irrelevant_chosen_scores": irrelevant_chosen_scores,
                    "irrelevant_rejected_scores": irrelevant_rejected_scores,
                    "original_loss": original_loss,
                    "irrelevant_loss": irrelevant_loss,
                }
        else:
            # Fallback to original behavior if no irrelevant pairs
            chosen_scores = scores[:original_batch_size]
            rejected_scores = scores[original_batch_size:]
            loss = - torch.nn.functional.logsigmoid(chosen_scores - rejected_scores).mean()
            
            if return_outputs:
                return loss, {
                    "chosen_scores": chosen_scores,
                    "rejected_scores": rejected_scores,
                }
        
        return loss

    def evaluate(
        self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"
    ) -> Dict[str, float]:
        # Only run evaluation on rank 0
        if not self.is_world_process_zero():
            metrics = {}
        else:
            # Add custom preference accuracy evaluation
            metrics = self.compute_preference_accuracy(eval_dataset)
            print(metrics)

        # Make sure all processes are synced after evaluation
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        return metrics

    def compute_preference_accuracy(self, eval_dataset=None) -> Dict[str, float]:
        """Compute accuracy of the reward model's preferences."""
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        dataloader = DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
        )

        model = self.model
        model.eval()

        total_pairs = 0
        correct_preferences = 0
        all_margins = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing preference accuracy"):
                batch = {
                    k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                values = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    image_grid_thw=batch["image_grid_thw"],
                    return_dict=True,
                    use_cache=False,
                )[-1]

                scores = values.gather(
                    dim=-1,
                    index=(batch["attention_mask"].sum(dim=-1, keepdim=True) - 1),
                )

                # Get batch structure information
                original_batch_size = batch.get("original_batch_size", scores.shape[0] // 2)
                irrelevant_batch_size = batch.get("irrelevant_batch_size", 0)
                
                # Only evaluate on original pairs (not irrelevant pairs)
                original_chosen_scores = scores[:original_batch_size]
                original_rejected_scores = scores[original_batch_size:original_batch_size * 2]

                # Calculate accuracy
                correct = (
                    (original_chosen_scores > original_rejected_scores).float().detach().cpu().numpy()
                )
                correct_preferences += correct.sum().item()
                total_pairs += original_batch_size

                # Calculate margins - convert to float32 before numpy conversion
                margins = (
                    (original_chosen_scores - original_rejected_scores).squeeze().float().cpu().numpy()
                )
                all_margins.extend(margins.tolist())
                    
        accuracy = correct_preferences / total_pairs
        avg_margin = np.mean(all_margins)
        margin_std = np.std(all_margins)

        return {
            "eval_preference_accuracy": accuracy,
            "eval_average_margin": avg_margin,
            "eval_margin_std": margin_std,
        }


@dataclass
class VLMDataCollator:
    """
    Data collator for VLM reward model training with irrelevant query pair support.
    
    This collator creates batches with the following structure:
    1. Original chosen responses (first half)
    2. Original rejected responses (second half) 
    3. Irrelevant chosen responses (same responses, different multimodal queries)
    4. Irrelevant rejected responses (same responses, different multimodal queries)
    
    The irrelevant pairs help improve model robustness by ensuring that
    responses maintain similar scores regardless of the multimodal query context
    (different images + different text prompts).
    """
    processor: AutoProcessor
    max_length: int = 2560
    use_irrelevant_queries: bool = False 
    num_irrelevant_pairs: int = 1 

    # {'chosen': [{'content': [{'text': 'To determine which material has the highest density, we examine the "Density, \\( \\rho \\) [kg/m³]" column for each option:  \n\n- **A. Compact bone**: 1376 kg/m³  \n- **B. Fat**: 911 kg/m³  \n- **C. Muscle**: 1090 kg/m³  \n- **D. Skin**: 1109 kg/m³    \n\nComparing these values, 1376 kg/m³ (Compact bone) is the largest.    \n\nThus, the answer is **A. Compact bone**.', 'type': 'text'}], 'role': 'assistant'}],
    # 'rejected': [{'content': [{'text': 'To determine which material has the highest density, we need to look at the "Density, ρ" column in the table.\n\n- Muscle: 1090 kg/m³\n- Skin: 1109 kg/m³\n- Fat: 911 kg/m³\n- Compact bone: 1376 kg/m³\n- Bone marrow: 115 kg/m³\n\nFrom the data, we can see that compact bone has the highest density at 1376 kg/m³.\n\nTherefore, the correct answer is:\nA. Compact bone', 'type': 'text'}], 'role': 'assistant'}], 'prompt': [{'content': {'text': 'Which material has the highest density?\nOptions:\nA. Compact bone\nB. Fat\nC. Muscle\nD. Skin', 'type': 'text'}, 'role': 'user'}],
    #  'image': ['/mnt/user-ssd/lishicheng3/rlhf/raw_data_w_image_path/images/livexiv_2k/1e5f364f-3616-4f3a-ab03-7b34d0c210a3.jpg']}
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Process each example in the batch
        batch_messages_chosen = []
        batch_messages_rejected = []
        batch_messages_irrelevant_chosen = []
        batch_messages_irrelevant_rejected = []

        # Extract all queries (image + text) for irrelevant pair generation
        all_queries = []
        for feature in features:
            query_image_content = [
                {"type": "image", "image": Image.open(img_path.replace("/mnt/user-ssd/lishicheng3/rlhf/raw_data_w_image_path/images",   "/mnt/ssd-qinghai/lilei35/rlhf_images/images"))   }
                for img_path in feature["image"]
            ]
            query_text = feature["prompt"][0]["content"]["text"]
            all_queries.append({
                "image_content": query_image_content,
                "text": query_text
            })

        for i, feature in enumerate(features):
            # Create messages for chosen response
            # image_content = [
            #     {"type": "image", "image": Image.open(img_path.replace("/mnt/user-ssd/lishicheng3/rlhf/raw_data_w_image_path/images", 
            #                                                            "/mnt/ssd-qinghai/lilei35/rlhf_images/images"))}
            #     for img_path in feature["image"]
            # ]
            # use all_queries to get image_content to avoid image loading
            image_content = all_queries[i]["image_content"]
            query = feature["prompt"][0]["content"]["text"]
            chosen_response = feature["chosen"][0]["content"][0]["text"]
            rejected_response = feature["rejected"][0]["content"][0]["text"]

            chosen_messages = [
                {
                    "role": "user",
                    "content": [
                        *image_content,
                        {
                            "type": "text",
                            "text": query,
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": chosen_response,
                },
            ]

            # Create messages for rejected response
            rejected_messages = [
                {
                    "role": "user",
                    "content": [
                        *image_content,
                        {
                            "type": "text",
                            "text": query,
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": rejected_response,
                },
            ]

            batch_messages_chosen.append(chosen_messages)
            batch_messages_rejected.append(rejected_messages)

            # Create irrelevant query pairs if enabled
            if self.use_irrelevant_queries and i == 0 :
                # NOTE: we only create one irrelevant pair for reducing memory footprint
                assert len(all_queries) >= 2, "not enough queries to create irrelevant pairs"
                for _ in range(self.num_irrelevant_pairs):
                    # Sample a random query (image + text) from other samples (excluding current one)
                    available_queries = [q for j, q in enumerate(all_queries) if j != i]
                    if available_queries:
                        irrelevant_query = random.choice(available_queries)
                        
                        # Create irrelevant chosen pair (same response, different query)
                        irrelevant_chosen_messages = [
                            {
                                "role": "user",
                                "content": [
                                    *irrelevant_query["image_content"],  # Use different images
                                    {
                                        "type": "text",
                                        "text": irrelevant_query["text"],  # Use different text
                                    },
                                ],
                            },
                            {
                                "role": "assistant",
                                "content": chosen_response,
                            },
                        ]

                        # Create irrelevant rejected pair (same response, different query)
                        irrelevant_rejected_messages = [
                            {
                                "role": "user",
                                "content": [
                                    *irrelevant_query["image_content"],  # Use different images
                                    {
                                        "type": "text",
                                        "text": irrelevant_query["text"],  # Use different text
                                    },
                                ],
                            },
                            {
                                "role": "assistant",
                                "content": rejected_response,
                            },
                        ]

                        batch_messages_irrelevant_chosen.append(irrelevant_chosen_messages)
                        batch_messages_irrelevant_rejected.append(irrelevant_rejected_messages)

        # Process all messages: original pairs + irrelevant pairs
        all_messages = (
            batch_messages_chosen + 
            batch_messages_rejected + 
            batch_messages_irrelevant_chosen + 
            batch_messages_irrelevant_rejected
        )
        
        batch_texts = [
            self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=False
            )
            for msg in all_messages
        ]

        # Process vision inputs
        image_inputs = []
        for messages in all_messages:
            img_inputs, _ = process_vision_info(messages)
            image_inputs.extend(img_inputs)

        # Create the final batch
        batch = self.processor(
            text=batch_texts,
            images=image_inputs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Add metadata about batch structure for loss computation
        original_batch_size = len(features)
        irrelevant_batch_size = len(batch_messages_irrelevant_chosen)
        total_batch_size = original_batch_size * 2 + irrelevant_batch_size * 2
        
        batch["original_batch_size"] = original_batch_size
        batch["irrelevant_batch_size"] = irrelevant_batch_size
        batch["total_batch_size"] = total_batch_size
        # print("original_batch_size", batch["original_batch_size"], "irrelevant_batch_size", batch["irrelevant_batch_size"], "total_batch_size", batch["total_batch_size"])

        return batch


def preload_image(sample):
    sample["pil_image"] = [Image.open(img_path) for img_path in sample["image"]]
    # do smart resize
    try:
        sizes = [
            smart_resize(pil_image.size[1], pil_image.size[0])
            for pil_image in sample["pil_image"]
        ]
        sample["pil_image"] = [
            pil_image.resize((size[1], size[0]))
            for pil_image, size in zip(sample["pil_image"], sizes)
        ]
    except Exception as e:
        return None
    # conduct smart resize
    return sample

def filter_large_image(sample):
    cnt = 0 
    for pil_image in sample["image"]:
        pil_image = pil_image.replace("/mnt/user-ssd/lishicheng3/rlhf/raw_data_w_image_path/images", 
                                    "/mnt/ssd-qinghai/lilei35/rlhf_images/images")
        width, height = imagesize.get(pil_image)
        if width * height > 4096 * 28 * 28:
            return False
        elif width < 28 or height < 28:
            return False 
        cnt += int( (width * height) / 28 / 28)
    if cnt > 4096 * 2 :
        return False 
    return True


def load_v1_dataset(
    dataset_path="/mnt/user-ssd/lishicheng3/rlhf/hf_pref_data/v1", split="train"
):
    if "balance" in dataset_path:
        dataset = load_from_disk(dataset_path)
        splits = dataset.train_test_split(test_size=1000, seed=42)
    else:
        dataset = load_dataset(
            dataset_path, split=split, cache_dir="/mnt/bos-multimodal/lilei35/.cache/hf", 
        )
        splits = dataset.train_test_split(test_size=0.001, seed=42)
    train = splits["train"]
    test = splits["test"]
    
    # Filter out invalid images
    train = train.filter(filter_large_image, num_proc=16)
    test = test.filter(filter_large_image, num_proc=16)

    # make sure test is even 
    if len(test) % 2 != 0:
        test = test.select(range(len(test) - 1))
    
    return train, test


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, RMTrainingArguments))
    if len(sys.argv) == 2:
        # If we pass only one argument to the script, it could be either json or yaml
        config_file = os.path.abspath(sys.argv[1])
        if config_file.endswith(".json"):
            model_args, data_args, training_args = parser.parse_json_file(config_file)
        elif config_file.endswith(".yaml") or config_file.endswith(".yml"):
            model_args, data_args, training_args = parser.parse_yaml_file(config_file)
        else:
            raise ValueError("Config file must be a json or yaml file.")
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    vision_process.MAX_PIXELS = model_args.max_pixels
    vision_process.MIN_PIXELS = model_args.min_pixels 
    # Load model and processor
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        min_pixels=model_args.min_pixels,
        max_pixels=model_args.max_pixels,
    )

    # Load base model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
        attn_implementation=model_args.attn_implementation,
    )


    # Wrap with value head
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model, 
                                                              torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32)
    model.warnings_issued = {}
    print(model.pretrained_model.dtype, model.pretrained_model.device)
        # freeze vit
    if training_args.freeze_vit:
        print("freezing ViT")
        for name, param in model.pretrained_model.visual.named_parameters():
            param.requires_grad = False

    # Load dataset
    train_dataset, val_dataset = load_v1_dataset(data_args.dataset_path)
    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(data_args.max_train_samples))
    if data_args.max_eval_samples is not None:
        val_dataset = val_dataset.select(range(data_args.max_eval_samples))

    # Initialize data collator
    data_collator = VLMDataCollator(
        processor=processor,
        max_length=data_args.max_seq_length,
        use_irrelevant_queries=training_args.use_irrelevant_queries,
        num_irrelevant_pairs=training_args.num_irrelevant_pairs,
    )

    # Initialize trainer with custom VLMRewardTrainer
    trainer = VLMRewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # overfit debugging
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()

    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
