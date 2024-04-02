# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/3/5 21:04
# @author  : Mo
# @function: qwen1.5


import traceback
import random
import sys
import os

path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print(path_root)
sys.path.append(path_root)
from qwen2_sft.ft_qwen2.config import CUDA_VISIBLE_DEVICES, USE_TORCH, CPU_NUMS  # from config
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:3072"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
os.environ["USE_TORCH"] = USE_TORCH
os.environ["OMP_NUM_THREADS"] = CPU_NUMS  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = CPU_NUMS  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = CPU_NUMS  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = CPU_NUMS  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = CPU_NUMS  # export NUMEXPR_NUM_THREADS=1

from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.modeling_utils import unwrap_model
from peft import LoraConfig, get_peft_model
from transformers import GenerationConfig
from tensorboardX import SummaryWriter
from datasets import load_dataset
from pydantic import BaseModel
from rouge import Rouge  # pip install rouge
from tqdm import tqdm
import transformers
import torch

# from qwen2_sft.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer as LLMTokenizer
# from qwen2_sft.models.qwen2.configuration_qwen2 import Qwen2Config as LLMConfig
# from qwen2_sft.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM as LLMModel
from transformers import Qwen2Tokenizer as LLMTokenizer
from transformers import Qwen2ForCausalLM as LLMModel
from transformers import Qwen2Config as LLMConfig
from qwen2_sft.ft_qwen2.config import PATH_MODEL_PRETRAIN, DATA_PATH, MODEL_SAVE_DIR, REPO_ID
from qwen2_sft.ft_qwen2.config import MICRO_BATCH_SIZE, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS
from qwen2_sft.ft_qwen2.config import LEARNING_RATE, EPOCHS, SAVE_STEPS, VAL_SET_SIZE, TARGET_MODULES
from qwen2_sft.ft_qwen2.config import IS_PARALLELIZABLE, MODEL_PARALLEL, USE_CACHE
from qwen2_sft.ft_qwen2.config import MAX_LENGTH_Q, MAX_LENGTH_A, MAX_LENGTH_QA
from qwen2_sft.ft_qwen2.config import LORA_DROPOUT, LORA_ALPHA, LORA_R
from qwen2_sft.ft_qwen2.config import USE_CUDA, WEIGHT_DECAY
from qwen2_sft.ft_qwen2.config import USE_ALL_LOSS


tensorboardx_witer = SummaryWriter(logdir=MODEL_SAVE_DIR)
# device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
use_all_loss = USE_ALL_LOSS or True
# world_size = int(os.environ.get("WORLD_SIZE", 1))
# device_map = "auto"
# ddp = world_size != 1
# print(device_map)
# print(ddp)


def save_model_state(model, config=None, model_save_dir="./", model_name="adapter_model.safetensors"):
    """  仅保存 有梯度 的 模型参数(推荐使用)  """
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    # save config
    if config:
        config.save_pretrained(model_save_dir)
        # config.to_dict()
    # save model
    path_model = os.path.join(model_save_dir, model_name)
    grad_params_dict = {k: v.to("cpu") for k, v in model.named_parameters()
                        if v.requires_grad == True}
    torch.save(grad_params_dict, path_model)
    print("******model_save_path is {}******".format(path_model))
def print_named_parameters(model, use_print_data=False):
    """   打印模型训练参数/数据类型信息   """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        if use_print_data:
            print((name, param.data.dtype, param.requires_grad, param.data))
        else:
            print((name, param.data.dtype, param.requires_grad))
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
def prepare_model_for_half_training(model, output_embedding_layer_name="lm_head",
        use_gradient_checkpointing=True, layer_norm_names=["layer_norm"]):
    r"""
    This method wrapps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    #  不要使用 model.half(), 这样会先截取精度再训练了, 最初data就要保持half
    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False
        # cast layer norm in fp32 for stability for 8bit models
        if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
            param.data = param.data.to(torch.float32)
        elif output_embedding_layer_name in name:  # lm_head也需要是tf.float32(最后一层)
            param.data = param.data.to(torch.float32)
        else:
            param.data = param.data.to(torch.half)

    if use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()
    return model
def generate_prompt(data_point, is_logger=False):
    """   指令微调:
    普通句子续写: bos + text + eos
    带 prompt:
    ‘<start_of_turn>user
    Knock knock.<end_of_turn>
    <start_of_turn>model
    Who’s there?<end_of_turn>model
    <start_of_turn>user
    Gemma.<end_of_turn>
    <start_of_turn>model
    Gemma who?<end_of_turn>model’
    """
    text_input = data_point.get("content", "")
    text_out = data_point.get("summary", "")

    prompt_text_1 = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{}<|im_end|>"""
    prompt_text_2 = """\n<|im_start|>assistant
{}<|im_end|>"""
    text_1 = prompt_text_1.format(text_input.strip())
    text_2 = prompt_text_2.format(text_out.strip())

    x = tokenizer.encode(text_1, add_special_tokens=False)
    y = tokenizer.encode(text_2, add_special_tokens=False)
    if len(x) + len(y) > (MAX_LENGTH_Q + MAX_LENGTH_A):
        x = x[:MAX_LENGTH_Q] + [ID_EOS]
        y = y[:MAX_LENGTH_A] + [ID_EOS]
    out = {"input_ids": x, "labels": y}
    if is_logger:
        print(text_1)
        print(text_2)
        print(out)
    return out

def data_collator(batch):
    # there's probably a way to do this with the tokenizer settings
    len_max_batch = [len(batch[i].get("input_ids")) + len(batch[i].get("labels"))
                    for i in range(len(batch))]
    len_max_batch = min(MAX_LENGTH_QA, max(len_max_batch))
    batch_attention_mask = []
    batch_input_ids = []
    batch_labels = []
    for ba in batch:
        x, y = ba.get("input_ids"), ba.get("labels")
        len_padding = len_max_batch - len(x) - len(y)
        if use_all_loss: ### 部分loss参与计算, output
            if tokenizer.padding_side and tokenizer.padding_side == "left":
                labels = [-100] * len_padding + [-100] * len(x) + y
                input_ids = [ID_PAD] * len_padding + x + y
                attention_mask = [1] * len_padding + [0] * (len_max_batch - len_padding)
            else:
                labels = [-100] * len(x) + y + [-100] * len_padding
                input_ids = x + y + [ID_PAD] * len_padding
                attention_mask = [0] * (len(x)+len(y)) + [1] * len_padding
        else:  ### 全部loss参与计算, input + output
            if tokenizer.padding_side and tokenizer.padding_side == "left":
                labels = [-100] * len_padding + x + y
                input_ids = [ID_PAD] * len_padding + x + y
                attention_mask = [1] * len_padding + [0] * (len_max_batch - len_padding)
            else:
                labels = x + y + [-100] * len_padding
                input_ids = x + y + [ID_PAD] * len_padding
                attention_mask = [0] * (len(x)+len(y)) + [1] * len_padding
        tensor_attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        tensor_input_ids = torch.tensor(input_ids, dtype=torch.long)
        tensor_labels = torch.tensor(labels, dtype=torch.long)
        batch_attention_mask.append(tensor_attention_mask)
        batch_input_ids.append(tensor_input_ids)
        batch_labels.append(tensor_labels)
    batch_attention_mask = torch.stack(batch_attention_mask)
    batch_input_ids = torch.stack(batch_input_ids)
    batch_labels = torch.stack(batch_labels)
    input_dict = {"attention_mask": batch_attention_mask,  # no use
                  "input_ids": batch_input_ids,
                  "labels": batch_labels,
                  }
    return input_dict
def dfs_file(path_dir):
    """
        递归获取某个目录下的所有文件(所有层, 包括子目录)
    Args:
        path_dir[String]:, path of dir, eg. "/home/data"
    Returns:
        data[List]: data of input, eg. ["2020_01_08.txt"]
    """
    path_files = []
    for root, dirs, files in os.walk(path_dir):  # 分别代表根目录、文件夹、文件
        for file in files:  # 遍历文件
            file_path = os.path.join(root, file)  # 获取文件绝对路径
            path_files.append(file_path)  # 将文件路径添加进列表
    files = list(set(path_files))
    files.sort()  # the same list
    return files


model = LLMModel.from_pretrained(PATH_MODEL_PRETRAIN, torch_dtype=torch.bfloat16)

# model = prepare_model_for_half_training(model,
#         use_gradient_checkpointing=True,
#         output_embedding_layer_name="lm_head",
#         layer_norm_names=["post_attention_layernorm",
#                           "input_layernorm",
#                           "norm"
#                           ],
#         )
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.is_parallelizable = IS_PARALLELIZABLE
model.model_parallel = MODEL_PARALLEL
model.config.use_cache = USE_CACHE
config = LoraConfig(target_modules=TARGET_MODULES,
                    lora_dropout=LORA_DROPOUT,
                    lora_alpha=LORA_ALPHA,
                    task_type="CAUSAL_LM",
                    bias="none",
                    r=LORA_R,
                    )
model = get_peft_model(model, config)
print_named_parameters(model)
model = model.cuda()
# for param in filter(lambda p: p.requires_grad, model.parameters()):
#     param.data = param.data.to(torch.float32)

for name, param in model.named_parameters():
    if "LoR" in name:   # 某些peft版本默认dtype=fp16, 这里全部转为 fp32
        param.data = param.data.to(torch.float32)

print_named_parameters(model)

tokenizer = LLMTokenizer.from_pretrained(PATH_MODEL_PRETRAIN, add_eos_token=True)
ID_END = 151643
ID_BOS = 151644
ID_EOS = 151645
ID_PAD = ID_EOS
ID_BR = 1699  # "\\n"
ID_USER = 872
ID_MODEL = 8948
tokenizer.pad_token_id = ID_EOS
tokenizer.eos_token_id = ID_EOS
tokenizer.padding_side = "left"
# tokenizer.padding_side = "right"  # NO use attention-mask
print(ID_PAD)
print(ID_BOS)
print(ID_EOS)
print(ID_BR)
print(ID_USER)
print(ID_MODEL)
"""
"<|endoftext|>": 151643,
"<|im_start|>": 151644,
"<|im_end|>": 151645
"system": 8948,
"user": 872,
"\\n": 1699,
"""
STOP_WORDS_IDS = [[ID_BOS], [ID_EOS], [ID_END]]



### 只有一个train的情况
data = load_dataset("json", data_files=DATA_PATH)
if VAL_SET_SIZE > 0:
    # train_val = data["train"].train_test_split(test_size=min(VAL_SET_SIZE,
    #                     int(len(data["train"])/10000)), shuffle=True, seed=42)
    VAL_SET_SIZE = max(min(VAL_SET_SIZE, int(len(data["train"])/10000)), 1)
    generate_prompt(data["train"][0], is_logger=True)
    train_val = data["train"].train_test_split(test_size=VAL_SET_SIZE, shuffle=True, seed=42)
    train_data = train_val["train"].shuffle().map(generate_prompt)
    val_data = train_val["test"].shuffle().map(generate_prompt)
else:
    generate_prompt(data["train"][0], is_logger=True)
    train_data = data["train"].shuffle().map(generate_prompt)
    val_data = None


class CustomTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """  newest loss """
        inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = model(**inputs)  # if contain labels, will calculate loss

        try:
            logs = {}
            tr_loss_scalar = self._nested_gather(outputs.loss.detach()).mean().item()
            logs["loss"] = round(tr_loss_scalar, 4)
            logs["lr"] = self.lr_scheduler.get_last_lr()[0]
            step = self.state.global_step
            for k, v in logs.items():
                tensorboardx_witer.add_scalar(k, v, step)
            self.log(logs)
        except Exception as e:
            print(traceback.print_exc())
            print(logs)

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss


trainer = CustomTrainer(
        # data_collator=transformers.DataCollatorForSeq2Seq(
        #             tokenizer, pad_to_multiple_of=8,
        #             return_tensors="pt", padding=True
        #         ),
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=val_data,
        model=model,
        args=transformers.TrainingArguments(
            weight_decay=WEIGHT_DECAY,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            num_train_epochs=EPOCHS,
            max_grad_norm=1.0,
            logging_steps=8,
            # warmup_steps=382,  # 618
            # warmup_ratio=0.01,
            warmup_steps=1,  # 618
            evaluation_strategy="no",
            lr_scheduler_type="cosine",  # "constant",  #'constant',  # "cosine",
            logging_first_step=True,
            # evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
            # eval_steps=SAVE_STEPS if VAL_SET_SIZE > 0 else None,
            save_strategy="steps",
            save_total_limit=12,
            save_steps=SAVE_STEPS,
            # load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
            ddp_find_unused_parameters=None,
            gradient_checkpointing=True,
            # group_by_length=True,  # group together samples of roughly the same length in training
            output_dir=MODEL_SAVE_DIR,
            report_to=[],  # ["tensorboard"],  # [], ["wandb"]
            optim="adamw_torch",  # "adamw_hf",
            # optim="adafactor",
            # fp16=True,
        )
    )

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)


## 加载训练好的权重
# files = dfs_file(MODEL_SAVE_DIR)
# files_name_str = str(files)
# flag_checkpoint = True if files and "checkpoint" in files_name_str else False
flag_checkpoint = False
trainer.train(resume_from_checkpoint=flag_checkpoint)
save_model_state(model=model, config=config, model_save_dir=MODEL_SAVE_DIR)
print_named_parameters(model, use_print_data=True)  # 查看LoRA层权重是不是为NAN溢出


# nohup python train.py > tc.train.py.log 2>&1 &
# tail -n 1000  -f tc.train.py.log
# |myz|

