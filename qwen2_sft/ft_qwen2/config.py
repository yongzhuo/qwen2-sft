# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/3/5 21:29
# @author  : Mo
# @function: config of qwen1.5


# optimized for RTX 4090. for larger GPUs, increase some of these?
MICRO_BATCH_SIZE = 4  # 4  # default=4  # this could actually be 5 but i like powers of 2
BATCH_SIZE = 128  # 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 5e-5  ### 1e-5
# LEARNING_RATE = 3e-4  # default=3e-4  # the Karpathy constant
EPOCHS = 3  # default=3  # we don't always need 3 tbh
# LORA_DROPOUT = 0.1
# LORA_ALPHA = 32
# LORA_R = 32
WEIGHT_DECAY = 0.01
LORA_DROPOUT = 0.1
LORA_ALPHA = 16
LORA_R = 8
SAVE_STEPS = 384
VAL_SET_SIZE = 0
MAX_LENGTH_Q = 128 - 1  # default=128 - 2
MAX_LENGTH_A = 128 - 1  # default=128 - 2
MAX_LENGTH_QA = MAX_LENGTH_Q + MAX_LENGTH_A + 2
TARGET_MODULES = ["q_proj",
                  "k_proj",
                  "v_proj",
                  # "o_proj",
                  # "down_proj",
                  # "gate_proj",
                  # "up_proj",
                  ]

PATH_MODEL_PRETRAIN = ""
REPO_ID = "Qwen/Qwen1.5-7B-Chat"
PATH_MODEL_PRETRAIN = PATH_MODEL_PRETRAIN if PATH_MODEL_PRETRAIN else REPO_ID
DATA_PATH = "../dataset/alpaca_gpt4_data_zh.json"
MODEL_SAVE_DIR = "model_sft"

IS_PARALLELIZABLE = True
MODEL_PARALLEL = True
USE_CACHE = False
CUDA_VISIBLE_DEVICES = "0"
USE_TORCH = "1"
CPU_NUMS = "9"
USE_CUDA = False if CUDA_VISIBLE_DEVICES == "-1" else True
USE_ALL_LOSS = True  # 计算loss时是否计算全部(False则为)

"""
('base_model.model.model.embed_tokens.weight', torch.bfloat16, False)
('base_model.model.model.layers.31.self_attn.q_proj.weight', torch.bfloat16, False)
('base_model.model.model.layers.31.self_attn.q_proj.bias', torch.bfloat16, False)
('base_model.model.model.layers.31.self_attn.q_proj.lora_A.default.weight', torch.float32, True)
('base_model.model.model.layers.31.self_attn.q_proj.lora_B.default.weight', torch.float32, True)
('base_model.model.model.layers.31.self_attn.k_proj.weight', torch.bfloat16, False)
('base_model.model.model.layers.31.self_attn.k_proj.bias', torch.bfloat16, False)
('base_model.model.model.layers.31.self_attn.k_proj.lora_A.default.weight', torch.float32, True)
('base_model.model.model.layers.31.self_attn.k_proj.lora_B.default.weight', torch.float32, True)
('base_model.model.model.layers.31.self_attn.v_proj.weight', torch.bfloat16, False)
('base_model.model.model.layers.31.self_attn.v_proj.bias', torch.bfloat16, False)
('base_model.model.model.layers.31.self_attn.v_proj.lora_A.default.weight', torch.float32, True)
('base_model.model.model.layers.31.self_attn.v_proj.lora_B.default.weight', torch.float32, True)
('base_model.model.model.layers.31.self_attn.o_proj.weight', torch.bfloat16, False)
('base_model.model.model.layers.31.mlp.gate_proj.weight', torch.bfloat16, False)
('base_model.model.model.layers.31.mlp.up_proj.weight', torch.bfloat16, False)
('base_model.model.model.layers.31.mlp.down_proj.weight', torch.bfloat16, False)
('base_model.model.model.layers.31.input_layernorm.weight', torch.bfloat16, False)
('base_model.model.model.layers.31.post_attention_layernorm.weight', torch.bfloat16, False)
('base_model.model.model.norm.weight', torch.bfloat16, False)
('base_model.model.lm_head.weight', torch.bfloat16, False)
"""
# layernorm_s = ["post_attention_layernorm",
#                "input_layernorm",
#                "norm"
#                ]
