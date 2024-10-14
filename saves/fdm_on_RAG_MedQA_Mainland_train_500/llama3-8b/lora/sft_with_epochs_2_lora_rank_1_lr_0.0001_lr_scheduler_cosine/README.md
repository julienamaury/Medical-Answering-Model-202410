---
license: other
library_name: peft
tags:
- llama-factory
- lora
- generated_from_trainer
base_model: /home/zhuxiang001/llamafactory/Meta-Llama-3-8B-Instruct/
model-index:
- name: sft_with_epochs_2_lora_rank_1_lr_0.0001_lr_scheduler_cosine
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# sft_with_epochs_2_lora_rank_1_lr_0.0001_lr_scheduler_cosine

This model is a fine-tuned version of [/home/zhuxiang001/llamafactory/Meta-Llama-3-8B-Instruct/](https://huggingface.co//home/zhuxiang001/llamafactory/Meta-Llama-3-8B-Instruct/) on the RAG_MedQA_Mainland_train_500(example) dataset.
It achieves the following results on the evaluation set:
- Loss: 0.7031

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 2.0
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| 0.8836        | 1.0   | 499  | 0.7141          |
| 0.9244        | 2.0   | 998  | 0.7031          |


### Framework versions

- PEFT 0.11.1
- Transformers 4.41.0
- Pytorch 2.3.0+cu121
- Datasets 2.18.0
- Tokenizers 0.19.1