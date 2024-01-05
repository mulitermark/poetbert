from torch import nn
import torch
from poem_dataset import PoemDataset
from custom_trainer import CustomTrainer
import json
import os
print(os.listdir())
from transformers import (
    DataCollatorForLanguageModeling,
    TrainingArguments,
    GPTNeoXForCausalLM,
    AutoTokenizer,
)

device = torch.device('cpu')

cache_dir = './models/'

base_model = GPTNeoXForCausalLM.from_pretrained("NYTK/PULI-GPT-3SX", cache_dir = cache_dir)
tokenizer = AutoTokenizer.from_pretrained("NYTK/PULI-GPT-3SX", 
                                          cache_dir = cache_dir)



if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

dataset = PoemDataset(
    file_path="data/dataset.json",
    tokenizer=tokenizer,
    #block_size=128,
)

base_model.to(device)
dataset.to(device)

training_args = TrainingArguments(
    output_dir="./puli_finetuned",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    save_steps=500,
    save_total_limit=2,
    device='cpu'
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = CustomTrainer(
    model=base_model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

print('training_started')
trainer.train()
trainer.save_model(training_args['output_dir'])
print('training_done')
