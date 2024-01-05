from torch import nn, optim
import torch
import json
from torch.utils.data import Dataset
from transformers import GPTNeoXForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader
from torch.profiler import ProfilerActivity, profile
from poem_dataset import PoemDataset
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cache_dir = './models/'

# base_model = GPTNeoXForCausalLM.from_pretrained("NYTK/PULI-GPT-3SX", cache_dir=cache_dir)
# tokenizer = AutoTokenizer.from_pretrained("NYTK/PULI-GPT-3SX", cache_dir=cache_dir)

base_model = GPTNeoXForCausalLM.from_pretrained("NYTK/PULI-GPTrio", cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained("NYTK/PULI-GPTrio", cache_dir=cache_dir)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

num_epochs = 1
batch_size = 2

dataset = PoemDataset(file_path="data/dataset.json", tokenizer=tokenizer)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

base_model.to(device)
base_model.train()

optimizer = optim.AdamW(base_model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()



base_model.train()
for epoch in range(num_epochs):
    profiler = profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA
        ],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=2
        ),
        on_trace_ready=None,
        record_shapes=True,
        with_stack=True,
        use_cuda=torch.cuda.is_available()
    )
    
    for idx, batch in enumerate(train_loader):
        print('loop: ', idx)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        with torch.autograd.profiler.emit_nvtx():
            outputs = base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs.loss
            loss.backward()
            optimizer.step()

        if (idx + 1) % 100 == 0: 
            print(f"Epoch [{epoch + 1}/{num_epochs}] - Batch [{idx + 1}/{len(train_loader)}]")

    profiler.export_chrome_trace(f'profiler_output_epoch_{epoch + 1}.json')
    profiler.__exit__(None, None, None)


output_dir = "./puli_finetuned"
base_model.save_pretrained(output_dir)
