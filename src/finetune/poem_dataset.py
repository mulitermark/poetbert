import json
from torch.utils.data import Dataset

class PoemDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.tokenizer = tokenizer

        # Load the data from the JSON file
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Extract prompts and stanzas
        self.prompts = [entry["prompt"] for entry in data]
        self.stanzas = [entry["stanza"] for entry in data]

    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        stanza = self.stanzas[idx]

        # Tokenize the prompt and stanza using the provided tokenizer
        tokenized_prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        tokenized_stanza = self.tokenizer(stanza, return_tensors="pt", padding=True, truncation=True)
        labels = tokenized_stanza.input_ids
        print('prompt')
        print(tokenized_prompt)
        #print(labels[0])
        print('stanza')
        print(tokenized_stanza)
        print('\n')

        return {
            "input_ids": tokenized_prompt.input_ids.squeeze(),
            "attention_mask": tokenized_prompt.attention_mask.squeeze(),
            "labels": tokenized_stanza.input_ids.squeeze()  # Convert to list of integers
        }