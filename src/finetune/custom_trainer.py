from transformers import Trainer
import torch.nn as nn

class CustomTrainer(Trainer):
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop('labels')

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits

        # Flatten logits and labels to handle variable lengths
        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = labels.view(-1)

        # Calculate cross-entropy loss
        loss = nn.functional.cross_entropy(flat_logits, flat_labels, ignore_index=-100)  # Ignoring padding tokens

        return (loss, outputs) if return_outputs else loss
