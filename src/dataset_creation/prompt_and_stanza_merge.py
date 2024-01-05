import random
import tqdm
import pandas as pd

from utils import read_json_dataframe

TEMPLATE_TO_DATA_DICT = {
    "[RHYME_PATTERN]": "rhyme",
    "[LINE_LENGTH]": "line_len",
    "[SYLLABLE_COUNT]": "syllables"
}

def replace_in_prompt(prompt, old_str, new_str):
    if old_str in prompt:
        prompt = prompt.replace(old_str, new_str)
    return prompt

def replace_all_in_prompt(prompt: str, data_dict, prompt_template_to_data_dict: dict):
    for key, value in prompt_template_to_data_dict.items():
        prompt = replace_in_prompt(prompt, key, str(data_dict[value]))
    return prompt

def merge_prompts_with_data(stanza_dataset_path, prompts_txt_path, max_stanza_length = 10, stanza_repetition_tuple = (1, 2)):
    out_df = {"prompt": [], "stanza": []}
    with open(prompts_txt_path, 'r', encoding='utf-8') as prompt_f:
        prompt_templates = prompt_f.readlines()
        prompt_templates = [pt.strip() for pt in prompt_templates]
        data = read_json_dataframe(stanza_dataset_path)
        for data_dict in tqdm(data):
            stanza = data_dict["text"]
            if len(stanza.split('\n')) >= max_stanza_length:
                continue
            for _ in range(random.randint(*stanza_repetition_tuple)):
                out_df['stanza'].append(stanza)
                prompt = random.choice(prompt_templates)
                prompt = replace_all_in_prompt(prompt, data_dict, TEMPLATE_TO_DATA_DICT)
                out_df['prompt'].append(prompt)
    out_df = pd.DataFrame(out_df)
    print(out_df)
    return out_df