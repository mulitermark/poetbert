import itertools
import random

# Function to load data from files
def load_data(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        return [line.strip().replace('.', '') for line in file.readlines()]

# Load data from files
instructions = load_data('hungarian_instructions/hun_instruction.txt') 
syllable_constraints = load_data('hungarian_instructions/hun_syllab.txt')
line_constraints = load_data('hungarian_instructions/hun_line_num.txt')
rhyme_schemes = load_data('hungarian_instructions/hun_rhyme.txt')
# Generate all possible combinations
all_combinations = list(itertools.product(instructions, line_constraints, syllable_constraints, rhyme_schemes))

def generate_texts(combinations):
    with open('templates.txt','w',encoding='utf-8') as file:
        for idx, combo in enumerate(combinations, start=1):
            instruction, line_constraint, syllable_constraint, rhyme_scheme = combo
            text = f"{instruction}!\n{line_constraint}: <<line_count>>\n{syllable_constraint}: <<syllable_count>>\n{rhyme_scheme}: <<rhyme_scheme>>\n"
            file.write(text + "\n\n")            

random.shuffle(all_combinations)

# Generate texts
generate_texts(all_combinations)