import argparse

from prompt_and_stanza_merge import merge_prompts_with_data
from utils import write_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script that creates a nicely formatted dataframe (stored as a .json file) from the provided prompt templates and stanza dataset. The created dataset includes prompt-stanza pairs")

    parser.add_argument("-s", "--stanza_dataset", required=True,type=str, help="The path of the stanza dataset created by the format_stanzas.py script")
    parser.add_argument("-p", "--prompts_file", required=True, type=str, help="The path of the file containing the prompt templates")
    parser.add_argument("-o", "--out_file_path", required=True, type=str, help="The path of the created output .json dataset file")
    parser.add_argument("-m", "--max_stanza_length", type=int, default=10, help="The maximum number of lines a stanza can have to be included in the created dataset")
    parser.add_argument("--min_stanza_repetition", type=int, default=1, help="The minimum number of times a stanza will be repeated in the created dataset (the actual number is randomized as randint(min, max))")
    parser.add_argument("--max_stanza_repetition", type=int, default=2, help="The maximum number of times a stanza will be repeated in the created dataset (the actual number is randomized as randint(min, max))")

    
    args = parser.parse_args()
    print(args)

    dataset = merge_prompts_with_data(args.stanza_dataset,
                                      args.prompts_file, 
                                      args.max_stanza_length, 
                                      (args.min_stanza_repetition, args.max_stanza_repetition))
    write_data(dataset, args.out_file_path)