import argparse

from stanza_data_creation import create_stanza_dataset
from utils import write_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script that creates a nicely formatted dataframe (stored as a .json file) from the xml files included in the poetry corpus. The created dataset includes stanzas of poems with their respective information.")

    parser.add_argument("-i", "--input_dir", required=True, type=str, help="The root directory of the poetry corpus, probably .../.../poetry-corpus")
    parser.add_argument("-o", "--out_file_path", required=True, type=str, help="The path of the created output .json dataset file")
    parser.add_argument("-f", "--filtered_poets_path", type=str, default="filtered_poets.txt",
                        help="Path to the .txt file containing the list of poets (their directory name in the repo) that should be included in the dataset")
    
    args = parser.parse_args()

    df = create_stanza_dataset(args.input_dir, args.filtered_poets_path)
    write_data(df, args.out_file_path)