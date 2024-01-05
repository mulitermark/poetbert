from bs4 import BeautifulSoup
import lxml
import os
import pandas as pd
import glob
from tqdm import tqdm

def get_stanza_texts(lvl1_tei_xml_file):
    with open(lvl1_tei_xml_file, 'r', encoding='utf-8') as tei:
        soup = BeautifulSoup(tei.read(), 'xml')
        stanzas = soup.findAll('lg')
        stanza_texts = [s.text[1:-1] for s in stanzas]
        return stanza_texts
    
def get_stanza_data(lvl4_tei_xml_file):
    with open(lvl4_tei_xml_file, 'r', encoding='utf-8') as tei:
        soup = BeautifulSoup(tei.read(), 'lxml')
        stanzas = soup.findAll('lg')
        rhyme_patterns = [s.attrs['rhyme'] for s in stanzas]
        line_lengths = [s.attrs['lg_numline'] for s in stanzas]
        syll_patterns = [s.attrs['lg_syllpattern'] for s in stanzas]
        return rhyme_patterns, line_lengths, syll_patterns

def get_data_for_one_poem(data_dir, poem_rel_path: str):
    stanza_texts = get_stanza_texts(os.path.join(data_dir, "level1", poem_rel_path))
    rhyme, line_l, syll = get_stanza_data(os.path.join(data_dir, "level4", poem_rel_path))
    data = {'poem_rel_path': poem_rel_path,'text': stanza_texts, 'rhyme': rhyme, 'line_len': line_l, 'syllables': syll}
    df = pd.DataFrame(data)
    return df

def create_stanza_dataset(data_dir, filtered_poets_filename):
    data_dir = os.path.normpath(data_dir)
    unfiltered_poem_file_paths = glob.glob(os.path.join(data_dir, "level1/*/*.xml"))
    poem_file_paths = []
    with open(filtered_poets_filename, "r", encoding='UTF-8') as filtered_poets_f:
        poets = [poet.strip() for poet in filtered_poets_f.readlines()]
        poem_file_paths = [poem_file_p for poem_file_p in unfiltered_poem_file_paths if poem_file_p.split(os.sep)[-2] in poets]
    poem_relative_paths = [os.sep.join(os.path.normpath(poem_file).split(os.sep)[-2:]) for poem_file in poem_file_paths]
    poem = poem_relative_paths[0]
    df = get_data_for_one_poem(data_dir, poem)
    print(df.head)
    for poem_rel_path in tqdm(poem_relative_paths[1:]):
        df = pd.concat([df, get_data_for_one_poem(data_dir, poem_rel_path)])
    print("DATASET_HOSSZA:", len(df))
    return df