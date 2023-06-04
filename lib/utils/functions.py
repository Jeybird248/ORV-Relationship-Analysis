import pandas as pd
import numpy as np
import spacy
from spacy import displacy
import networkx as nx
import matplotlib.pyplot as plt
import os
import re

from pyvis.network import Network
from community import community_louvain

# function definitions

def load_books(book_folder):
    texts = []
    for file_name in os.listdir(book_folder):
        file_path = os.path.join(book_folder, file_name)
    
        # Check if the path is a file (not a directory)
        if os.path.isfile(file_path):
            texts.append(file_path)
    return texts

def filter_entity(ent_list, character_df):
    return [ent for ent in ent_list
            if ent in list(character_df.character)
            or ent in list(character_df.character_firstname)
            or any(ent in alias_list for alias_list in character_df.aliases)]

def extract_last_name(full_name, last_names):
    for last_name in last_names:
        if last_name in full_name[0]:
            return last_name
    return full_name[0]

def ner(file):
    # load in text and langauge model

    NER = spacy.load("en_core_web_sm")
    book_doc = NER(open(file, 'r', encoding='utf-8').read())
    return book_doc

def ne_list(spacy_doc):
    # use NER to process and grab the list of entities and their sentences
    sent_entity_df = []

    for sent in spacy_doc.sents:
        entity_list = [ent.text for ent in sent.ents]
        sent_entity_df.append({"sentence": sent, "entities": entity_list})

    # turn list into df
    sent_entity_df = pd.DataFrame(sent_entity_df)
    return sent_entity_df

def create_relationships(df, window_size):
    # scroll through 5 lines at a time to aggregate relationships between characters and reorder to make sure A -> B is the same as B -> A
    relationships = []

    for i in range(df.index[-1]):
        end_i = min(i+5, df.index[-1])
        char_list = sum((df.loc[i : end_i].character_entities), [])

        char_unique = [char_list[i] for i in range(len(char_list))
                    if (i == 0) or char_list[i] != char_list[i-1]]
        if len(char_unique) > 1:
            for idx, a in enumerate(char_unique[:-1]):
                b = char_unique[idx + 1]
                relationships.append({"source": a, "target": b})

    # count the occurences of relationships between characters
    relationship_df = pd.DataFrame(relationships)
    relationship_df.head(10)
    relationship_df = pd.DataFrame(np.sort(relationship_df.values, axis = 1), columns = relationship_df.columns)

    relationship_df["value"] = 1
    relationship_df = relationship_df.groupby(["source", "target"], sort=False, as_index = False).sum()
    return relationship_df