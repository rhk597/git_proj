import pandas as pd
import pickle
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def train(df):
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(df['All_Data'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    filename = 'trained_model.sav'
    pickle.dump(cosine_sim, open(filename, 'wb'))
    print("----  trained and stored successfully ---- ")

def extract_titles(df):
    df = df.reset_index()
    indices = pd.Series(df.index, index=df['Title'])
    all_Titles = [df['Title'][i] for i in range(len(df['Title']))]
    with open('titles.txt', 'w') as outfile:
        json.dump({'titles': all_Titles}, outfile)
    print("---- Titles exported successfully ----")

if __name__ == '__main__':
    df = pd.read_csv('./model/final_steam_data.csv')
    train(df)
    extract_titles(df)
    print("---- successfully completed all tasks ---- ")