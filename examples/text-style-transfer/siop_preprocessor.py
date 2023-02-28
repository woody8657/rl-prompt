import os
import pandas as pd
from sklearn.model_selection import train_test_split

class SIOP_DATASET:
    def __init__(self, csv_path="./data/siop/SIOP_dataset.csv"):
        self.df_siop = pd.read_csv(csv_path)
        self.response_col_LUT = {
            "O": "open_ended_5_Openness",
            "C": "open_ended_2_Conscientiousness",
            "E": "open_ended_3_Extraversion",
            "A": "open_ended_1_Agreeableness",
            "N": "open_ended_4_Neuroticism",
        }
        self.score_col_LUT = {
            "O": "O_Scale_score",
            "C": "C_Scale_score",
            "E": "E_Scale_score",
            "A": "A_Scale_score",
            "N": "N_Scale_score",
        }

    def threshold(self, personality="O", threshold=2.5):
        df_filtered = self.df_siop.loc[:,[self.response_col_LUT[personality],self.score_col_LUT[personality]]]
        df_positive = df_filtered.loc[df_filtered[self.score_col_LUT[personality]]>threshold,:]
        df_negative = df_filtered.loc[df_filtered[self.score_col_LUT[personality]]<=threshold,:]
        return list(df_positive.loc[:,self.response_col_LUT[personality]]), list(df_negative.loc[:,self.response_col_LUT[personality]])

    def plot_distribution(self, personality="O"):
        import numpy as np
        from matplotlib import pyplot as plt 
        score_list = np.array(self.df_siop.loc[:,[self.score_col_LUT[personality]]])
        plt.hist(score_list, bins=20) 
        plt.xlim([1, 5])
        plt.title(f"Distribution of personality: {personality}") 
        plt.savefig(f"./figures/hist_{personality}.png")
        plt.clf()

def split_dataset(data):
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    dev, test = train_test_split(test, test_size=0.5, random_state=42)
    return train, dev, test

def write_txt(file_name, data_list):
    with open(file_name, 'w') as f:
        for line in data_list:
            f.write(f"{line}\n")

if __name__ == "__main__":
    siop = SIOP_DATASET()
    positive_list, negative_list = siop.threshold(personality="N", threshold=2)
    
    for category in ["positive", "negative"]:
        if category == "positive":
            data_list = positive_list
        elif category == "negative":
            data_list = negative_list
        else:
            raise
        train, dev, test = split_dataset(data_list)
        write_txt(os.path.join(f"./data/siop/train_{category}.txt"), train)
        write_txt(os.path.join(f"./data/siop/dev_{category}.txt"), dev)
        write_txt(os.path.join(f"./data/siop/test_{category}.txt"),test)