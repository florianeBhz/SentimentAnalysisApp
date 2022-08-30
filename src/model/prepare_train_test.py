import sys
import os 
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath('../..'))
import config.model_config as model_config

text_col,labels_col = model_config.TEXT_COL, model_config.LABELS_COL
labels = list(model_config.LABELS_DICT.keys())

def prepare_data(data_file=model_config.DATA_FILE_LABELLED):
    print(data_file)
    df = pd.read_csv(data_file,index_col=None, 
                           header=0, engine='python',usecols=[text_col,labels_col])
    df_list = []
    for label in labels:
        df_list.append(df[df[labels_col]==label])

        # make the dataset balanced before splitting
        min_len = min([len(df_i) for df_i in df_list])
        tweet_df = pd.concat([df_i.iloc[:min_len] for df_i in df_list])
        tweet_df = shuffle(tweet_df,random_state=42)

        # Data spllitting 
        X = list(tweet_df[text_col])
        y = list(tweet_df[labels_col])
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)

        df_train = pd.DataFrame(list(zip(X_train, y_train)),
                columns =[text_col, labels_col])
        df_val = pd.DataFrame(list(zip(X_val, y_val)),
                columns =[text_col, labels_col])

        print("saving training data to ",model_config.TRAIN_FILE)
        df_train.to_csv(model_config.TRAIN_FILE,columns=[text_col,labels_col])
        print("saving test data to ",model_config.TEST_FILE)
        df_val.to_csv(model_config.TEST_FILE,columns=[text_col,labels_col])


#if __name__ == "main":
prepare_data()