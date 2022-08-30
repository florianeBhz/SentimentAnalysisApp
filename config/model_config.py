import os 

LABELS_DICT = {0:'Negative',1:'Neutral',2:'Positive'}

TEXT_COL = "text"
LABELS_COL = "labels"

# PATHS 
PROJECT_DIR = os.path.abspath('../..')
DATA_DIR = os.path.join(PROJECT_DIR,'data')
DATA_FILE_LABELLED = os.path.join(DATA_DIR, "labelled_COVID-19_vaccine.csv")
TRAIN_FILE = os.path.join(DATA_DIR,"train.csv")
TEST_FILE = os.path.join(DATA_DIR,"test.csv")

MODEL_DIR = os.path.join(os.path.abspath('..'),'model')


MODELS_path = os.path.join(PROJECT_DIR,"models")
LOGS_path = os.path.join(PROJECT_DIR,'logs')
RAY_path = os.path.join(PROJECT_DIR,"ray_results")