##################################
### Paths and general settings ###
##################################
DEBUG = True
PATH_BCN = "/Users/jcasasr/Library/CloudStorage/Dropbox/CVC_MRI-Data/BCN_output"
PATH_NAP = "/Users/jcasasr/Library/CloudStorage/Dropbox/CVC_MRI-Data/NAP_output"
PATH_1 = "./data_1"
PATH_2 = "./data_2"
MATRIX_TYPES = ["FA", "GM", "RS", "ML"] # Types of valid matrices

####################
### Data sources ###
####################
# All: "*.pkl" | BCN: "BCN-*.pkl" | NAP: "NAP-*.pkl"
SUBJECT_PATTERN = "BCN-*.pkl"
# Data models
DATA_TYPES = ["FA", "GM", "RS", "ML"]

######################
### Graph settings ###
######################
# List of metrics to be computed
METRIC_LIST = ['Degree', 'Strength', 'LocalEfficiency', 'ClosenessCentrality', 'BetweennessCentrality']

####################
### GNN settings ###
####################
# Data types to be tested
TEST_DATA_TYPES = [("FA", ["FA"]), 
                   ("GM", ["GM"]), 
                   ("RS", ["RS"]), 
                   ("ML", ["ML"]), 
                   ("MM", ["GM", "RS"])]
# Number of classes (binary or multiclass)
NUM_CLASSES = 2
# PARAMS
NUM_EPOCHS = 100
N_SPLITS = 4
THR = 0.4
# GAT params
NUM_HEADS = 4
DROPOUT = 0.6
# Model architecture
# Tuple: (model, [dim_in, dim_out, dim_h1, dim_h2], [learning_rate, weight_decay])
MODEL_ARCHITECTURES = [("GCN", [32, 0]),
                       ("GCN", [32, 16]),
                       ("GAT", [32, 0]),
                       ("GAT", [32, 16])]
LEARNING_RATES = [0.01, 0.001, 0.0001]
WEIGHT_DECAYS = [5e-4, 5e-6]