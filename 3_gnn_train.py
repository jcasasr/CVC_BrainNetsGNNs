import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime

from Subject import Subject
from GCN import GCN
from GAT import GAT
import utils
import utils_gnn
import utils_score
import utils_graphs
from config import *


if __name__ == "__main__":
    # for each data type
    for DATA_TYPE in TEST_DATA_TYPES:
        # graph structure (topology)
        data_type_structure = DATA_TYPE[0]
        data_type_embbedings = DATA_TYPE[1]

        # for each model architecture
        for model_arch in MODEL_ARCHITECTURES:
            MODEL = model_arch[0]
            # DIM_IN is the number of embeddings / node
            DIM_IN = len(METRIC_LIST) * len(data_type_embbedings)
            # DIM_OUT is the number of classes
            DIM_OUT = NUM_CLASSES
            # Hidden layers
            DIM_H1 = model_arch[1][0]
            DIM_H2 = model_arch[1][1]

            # for each Learning Rate and Weight Decay
            for LEARNING_RATE in LEARNING_RATES:
                # for each weight decay
                for WEIGHT_DECAY in WEIGHT_DECAYS:
                    # Set logger
                    LOGGING_NAME = datetime.today().strftime("%Y%m%d-%H%M%S") +"_"+ data_type_structure +"_"+ MODEL + "-"+ str(DIM_H1) + "-" + str(DIM_H2) + "_lr" + str(LEARNING_RATE) + "_wd" + str(WEIGHT_DECAY)
                    logger = utils.set_logging(logging_level=logging.DEBUG, logging_name=LOGGING_NAME)
                    logger.info("Model: {} | Dim_in: {} | Dim_out: {} | Dim_h1: {} | Dim_h2: {} | LR: {} | WD: {}".format(MODEL, DIM_IN, DIM_OUT, DIM_H1, DIM_H2, LEARNING_RATE, WEIGHT_DECAY))
                    
                    # load data
                    subjects = utils.load_subjects(path_input=PATH_2, subject_pattern=SUBJECT_PATTERN, debug=DEBUG)
                    if NUM_CLASSES==2:
                        target = [subject.get_mstype(type="binary") for subject in subjects]
                    else:
                        target = [subject.get_mstype() for subject in subjects]
                    preds = np.zeros(len(subjects))

                    # k-fold cross-validation
                    skf = StratifiedKFold(n_splits=N_SPLITS)
                    fold = 0
                    for train_index, test_index in skf.split(subjects, target):
                        fold += 1
                        logging.info("Fold: {}".format(fold))

                        # split dataset
                        X_train = [subjects[i] for i in train_index]
                        X_test  = [subjects[i] for i in test_index]
                        y_train, y_test = np.array(target)[train_index], np.array(target)[test_index]
                        
                        prop_train = np.where(y_train == 1)[0].shape[0] / y_train.shape[0]
                        prop_test = np.where(y_test == 1)[0].shape[0] / y_test.shape[0]
                        logger.info("Train set size     : {}".format(len(X_train)))
                        logger.info("Test set size      : {}".format(len(X_test)))
                        logger.info("Train set % of pwMS: {:.4f} ({})".format(prop_train, y_train.sum()))
                        logger.info("Test set % of pwMS : {:.4f} ({})".format(prop_test, y_test.sum()))

                        # list of Data structures (one for each subject)
                        train_graphs = []
                        for i in range(len(X_train)):
                            g = utils_gnn.array_to_graph(subject=X_train[i], data_type_structure=data_type_structure, data_type_embbedings=data_type_embbedings, num_classes=NUM_CLASSES, thr=THR, logger=logger)
                            train_graphs.append(g)
                            
                        test_graphs = []
                        for i in range(len(X_test)):
                            g = utils_gnn.array_to_graph(subject=X_test[i], data_type_structure=data_type_structure, data_type_embbedings=data_type_embbedings, num_classes=NUM_CLASSES, thr=THR, logger=logger)
                            test_graphs.append(g)

                        # create the model
                        if MODEL == "GCN":
                            model = GCN(dim_in=DIM_IN, dim_out=DIM_OUT, dim_h1=DIM_H1, dim_h2=DIM_H2, logger=logger)
                        elif MODEL == "GAT":
                            model = GAT(dim_in=DIM_IN, dim_out=DIM_OUT, dim_h1=DIM_H1, dim_h2=DIM_H2, heads=NUM_HEADS, dropout=DROPOUT, lr=LEARNING_RATE, wd=WEIGHT_DECAY, logger=logger)
                        else:
                            logger.error("Model not implemented!")
                            raise NotImplementedError("Model not implemented!")
                        
                        model.fit(num_epocs=NUM_EPOCHS, num_classes=NUM_CLASSES, train_graphs=train_graphs)
                        y_pred = model.test(test_graphs=test_graphs, y_test=y_test)
                        # Save predictions to compute final test scores
                        preds[test_index] = y_pred

                    # Compute metrics
                    utils_score.compute_auc(target=target, preds=preds, logger=logger)

                    # Plot ROC curve
                    utils_score.plot_auc_roc(target=target, preds=preds, name_out=LOGGING_NAME)