from Subject import Subject
from config import *
import utils
import os
import numpy as np


# load data
subjects = utils.load_subjects(path_input=PATH_2, subject_pattern=SUBJECT_PATTERN, debug=DEBUG)

subject = subjects[0]

# check all available attributes
print("Attributes:")
for key in subject._attributes.keys():
    print("   {}".format(key))

# check all available matrices
print("Matrices  :")
for key in subject._matrices.keys():
    print("   {}".format(key))

# check all available metrics
print("Metrics   :")
for key in subject._metrics.keys():
    print("   {}".format(key))
