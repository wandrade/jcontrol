#!/usr/bin/env python

import pandas as pd
import os

path = os.path.dirname(os.path.realpath(__file__))
path = path + "/model/Dataset/Dataset_evo.txt"
pd.read_csv(path, sep=" ")