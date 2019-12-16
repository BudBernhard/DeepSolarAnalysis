"""
DOCSTRING:

This module contains functions and pipelines for different classification models and parameter gridsearches.

"""

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report