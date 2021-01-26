import neat
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import DataConversionWarning
import os
import warnings
import gzip
import pickle

### Data Preparation ########################################################################################################################################################################################


def problem_datasets(directory, num_jobs=None):

    warnings.filterwarnings(action="ignore", category=DataConversionWarning)

    feature_columns = ["due date", "family", "t_smd", "t_aoi"]

    input_1 = pd.read_excel(
        os.path.join(directory, "input.xlsx"), sheet_name="dataset 1", nrows=num_jobs
    )
    input_2 = pd.read_excel(
        os.path.join(directory, "input.xlsx"), sheet_name="dataset 2", nrows=num_jobs
    )
    input_3 = pd.read_excel(
        os.path.join(directory, "input.xlsx"), sheet_name="dataset 3", nrows=num_jobs
    )
    input_4 = pd.read_excel(
        os.path.join(directory, "input.xlsx"), sheet_name="dataset 4", nrows=num_jobs
    )

    features_1 = MinMaxScaler().fit_transform(input_1[feature_columns].values.tolist())
    features_2 = MinMaxScaler().fit_transform(input_2[feature_columns].values.tolist())
    features_3 = MinMaxScaler().fit_transform(input_3[feature_columns].values.tolist())
    features_4 = MinMaxScaler().fit_transform(input_4[feature_columns].values.tolist())

    dataset_1 = {
        i: {
            "id": i + 1,
            "due date": input_1["due date"][i],
            "family": input_1["family"][i],
            "t_smd": input_1["t_smd"][i],
            "t_aoi": input_1["t_aoi"][i],
            "scaled due date": features_1[i][0],
            "scaled family": features_1[i][1],
            "scaled t_smd": features_1[i][2],
            "scaled t_aoi": features_1[i][3],
            "alloc_to_smd": None,
        }
        for i in range(len(input_1))
    }

    dataset_2 = {
        i: {
            "id": i + 1,
            "due date": input_2["due date"][i],
            "family": input_2["family"][i],
            "t_smd": input_2["t_smd"][i],
            "t_aoi": input_2["t_aoi"][i],
            "scaled due date": features_2[i][0],
            "scaled family": features_2[i][1],
            "scaled t_smd": features_2[i][2],
            "scaled t_aoi": features_2[i][3],
            "alloc_to_smd": None,
        }
        for i in range(len(input_2))
    }

    dataset_3 = {
        i: {
            "id": i + 1,
            "due date": input_3["due date"][i],
            "family": input_3["family"][i],
            "t_smd": input_3["t_smd"][i],
            "t_aoi": input_3["t_aoi"][i],
            "scaled due date": features_3[i][0],
            "scaled family": features_3[i][1],
            "scaled t_smd": features_3[i][2],
            "scaled t_aoi": features_3[i][3],
            "alloc_to_smd": None,
        }
        for i in range(len(input_3))
    }

    dataset_4 = {
        i: {
            "id": i + 1,
            "due date": input_4["due date"][i],
            "family": input_4["family"][i],
            "t_smd": input_4["t_smd"][i],
            "t_aoi": input_4["t_aoi"][i],
            "scaled due date": features_4[i][0],
            "scaled family": features_4[i][1],
            "scaled t_smd": features_4[i][2],
            "scaled t_aoi": features_4[i][3],
            "alloc_to_smd": None,
        }
        for i in range(len(input_4))
    }

    datasets = {
        "dataset 1": dataset_1,
        "dataset 2": dataset_2,
        "dataset 3": dataset_3,
        "dataset 4": dataset_4,
    }

    return datasets


#############################################################################################################################################################################################################

### Restore ANN #############################################################################################################################################################################################


def restore_genome(filename, directory):

    genome_file = os.path.join(directory, filename + ".bin")
    with gzip.open(genome_file) as f:
        genome, config = pickle.load(f)
        return genome, config


#############################################################################################################################################################################################################
