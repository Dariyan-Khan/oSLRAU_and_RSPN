import numpy as np
import torch

from collections import namedtuple
from hmmlearn.hmm import MultinomialHMM

from rspnlib.models.iohmm import IoHmmRSPN, ContinuousIoHmmRSPN
from sklearn.model_selection import KFold
import pickle

CausalHealthDataset = namedtuple(
    "CausalHealthDataset",
    [
        "inputs_raw",
        "observations_raw_continuous",
        "spn_data_continuous",
    ],
)

def get_CH_parameters():
    # Specify the IOHMM parameters
    prior_proba = [0.7, 0.3]
    start_proba = [[0.9, 0.1], [0.7, 0.3]]
    trans_proba = [[[0.9, 0.1], [0.5, 0.5]], [[0.6, 0.4], [0.1, 0.9]]]

    # Continuous case
    emission_means = [[[50, 50], [40, 70]], [[70, 75], [60, 100]]]
    emission_covars = [
        [[[15, 0], [0, 15]], [[15, 0], [0, 15]]],
        [[[15, 0], [0, 15]], [[15, 0], [0, 15]]],
    ]

    prior_proba = torch.tensor(prior_proba, dtype=torch.float64)
    start_proba = torch.tensor(start_proba, dtype=torch.float64)
    trans_proba = torch.tensor(trans_proba, dtype=torch.float64)
    emission_means = torch.tensor(emission_means, dtype=torch.float64)
    emission_covars = torch.tensor(emission_covars, dtype=torch.float64)

    return (
        prior_proba,
        start_proba,
        trans_proba,
        emission_means,
        emission_covars,
    )

def create_causal_health_iohmm():
    (
        prior_proba,
        start_proba,
        trans_proba,
        emission_means,
        emission_covars,
    ) = get_CH_parameters()
    reference_io_hmm_rspn = ContinuousIoHmmRSPN(
        prior_proba, start_proba, trans_proba, emission_means, emission_covars
    )
    return reference_io_hmm_rspn

def load_causal_health_dataset(folds=5):
    """
    Loads the vasopressors dataset derived from MIMIC

    :param seq_length: The length of a sequence to use for each patient
    :param folds: The number of folds to split the dataset into
    """

    causal_health_data = pickle.load(open('c:/Users/jonat/Desktop/UROP/Github/rspn_policy_learn/RSPN/experiment_utils/datasets/CH_one_step_N10000.pkl', 'rb'))

    def construct_causal_health_dataset(raw_data):
        inputs_raw = raw_data[:, 0::6].astype(int)
        observations_raw_continuous = np.hstack([raw_data[:, 2:6], raw_data[:, 8:12]])
        return CausalHealthDataset(
            inputs_raw=inputs_raw,
            observations_raw_continuous=observations_raw_continuous,
            spn_data_continuous=raw_data,
        )
    
    train_causal_health_datasets = []
    test_causal_health_datasets = []

    kf = KFold(n_splits=folds)
    for train_index, test_index in kf.split(causal_health_data):
        train_data = construct_causal_health_dataset(causal_health_data[train_index])
        train_causal_health_datasets.append(train_data)
        test_data = construct_causal_health_dataset(causal_health_data[test_index])
        test_causal_health_datasets.append(test_data)
    
    return train_causal_health_datasets, test_causal_health_datasets


# def generate_engine_datasets(
#     seeds=range(42, 42 + 5), development_samples=1000, test_samples=1000, seq_length=12
# ):
#     """
#     Generates engine datasets with the specified number of seed partitions, samples and sequence lengths

#     :param seeds: The seeds to use when generating partitions
#     :param development_samples: The number of samples in the development set (to be used for training and validation)
#     :param test_samples: The number of samples in the withheld test set
#     :param seq_length: The length of the temporal sequences to generate
#     """
#     development_engine_datasets = []
#     test_engine_datasets = []

#     for seed in seeds:
#         torch.manual_seed(seed)
#         development_engine_dataset = generate_engine_dataset(
#             num_samples=development_samples, seq_length=seq_length
#         )
#         development_engine_datasets.append(development_engine_dataset)
#         test_engine_dataset = generate_engine_dataset(
#             num_samples=test_samples, seq_length=seq_length
#         )
#         test_engine_datasets.append(test_engine_dataset)

#     return development_engine_datasets, test_engine_datasets