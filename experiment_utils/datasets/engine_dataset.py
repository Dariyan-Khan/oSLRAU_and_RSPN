import numpy as np
import torch

from collections import namedtuple
from hmmlearn.hmm import MultinomialHMM

from rspnlib.models.iohmm import IoHmmRSPN, ContinuousIoHmmRSPN

EngineDataset = namedtuple(
    "EngineDataset",
    [
        "inputs_raw",
        "states_raw",
        "observations_raw_discrete",
        "observations_raw_continuous",
        "spn_data_discrete",
        "spn_data_continuous",
    ],
)


def get_engine_parameters():
    # Specify the IOHMM parameters
    prior_proba = [0.7, 0.3]
    start_proba = [[0.9, 0.1], [0.7, 0.3]]
    trans_proba = [[[0.9, 0.1], [0.5, 0.5]], [[0.6, 0.4], [0.1, 0.9]]]

    # Discrete case
    emission_proba = [
        [[0.7, 0.2, 0.05, 0.05], [0.2, 0.78, 0.01, 0.01]],
        [[0.09, 0.12, 0.05, 0.74], [0.1, 0.68, 0.02, 0.2]],
    ]

    # Continuous case
    emission_means = [[[50, 50], [40, 70]], [[70, 75], [60, 100]]]
    emission_covars = [
        [[[15, 0], [0, 15]], [[15, 0], [0, 15]]],
        [[[15, 0], [0, 15]], [[15, 0], [0, 15]]],
    ]

    prior_proba = torch.tensor(prior_proba, dtype=torch.float64)
    start_proba = torch.tensor(start_proba, dtype=torch.float64)
    trans_proba = torch.tensor(trans_proba, dtype=torch.float64)
    emission_proba = torch.tensor(emission_proba, dtype=torch.float64)
    emission_means = torch.tensor(emission_means, dtype=torch.float64)
    emission_covars = torch.tensor(emission_covars, dtype=torch.float64)

    return (
        prior_proba,
        start_proba,
        trans_proba,
        emission_proba,
        emission_means,
        emission_covars,
    )


def create_engine_iohmm():
    (
        prior_proba,
        start_proba,
        trans_proba,
        emission_proba,
        _,
        _,
    ) = get_engine_parameters()
    reference_io_hmm_rspn = IoHmmRSPN(
        prior_proba, start_proba, trans_proba, emission_proba
    )
    return reference_io_hmm_rspn


def create_continuous_engine_iohmm():
    (
        prior_proba,
        start_proba,
        trans_proba,
        _,
        emission_means,
        emission_covars,
    ) = get_engine_parameters()
    reference_io_hmm_rspn = ContinuousIoHmmRSPN(
        prior_proba, start_proba, trans_proba, emission_means, emission_covars
    )
    return reference_io_hmm_rspn


def generate_engine_dataset(num_samples=1000, seq_length=12):
    """
    Generates an engine dataset

    :param num_samples: The number of samples in the dataset
    :param seq_length: The length of the temporal sequences to generate
    """
    (
        prior_proba,
        start_proba,
        trans_proba,
        emission_proba,
        emission_means,
        emission_covars,
    ) = get_engine_parameters()
    num_continuous_observations = emission_means.shape[-1]

    inputs_raw = []
    states_raw = []
    observations_raw_discrete = []
    observations_raw_continuous = []

    prior_dist = torch.distributions.Categorical(prior_proba)
    for _ in range(num_samples):
        prev_state = None
        inputs_raw_sample = []
        states_raw_sample = []
        observations_raw_discrete_sample = []
        observations_raw_continuous_sample = []

        current_discrete_sample = []
        current_continuous_sample = []
        for _ in range(seq_length):
            current_input = prior_dist.sample().item()
            inputs_raw_sample.append(current_input)
            if prev_state is None:
                start_dist = torch.distributions.Categorical(start_proba[current_input])
                current_state = start_dist.sample().item()
            else:
                trans_dist = torch.distributions.Categorical(
                    trans_proba[prev_state][current_input]
                )
                current_state = trans_dist.sample().item()
            states_raw_sample.append(current_state)
            prev_state = current_state
            discrete_obs_dist = torch.distributions.Categorical(
                emission_proba[current_input][current_state]
            )
            current_discrete_observation = discrete_obs_dist.sample().tolist()
            observations_raw_discrete_sample.append(current_discrete_observation)
            continuous_obs_dist = torch.distributions.MultivariateNormal(
                emission_means[current_input][current_state],
                covariance_matrix=emission_covars[current_input][current_state],
            )
            current_continuous_observation = continuous_obs_dist.sample().tolist()
            observations_raw_continuous_sample.append(current_continuous_observation)

        inputs_raw.append(inputs_raw_sample)
        states_raw.append(states_raw_sample)
        observations_raw_discrete.append(observations_raw_discrete_sample)
        observations_raw_continuous.append(observations_raw_continuous_sample)

    spn_data_discrete = np.full((num_samples, 3 * seq_length), np.nan)
    spn_data_discrete[:, 0::3] = np.array(inputs_raw).reshape(num_samples, seq_length)
    spn_data_discrete[:, 2::3] = np.array(observations_raw_discrete).reshape(
        num_samples, seq_length
    )

    observations_raw_continuous = np.array(observations_raw_continuous)
    spn_data_continuous = np.full(
        (num_samples, (2 + num_continuous_observations) * seq_length), np.nan
    )
    spn_data_continuous[:, 0 :: 2 + num_continuous_observations] = np.array(
        inputs_raw
    ).reshape(num_samples, seq_length)
    for i in range(num_continuous_observations):
        spn_data_continuous[:, 2 + i :: 2 + num_continuous_observations] = np.array(
            observations_raw_continuous[:, :, i]
        )

    return EngineDataset(
        inputs_raw=inputs_raw,
        states_raw=states_raw,
        observations_raw_discrete=observations_raw_discrete,
        observations_raw_continuous=observations_raw_continuous,
        spn_data_discrete=spn_data_discrete,
        spn_data_continuous=spn_data_continuous,
    )


def generate_engine_datasets(
    seeds=range(42, 42 + 5), development_samples=1000, test_samples=1000, seq_length=12
):
    """
    Generates engine datasets with the specified number of seed partitions, samples and sequence lengths

    :param seeds: The seeds to use when generating partitions
    :param development_samples: The number of samples in the development set (to be used for training and validation)
    :param test_samples: The number of samples in the withheld test set
    :param seq_length: The length of the temporal sequences to generate
    """
    development_engine_datasets = []
    test_engine_datasets = []

    for seed in seeds:
        torch.manual_seed(seed)
        development_engine_dataset = generate_engine_dataset(
            num_samples=development_samples, seq_length=seq_length
        )
        development_engine_datasets.append(development_engine_dataset)
        test_engine_dataset = generate_engine_dataset(
            num_samples=test_samples, seq_length=seq_length
        )
        test_engine_datasets.append(test_engine_dataset)

    return development_engine_datasets, test_engine_datasets
