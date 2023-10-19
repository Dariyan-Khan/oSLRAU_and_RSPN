import numpy as np

from collections import namedtuple
from hmmlearn.hmm import MultinomialHMM


WeatherDataset = namedtuple(
    "WeatherDataset", ["states_raw", "observations_raw", "spn_data"]
)


def create_weather_hmm():
    start_proba = [0.65, 0.35]
    trans_proba = [[0.7, 0.3], [0.25, 0.75]]
    emission_proba = [[0.2, 0.8], [0.6, 0.4]]

    reference_hmm = MultinomialHMM(n_components=2)
    reference_hmm.startprob_ = np.array(start_proba)
    reference_hmm.transmat_ = np.array(trans_proba)
    reference_hmm.emissionprob_ = np.array(emission_proba)

    return reference_hmm


def generate_weather_dataset(
    reference_hmm, random_gen, num_samples=1000, seq_length=12
):
    """
    Generates a weather dataset

    :param reference_hmm: The reference HMM to use for generating the dataset
    :param random_gen: The random generator used for sampling
    :param num_samples: The number of samples in the dataset
    :param seq_length: The length of the temporal sequences to generate
    """
    states_raw = []
    observations_raw = []
    for i in range(num_samples):
        sample_observations, sample_states = reference_hmm.sample(
            n_samples=seq_length, random_state=random_gen
        )
        sample_observations = sample_observations.flatten()
        states_raw.append(sample_states)
        observations_raw.append(sample_observations)
    states_raw = np.array(states_raw)
    observations_raw = np.array(observations_raw)
    spn_data = np.full((num_samples, 2 * seq_length), np.nan)
    spn_data[:, 1::2] = observations_raw
    return WeatherDataset(
        states_raw=states_raw,
        observations_raw=observations_raw,
        spn_data=spn_data,
    )


def generate_weather_datasets(
    seeds=range(42, 42 + 5), development_samples=1000, test_samples=1000, seq_length=12
):
    """
    Generates weather datasets with the specified number of seed partitions, samples and sequence lengths

    :param seeds: The seeds to use when generating partitions
    :param development_samples: The number of samples in the development set (to be used for training and validation)
    :param test_samples: The number of samples in the withheld test set
    :param seq_length: The length of the temporal sequences to generate
    """
    development_weather_datasets = []
    test_weather_datasets = []
    reference_hmm = create_weather_hmm()

    for seed in seeds:
        random_gen = np.random.RandomState(seed)

        development_dataset = generate_weather_dataset(
            reference_hmm,
            random_gen,
            num_samples=development_samples,
            seq_length=seq_length,
        )
        test_dataset = generate_weather_dataset(
            reference_hmm,
            random_gen,
            num_samples=test_samples,
            seq_length=seq_length,
        )

        development_weather_datasets.append(development_dataset)
        test_weather_datasets.append(test_dataset)

    return development_weather_datasets, test_weather_datasets
