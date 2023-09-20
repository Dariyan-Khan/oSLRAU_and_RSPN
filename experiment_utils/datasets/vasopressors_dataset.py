import numpy as np
import pandas as pd

from collections import namedtuple
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

VasoDataset = namedtuple(
    "VasoDataset",
    [
        "inputs_raw",
        "observations_raw_discrete",
        "spn_data_discrete",
    ],
)


def split_vasopressors_dataset(dataset, split_ratio):
    """
    Splits the vasopressors dataset into two sets according to the split ratio

    :param dataset: The VasoDataset to split
    :param split_ratio: The portion of samples in the first set
    """
    num_samples = len(dataset.inputs_raw)
    num_samples_first = int(num_samples * split_ratio)

    return VasoDataset(
        dataset.inputs_raw[:num_samples_first],
        dataset.observations_raw_discrete[:num_samples_first],
        dataset.spn_data_discrete[:num_samples_first],
    ), VasoDataset(
        dataset.inputs_raw[num_samples_first:],
        dataset.observations_raw_discrete[num_samples_first:],
        dataset.spn_data_discrete[num_samples_first:],
    )


def split_vasopressors_dataset(dataset, split_ratio):
    """
    Splits the vasopressors dataset into two sets according to the split ratio

    :param dataset: The VasoDataset to split
    :param split_ratio: The portion of samples in the first set
    """
    num_samples = len(dataset.inputs_raw)
    num_samples_first = int(num_samples * split_ratio)

    return VasoDataset(
        dataset.inputs_raw[:num_samples_first],
        dataset.observations_raw_discrete[:num_samples_first],
        spn_data_discrete[:num_samples_first],
    ), VasoDataset(
        dataset.inputs_raw[num_samples_first:],
        dataset.observations_raw_discrete[num_samples_first:],
        spn_data_discrete[num_samples_first:],
    )


def load_vasopressors_dataset(seq_length, folds=5):
    """
    Loads the vasopressors dataset derived from MIMIC

    :param seq_length: The length of a sequence to use for each patient
    :param folds: The number of folds to split the dataset into
    """

    def construct_vaso_dataset(raw_data):
        inputs_raw = raw_data[:, 0::3].astype(int)
        observations_raw_discrete = raw_data[:, 2::3].astype(int)
        return VasoDataset(
            inputs_raw=inputs_raw,
            observations_raw_discrete=observations_raw_discrete,
            spn_data_discrete=raw_data,
        )

    # Range of hours to use for each patient (additional hour for shifting the data)
    HOURS_RANGE = range(1, seq_length + 1 + 1)

    vaso_df = pd.read_csv(
        "../data/rebinned_sepsis_24hrs_20220122_final.csv", low_memory=False
    )

    # Select only patients who do not start on vasopressors
    vaso_startoff_df = vaso_df[
        (vaso_df["hour"] == 1) & (vaso_df["vaso_indicator"] == 0)
    ]
    stay_ids = vaso_startoff_df.stay_id.unique()

    # for each patient, extract their trajectory [(action, state, observation), ...] over time
    vaso_data = []
    for stay_id in tqdm(stay_ids):
        # The trajectory for the given stay
        stay_trajectory = []

        # Select entries for this stay
        stay_df = vaso_df[vaso_df.stay_id == stay_id]
        stay_df = stay_df.dropna(subset=["min_mbp", "total_fluids"])

        # Initialize tracking variable for data shift
        prev_action = None

        for hour in HOURS_RANGE:
            # Select the row for this hour
            hour_df = stay_df[stay_df.hour == hour]
            if hour_df.empty:
                break

            # Use vasopressor tratment as the action
            action = 0 if hour_df.iloc[0]["vaso_indicator"] == 0 else 1

            if prev_action is not None:
                # Use manual categories for observations, combining min_mbp and total_fluids
                min_mbp = hour_df.iloc[0]["min_mbp"]
                observation_mbp = 0 if min_mbp > 75 else 1 if min_mbp > 65 else 2
                total_fluids = hour_df.iloc[0]["total_fluids"]
                observation_fluids = (
                    3 if total_fluids > 9000 else int(total_fluids // 3000)
                )  # Max 4 categories
                observation = 4 * observation_mbp + observation_fluids

                stay_trajectory.append(prev_action)
                stay_trajectory.append(np.nan)  # Latent state
                stay_trajectory.append(observation)

            prev_action = action

        if len(stay_trajectory) // 3 == len(HOURS_RANGE) - 1:
            # Only consider complete data
            vaso_data.append(stay_trajectory)

    vaso_data = np.array(vaso_data)
    print(f"Loaded {len(vaso_data)} data points")

    print("Partitioning into folds...")
    train_vaso_datasets = []
    test_vaso_datasets = []

    kf = KFold(n_splits=folds)
    for train_index, test_index in kf.split(vaso_data):
        train_data = construct_vaso_dataset(vaso_data[train_index])
        train_vaso_datasets.append(train_data)
        test_data = construct_vaso_dataset(vaso_data[test_index])
        test_vaso_datasets.append(test_data)
    print("Done!")

    return train_vaso_datasets, test_vaso_datasets
