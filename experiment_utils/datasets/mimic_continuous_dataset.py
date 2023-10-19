from cmath import isnan
import numpy as np
import pandas as pd

from collections import namedtuple
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

MimicDataset = namedtuple(
    "MimicDataset",
    ["inputs_raw", "observations_raw_continuous", "spn_data_continuous"],
)

csv = "../data/rebinned_sepsis_24hrs_20220122_final.csv"


def load_mimic_dataset(seq_length, folds=5):
    """
    Loads the vasopressors dataset derived from MIMIC

    :param seq_length: The length of a sequence to use for each patient
    :param folds: The number of folds to split the dataset into
    """

    def construct_mimic_dataset(raw_data, observations_data):
        inputs_raw = raw_data[:, 0::5].astype(int)
        # FIXME: Shape possibly inconsistent with other datasets
        observations_raw_continuous = observations_data
        return MimicDataset(
            inputs_raw=inputs_raw,
            observations_raw_continuous=observations_raw_continuous,
            spn_data_continuous=raw_data,
        )

    # Range of hours to use for each patient
    HOURS_RANGE = range(1, seq_length + 1 + 1)  # original data is shifted, prev_action

    vaso_df = pd.read_csv(csv, low_memory=False)  # change this later

    # Select only patients who do not start on vasopressors
    vaso_startoff_df = vaso_df[
        (vaso_df["hour"] == 1) & (vaso_df["vaso_indicator"] == 0)
    ]
    stay_ids = vaso_startoff_df.stay_id.unique()

    # for each patient, extract their trajectory [(action, state, observation), ...] over time
    vaso_data = []
    observations_data = []
    for stay_id in tqdm(stay_ids):
        # The trajectory for the given stay
        stay_trajectory = []
        observation_traj = []

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

            currRow = hour_df.iloc[0]

            # (fluid, vaso) as the action.
            action = None
            vaso = currRow["vaso_indicator"] == 1
            fluid = currRow["bolus_indicator"] == 1
            if not vaso and not fluid:
                action = 0
            elif fluid and not vaso:
                action = 1
            elif vaso and not fluid:
                action = 2
            elif fluid and vaso:
                action = 3

            if prev_action is not None:
                observation = [
                    currRow["heart_rate"],
                    currRow["min_mbp"],
                    currRow["resp_rate"],
                    #currRow['sbp'],
                    #currRow['dbp'],
                    #currRow['spo2']
                ]

                isValid = True
                for val in observation:
                    if isnan(val):
                        isValid = False
                        break
                if isValid:
                    stay_trajectory.append(prev_action)
                    stay_trajectory.append(np.nan)  # Latent state
                    stay_trajectory.extend(observation)
                    observation_traj.append(observation)

            prev_action = action

        if len(stay_trajectory) // 5 == len(HOURS_RANGE) - 1:
            # Only consider complete data
            vaso_data.append(stay_trajectory)
            observations_data.append(observation_traj)

    vaso_data = np.array(vaso_data)
    observations_data = np.array(observations_data)
    print(f"Loaded {len(vaso_data)} data points")
    print("Partitioning into folds...")
    train_vaso_datasets = []
    test_vaso_datasets = []

    kf = KFold(n_splits=folds)
    for train_index, test_index in kf.split(vaso_data):
        train_data = construct_mimic_dataset(
            vaso_data[train_index], observations_data[train_index]
        )
        train_vaso_datasets.append(train_data)
        test_data = construct_mimic_dataset(
            vaso_data[test_index], observations_data[test_index]
        )
        test_vaso_datasets.append(test_data)
    print("Done!")

    return train_vaso_datasets, test_vaso_datasets


if __name__ == "__main__":
    csv = "rebinned_sepsis_24hrs_20220122_final.csv"
    load_mimic_dataset(23, folds=5)
