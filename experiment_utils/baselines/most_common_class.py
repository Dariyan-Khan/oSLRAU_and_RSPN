import numpy as np

from rspnlib.evaluation import compute_metrics


def train_eval_most_common_baseline(
    train_dataset,
    test_dataset,
    predicted_variable,
    identifier="UNKNOWN",
    print_report=True,
):
    """
    Trains and evaluates a baseline model that always predicts the most common class.

    :param train_dataset: A dataset used to estimate the most common class.
    :param test_dataset: A dataset on which to evaluate the model.
    :param predicted_variable: The ID of the variable being predicted (starting from 0).
    :param identifier: The identifier for the experiment to be used when printing the report.
    :param print_report: Whether to print report about the evaluation results
    """
    if print_report:
        print(
            f"——————————[ Running most common class classification evaluation for {identifier} ]——————————"
        )

    seq_length = len(train_dataset.inputs_raw[0])
    num_variables = len(train_dataset.spn_data_discrete[0]) // seq_length

    # Create a list for storing the class predictions and ground truth values
    # Format: [(ground_truth, [pred_class_1_prob, pred_class_2_prob, ...]), ...]
    train_data = train_dataset.spn_data_discrete
    all_observations = []

    for input_seq_length in range(1, seq_length):
        seq_data = np.array(train_data[:, : (input_seq_length + 1) * num_variables])
        ground_truth_observations = np.array(
            seq_data[:, -(num_variables - predicted_variable)]
        )
        all_observations.extend(ground_truth_observations)

    most_common_observation = int(
        max(set(all_observations), key=all_observations.count)
    )
    unique_observations, counts = np.unique(
        np.array(all_observations), return_counts=True
    )
    observation_counts_dict = dict(zip(unique_observations, counts))
    observation_counts = []
    for obs in range(int(max(unique_observations) + 1)):
        observation_counts.append(observation_counts_dict[obs])
    observation_counts = np.array(observation_counts)
    observation_probas = observation_counts / observation_counts.sum()

    all_labels = []
    test_data = test_dataset.spn_data_discrete
    for input_seq_length in range(1, seq_length):
        seq_data = np.array(test_data[:, : (input_seq_length + 1) * num_variables])
        ground_truth_observations = np.array(
            seq_data[:, -(num_variables - predicted_variable)]
        )
        all_labels.extend(ground_truth_observations)
    all_labels = [int(l) for l in all_labels]

    all_argmax_predictions = [most_common_observation] * len(all_labels)
    num_classes = int(max(all_labels)) + 1
    proba = observation_probas
    all_proba_predictions = [proba] * len(all_labels)

    metrics = compute_metrics(
        all_labels,
        all_argmax_predictions,
        all_proba_predictions,
        is_binary=num_classes == 2,
        print_report=print_report,
    )

    return all_labels, all_proba_predictions, all_argmax_predictions, metrics
