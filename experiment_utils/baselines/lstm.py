import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from tqdm.auto import tqdm

from rspnlib.evaluation import compute_metrics, compute_regression_metrics

from torch import optim

DEVICE = "cpu"


class OneStepPredictionDataset(data.Dataset):
    def __init__(
        self,
        inputs_raw,
        observations_raw,
        num_inputs,
        num_observations,
        action_prediction=False,
        continuous=False,
    ):
        if action_prediction and continuous:
            raise NotImplementedError("Continuous action prediction not implemented")

        super(OneStepPredictionDataset, self).__init__()
        inputs_raw = np.array(inputs_raw)
        observations_raw = np.array(observations_raw)
        obs_multiplier = 1 if not continuous else num_observations
        seq_length = len(inputs_raw[0])
        all_data = []
        all_labels = []
        for input_seq_length in range(1, seq_length):
            current_inputs = np.array(inputs_raw[:, : input_seq_length + 1])
            current_observations = np.array(
                observations_raw[
                    :,
                    : (input_seq_length + 1) * obs_multiplier,
                ]
            )

            # Retrieve the label and remove the associated data from the time series
            if action_prediction:
                current_labels = current_inputs[:, -1]
                current_inputs = current_inputs[:, :-1]
            else:
                current_labels = np.array(current_observations[:, -obs_multiplier:])
            current_observations = current_observations[:, :-obs_multiplier]
            inputs_padded = np.pad(
                current_inputs,
                (
                    (0, 0),
                    (
                        0,
                        seq_length
                        - input_seq_length
                        - (1 if not action_prediction else 0),
                    ),
                ),
                constant_values=num_inputs,
            )
            inputs_encoded = F.one_hot(torch.tensor(inputs_padded), num_inputs + 1)

            if continuous:
                # Insert ones to indicate that the observations were measured at these time steps
                current_observations = np.insert(
                    current_observations,
                    np.arange(
                        num_observations,
                        len(current_observations[0]) + 1,
                        num_observations,
                    ),
                    1,
                    axis=1,
                )
                # Pad the observations with zeros (including the values for the indicator variable)
                observations_padded = np.pad(
                    current_observations,
                    (
                        (0, 0),
                        (0, (seq_length - input_seq_length) * (num_observations + 1)),
                    ),
                    constant_values=0,
                )
                observations_encoded = torch.tensor(
                    observations_padded.reshape(-1, seq_length, num_observations + 1)
                )
            else:
                # Pad the observations with a special not measured value
                observations_padded = np.pad(
                    current_observations,
                    ((0, 0), (0, seq_length - input_seq_length)),
                    constant_values=num_observations,
                )
                observations_encoded = F.one_hot(
                    torch.tensor(observations_padded), num_observations + 1
                )

            data_concat = torch.cat((inputs_encoded, observations_encoded), -1)
            all_data.append(data_concat)
            all_labels.append(torch.tensor(current_labels))

        self.all_data = torch.cat(all_data, 0).double()
        self.all_labels = torch.cat(all_labels, 0).double()

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        return self.all_data[index], self.all_labels[index]


class LstmPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LstmPredictor, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False
        )
        self.output_layers = nn.Sequential(nn.Linear(hidden_size, output_size))

    def forward(self, inputs):
        lstm_out, _ = self.lstm(inputs)
        lstm_final_out = lstm_out[:, -1, :]
        raw_predictions = self.output_layers(lstm_final_out)
        return raw_predictions
    
    
def train_eval_lstm(
    train_dataset,
    test_dataset,
    num_inputs,
    num_observations,
    seed,
    lstm_hidden_size=32,
    num_epochs=30,
    action_prediction=False,
    continuous=False,
):
    """
    Trains an LSTM on the train dataset and evaluates it on the test dataset

    :param train_dataset: The train dataset
    :param test_dataset: The test dataset
    :param num_inputs: The number of inputs
    :param num_observations: The number of observations if discrete or their dimension if continuous
    :param seed: The seed to use for reproducibility
    :param num_epochs: The number of epochs to train for
    :param lstm_hidden_size: The hidden size of the LSTM
    :param action_prediction: Whether to predict treatment actions instead of the outcomes
    :param continuous: Whether the observations are continuous
    """
    print(
        f"——————————[ Running LSTM classification evaluation for seed {seed} ]——————————"
    )
    print("Training LSTM...")
    torch.manual_seed(seed)

    dataset = OneStepPredictionDataset(
        train_dataset.inputs_raw,
        train_dataset.observations_raw_discrete
        if not continuous
        else train_dataset.observations_raw_continuous.reshape((-1, train_dataset.observations_raw_continuous.shape[1]*train_dataset.observations_raw_continuous.shape[2])),
        num_inputs,
        num_observations,
        action_prediction=action_prediction,
        continuous=continuous,
    )
    data_loader = data.DataLoader(dataset, batch_size=128, num_workers=0, shuffle=True)
    test_dataset = OneStepPredictionDataset(
        test_dataset.inputs_raw,
        test_dataset.observations_raw_discrete
        if not continuous
        else test_dataset.observations_raw_continuous.reshape((-1, test_dataset.observations_raw_continuous.shape[1]*test_dataset.observations_raw_continuous.shape[2])),
        num_inputs,
        num_observations,
        action_prediction=action_prediction,
        continuous=continuous,
    )
    test_data_loader = data.DataLoader(
        test_dataset, batch_size=128, num_workers=0, shuffle=False
    )

    # Two special values for missing (padded) inputs and observations
    input_size = num_inputs + num_observations + 2
    output_size = num_observations if not action_prediction else num_inputs
    model = LstmPredictor(input_size, lstm_hidden_size, output_size)
    model = model.double()
    if not continuous:
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    losses = []
    for epoch in tqdm(range(num_epochs)):

        model.train()
        epoch_loss = 0
        for i, (inputs, labels) in enumerate(data_loader):
            labels_cpu = labels
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            predictions = model(inputs)
            loss = loss_fn(predictions, labels)
            epoch_loss += loss.to("cpu").item()

            opt.zero_grad()
            loss.backward()
            opt.step()

        losses.append(epoch_loss)

    sns.lineplot(x=range(0, num_epochs), y=losses)
    plt.show()
    print("Training complete.")

    # Evaluation
    print("Evaluating...")
    model.eval()
    all_labels = []
    all_predictions = []
    if not continuous:
        all_argmax_predictions = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_data_loader):
            inputs = inputs.to(DEVICE)

            predictions = model(inputs)
            if not continuous:
                argmax_predictions = torch.argmax(predictions, dim=-1)
                predictions = torch.nn.functional.softmax(predictions, dim=-1).to("cpu")

            all_labels.extend(labels.tolist())
            all_predictions.extend(predictions.tolist())
            if not continuous:
                all_argmax_predictions.extend(argmax_predictions.tolist())

    if not continuous:
        metrics = compute_metrics(
            all_labels,
            all_argmax_predictions,
            all_proba_predictions,
            is_binary=output_size == 2,
        )
        return all_labels, all_predictions, all_argmax_predictions, metrics
    else:
        metrics = compute_regression_metrics(all_labels, all_predictions)
        return all_labels, all_predictions, metrics
