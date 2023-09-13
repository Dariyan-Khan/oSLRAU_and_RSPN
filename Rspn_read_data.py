
import arff
import numpy as np
import pandas as pd


def read_arff_data(file_path):
    with open(file_path) as f:
        dataDict = arff.load(f)
        f.close()

    data = np.array(dataDict['data'])

    data = data[:, :-1]
    data = data.astype(np.float)
    print(data)

    return data


def read_csv_data(file_path):
    df = pd.read_csv(file_path, sep=',')
    # df = df.fillna(df.mean())
    df = df.iloc[:]

    # print(df.dtypes)
    #df = df.convert_objects(convert_numeric=True)
    #df = pd.to_numeric(df)
    df = df.apply(pd.to_numeric)
    # pd.to_numeric(df)
    # print(df)

    data = df.values
    print(data.dtype)

    data = data.astype(float)

    data = data[:, 0:-1]
    print(data)

    return data


def read_varying_seq_data(file_path):
    with open(file_path) as file:

        bulk = file.readlines()
        data_list = []
        seq = []
        for line in bulk:
            # when the whole sequence ends
            if line.startswith('\n'):
                data_list.append(seq)
                seq = []
            else:
                time_step = line.split()
                for col_val in time_step:
                    seq.append(col_val)  # add each value of each time step of the whole sequence to a seq list

    for i, seq in enumerate(data_list):
        for j, val in enumerate(seq):
            data_list[i][j] = float(val)

    data = [np.reshape(np.array(xi), (1, -1)) for xi in data_list]
    print(data[0:5])

    return data
