import sys
sys.path.append('path/to/SPFlow')

import numpy as np
import pandas as pd
from spn.structure.Base import Context
from spn.algorithms.oSLRAU import oSLRAU, oSLRAUParams
from spn.structure.leaves.parametric.Parametric import Gaussian, In_Latent
from spn.algorithms.LearningWrappers import learn_parametric
from spn.io.Graphics import plot_spn
from spn.algorithms.Inference import log_likelihood
from spn.algorithms.TransformStructure import Prune ,Prune_oSLRAU

def run_oSLRAU(dataset, update_after_no_min_batches, prune_after):
    
    data = get_data(dataset)
    data = np.where(np.isnan(data), np.ma.array(data, mask=np.isnan(data)).mean(axis=0), data)

    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(data, test_size=0.33, random_state=42)

    # make first mini_batch from data
    mini_batch_size = 50
    first_mini_batch = data[0:mini_batch_size]

    n = first_mini_batch.shape[1]  # num of variables 
    print(n)
    context = [Gaussian] * n
    ds_context = Context(parametric_types=context).add_domains(first_mini_batch)

    # Learn initial spn 
    spn = learn_parametric(first_mini_batch, ds_context)
    assert False
    plot_spn(spn, 'intitial_spn.pdf')
    print(np.mean(log_likelihood(spn, test_data)))

    oSLRAU_params = oSLRAUParams(mergebatch_threshold=128, corrthresh=0.1, mvmaxscope=1, equalweight=True,
                                 currVals=True)
    no_of_minibatches = int(data.shape[0] / mini_batch_size)
    print(f"==>> no_of_minibatches: {no_of_minibatches}")
    assert False

    # update using oSLRAU

    for i in range(1, no_of_minibatches):
        mini_batch = data[i * mini_batch_size: (i+1) * mini_batch_size]

        update_structure = False
        if update_after_no_min_batches//i == 0:
            print(i)
            update_structure = True
        spn = oSLRAU(spn, mini_batch, oSLRAU_params, update_structure)

        if i == prune_after:
            spn = Prune_oSLRAU(spn)

    print(np.mean(log_likelihood(spn, test_data)))
    plot_spn(spn, 'final_spn.pdf')


def get_data(dataset):

    csv_file_path_hh_power = 'path/to/file'
    csv_file_path_other_power = 'path/to/file'
    csv_file_path_wine_qual = 'oSLRAU_and_RSPN/datasets/winequality-red.csv'
    csv_file_path_japan_vowels = "./oSLRAU_and_RSPN/datasets/japan_vowels.train"
    csv_file_path_libras = "datasets/movement_libras.csv"

    if dataset == 'hh_power':
        file_path = csv_file_path_hh_power
        df = pd.read_csv(file_path, sep=',')

        df = df.iloc[:, 2:6]
        df = df.convert_objects(convert_numeric=True)

        data = df.values
        data = data.astype(float)

        print(data)
        return data

    elif dataset == 'other_power':
        file_path = csv_file_path_other_power
        df = pd.read_csv(file_path, sep=',')

        df = df.iloc[:]
        df = df.convert_objects(convert_numeric=True)

        data = df.values
        data = data[0:-1]
        data = data.astype(float)

        print(data)
        return data

    elif dataset == 'wine_qual':
        file_path = csv_file_path_wine_qual
        df = pd.read_csv(file_path, sep=';')

        df = df.iloc[:]
        # df = df.convert_objects(convert_numeric=True)
        df = df.apply(pd.to_numeric)

        data = df.values
        data = data[0:-1]
        data = data.astype(float)

        print(data)
        return data
    
    elif dataset == 'japan_vowels':
        file_path = csv_file_path_japan_vowels
        df = pd.read_csv(file_path, sep='')

        df = df.iloc[:]
        # df = df.apply(pd.to_numeric)

        data = df.values
        data = data[0:-1]
        data = data.astype(float)
        return data
    
    elif dataset == 'libras':
        file_path = csv_file_path_libras
        df = pd.read_csv(file_path, sep=',')

        df = df.iloc[:]
        df = df.apply(pd.to_numeric)

        data = df.values

        print(f"==>> data.shape: {data.shape}")
        print(f"==>> data: {data}")
        #assert False
        

        data = data[0:-1]
        data = data.astype(float)

        print(data)
        print(f"==>> data.shape: {data.shape}")
        print(f"==>> data: {data}")
        return data



def main():
    dataset = 'wine_qual'
    update_after_no_min_batches = 15
    prune_after = 50
    run_oSLRAU(dataset, update_after_no_min_batches, prune_after)

if __name__ == "__main__":

    main()
