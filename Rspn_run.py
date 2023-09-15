import sys
sys.path.append('path/to/SPFlow')
#from Rspn_final import *

from Rspn_read_data import read_arff_data, read_csv_data, read_varying_seq_data
from spn.structure.leaves.parametric.Parametric import Gaussian, In_Latent
from spn.algorithms.RSPN import RSPN
from spn.structure.Base import Context
from spn.algorithms.oSLRAU import oSLRAU, oSLRAUParams
from spn.io.Graphics import plot_spn
import numpy as np


def run_rspn(dataset, num_variables, num_latent_variables, num_latent_values, unroll, oSLRAU_params,
             update_after_no_min_batches, full_update, update_leaves, len_sequence_varies):

    data = get_data(dataset)

    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(data, test_size=0.33, random_state=42)
    print("train:", len(train_data))
    print("test:", len(test_data))

    rspn = RSPN(num_variables=num_variables, num_latent_variables=num_latent_variables,
                num_latent_values=num_latent_values)

    # make first mini_batch from data
    mini_batch_size = 10
    first_mini_batch = train_data[0:mini_batch_size]

    dat = first_mini_batch

    print(f"==>> num_variables: {num_variables}")
    print(f"==>> type(num_variables): {type(num_variables)}")
    print(f"==>> dat.shape: {dat[0].shape}")

    if len_sequence_varies:
        dat = [dat[i][0:num_variables] for i in range(len(dat))]
    
    else:
        dat = dat[:, 0:num_variables]

    n = len(dat[0])  # num of variables in each time step
    print(n)
    context = [Gaussian] * n
    # ds_context = Context(parametric_types=context).add_domains(first_mini_batch[:, 0:num_variables])
    ds_context = Context(parametric_types=context).add_domains(dat)

    print(dat)
    assert False

    spn, initial_template_spn, top_spn = rspn.build_initial_template(dat, ds_context,
                                                                     len_sequence_varies)

    plot_spn(spn, 'rspn_initial_spn.pdf')
    plot_spn(initial_template_spn, 'rspn_initial_template_spn.pdf')
    plot_spn(top_spn, 'rspn_top_spn.pdf')

    print(np.mean(rspn.log_likelihood(test_data, unroll, len_sequence_varies)))

    # Determine number of mini batches
    if not len_sequence_varies:
        no_of_minibatches = int(train_data.shape[0] / mini_batch_size)

    else:
        no_of_minibatches = int(len(train_data) / mini_batch_size)

    for i in range(1, no_of_minibatches):
        mini_batch = train_data[i * mini_batch_size: (i+1) * mini_batch_size]

        update_template = False
        if i % update_after_no_min_batches == 0:
            print(i)
            update_template = True

        template_spn = rspn.learn_rspn(mini_batch, update_template, oSLRAU_params, unroll, full_update, update_leaves,
                                       len_sequence_varies)
        
    plot_spn(template_spn, 'rspn_final_template.pdf')
    print(np.mean(rspn.log_likelihood(test_data, unroll, len_sequence_varies)))

    unrolled_rspn_full = rspn.get_unrolled_rspn(rspn.get_len_sequence())
    plot_spn(unrolled_rspn_full, 'rspn_unrolled_full.pdf')

    unrolled_rspn = rspn.get_unrolled_rspn(2)
    plot_spn(unrolled_rspn, 'rspn_unrolled_2.pdf')


def get_data(dataset):

    csv_file_path_libras = './oSLRAU_and_RSPN/datasets/movement_libras.csv' 
    csv_file_path_hill_valley = 'path/to/file'
    arff_file_path_eeg_eye = ''

    varying_seq_file_path_japan_vowels = './datasets/japan_vowels.train'

    if dataset == 'libras':
        file_path = csv_file_path_libras
        data = read_csv_data(file_path)
        return data

    elif dataset == 'hill_valley':
        file_path = csv_file_path_hill_valley
        data = read_csv_data(file_path)
        return data

    elif dataset == 'eeg_eye':
        file_path = arff_file_path_eeg_eye
        data = read_arff_data(file_path)
        return data

    elif dataset == 'japan_vowels':
        file_path = varying_seq_file_path_japan_vowels
        data = read_varying_seq_data(file_path)
        return data

    


def main():
    dataset = 'japan_vowels'
    num_variables = 1
    num_latent_variables = 2
    num_latent_values = 2
    unroll = 'backward'
    oSLRAU_params = oSLRAUParams(mergebatch_threshold=10, corrthresh=0.1, mvmaxscope=1, equalweight=True,
                                 currVals=True)
    update_after_no_min_batches = 1
    full_update = False
    update_leaves = True
    len_sequence_varies = True

    run_rspn(dataset, num_variables, num_latent_variables, num_latent_values, unroll, oSLRAU_params,
             update_after_no_min_batches, full_update, update_leaves, len_sequence_varies)



if __name__ == "__main__":

    main()
