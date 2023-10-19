from oSLRAU_and_RSPN.mixed_state_hmm import MixedHMM

# Define MHMM
mixed_states = [[0.0, 1.0, 0.5], [10.0, 1.0, 0.5], [20.0, 1.0, 0.5]]
mhmm = MixedHMM(mixed_states)


from oSLRAU_run import get_data
from spn.structure.leaves.parametric.Parametric import Gaussian, Bernoulli,  In_Latent
from spn.structure.Base import Context
from spn.algorithms.LearningWrappers import learn_parametric
from spn.io.Graphics import plot_spn
import numpy as np
from spn.algorithms.Inference import log_likelihood
from sklearn.model_selection import train_test_split
from spn.algorithms.oSLRAU import oSLRAU, oSLRAUParams
from spn.algorithms.RSPN import RSPN
from spn.algorithms.TransformStructure import Prune ,Prune_oSLRAU

from hmmlearn import hmm
from spn.algorithms.Inference import log_likelihood


n_states = 3
n_dim = 2

num_time_steps_mix = 5
num_seq_mix = 20
rspn_data_mix= np.array([mhmm.sample_obs(num_time_steps_mix) for _ in range(num_seq_mix)])
rspn_data_mix.shape

rspn_data_mix = rspn_data_mix.reshape((rspn_data_mix.shape[0], -1))
print(f"==>> rspn_data_mix.shape: {rspn_data_mix.shape}")



num_variables = num_time_steps_mix * n_dim
num_latent_variables = n_states
num_latent_values = n_dim
unroll = 'backward'
full_update = False
update_leaves = True
len_sequence_varies = False
oSLRAU_params = oSLRAUParams(mergebatch_threshold=10, corrthresh=0.7, mvmaxscope=1, equalweight=True, currVals=True)


mini_batch_size = 2
update_after_no_min_batches = 2




def train_rspn_mix(train_data, test_data, rspn=None):

    if len(train_data.shape) == 3: train_data = np.squeeze(train_data)
    if len(test_data.shape) == 3: test_data = np.squeeze(test_data)

    if rspn is None:
    
        rspn = RSPN(num_variables=num_variables, num_latent_variables=num_latent_variables, num_latent_values=num_latent_values)
        first_mini_batch = train_data[0:mini_batch_size]
        n = first_mini_batch.shape[1]
        print(n)
        
        context = [Gaussian, Bernoulli]*(int(n/2)) # Bernoulli
        ds_context = Context(parametric_types=context).add_domains(first_mini_batch[:, :num_variables])
        spn, initial_template_spn, top_spn = rspn.build_initial_template(first_mini_batch, ds_context, len_sequence_varies)

    
    no_of_minibatches = int(train_data.shape[0] / mini_batch_size)

    print(f"no of minibatches: {no_of_minibatches}")

    
    for i in range(1, no_of_minibatches):
        mini_batch = train_data[i * mini_batch_size: (i+1) * mini_batch_size]
    
        update_template = False
        
        if i % update_after_no_min_batches == 0:
            print(i)
            update_template = True
    
        template_spn = rspn.learn_rspn(mini_batch, update_template, oSLRAU_params, unroll, full_update, update_leaves,
                                       len_sequence_varies)
           
            
    test_ll = np.mean(rspn.log_likelihood(test_data, unroll, len_sequence_varies))


    return test_ll, rspn, template_spn



def avg_ll_mix(data, num_epochs=1, do_plot_spn=True):
    ll_list = []
    rspn=None
    train_data, val_data= train_test_split(data, test_size=0.1, random_state=42)
    for i in range(num_epochs):
        print(train_data.shape)
        
        
        epoch_ll, rspn, template_spn = train_rspn_mix(train_data, val_data, rspn)        
        print(f"epoch_ll: {epoch_ll}")
        
        ll_list.append(epoch_ll)

    average_ll = np.mean(np.array(ll_list))
    print(f" \n\n\n average ll: {average_ll}\n\n\n")

    if do_plot_spn:
        plot_spn(template_spn, 'rspn_final_template.pdf')
    
    return rspn


if __name__ == "__main__":
    rspn_mix = avg_ll_mix(rspn_data_mix, do_plot_spn=True)

