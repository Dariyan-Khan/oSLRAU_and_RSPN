# oSLRAU and RSPN

oSLRAU module in SPFlow library implements the Online Structure Learning with Running Average Update algorithm (**oSLRAU**) 
which can be used for learning structure and parameters of an SPN in an online fashion.

RSPN module in SPFlow library uses the **oSLRAU** algorithm to learn the structure and parameters of the Template Network
of Recurrent Sum-Product-Networks (**RSPN**)  

## Getting Started

Use the forked version of [SPFlow](https://github.com/c0derzer0/SPFlow) for installation.

## Using oSLRAU Module

Look at *oSLRAU_AN_RSPN/datasets* folder for a list of sample datasets to use with oSLRAU algorithm. 
*oSLRAU_AN_RSPN/oSLRAU_run.py* contains example code for running the algorithm
```python
    data = get_data(dataset) # load data as an np ndarray
 ```
Learn initial spn using first mini batch of data using learn_spn algorithm
```python
    # make first mini_batch from data
    mini_batch_size = 50
    first_mini_batch = data[0:mini_batch_size]

    n = first_mini_batch.shape[1]  # num of variables 
    print(n)
    context = [Gaussian] * n
    ds_context = Context(parametric_types=context).add_domains(first_mini_batch)

    # Learn initial spn 
    spn = learn_parametric(first_mini_batch, ds_context)
    
    # You can plot and get the log likelihood on some test data
    plot_spn(spn, 'intitial_spn.pdf')
    print(np.mean(log_likelihood(spn, test_data)))
```
Specify the parameters for oSLRAU algorithm
```python
    oSLRAU_params = oSLRAUParams(mergebatch_threshold=128, corrthresh=0.1, mvmaxscope=1, equalweight=True,
                                 currVals=True)
```                                
Use rest of the mini batches for updating the structure of learned spn from learn spn algorithm
```python
    no_of_minibatches = int(data.shape[0] / mini_batch_size)

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
            
    # Use SPFlow modules for analysis on the learned sturcture
    print(np.mean(log_likelihood(spn, test_data)))
    plot_spn(spn, 'final_spn.pdf')
```
## Using RSPN Module

Look at *oSLRAU_AN_RSPN/datasets* folder for a list of sample datasets to learn RSPNs. 
*oSLRAU_AN_RSPN/RSPN_run.py* and *oSLRAU_AN_RSPN/RSPN_read_data.py* contains example code for running the algorithm
```python
    # If sequence length in the data does not vary, load dataset as an np ndarray
    # If sequence length in the data varies, load dataset as a list of ndarrays
    data = get_data(dataset) 
 ``` 
 Specify the parameters for RSPN
```python
    num_variables = 1
    num_latent_variables = 2
    num_latent_values = 2
    unroll = 'backward'
    full_update = False
    update_leaves = True
    len_sequence_varies = False
    oSLRAU_params = oSLRAUParams(mergebatch_threshold=128, corrthresh=0.1, mvmaxscope=1, equalweight=True,
                                 currVals=True)
```
Initialise RSPN 
```python
   rspn = RSPN(num_variables=num_variables, num_latent_variables=num_latent_variables,
                num_latent_values=num_latent_values)
```
Initialise template structure using first mini batch of data
```python
    # make first mini_batch from data
    mini_batch_size = 50
    first_mini_batch = data[0:mini_batch_size]

    # num of variables in each time step
    n = first_mini_batch.shape[1]  # account for change in the type of data, if length os sequence varies
    print(n)
    context = [Gaussian] * n
    ds_context = Context(parametric_types=context).add_domains(first_mini_batch[:, 0:num_variables]

    # Build initial template
     spn, initial_template_spn, top_spn = rspn.build_initial_template(first_mini_batch, ds_context,
                                                                       len_sequence_varies)
    
    # You can plot and get the log likelihood on some test data
    plot_spn(spn, 'rspn_initial_spn.pdf')
    plot_spn(initial_template_spn, 'rspn_initial_template_spn.pdf')
    plot_spn(top_spn, 'rspn_top_spn.pdf')

    print(np.mean(rspn.log_likelihood(test_data, unroll, len_sequence_varies=False)))
```                                
Use rest of the mini batches for updating the structure of template
```python
    no_of_minibatches = int(data.shape[0] / mini_batch_size)  # account for change in the type of data, if length os sequence varies

    # update using oSLRAU
    for i in range(1, no_of_minibatches):
        mini_batch = train_data[i * mini_batch_size: (i+1) * mini_batch_size]

        update_template = False
        if i % update_after_no_min_batches == 0:
            print(i)
            update_template = True

        template_spn = rspn.learn_rspn(mini_batch, update_template, oSLRAU_params, unroll, full_update, update_leaves,
                                       len_sequence_varies)
           
            
    # You can plot and get the log likelihood on some test data
    plot_spn(template_spn, 'rspn_final_template.pdf')
    print(np.mean(rspn.log_likelihood(test_data, unroll, len_sequence_varies)))

```
You can also plot the unrolled spn
```python
   unrolled_rspn_full = rspn.get_unrolled_rspn(rspn.get_len_sequence())
   plot_spn(unrolled_rspn_full, 'rspn_unrolled_full.pdf')

   unrolled_rspn = rspn.get_unrolled_rspn(2)
   plot_spn(unrolled_rspn, 'rspn_unrolled_2.pdf')
 ```
## Papers implemented
Agastya Kalra, Abdullah Rashwan, Wilson Hsu, Pascal Poupart, Prashant Doshi, George Trimponias. 
"Online Structure Learning for Feed-Forward andRecurrent Sum-Product Networks". 
Advances in Neural Information Processing Systems 31 (NIPS 2018)


    

