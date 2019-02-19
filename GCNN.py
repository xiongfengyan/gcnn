#

from lib import models, dataLoading 

import tensorflow as tf
import matplotlib.pyplot as plt

# # 1 Load Data.
# train_test_dataset(None)
# training, validation, test data.
# [vertices_geometry, vertices_fourier, adjacencies, labels]
# train_dataset,val_dataset,test_dataset=load_data('./data/test_data_batch_mini.json')
# train_dataset, val_dataset, test_dataset = input_data.load_data('./data/testExtra0725r.json', dataSeparation=[0.6, 0.2, 0.2]) #
train_dataset, val_dataset, test_dataset = dataLoading.load_data('./lib/data/gz1124rrri.json', dataSeparation=[0.8, 0.18, 0.02])     # test228r
# Number of samples.
n_train = train_dataset[4]
# Number of classes.
n_class = len(set(train_dataset[2]))
# GeometryDescriptors or Fourier
i_type  = 0

# print the Graph structure.
if True:
    print('Graph structure: {0}x{0} Graph with {1} dimensionaal vecter.'.format(train_dataset[1].shape[1], train_dataset[0].shape[2]))
    print('labels set: {}'.format(set(train_dataset[2])))
    print('  train dataset:')
    print('    vertices   : {0}'.format(train_dataset[0].shape))
    print('    adjacencies: {0}'.format(train_dataset[1].shape))
    print('    labels     : {0}'.format(train_dataset[2].shape))
    print('  validation dataset:')
    print('    vertices   : {0}'.format(val_dataset[0].shape))
    print('    adjacencies: {0}'.format(val_dataset[1].shape))
    print('    labels     : {0}'.format(val_dataset[2].shape))
    print('  test dataset:')
    print('    vertices   : {0}'.format(test_dataset[0].shape))
    print('    adjacencies: {0}'.format(test_dataset[1].shape))
    print('    labels     : {0}'.format(test_dataset[2].shape))

# # 2 Graph Convolution.
params = dict()
# The dimensions of vectives.
params['Vs']             = train_dataset[0].shape[2]

# The size of adjacencies (Number of vectices).
params['As']             = train_dataset[1].shape[1]

# The parameters of convolation and pooling layer. Two layers
params['Fs']             = [16, 16, 16, 16, 16]
params['Ks']             = [3, 3, 3, 3, 3]
params['Ps']             = [1, 1, 1, 1, 1]

# The parameters of full connection layer.
params['Cs']             = [16, n_class]

params['filter']         = 'monomial'
params['brelu']          = 'b1relu'
params['pool']           = 'm1pool'
params['num_epochs']     = 150
params['batch_size']     = 100
params['decay_steps']    = n_train / params['batch_size']
params['eval_frequency'] = 50


# Hyper-parameters.
params['regularization'] = 5e-4
params['dropout']        = 0.5
params['learning_rate']  = 0.001
params['decay_rate']     = 0.95
params['momentum']       = 0
params['dir_name']       = 'buildings'
# momentum,learning_rate = [0.9, 0.01]
# momentum,learning_rate = [0,   0.002, 0.003, 0.001]


# # 3 TRAINING AND EVALUTING NETWORK
  # We often want to monitor:
  # 1) The convergence, i.e. the training loss and the classification accuracy on the validation set.
  # 2) The performance, i.e. the classification accuracy on the testing set (to be compared with the training set accuracy to spot overfitting).

# DIFINE THE MODEL COLLECTION
# multi train networks.
# The `model_perf` class in [models.py](models.py) can be used to compactly evaluate multiple models.
model_perf = models.model_perf()

if True:
    # Test depth of network structures.
    for i in [1,2,3,4,5,6,7,8]:
        params['Fs']             = [16]*i
        params['Ks']             = [3] *i
        params['Ps']             = [1] *i
        name = 'depth={}'.format(i)
        params['dir_name'] = name
        model_perf.test(models.gcnn(**params), name, params, train_dataset, val_dataset, test_dataset)

    model_perf.show()

# Test input variable important
if False:
    # DEFINE THE CNN ARCHITECTURE
    params['Fs']             = [16, 16, 16, 16, 16]
    params['Ks']             = [3, 3, 3, 3, 3]
    params['Ps']             = [1, 1, 1, 1, 1]
    name = 'newfs_24'

    import copy
    
    # GET THE DIMENSION OF INPUT VARIABLES
    dim_variables = train_dataset[0].shape[2]

    for i in range(0, dim_variables):
        # Only variable i
        params['Vs']         = 1
        filtered_list        = [i]

        train_copy, val_copy, test_copy = copy.deepcopy(train_dataset), copy.deepcopy(val_dataset), copy.deepcopy(test_dataset)
        
        train_copy[0]        = train_dataset[0][:,:,filtered_list]
        val_copy[0]          = val_dataset[0][:,:,filtered_list]
        test_copy[0]         = test_dataset[0][:,:,filtered_list]
        params['dir_name']   = name + '_one_' + str(i).zfill(2)
        model_perf.test(models.gcnn(**params), name, params, i_type, train_copy, val_copy, test_copy)
        

        # Except variable i
        params['Vs']         = dim_variables - 1
        filtered_list        = [j for j in range(dim_variables)]
        del filtered_list[i]

        train_copy[0]        = train_dataset[0][:,:,filtered_list]
        val_copy[0]          = val_dataset[0][:,:,filtered_list]
        test_copy[0]         = test_dataset[0][:,:,filtered_list]

        params['dir_name']   = name + '_exc_' + str(i).zfill(2)
        model_perf.test(models.gcnn(**params), name, params, i_type, train_copy, val_copy, test_copy)

        
        del train_copy, val_copy, test_copy

    model_perf.show()

# Test input variable important
if False:
    # DEFINE THE CNN ARCHITECTURE
    params['Fs']         = [16, 16, 16, 16, 16]
    params['Ks']         = [3, 3, 3, 3, 3]
    params['Ps']         = [1, 1, 1, 1, 1]
    name = 'newfs_21'
    
    params['Vs']         = 21
    filtered_list        = [i for i in range(0,21)]

    import copy
    train_copy, val_copy, test_copy = copy.deepcopy(train_dataset), copy.deepcopy(val_dataset), copy.deepcopy(test_dataset)
    
    train_copy[0]        = train_dataset[0][:,:,filtered_list]
    val_copy[0]          = val_dataset[0][:,:,filtered_list]
    test_copy[0]         = test_dataset[0][:,:,filtered_list]

    for i in [1,2,3,4,5,6]:
        params['Ks']         = [i, i, i, i, i]
        name                 = 'newfs_24_MST3_k={}'.format(i)
        params['dir_name']   = name
        model_perf.test(models.gcnn(**params), name, params, i_type, train_copy, val_copy, test_copy)

    
    del train_copy, val_copy, test_copy

    model_perf.show()

if False:
    # DEFINE THE CNN ARCHITECTURE
    name = 'oldfs_25_3'
    
    params['dir_name']   = name
    model_perf.test(models.gcnn(**params), name, params, i_type, train_dataset, val_dataset, test_dataset)

    model_perf.show()

# Test graph structure and neighbors.
if False:
    # For DT or MST model
    for i in [1,2,3,4,5,6]:
        params['Fs']             = [16, 16, 16, 16, 16]
        params['Ks']             = [i, i, i, i, i]
        params['Ps']             = [1, 1, 1, 1, 1]
        name = 'newfs_24_MST2_k={}'.format(i)
        params['dir_name'] = name
        model_perf.test(models.gcnn(**params), name, params, i_type, train_dataset, val_dataset, test_dataset)

    model_perf.show()

if False:
    if True:
        params['Fs']             = [16, 16, 16, 16, 16]
        params['Ks']             = [3, 3, 3, 3, 3]
        params['Ps']             = [1, 1, 1, 1, 1]
        name = 'newfeatures_24'
        params['dir_name'] = name
        model_perf.test(models.gcnn(**params), name, params, i_type, train_dataset, val_dataset, test_dataset)

    if False:
        params['Fs']             = [16, 16, 16, 16, 16]
        params['Ks']             = [3, 3, 3, 3, 3]
        params['Ps']             = [1, 1, 1, 1, 1]
        name = 'Convolutions_5_16_normalizing'
        params['dir_name'] = name
        model_perf.test(models.gcnn(**params), name, params, i_type, train_dataset, val_dataset, test_dataset)

    if False:
        # test the order K.
        for i in [1,2,3,4,5,6]:
            params['Fs']             = [16, 16, 16, 16, 16]
            params['Ks']             = [i, i, i, i, i]
            params['Ps']             = [1, 1, 1, 1, 1]
            name = 'MST_k_2={}'.format(i)
            params['dir_name'] = name
            model_perf.test(models.gcnn(**params), name, params, i_type, train_dataset, val_dataset, test_dataset)

    if False:
        # test the order K.
        for i in [1,2,3,4,5,6]:
            params['Fs']             = [16, 16, 16, 16, 16]
            params['Ks']             = [i, i, i, i, i]
            params['Ps']             = [1, 1, 1, 1, 1]
            name = 'DT_k={}'.format(i)
            params['dir_name'] = name
            model_perf.test(models.gcnn(**params), name, params, i_type, train_dataset, val_dataset, test_dataset)

    if False:
        # test input features.
        for i in range(0,12):
            if i==1:continue
            i_type = i
            params['Vs']             = train_dataset[i_type].shape[2]
            params['As']             = train_dataset[2].shape[1]
            params['Fs']             = [24, 24, 24, 24]
            params['Ks']             = [3, 3, 3, 3]
            params['Ps']             = [1, 1, 1, 1]
            name = 'Convolutions_features2={}'.format(i)
            params['dir_name'] = name
            model_perf.test(models.gcnn(**params), name, params, i_type, train_dataset, val_dataset, test_dataset)
    if False:
        # test the order K.
        for i in [1,2,3,4,5,6]:
            params['Fs']             = [24, 24, 24, 24]
            params['Ks']             = [i, i, i, i]
            params['Ps']             = [1, 1, 1, 1]
            name = 'Convolutions_k={}'.format(i)
            params['dir_name'] = name
            model_perf.test(models.gcnn(**params), name, params, i_type, train_dataset, val_dataset, test_dataset)

    # The following models will be tested at night.
    if False:
        params['Fs']             = [16, 16]
        params['Ks']             = [3, 3]
        params['Ps']             = [1, 1]
        for rate in [1]:
            name = 'test_{}'.format(rate)
            params['dir_name'] = name
            model_perf.test(models.gcnn(**params), name, params, i_type, train_dataset, val_dataset, test_dataset)
    if False:
        params['Fs']             = [12, 12, 12]
        params['Ks']             = [3, 3, 3]
        params['Ps']             = [1, 1, 1]
        for rate in [1, 2]:
            name = 'Convolutions_3_12_{}_{}'.format(rate, 0.001)
            params['dir_name'] = name
            model_perf.test(models.gcnn(**params), name, params, i_type, train_dataset, val_dataset, test_dataset)
    if False:
        params['Fs']             = [16, 16, 16]
        params['Ks']             = [3, 3, 3]
        params['Ps']             = [1, 1, 1]
        for rate in [3, 4]:
            name = 'Convolutions_3_16_{}_{}'.format(rate, 0.001)
            params['dir_name'] = name
            model_perf.test(models.gcnn(**params), name, params, i_type, train_dataset, val_dataset, test_dataset)
    if False:
        params['Fs']             = [24, 24, 24, 24, 24, 24, 24, 24, 24, 24]
        params['Ks']             = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        params['Ps']             = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        for rate in [3]:
            name = 'Convolutions_10_24_test{}'.format(rate)
            params['dir_name'] = name
            model_perf.test(models.gcnn(**params), name, params, i_type, train_dataset, val_dataset, test_dataset)

    if False:
        params['Fs']             = [36, 36, 36, 36, 36, 36, 36, 36, 36, 36]
        params['Ks']             = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        params['Ps']             = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        for i_type in [0, 1]:
            params['Vs']         = train_dataset[i_type].shape[2]
            name = 'Fourier_Geometry{}'.format(i_type)
            params['dir_name'] = name
            model_perf.test(models.gcnn(**params), name, params, i_type, train_dataset, val_dataset, test_dataset)

    if False:
        params['Fs']             = [16, 16, 16, 16]
        params['Ks']             = [3, 3, 3, 3]
        params['Ps']             = [1, 1, 1, 1]
        for rate in [1, 2]:
            name = 'Convolutions_4_16_{}_{}'.format(rate, 0.001)
            params['dir_name'] = name
            model_perf.test(models.gcnn(**params), name, params, i_type, train_dataset, val_dataset, test_dataset)
    
    if False:
        params['Fs']             = [36, 36, 36, 36, 36]
        params['Ks']             = [3, 3, 3, 3, 3]
        params['Ps']             = [1, 1, 1, 1, 1]
        for rate in [6]:
            name = 'Convolutions_5_36_test{}'.format(rate)
            params['dir_name'] = name
            model_perf.test(models.gcnn(**params), name, params, i_type, train_dataset, val_dataset, test_dataset)

    model_perf.show()
