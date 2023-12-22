import util
from numba import jit
import re
import os
import numpy as np
from filelock import FileLock


blockDict = {}
layerNames = []
pca_components = 0
ica_components = 0
num_channel_chunks = 0
use_final_hidden_layer = True # defaulted to true as it was included in the original paper
num_final_hidden_layer_chunks = 0 # returns all values from the final hidden layer
pool_layer_threshold = 0.5 # threshold at which activation values from the pooling layer are stored (0.5 used in paper)
use_pool_layer_threshold = True  # threshold used in paper, therefore defaulted to true here
block_map = {}

# CBIR - Content Based Image Retrieval - descriptors generation for image retrieval by analyzing activations of deep neural network layers
# https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9451541&casa_token=SjYhfTbe85cAAAAA:DEj6uWr0--chlnptQokS0JLtEYxiaeu_KjCRGYidZfZQFz4UNo1jJv0-VOAYxJLFcfDk61rFm-li&tag=1
# github: https://github.com/pstaszewski/cbir202004

# runs without using numba, but is slow when dealing with 1000 instances at a time. Options:
# 1. Sort out numba (needs extra type information to run on GPU)
# 2. Use CuPy instead of numpy to get it running on GPU
# 3. Can the code be written more efficiently?

def extractSingleLayer(key, layer, reset=False, layerActivationReduction='cbir', model=None, inData=None, y_inData=None):
    # store the pooling layers only
    # do not use 'flatten' with this method
    global blockDict, layerNames, pca_components, block_map
    global num_channel_chunks, num_final_hidden_layer_chunks, use_pool_layer_threshold
    num_channel_chunks = 0
    num_final_hidden_layer_chunks = 32     # change this value depending on how big the final layer is for mobilenet.
    use_pool_layer_threshold = False

    # print('key: %s'%(key))
    if reset:
        blockDict = get_block_map(model)
        layerNames = [layers for blocks in blockDict.values() for layers in blocks.keys()]


    # flatten dictionary to get listo of layers
    result = np.zeros(1)
    # print('layer_name: %s'%(key))
    # print('layer_names: %s' % (layerNames))
    if key in layerNames:
        try:
            result = jit_calc(key, layer, blockDict)
        except ZeroDivisionError as z:
            util.thisLogger.logInfo('ZeroDivisionError %s'%(z))
            util.thisLogger.logInfo('key %s, layer %s, blockDict %s' % (key, layer, blockDict))
            result = jit_calc(key, layer, blockDict)

    return result

def is_final_layer(model, key):
    is_last = model.get_layer_names()[-2] == key
    # use -2 as activations are never extracted from first and last layers
    return is_last


def extractMultiLayer(actDict, layerActivationReduction='jsdiverge', reset=False):
    # not implemented
    return actDict

def get_block_map(model):
    dnn = util.getParameter('Dnn')
    # get block mapping data
    full_filename = os.path.join(os. getcwd(), 'modules_reduce', 'data', 'cbir_blockmap_%s_blockcbir' % (dnn)) + '.csv'
    lock_filename = '%s.lock'%(full_filename)
    if os.path.isfile(full_filename):
        with FileLock(lock_filename):
            with open(full_filename, 'r') as f:
                lines = f.readlines()
        block_map = {}
        for line in lines[1:]:
            l = line.replace('\n','').split(',')
            if l[1] not in block_map:
                block_map[l[1]] = {}
            block_map[l[1]][l[0]] = None
        del block_map['none']
    else:
        # create file
        # get the dnn layer numbers
        layer_csv_lines = []
        layer_csv_lines.append('layer_name,layer_block\n')
        layer_names = model.get_layer_names()
        block_count = 1
        is_conv_detected = False
        is_pool_detected = False
        for l in layer_names:
            if 'conv' in l.lower():
                is_conv_detected = True
                layer_csv_lines.append('%s,%s\n' % (l, 'block%s' % (block_count)))
            elif 'pool' in l.lower():
                is_pool_detected = True
                layer_csv_lines.append('%s,%s\n' % (l, 'block%s' % (block_count)))
            else:
                layer_csv_lines.append('%s,%s\n' % (l, 'none'))
            if is_conv_detected and is_pool_detected:
                is_conv_detected = False
                is_pool_detected = False
                block_count += 1
        with FileLock(lock_filename):
            with open(full_filename, 'w') as f:
                f.writelines(layer_csv_lines)
        util.thisLogger.logInfo('cbir block mapping file created in %s. Check that it is correct.'%(full_filename))
        block_map = get_block_map(model)
    return block_map

@jit(nopython=False)
def get_block_name(layer_name, block_map):
    block_name = ''
    for k, v in block_map.items():
        if layer_name in v:
            block_name = k
            break
    if block_name == '':
        raise Exception('Layer name %s is not associated with a block'%(layer_name))
    return block_name


def isValidLayer(l, model, layerActivationReduction):
    isValid = False
    dnn = util.getParameter('Dnn')

    if dnn == 'vgg16' or dnn == 'vgg16bn':
        if 'block' in l:
            isValid = True
    elif dnn == 'mobilenet':
        regex = re.compile('conv_pw_._relu')
        if re.match(regex, l):
            isValid = True
    else:
        raise Exception('DNN %s not handled in %s'%(dnn, layerActivationReduction))

    return isValid

@jit(nopython=False)
def flatten(t):
    # return [item for sublist in t for item in sublist]
    flat = np.ravel(t)
    return np.reshape(flat, (1, len(flat)))

@jit(nopython=False)
def allTrue(t):
    isTrue = False
    trueItems = [i for i in t if i == True]
    if len(t) == len(trueItems):
        isTrue = True
    return isTrue

@jit(nopython=False) #  uses GPU
def jit_calc(layer_name, layer, blockDict):
    result = np.zeros(1)
    all_unflattened = []

    # invert map to get teh block name associated with the layer
    block_name = get_block_name(layer_name, blockDict)
    if blockDict[block_name][layer_name] is None:
        blockDict[block_name][layer_name] = layer
    else:
        blockDict[block_name][layer_name] = np.vstack((blockDict[block_name][layer_name], layer))

    # collect information when we have all block layers
    isComplete = []
    for blockLayers in blockDict.values():
        hasData = False
        for val in blockLayers.values():
            if val is None:
                hasData = False
                break
            else:
                hasData = True

        isComplete.append(hasData)
    # print(isComplete)

    if allTrue(isComplete):
        for n in range(0, list(list(blockDict.values())[0].values())[0].shape[0]):
            convResults = np.array([0])
            unflattened = []

            for block_name, block in blockDict.items():
                # slice the layers based on the instance number
                slicedBlockLayers = []
                for bl in list(block.values()):
                    slicedBlockLayers.append(bl[n:n + 1])
                if block_name != 'final_hidden':
                    if convResults.size == 1 and convResults[0] == 0:
                        info = jit_collect_information(slicedBlockLayers)
                        convResults = info
                        unflattened.append(info)
                    else:
                        info = jit_collect_information(slicedBlockLayers)
                        convResults = np.append(convResults, info)
                        unflattened.append(info)
                else:
                    res = slicedBlockLayers[0].reshape((slicedBlockLayers[0].shape[1]))
                    if num_final_hidden_layer_chunks > 0:
                        res = average_chunk_jit(res, num_final_hidden_layer_chunks)
                        res = np.array(res)
                    unflattened.append(res)
                    pass

            if convResults.size > 0:
                convResults = flatten(convResults)
                convResults = np.reshape(convResults, (convResults.shape[1]))

                if result.size == 1 and result[0] == 0:
                    result = convResults
                else:
                    result = np.vstack([result, convResults])
                all_unflattened.append(unflattened)

    return_unflattened_result = True
    if return_unflattened_result and len(all_unflattened) > 0:
        return_result = all_unflattened
    else:
        return_result = result

    return return_result


# this method is called once for each pooling layer
# produces a matrix that contains info about which neurons are significant
# take last pool layer, divide by max value to normalize the values
# loop round each value in the normalized final pool layer
# if the value is > 0.5 and > any other value stored for that x y position, store the value in x y
# (if the value is > 0.5 and is the largest in all of the channels, keep it).
# we end up with output_conv_map that is a map the same x y size as the final pooling layers, but 1-dimensional
# but if the value was > 0.5, the max value for that x y position out of all the channels is stored
# The rest of the values are zero
# output_conv_map is the last layer with important neurons selected
@jit(nopython=False)
def jit_collect_information(blockLayers):
    # Begin - Initialize memory for collecting informations
    # we only have one label for each image, so number_of_info_samples = 1
    number_of_info_samples = 1 # todo change back to 1

    # select the convolutional layers that can have the importances calculated
    # only blocks that have feature map dimensions > number_of_info_samples can have the importances calculated
    # if this block does not have enough feature map dimensions, just take the values
    calculate_importances = True
    if blockLayers[0].shape[1] <= number_of_info_samples*2:
        calculate_importances = False

    if calculate_importances:
        # only use the first conv layer of the block as subsequent layers come out with the same calculations
        # (this is what is done in the original code)
        convLayers = blockLayers[:1] # alter this to use > 1 convolutional layer
        poolLayer = blockLayers[-1]  # last layer of the block is the pooling layer

        val_info_pools = []
        for n in range(0, len(convLayers)): # the first conv layer only (looks like the original code only did the first conv layer)
            val_info_pools.append(np.zeros((number_of_info_samples, np.array(blockLayers[n]).shape[-1])))

        # calculate important activations
        output_conv_eval = poolLayer
        output_conv_eval = output_conv_eval / output_conv_eval.max() # normalize the activations
        xdim = blockLayers[-1].shape[1]
        ydim = blockLayers[-1].shape[2]
        nchannels = blockLayers[-1].shape[3]
        output_conv_map = np.zeros((xdim, ydim), dtype=np.float32)

        for iZ in range(nchannels):
            for iY in range(ydim):
                for iX in range(xdim):
                    src_value = output_conv_eval[0][iY][iX][iZ]

                    if use_pool_layer_threshold:
                        if src_value > pool_layer_threshold and src_value > output_conv_map[iY][iX]:
                            output_conv_map[iY][iX] = src_value
                    else:
                        if src_value > output_conv_map[iY][iX]:
                            output_conv_map[iY][iX] = src_value

        for c in range(0, len(convLayers)):
            for i in range(0, number_of_info_samples):
                jit_calc_importances(output_conv_map, val_info_pools[c][i], convLayers[c])

        flatVal = np.asarray(val_info_pools).flatten()
    else:
        flatVal = blockLayers[-1].flatten()


    if num_channel_chunks > 0:
        # split the channel values into chunks and average the channel values in each chunk
        averages = average_chunk_jit(flatVal, num_channel_chunks)
        flatVal = np.reshape(averages, (1, averages.shape[0]))

    return flatVal

@jit(nopython=False)
def average_chunk_jit(array, num_chunks):
    # splits array into chunks and takes the average of each chunk
    split = np.array_split(array, num_chunks)
    averages = []
    for chunk in split:
        averages.append(np.average(chunk))
    averages = np.array(averages)
    return averages


def average_chunk(array, num_chunks):
    # splits array into chunks and takes the average of each chunk
    split = np.array_split(array, num_chunks)
    averages = []
    for chunk in split:
        averages.append(np.average(chunk))
    averages = np.array(averages)
    return averages



# ------------------------------
# calc_importances selects the important neurons from previous pooled layers
# (these are neurons that correspond spatially to significant neurons in the Nth pooling layer)
# only these neurons have an influence on the values of the significant neurons in the last layer
# 1 characterisic value is computed for each feature map (channel) in each layer (imp_array)
# eval_array = pooling layer
# imp_array = 1 value for each feature map (channel) in the input pooling layer (eval_array)
# output_conv_map = the merged max values above 0.5 for each channel in the final pooling layer
# (output_conv_map determines which values from the previous pooling layers are determined as important)
@jit(nopython=False) # uncomment
def jit_calc_importances(output_conv_map, imp_array, eval_array):
    number_of_importances = eval_array.shape[3] # number of channels

    scale_param_y = int(eval_array.shape[1] / output_conv_map.shape[0]) # 32/16
    scale_param_x = int(eval_array.shape[2] / output_conv_map.shape[1]) # 32/16

    for iZ in range(number_of_importances):
        c = 0

        # For max pool layer, if value > adaptation threshold, put 1 in maxpool adapt array
        # put 1 in coesponding adapt conv array
        # accumulate 1's for each instance
        for iY in range(output_conv_map.shape[0]):
            for iX in range(output_conv_map.shape[1]):
                if output_conv_map[iY][iX] > pool_layer_threshold:
                    k = 0.0

                    for iYY in range(scale_param_y):
                        for iXX in range(scale_param_x):
                            k += eval_array[0][iY * scale_param_y + iYY][iX * scale_param_x + iXX][iZ] # [16*2 + 0] [16*2 + 0] [importance_number]

                    k /= number_of_importances
                    c += 1

                    imp_array[iZ] += k

        imp_array[iZ] /= c

    return imp_array



