from framework import analyse
from modules_reduce import blockcbir
from modules_detect_adapt.dse_adapt_ensemble import AdaptEnsemble
from modules_detect_adapt.dse_dnnblockbase import DnnBlock

data_dir = ''
dnn_block = None
adapt_ens = None
adaptive_dnn = None
is_adapted = False
adapt_times = []

def setData(dir):
    global data_dir
    data_dir = dir
    pass

# Hoeffding tree for each block, no ensemble
def setup(indnn, act_train_batch, y_train_batch, inData=[]):
    global data_dir, dnn_block, adapt_ens, adaptive_dnn, is_adapted, adapt_times
    is_adapted = False
    adapt_times = []

    # truncate to 150 as that's what we did for the training data
    for i in range(len(act_train_batch)):
        for b in range(len(act_train_batch[0])):
            act_train_batch[i][b] = blockcbir.average_chunk(act_train_batch[i][b], 150)

    # setup drift detection
    dnn_block = DnnBlock(data_dir=data_dir, base_classifier_type='HT', is_ensemble=False,
                         drift_detector_name='DIFF_PREV', vote_method='ANY_1')
    dnn_block.num_threshold_windows = 3
    dnn_block.setup(indnn, act_train_batch, y_train_batch, inData)

    # setup adaption
    adapt_ens = AdaptEnsemble()
    adapt_ens.num_window_buffers = 2
    adapt_ens.num_previous_buffers = 5
    adapt_ens.augment = False
    adapt_ens.use_clustering = True
    adapt_ens.use_class_buffer = True
    adapt_ens.first_sequential = True
    adapt_ens.inline_adaptation = False
    adapt_ens.first_sequential = True
    adapt_ens.set_activation_ensemble(inData, act_train_batch, y_train_batch, dnn_block)
    adaptive_dnn = adapt_ens.setup_adaptive_dnn(inData, y_train_batch, ['always_update', 'lin'], 'none', freeze_layer_type='classification', epochs=3) # classification  block

    pass

def logTrainAccuracy(act_train, y_train):
    return 0

def logTestAccuracy(act_unseen, y_unseen):
    return 0

def detect(xdata_unseen_batch, act_unseen_batch, dnnPredict_batch):
    global dnn_block, adapt_ens, adaptive_dnn

    # truncate to 150 as that's what we did for the training data
    for i in range(len(act_unseen_batch)):
        for b in range(len(act_unseen_batch[0])):
            act_unseen_batch[i][b] = blockcbir.average_chunk(act_unseen_batch[i][b], 150)  # chunk up and average to 150

    # Todo: maybe predicting from DNN twice?  I think this is needed if novel classes are added
    dnnPredict_batch = analyse.getPredictions(adaptive_dnn, xdata_unseen_batch, adaptive_dnn.all_classes)

    drift_result, streaming_classifier_predictions = dnn_block.processUnseenStreamBatch(xdata_unseen_batch, act_unseen_batch, dnnPredict_batch)

    return dnnPredict_batch, streaming_classifier_predictions, drift_result


def adapt(xdata_unseen_batch, act_unseen_batch, dnnPredict_batch, drift_result, sc_predicts, true_classes):
    global dnn_block, adapt_ens, adaptive_dnn, is_adapted

    win_drift_val = 0
    if 'D' in drift_result:
        win_drift_val = 1

    # adapt
    adaptive_dnn, majority_result, y_classifier_batch, adapt_discrepancy, adapt_class = \
        adapt_ens.adaptation(win_drift_val, act_unseen_batch, xdata_unseen_batch, dnnPredict_batch, sc_predicts, drift_result, true_classes)
    is_adapted = adapt_ens.is_adapted

    return majority_result, y_classifier_batch, adapt_discrepancy, adapt_class

def processStreamInstances():
    # only used if instances need to be processed on a separate thread
    pass

def stopProcessing():
    global data_dir, dnn_block, adapt_ens, adaptive_dnn, adapt_times
    adapt_times = adapt_ens.adapt_times
    dnn_block.stopProcessing()
    adapt_ens.stopProcessing()
    data_dir = ''
    dnn_block = None
    adapt_ens = None
    adaptive_dnn = None

def getAnalysisParams():
    parameters = []
    # return a list of variable parameters in the format [[name,value],[name,value]...[name,value]]
    parameters.append(['none', 0])
    return parameters
