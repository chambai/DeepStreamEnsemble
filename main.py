import util
from core import dataset, train, analyse, datastream, pretrain_dnn
import numpy as np
from collections import defaultdict
import os

def start(dnn_name='vgg16', dataset_name='mnistfashion', data_combination='23456789-01', drift_pattern='categorical-abrupt', reduction='dscbir', adaptation='dsadapt'):
    if __name__ == '__main__':
        trainedClasses = data_combination.split('-')[0]
        unknownClasses = data_combination.split('-')[1]

        data_dir = 'output/intermediate'
        run_data_file = os.path.join(data_dir,'activations_%s_%s_%s.npz'%(dnn_name,dataset_name,trainedClasses))

        # load setup file
        util.params = None
        util.usedParams = []
        util.setupFile = 'input/params/%s_%s_%s_%s.txt'%(dnn_name,dataset_name,trainedClasses,unknownClasses)
        util.setParameter('Dnn', dnn_name)
        util.setParameter('DatasetName', dataset_name)
        util.setParameter('DriftType', drift_pattern.split('-')[0])
        util.setParameter('DriftPattern', drift_pattern.split('-')[1])
        util.setParameter('DataClasses', '[%s]'%(','.join(list(trainedClasses))))
        if adaptation == 'rsb':
            util.setParameter('AnalysisModule', 'modules_compare.%s' % (adaptation))
            util.setParameter('LayerActivationReduction', 'none')
        else:
            util.setParameter('AnalysisModule', 'modules_adapt.%s' % (adaptation))
            if reduction == 'jsdl':
                # JSDL requires that the data is flattened and padded prior to jsdl calculations
                reduction = 'flatten,pad,jsdl'
            util.setParameter('LayerActivationReduction', reduction)

        util.thisLogger = util.Logger()

        # Get the training class data
        x_train, y_train, x_test, y_test = dataset.getFilteredData()

        # Load the DNN
        model = pretrain_dnn.loadDnn(dataset_name, list(map(int, list(trainedClasses))), x_train, y_train, x_test, y_test)

        if adaptation == 'rsb':
            analyse.loadModule('modules_compare.' + adaptation)
            analyse.setupCompare(model, x_train, y_train)
            # pixel data is used and this is already normalized
            # when unseen data is extracted this will be normalized, so we can pass 1 in as the maxValue
            unseenData = datastream.getData()
            processDataStream(unseenData, model, 1)
        else:
            # get activations
            if os.path.exists(run_data_file):
                data = np.load(run_data_file, allow_pickle=True)
                activations = data['a']
                xData = data['x']
                yData = data['y']
            else:
                activations, xData, yData = train.getActivations(x_train, -1, model, y_train)
                np.savez(run_data_file, a=activations, x=xData, y=yData, allow_pickle=True)

            # normalize
            activations, max_values = normalize(activations)

            analyse.loadModule('modules_adapt.' + adaptation, data_dir)
            analyse.setup(model, activations, yData, xData)
            unseenData = datastream.getData()
            processDataStream(unseenData, model, max_values)

# ------------------------------------------------------------------------------------------
def normalize(activations):
    max_values = []
    if len(activations.shape) == 1:
        maxValue = np.amax(activations)
        activations = activations / maxValue
        max_values.append(maxValue)
    else:
        # list of activations
        for n,a in enumerate(activations[0]):
            maxValue = np.amax(a)
            activations[0][n] = a/maxValue
            max_values.append(maxValue)
    return activations, max_values


# ------------------------------------------------------------------------------------------
def processDataStream(all_unseen_data, model, maxValue):
    unseenDataDict = defaultdict(list)
    for item in all_unseen_data:
        unseenDataDict[item.adaptState].append(item)

    unseenInstancesList = []
    for adapt_state, unseenData in unseenDataDict.items():
        unseenInstancesObjs = analyse.startDataInputStream(model, maxValue,
                                                           unseenData)
        unseenInstancesList.append(unseenInstancesObjs)
        if unseenInstancesObjs[0].adaptClass != '-':
            # calculate accuracy of each section
            correct = [x.correctResult for x in unseenInstancesObjs if x.correctResult == int(x.adaptClass)]
            acc = len(correct) / len(unseenInstancesObjs)
            util.thisLogger.logInfo('%sAcc=%s' % (adapt_state, acc))

    unseenInstancesObjList = [item for sublist in unseenInstancesList for item in sublist]
    util.thisLogger.logInfo('TotalNumberOfUnseenInstances=%s' % (len(unseenInstancesObjList)))
    drift_instances = [u.id for u in unseenInstancesObjList if u.driftDetected]
    util.thisLogger.logInfo('DriftDetectionInstances=%s' % (drift_instances))

    analyse.stopProcessing()


# Example
# mobilenet fashion class data example (our dseadapt method)

# DeepStreamEnsemble method (Ours)
start(dnn_name='vgg16',
      dataset_name='mnistfashion',
      data_combination='05-12',
      drift_pattern='categorical-abrupt',
      reduction='blockcbir',
      adaptation='dseadapt')




