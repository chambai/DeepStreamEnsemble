import util
import threading
import copy
import time
import datetime
from core import pretrain_dnn as dnnModel
from core.dnn_models import DNNPytorchAdaptive
from core import dataset
import numpy as np
from core import extract
from modules_adapt import adapt_ensemble_buffer

class AdaptEnsemble:
    def __init__(self):
        self.classifier_dict = []  # trained on true label at training time
        self.streamClassifierNames = []
        self.threads = []
        self.drift_detectors = []
        self.adaptive_dnn = None
        self.run_update_thread = False
        self.update_thread_started = False
        self.adapt = False
        self.adapt_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.data_lock = threading.Lock()
        self.clus_lock = threading.Lock()
        self.buffer_windows_count = 0
        self.num_window_buffers = 0
        self.x_act_buffer_dict = np.array([0])
        self.x_data_buffer = np.array([0])
        self.y_data_buffer = np.array([0])
        self.num_previous_buffers = 0      # saves the previous mem buffers from previous drifts and uses them in all subsequent adaptions
        self.buffer_prev_windows_count = 0
        self.x_act_prev_buffer_dict = np.array([0])
        self.x_data_prev_buffer = np.array([0])
        self.y_data_prev_buffer = np.array([0])
        self.adaption_thread_error = False
        self.adaption_error_message = ''
        self.adaption_times = []
        self.is_adapted = False
        self.use_artificial_drift_detection = False # instead of detecting drift, it uses true values to determine when drift has actually started
        self.adapt_every_n_windows = 0    # once drift has been detected, adapt every n windows
        self.drift_triggered = False
        self.drift_triggered_window = 0
        self.drift_trigger_count = 0
        self.use_clustering = False
        self.clusterer = None
        self.x_train = None
        self.y_train = None
        self.first_sequential = False
        self.use_class_buffer = False
        self.is_adapting = False
        self.adaptation_count = 0
        self.dnn_block = None
        self.inline_adaptation = False
        self.no_adaptation = False # if set to true, no adaptation thread is started
        self.applied_adaptation = []
        self.win_number = 0
        self.adapt_win_numbers = []
        self.train_acc = 0

    def set_activation_ensemble(self, x_train, act_train_data, y_train, dnn_block):
        self.dnn_block = dnn_block
        self.classifier_dict = dnn_block.ens_dict
        if self.use_clustering:
            # use a copy of the ensemble classifiers so it doesn't have to be trained again.
            self.clusterer = adapt_ensemble_buffer.AdaptationEnsemble(copy.deepcopy(dnn_block.ens_dict))
            self.clus_lock.acquire()
            self.clusterer.partial_fit(x_train, act_train_data, y_train)
            self.clusterer.weights = dnn_block.weights
            self.clus_lock.release()

        if not self.no_adaptation:
            self.start_model_update_thread()


    def logTrainAccuracy(self, act_train, y_train):
        accuracies = self.getAccuracy(act_train[:1000], y_train[:1000])
        [util.thisLogger.logInfo('StreamClassifierTrainAccuracy%s=%s' % (i, a)) for i, a in
         enumerate(accuracies)]
        return accuracies

    def logTestAccuracy(self, act_unseen, y_unseen):
        accuracies = self.getAccuracy(act_unseen[:1000], y_unseen[:1000])
        [util.thisLogger.logInfo('StreamClassifierTestAccuracy%s=%s' % (i, a)) for i, a in
         enumerate(accuracies)]
        return accuracies

    def getAccuracy(self, x, y):
        accuracies = []
        for i, (k, v) in enumerate(self.classifier_dict.items()):
            y_predict = v.predict(x[:,i])
            bool_array = y == y_predict
            num_correct = np.count_nonzero(bool_array)
            accuracies.append(num_correct / len(y))
        return np.array(accuracies)


    def get_training_data(self):
        # get original training data
        mapOriginalYValues = util.get_mapping_param('MapOriginalYValues')
        if len(mapOriginalYValues) == 0:
            self.x_train, self.y_train, _, _ = dataset.getFilteredData(isMap=False, do_normalize=True)
        else:
            self.x_train, self.y_train, _, _ = dataset.getFilteredData(isMap=True, do_normalize=True)
        return self.x_train, self.y_train

    def get_discrepancy_data(self):
        # get original training data
        mapOriginalYValues = util.get_mapping_param('MapOriginalYValues')
        if len(mapOriginalYValues) == 0:
            x_train_orig, y_train_orig, _, _ = dataset.getOutOfFilteredData(isMap=False)
        else:
            x_train_orig, y_train_orig, _, _ = dataset.getOutOfFilteredData(isMap=True)
        return x_train_orig, y_train_orig

    def setup_adaptive_dnn(self, x_train_orig, y_train_orig, adaption_strategies, buffer_type='none', freeze_layer_type='classification', epochs=3):
        self.adaptive_dnn = DNNPytorchAdaptive(adaption_strategies=adaption_strategies, buffer_type=buffer_type, freeze_layer_type=freeze_layer_type, epochs=epochs)
        self.adaptive_dnn.load((dnnModel.getFullDnnModelPathAndName()))
        # x_train_orig, y_train_orig = self.get_training_data()
        self.adaptive_dnn.fit(x_train_orig, y_train_orig)  # applies extra training if required
        return self.adaptive_dnn

    def set_dnn_training_accuracy(self, X, y):
        # get accuracy value
        predicts = self.adaptive_dnn.predict(X)
        self.train_acc = np.count_nonzero(predicts == y) / len(y)
        print('train acc: %s' % self.train_acc)


    def start_model_update_thread(self):
        if not self.inline_adaptation:
            # start model update thread ready for when the stream batches start arriving
            self.run_update_thread = True
            update_thread = threading.Thread(target=self.run_adapt_thread, daemon=False)
            util.thisLogger.logInfo('run_adapt_thread thread starting...')
            update_thread.start()
            util.thisLogger.logInfo('run_adapt_thread thread started...')

    def run_adapt_thread(self):
        while self.run_update_thread:
            # print('waiting...')
            if not self.is_adapting and self.adapt:
                print('adapting...')
                self.adapt_dnn()
                # time.sleep(0.01)
            time.sleep(1)
        util.thisLogger.logInfo('adaption thread ended')

    def adapt_dnn(self):
        try:
            # self.adapt_lock.acquire()
            self.is_adapting = True
            self.adapt = False

            # concatenate current data with any stored data
            self.data_lock.acquire()
            activations, xdata, ydata = self.get_buffer_data()
            prev_act, prev_xdata, prev_ydata = self.get_previous_buffer_data()
            if activations.shape[0] > 1 and self.num_previous_buffers > 1:
                self.concatenate_previous_buffer_data(activations, xdata, ydata)    # store this data that is used for the adaption for future use
                # self.clear_previous_buffer_data()   # removes previous data buffers if it is over the specified buffer limit
                # concatenate previous adaption data with current buffer data
                activations, xdata, ydata = self.concatenate(prev_act, prev_xdata, prev_ydata, activations, xdata, ydata)

            if len(activations) > 1:
                util.thisLogger.logInfo('number of adaption activations: %s' % (activations.shape[0]))
                self.clear_buffer_data()

                self.data_lock.release()

                classifier_copies_dict = {}
                # re-train on window data
                start = datetime.datetime.now()
                self.model_lock.acquire()
                for k, classifier in self.classifier_dict.items():
                    copy_method = getattr(classifier, "get_copy", None)
                    if callable(copy_method):
                        classifier_copies_dict[k] = classifier.get_copy()
                    else:
                        classifier_copies_dict[k] = copy.deepcopy(classifier)
                adaptive_dnn_copy = self.adaptive_dnn.get_copy()
                self.model_lock.release()


                if self.use_clustering:
                    self.clus_lock.acquire()
                    ydata = self.clusterer.predict_act(activations)
                    util.thisLogger.logInfo('More samples on classes: %s' % (np.unique(ydata)))
                    if self.use_class_buffer:
                        xdata, activations, ydata = self.clusterer.get_buffer_data(ydata, 100)
                    self.clus_lock.release()

                dnn_threads = []
                threads = []
                # update adaptive DNN copy
                if util.getParameter('SimulateDriftDetection'):
                    adapt_blocks = []
                else:
                    adapt_blocks = self.dnn_block.get_max_diff_block_indexes(3)
                    util.thisLogger.logInfo('AdaptBlockIndexes=%s'%(adapt_blocks))
                    # adapt_blocks = [self.dnn_block.block_with_max_diff]
                    # adaptive_dnn_copy.partial_fit(xdata, ydata, adapt_blocks=adapt_blocks)

                if 'ocl' in util.getParameter('AnalysisModule'):
                    args = (xdata, ydata)
                else:
                    args = (xdata, ydata, adapt_blocks)

                thread = threading.Thread(target=adaptive_dnn_copy.partial_fit,
                                          args=args)
                dnn_threads.append(thread)

                # update streaming classifier copy
                if not util.getParameter('SimulateDriftDetection'):
                    for k, classifier_copy in classifier_copies_dict.items():
                        # classifier_copy.partial_fit(np.asarray(activations), np.asarray(ydata))
                        thread = threading.Thread(target=classifier_copy.partial_fit,
                                                  args=(np.asarray(activations[k]), np.asarray(ydata)))
                        threads.append(thread)
                        # pass


                for i, t in enumerate(dnn_threads):
                    t.start()
                    time.sleep(1)

                for i, t in enumerate(threads):
                    t.start()
                    time.sleep(1)

                for t in dnn_threads:
                    t.join()
                    time.sleep(1)
                stop = datetime.datetime.now()
                time_diff = stop - start
                util.thisLogger.logInfo('DnnAdaptionTimeNoEns=%s' % (time_diff.total_seconds()))

                for t in threads:
                    t.join()
                    time.sleep(1)

                # replace original dnn with the trained dnn
                self.model_lock.acquire()
                self.adaptive_dnn = adaptive_dnn_copy
                self.classifier_dict = classifier_copies_dict
                self.adapt_win_numbers.append(self.win_number + 1)
                self.model_lock.release()
                util.thisLogger.logInfo('Models updated')

                stop = datetime.datetime.now()
                time_diff = stop - start
                util.thisLogger.logInfo('DnnAdaptionTime=%s' % (time_diff.total_seconds()))
            else:
                self.data_lock.release()
        except Exception as e:
            util.thisLogger.logInfo('Error in adaption thread: %s'%(e))
            self.adaption_thread_error = True
            self.adaption_error_message = str(e)
            raise(e)
        finally:
            self.is_adapting = False
            self.is_adapted = True



    def concatenate_buffer_data(self, x_act, x_data, y):
        if len(self.y_data_buffer) == util.getParameter('StreamBatchLength') * self.num_window_buffers:
            # window buffer reached - remove first window
            self.x_act_buffer_dict = self.x_act_buffer_dict[len(x_data):]
            self.x_data_buffer = self.x_data_buffer[len(x_data):]
            self.y_data_buffer = self.y_data_buffer[len(x_data):]
        if len(self.y_data_buffer) == 1:
            self.x_act_buffer_dict = np.zeros((0, x_act.shape[1]))
            self.x_data_buffer = np.zeros((0, x_data.shape[1], x_data.shape[2], x_data.shape[3]))
            self.y_data_buffer = np.zeros((0), dtype=np.int64)
        self.x_act_buffer_dict = np.vstack((self.x_act_buffer_dict, x_act))
        self.x_data_buffer = np.vstack((self.x_data_buffer, x_data))
        self.y_data_buffer = np.hstack((self.y_data_buffer, y))
        self.buffer_windows_count += 1


    def concatenate(self, x_act_1, x_data_1, y_1, x_act_2, x_data_2, y_2):
        if len(x_act_1) == 1 and len(x_act_2) == 1:
            # no concatenation required
            return x_act_1, x_data_1, y_1
        else:
            if len(x_act_1) == 1:
                x_act_1 = np.zeros((0, x_act_2.shape[1]))
                x_data_1 = np.zeros((0, x_data_2.shape[1], x_data_2.shape[2], x_data_2.shape[3]))
                y_1 = np.zeros((0), dtype=np.int64)
            x_act = np.vstack((x_act_1, x_act_2))
            x_data = np.vstack((x_data_1, x_data_2))
            y = np.hstack((y_1, y_2))
            return x_act, x_data, y



    def get_buffer_data(self):
        return self.x_act_buffer_dict, self.x_data_buffer, self.y_data_buffer



    def clear_buffer_data(self):
        if self.num_window_buffers > 0:
            # clear if the number of buffers has been reached
            if self.buffer_windows_count >= self.num_window_buffers:
                self.x_act_buffer_dict = np.array([0])
                self.x_data_buffer = np.array([0])
                self.y_data_buffer = np.array([0])
                self.buffer_windows_count = 0
        else:
            self.x_act_buffer_dict = np.array([0])
            self.x_data_buffer = np.array([0])
            self.y_data_buffer = np.array([0])
            self.buffer_windows_count = 0


    def concatenate_previous_buffer_data(self, x_act, x_data, y):
        if len(self.y_data_prev_buffer) == util.getParameter('StreamBatchLength') * self.num_previous_buffers:
            # window buffer reached - remove first window
            self.x_act_prev_buffer_dict = self.x_act_prev_buffer_dict[len(x_act):]
            self.x_data_prev_buffer = self.x_data_prev_buffer[len(x_data):]
            self.y_data_prev_buffer = self.y_data_prev_buffer[len(x_data):]
        if len(self.y_data_prev_buffer) == 1:
            self.x_act_prev_buffer_dict = np.zeros((0, x_act.shape[1]))
            self.x_data_prev_buffer = np.zeros((0, x_data.shape[1], x_data.shape[2], x_data.shape[3]))
            self.y_data_prev_buffer = np.zeros((0), dtype=np.int64)
        self.x_act_prev_buffer_dict = np.vstack((self.x_act_prev_buffer_dict, x_act))
        self.x_data_prev_buffer = np.vstack((self.x_data_prev_buffer, x_data))
        self.y_data_prev_buffer = np.hstack((self.y_data_prev_buffer, y))
        self.buffer_prev_windows_count += 1



    def get_previous_buffer_data(self):
        return self.x_act_prev_buffer_dict, self.x_data_prev_buffer, self.y_data_prev_buffer



    def clear_previous_buffer_data(self):
        if self.num_previous_buffers > 0:
            # clear if the number of buffers has been reached
            if self.buffer_prev_windows_count >= self.num_previous_buffers:
                self.x_act_prev_buffer_dict = np.array([0])
                self.x_data_prev_buffer = np.array([0])
                self.y_data_prev_buffer = np.array([0])
                self.buffer_prev_windows_count = 0
        else:
            self.x_act_prev_buffer = np.array([0])
            self.x_data_prev_buffer = np.array([0])
            self.y_data_prev_buffer = np.array([0])
            self.buffer_prev_windows_count = 0


    def get_training_data_sample(self):
        idxs = np.random.choice(self.x_train.shape[0], self.num_window_buffers*util.getParameter('StreamBatchLength'), replace=False)
        x_sample = self.x_train[idxs]
        y_sample = self.y_train[idxs]

        # get activations
        act_sample = extract.getActivationData2(self.adaptive_dnn, x_sample, y_sample)

        return act_sample, x_sample, y_sample

    def adaptation(self, has_drift, act_unseen_batch, xdata_unseen_batch, dnnPredict_batch, true_values, streaming_classifier_predictions, result, batchTrueSuper):
        majority_result = result
        classifier_predictions = []

        if has_drift == 1:
            # get accuracy of DNN for this window
            window_dnn_acc = np.count_nonzero((dnnPredict_batch == batchTrueSuper)) / len(batchTrueSuper)
            util.thisLogger.logInfo('window_dnn_acc: %s, train_acc: %s' %(window_dnn_acc, self.train_acc))

            # if window_dnn_acc < self.train_acc:
            self.drift_trigger_count += 1

            self.data_lock.acquire()
            self.concatenate_buffer_data(act_unseen_batch, xdata_unseen_batch, batchTrueSuper)

            util.thisLogger.logInfo('adapt set to true')

            if self.use_clustering:
                self.clus_lock.acquire()
                self.clusterer.partial_fit(xdata_unseen_batch, act_unseen_batch, batchTrueSuper)
                self.clus_lock.release()

            self.data_lock.release()

            if self.is_adapting:
                classifier_predictions = streaming_classifier_predictions
            else:
                self.adaptation_count += 1
                if self.inline_adaptation or (self.first_sequential and self.adaptation_count == 1):
                    self.adapt_dnn()
                    for k, v in self.classifier_dict.items():
                            y_classifier_batch = v.predict(act_unseen_batch[:,k])
                            classifier_predictions.append(y_classifier_batch)
                    # get the voted prediction
                    classifier_predictions = np.array(classifier_predictions)
                    classifier_predictions = [np.bincount(classifier_predictions[:,p]).argmax() for p in range(len(classifier_predictions[0]))]
                else:
                    self.adapt = True
                    classifier_predictions = streaming_classifier_predictions
        else:
            classifier_predictions = streaming_classifier_predictions
            if self.num_window_buffers > 0:
                if self.use_clustering:
                    self.clus_lock.acquire()
                    cp = self.clusterer.predict_act(act_unseen_batch)
                    self.clus_lock.release()
                    self.concatenate_buffer_data(act_unseen_batch, xdata_unseen_batch, cp)
                    # todo: but if there's new classes in these preceeding windows, they will be incorrectly classified
                else:
                    self.concatenate_buffer_data(act_unseen_batch, xdata_unseen_batch, dnnPredict_batch)
                    # TODO: This lowers the accuracy as the DNN has started mis-predicting.


        if self.adaption_thread_error:
            raise Exception(self.adaption_error_message)

        adapt_discrepancy = []
        adapt_class = []
        discrepancyType = util.getParameter('DataDiscrepancy')

        for r, d, t in zip(majority_result, dnnPredict_batch, true_values):
            adapt_class.append(int(d))
            if r == 'D':
                adapt_discrepancy.append(discrepancyType)
            if r == 'N':
                adapt_discrepancy.append('ND')

        y_classifier_batch = classifier_predictions

        self.calc_applied_adaptation(true_values)
        if self.is_adapting:
            delay = util.getParameter('AdaptationTimeDelay')
            util.thisLogger.logInfo('AdaptationTimeDelay of %ss' % (delay))
            time.sleep(delay)

        self.win_number += 1
        return self.adaptive_dnn, majority_result, y_classifier_batch, adapt_discrepancy, adapt_class

    def calc_applied_adaptation(self, true_values):
        if self.is_adapted:
            self.applied_adaptation.extend([1])
            self.applied_adaptation.extend([-1 for i in range(len(true_values) - 1)])
        else:
            self.applied_adaptation.extend([-1 for i in range(len(true_values))])
        self.is_adapted = False

    def predict_act(self, X):
        # get predicts from each classifier
        predicts = []
        for k in range(len(X[0])):
            predicts.append(self.classifier_dict[k].predict(X[:, k]))
        predicts = np.array(predicts)

        weighted_predicts = []
        # append predicts based on weights
        for i, p in enumerate(predicts):
            for _ in range(self.clusterer.weights[i]):
                weighted_predicts.append(p)
        weighted_predicts = np.array(weighted_predicts)

        weighted_predicts = [np.bincount(weighted_predicts[:, p]).argmax() for p in
                    range(len(weighted_predicts[0]))]

        return weighted_predicts


    def get_drift(self, dnn_predict, streamClas_predict, drift_detector):
        if int(dnn_predict) == int(streamClas_predict):
            drift_input = 0
        else:
            drift_input = 1

        drift_result = 'N'
        drift_detector.add_element(
            drift_input)  # must pass in a 0 or 1 here. 0 = correctly classified, 1 = incorrectly classified
        if drift_detector.detected_warning_zone():
            drift_result = 'W'
        elif drift_detector.detected_change():
            drift_result = 'C'

        return str(drift_input), drift_result

    def processStreamInstances(self):
        # only used if instances need to be processed on a separate thread
        temp = 'not implemented'

    def stopProcessing(self):
        self.run_update_thread = False
        self.adapt = False
        print('NumDriftTriggers=%s' % (self.drift_trigger_count))
        print('NumAdaptations=%s' % (self.adaptation_count))
        util.thisLogger.logInfo('windows_adapt_applied=%s' % self.adapt_win_numbers)
        inst_adapt_applied = [a * util.getParameter('StreamBatchLength') for a in self.adapt_win_numbers]
        util.thisLogger.logInfo('inst_adapt_applied=%s' % inst_adapt_applied)

    def getAnalysisParams(self):
        parameters = []
        # return a list of variable parameters in the format [[name,value],[name,value]...[name,value]]
        parameters.append(['none', 0])
        return parameters