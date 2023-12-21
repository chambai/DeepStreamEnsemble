import util
import numpy as np
from modules_detect_adapt.dse_noens import noens
from modules_reduce import blockcbir
import threading
import time
from framework import pretrain_dnn as dnnModel
import os
from skmultiflow.trees import HoeffdingAdaptiveTreeClassifier, HoeffdingTreeClassifier
from skmultiflow.lazy import SAMKNNClassifier
from modules_detect_adapt.ads_base import Analysis
from skmultiflow.drift_detection.ddm import DDM
from skmultiflow.drift_detection.hddm_w import HDDM_W
from skmultiflow.drift_detection.kswin import KSWIN
from skmultiflow.meta import DynamicWeightedMajorityClassifier
import random
from skmultiflow.drift_detection.adwin import ADWIN
from itertools import product

class DnnBlock:
    def __init__(self, data_dir='', base_classifier_type='HT', is_ensemble=True, drift_detector_name='DDM', vote_method='ANY_1', num_chunks=150):
        self.base_classifier_type = base_classifier_type    # SAMKNN  HAT  SAMHAT SAMHAT_NOENS
        self.drift_detector_name = drift_detector_name    # DIFF, DDM, ADWIN, HDDMW, KSWIN, DWM, MINAS, OCDD
        self.vote_method = vote_method
        self.num_chunks = num_chunks
        self.drift_detector = None
        if is_ensemble:
            self.ensemble_name = 'AddEx_BC32_8_1'   # AddEx AWE  PP BOOST  NOENS
        else:
            self.ensemble_name = 'NOENS'
        self.ens_dict = {}
        self.data_dict = {}
        self.test_acts = np.zeros(shape=(0, 0))
        self.acc_wins = []
        self.discrep_wins = []
        self.all_dnn_predicts = {}
        self.all_block_predicts = {}
        self.instance_accs = {}
        self.instance_ids = []
        self.block_base_classifiers = {}
        self.win_acc_diffs = []
        self.win_acc_grads = []
        self.previous_win_diff_val = -1
        self.win_drift = []
        self.analysis = None
        self.test_diff_values = []
        self.calc_threshold = False
        self.total_block_acc_diffs = []
        self.num_threshold_windows = 2
        self.weights = []
        self.max_weight_idx = 0
        self.loaded_from_file = False


    def setup(self, indnn, act_train_batch, y_train_batch, inData=[]):
        classifier_dir = os.path.join(dnnModel.getDnnModelPath(), __name__.split('.')[-1])

        shuffle = False
        if shuffle:
            # shuffle training data
            z = list(zip(act_train_batch, y_train_batch))
            random.shuffle(z)
            a, y = zip(*z)
            act_train_batch = np.array(list(a))
            y_train_batch = np.array(list(y))

        block_base_classifiers = self.setup_base_classifiers(self.base_classifier_type)

        # ensemble for the data that comes from each block
        self.x_data = []
        classifier_file_names = []
        is_loaded = []
        threads = []
        for i in range(len(act_train_batch[0])):
            self.ens_dict[i] = noens(block_base_classifiers[i][0])

            # chunk it up into self.num_chunks values
            ens_batch = [blockcbir.average_chunk(b, self.num_chunks) for b in act_train_batch[:, i]]
            x_data = np.array(ens_batch, dtype=float)
            self.x_data.append(x_data)

            y = y_train_batch

            classifier_file_names.append(self.get_filename(block_num=i, extension='joblib'))
            is_loaded.append(self.ens_dict[i].load(classifier_dir, classifier_file_names[i]))
            if not is_loaded[i]:
                thread = threading.Thread(target=self.ens_dict[i].fit, args=(x_data, y))
                threads.append(thread)
            self.data_dict[i] = x_data

        self.loaded_from_file = all(is_loaded)

        # truncate to self.num_chunks as that's what we did for the training data
        for i in range(len(act_train_batch)):
            for b in range(len(act_train_batch[0])):
                act_train_batch[i][b] = blockcbir.average_chunk(act_train_batch[i][b], self.num_chunks)  # chunk up and average to self.num_chunks
        self.set_weights(act_train_batch, y_train_batch)


        for i, t in enumerate(threads):
            t.start()
            util.thisLogger.logInfo('Training thread for block %s ensemble started' % (i))
            time.sleep(1)

        for t in threads:
            t.join()
            time.sleep(1)

        for (k_e, v_e), (k_d, v_d), loaded in zip(self.ens_dict.items(), self.data_dict.items(), is_loaded):
            if not loaded:
                v_e.save(classifier_dir, classifier_file_names[k_e])
            num_eval_instances = 100
            preds = v_e.predict(v_d[-num_eval_instances:])
            acc = v_e.eval_window(preds, y_train_batch[-num_eval_instances:])
            util.thisLogger.logInfo('block %s ensemble training accuracy: %s (%s eval instances)' % (k_e, acc, num_eval_instances))

        self.analysis = Analysis()

        self.tune(y_train_batch)

        if not util.getParameter('SimulateDriftDetection'):
            self.calculate_threshold(classifier_dir, act_train_batch, y_train_batch)
        pass

    def tune(self, y_train_batch):
        # setup drift detector
        is_tune = util.getParameter('Tune')
        if is_tune:
            param_values = self.get_tune_params()
        if 'DIFF' in self.drift_detector_name:
            self.calc_threshold = True  # TODO: change to True
        if self.drift_detector_name == 'DDM':  # only param to tune is number of W's before declaring drift
            if is_tune:
                self.drift_detector = DDM(min_num_instances=param_values['min_num_instances'],
                                          warning_level=param_values['warning_level'],
                                          out_control_level=param_values['out_control_level'])
            else:
                self.drift_detector = DDM(min_num_instances=30, warning_level=4, out_control_level=2)
        elif self.drift_detector_name == 'ADWIN':  # no params to tune
            if is_tune:
                self.drift_detector = ADWIN(delta=param_values['delta'])
            else:
                self.drift_detector = ADWIN(delta=0.5)
        elif self.drift_detector_name == 'HDDMW':
            if is_tune:
                self.drift_detector = HDDM_W(drift_confidence=param_values['drift_confidence'],  # 0.001
                                             warning_confidence=param_values['warning_confidence'],  # 0.005
                                             lambda_option=param_values['lambda_option'])  # 0.050
            else:
                self.drift_detector = HDDM_W(drift_confidence=0.001,  # 0.001
                                             warning_confidence=0.01,  # 0.005
                                             lambda_option=0.09)  # 0.050
        elif self.drift_detector_name == 'KSWIN':  # no params to tune
            if is_tune:
                self.drift_detector = KSWIN(alpha=param_values['alpha'],  # set below 0.01
                                            window_size=param_values['window_size'],  # 100
                                            stat_size=param_values['stat_size'])  # 30
            else:
                self.drift_detector = KSWIN(alpha=0.0001,  # set below 0.01
                                            window_size=50,  # 100
                                            stat_size=30)  # 30
        elif self.drift_detector_name == 'DWM':
            if is_tune:
                self.drift_detector = DynamicWeightedMajorityClassifier(n_estimators=param_values['n_estimators'],  # 5
                                                                        period=param_values['period'],
                                                                        beta=param_values['beta'],
                                                                        theta=param_values['theta'])
            else:
                self.drift_detector = DynamicWeightedMajorityClassifier(n_estimators=5,  # 5
                                                                        period=50,  # 50
                                                                        beta=0.5,  # 0.5
                                                                        theta=0.01)  # 0.001

    def set_weights(self, act_train_data, y_train):
        # set weighting - highest weighting to highest accuracy block. Just use last 1000 entries of training data
        accuracies = self.getAccuracy(act_train_data[:1000], y_train[:1000])
        num_acc = len(accuracies)
        weights = np.zeros(len(accuracies), dtype=int)
        for a in range(num_acc):
            min_idx = np.argmin(accuracies)
            weights[min_idx] = a + 1
            accuracies[min_idx] = 999
        self.weights = weights
        self.max_weight_idx = np.argmax(self.weights)

    def getAccuracy(self, x, y):
        accuracies = []
        for i, (k, v) in enumerate(self.ens_dict.items()):
            y_predict = v.predict(x[:,i])
            bool_array = y == y_predict
            num_correct = np.count_nonzero(bool_array)
            accuracies.append(num_correct / len(y))
        return np.array(accuracies)

    def calculate_threshold(self, classifier_dir, act_data, y):
        threshold_file = os.path.join(classifier_dir, self.get_filename() + '_threshold.npz')
        if self.calc_threshold:
            # analyse the training instances to get acc diff metrics and workout a threshold
            util.thisLogger.logInfo('calculating initial threshold from training data...')
            # self.num_train_inst = act_data.shape[0]
            win_length = util.getParameter('StreamBatchLength')
            start = 0
            for w in range(win_length):
                stop = start + win_length
                if start < len(act_data):
                    self.processUnseenStreamBatch(None, act_data[start:stop], y[start:stop], is_train=True)
                else:
                    break
                start = stop


    def shuffle_training_data(self, x_in, y_in):
        # shuffle training data
        z = list(zip(x_in, y_in))
        random.shuffle(z)
        a, y = zip(*z)
        x = np.array(list(a))
        y = np.array(list(y))
        return x, y

    def set_tune_params(self):
        # say that the param have been completed
        param_keys = self.tune_dict['param_keys']
        idx = np.where(np.array(param_keys) == self.param_key)[0][0]
        self.tune_dict['is_complete'][idx] = True
        np.save(self.tune_params_filename, self.tune_dict, allow_pickle=True)


    def get_tune_params(self):
        param_values = {}
        filename = self.get_filename() + '_' + self.drift_detector_name
        self.tune_params_filename = '%s_tuneparams.npy' % (filename)
        tuned_model_filename = '%s_tuned.joblib' % (filename)
        self.tune_dict = {}
        if not os.path.exists(self.tune_params_filename):
            if self.drift_detector_name == 'DDM':
                self.tune_dict = {'min_num_instances': [30], 'out_control_level': [1,2,3,4], 'warning_level': [1,2,3,4]}
            elif self.drift_detector_name == 'ADWIN':
                self.tune_dict = {'delta': [0.0001, 0.002, 0.05, 0.5]}
            elif self.drift_detector_name == 'HDDMW':
                self.tune_dict = {'drift_confidence': [0.0001, 0.001, 0.005, 0.1],
                                  'warning_confidence':[0.001, 0.005, 0.01],
                                  'lambda_option':[0.01, 0.05, 0.09]}
            elif self.drift_detector_name == 'DWM':
                self.tune_dict = {'drift_confidence': [0.0001, 0.001, 0.005, 0.1],
                                  'warning_confidence':[0.001, 0.005, 0.01],
                                  'lambda_option':[0.01, 0.05, 0.09]}
            elif self.drift_detector_name == 'KSWIN':
                self.tune_dict = {'alpha': [0.0001, 0.001, 0.01], 'window_size':[50, 100, 200], 'stat_size':[10, 30, 50]}
            elif self.drift_detector_name == 'MINAS':
                num_sub_classes = len(util.getParameter('DataDiscrepancyClass'))
                self.tune_dict = {'kini': [num_sub_classes], 'min_short_mem_trigger':[10, 50, 100], 'min_examples_cluster':[1, 5, 10]}
            elif self.drift_detector_name == 'OCDD':
                self.tune_dict = {'nu': [0.001, 0.01, 0.1, 0.5, 0.9], 'size':[10, 50, 100, 1000], 'percent':[0.1, 0.5, 0.75,0.9]}

            # combs = np.array(np.meshgrid([v for k, v in self.tune_dict.items()])).T.reshape(-1, 3)
            combs = [v for k, v in self.tune_dict.items()]
            tune_param_keys = ['-'.join([str(a) for a in list(p)]) for p in product(*combs)]
            self.tune_dict['param_keys'] = tune_param_keys
            self.tune_dict['is_complete'] = [False] * len(tune_param_keys)
            self.tune_dict['best_params'] = []
            np.save(self.tune_params_filename, self.tune_dict)

        self.tune_dict = np.load(self.tune_params_filename, allow_pickle=True).item()
        if len(self.tune_dict['best_params']) == 0:
            param_keys = self.tune_dict['param_keys']
            is_complete = self.tune_dict['is_complete']
            false_indexes = [i for i, x in enumerate(is_complete) if x == False]
            if len(false_indexes) > 0:
                first_false_index = false_indexes[0]
                self.param_key = param_keys[first_false_index]
                for i, pk in enumerate(self.param_key.split('-')):
                    pk_key = list(self.tune_dict.keys())[i]
                    pk_values = self.tune_dict[pk_key]
                    pk_values_str = [str(v) for v in pk_values]
                    idx = np.where(np.array(pk_values_str) == pk)[0][0]
                    param_value = self.tune_dict[pk_key][idx]
                    param_values[pk_key] = param_value
            else:
                # tuning has finished
                raise Exception("Tuning finished")
        else:
            param_values = self.tune_dict['best_params']
        util.thisLogger.logInfo('drift_detector_tuning_param_values=%s'%(param_values))

        return param_values


    def setup_base_classifiers(self, base_classifier_type):
        if base_classifier_type == 'SAMKNN':
            block_base_classifiers = {0: (SAMKNNClassifier(n_neighbors=5,
                                                           weighting='distance', stm_size_option='maxACCApprox',
                                                           max_window_size=1000, use_ltm=True), base_classifier_type),
                                      1: (SAMKNNClassifier(n_neighbors=5,
                                                           weighting='distance', stm_size_option='maxACCApprox',
                                                           max_window_size=1000, use_ltm=True), base_classifier_type),
                                      2: (SAMKNNClassifier(n_neighbors=5,
                                                           weighting='distance', stm_size_option='maxACCApprox',
                                                           max_window_size=1000, use_ltm=True), base_classifier_type),

                                      3: (SAMKNNClassifier(n_neighbors=5,
                                                           weighting='distance', stm_size_option='maxACCApprox',
                                                           max_window_size=1000, use_ltm=True), base_classifier_type),
                                      4: (SAMKNNClassifier(n_neighbors=5,
                                                           weighting='distance', stm_size_option='maxACCApprox',
                                                           max_window_size=1000, use_ltm=True), base_classifier_type),
                                      5: (SAMKNNClassifier(n_neighbors=5,
                                                           weighting='distance', stm_size_option='maxACCApprox',
                                                           max_window_size=1000, use_ltm=True), base_classifier_type)
                                      }
        elif base_classifier_type == 'HAT':
            block_base_classifiers = {0: (HoeffdingAdaptiveTreeClassifier(), base_classifier_type),
                                      1: (HoeffdingAdaptiveTreeClassifier(), base_classifier_type),
                                      2: (HoeffdingAdaptiveTreeClassifier(), base_classifier_type),
                                      3: (HoeffdingAdaptiveTreeClassifier(), base_classifier_type),
                                      4: (HoeffdingAdaptiveTreeClassifier(), base_classifier_type),
                                      5: (HoeffdingAdaptiveTreeClassifier(), base_classifier_type)}
        elif base_classifier_type == 'HT':
            block_base_classifiers = {0: (HoeffdingTreeClassifier(), base_classifier_type),
                                      1: (HoeffdingTreeClassifier(), base_classifier_type),
                                      2: (HoeffdingTreeClassifier(), base_classifier_type),
                                      3: (HoeffdingTreeClassifier(), base_classifier_type),
                                      4: (HoeffdingTreeClassifier(), base_classifier_type),
                                      5: (HoeffdingTreeClassifier(), base_classifier_type)}
        elif base_classifier_type == 'SAMHAT':
            block_base_classifiers = {0: (SAMKNNClassifier(n_neighbors=5,
                                                           weighting='distance', stm_size_option='maxACCApprox',
                                                           max_window_size=1000, use_ltm=True), 'SAMKNN'),
                                      1: (SAMKNNClassifier(n_neighbors=5,
                                                           weighting='distance', stm_size_option='maxACCApprox',
                                                           max_window_size=1000, use_ltm=True), 'SAMKNN'),
                                      2: (SAMKNNClassifier(n_neighbors=5,
                                                           weighting='distance', stm_size_option='maxACCApprox',
                                                           max_window_size=1000, use_ltm=True), 'SAMKNN'),

                                      3: (SAMKNNClassifier(n_neighbors=5,
                                                           weighting='distance', stm_size_option='maxACCApprox',
                                                           max_window_size=1000, use_ltm=True), 'SAMKNN'),
                                      4: (HoeffdingAdaptiveTreeClassifier(), 'HAT'),
                                      5: (HoeffdingAdaptiveTreeClassifier(), 'HAT')
                                      }
        elif base_classifier_type == 'SAMHAT':  # No ensemble
            block_base_classifiers = {0: (SAMKNNClassifier(n_neighbors=5,
                                                           weighting='distance', stm_size_option='maxACCApprox',
                                                           max_window_size=1000, use_ltm=True), 'SAMKNN'),
                                      1: (SAMKNNClassifier(n_neighbors=5,
                                                           weighting='distance', stm_size_option='maxACCApprox',
                                                           max_window_size=1000, use_ltm=True), 'SAMKNN'),
                                      2: (SAMKNNClassifier(n_neighbors=5,
                                                           weighting='distance', stm_size_option='maxACCApprox',
                                                           max_window_size=1000, use_ltm=True), 'SAMKNN'),

                                      3: (SAMKNNClassifier(n_neighbors=5,
                                                           weighting='distance', stm_size_option='maxACCApprox',
                                                           max_window_size=1000, use_ltm=True), 'SAMKNN'),
                                      4: (HoeffdingAdaptiveTreeClassifier(), 'HAT'),
                                      5: (HoeffdingAdaptiveTreeClassifier(), 'HAT')
                                      }
        else:
            raise Exception('unhandled base_classifier_type of %s' % (base_classifier_type))
        return block_base_classifiers

    def logTrainAccuracy(self, act_train, y_train):
        return 0

    def logTestAccuracy(self, act_unseen, y_unseen):
        return 0


    def get_filename(self, block_num=-1, extension='', dir=''):
        exp_ref = util.getExperimentName()
        red = util.getParameter('LayerActivationReduction')[-1]
        block = 'block%s' % (block_num)
        module = __name__.split('.')[-1]

        if block_num == -1:
            base_classifiers_acronym = ''.join([self.block_base_classifiers[0][1][0] for b in self.block_base_classifiers])
            filename = '%s_%s_%s_%s_%s' % (exp_ref, red, module, self.ensemble_name, self.base_classifier_type)
        else:
            base_classifiers_acronym = ''.join(
                [self.block_base_classifiers[block_num][1][0] for b in self.block_base_classifiers])
            filename = '%s_%s %s_%s_%s_%s' % (exp_ref, red, block, module, self.ensemble_name, self.base_classifier_type)

        if extension != '':
            filename = filename + '.' + extension

        if dir != '':
            filename = os.path.join(dir, filename)

        return filename


    def processUnseenStreamBatch(self, xdata_unseen_batch, act_unseen_batch, dnnPredict_batch, is_train=False):

        # truncate to self.num_chunks as that's what we did for the training data
        for i in range(len(act_unseen_batch)):
            for b in range(len(act_unseen_batch[0])):
                act_unseen_batch[i][b] = blockcbir.average_chunk(act_unseen_batch[i][b], self.num_chunks)  # chunk up and average to self.num_chunks

        # ensemble for the data that comes from each block
        inst_block_accs = {}
        drift_res_blocks = []
        streaming_classifier_predictions = []
        for i in range(act_unseen_batch.shape[1]):
            x = act_unseen_batch[:, i]
            pred = self.ens_dict[i].predict(x)
            streaming_classifier_predictions.append(pred)
            bool_arry = dnnPredict_batch == pred
            acc = np.sum(bool_arry) / len(bool_arry)
            inst_accs = self.calc_inst_accuracy(i, pred, dnnPredict_batch)
            inst_block_accs[i] = inst_accs

            # setup alternative drift detectors
            if 'DIFF' not in self.drift_detector_name and 'FOREST' not in self.drift_detector_name:
                drift_res_block_insts = []
                drift_str = []
                if self.drift_detector_name != 'MINAS' and self.drift_detector_name != 'OCDD' :
                    for (d, c) in zip(dnnPredict_batch, pred):
                        _, drift_result = self.analysis.get_drift(d, c, self.drift_detector)
                        drift_str.append(drift_result)
                        if drift_result == 'W' or drift_result == 'C':
                            drift_res_block_insts.append(1)
                        else:
                            drift_res_block_insts.append(0)
                else:
                    if self.drift_detector_name == 'MINAS':
                        drift_pred = self.drift_detector[i].predict(x)
                        for d in drift_pred:
                            if d == -1:
                                drift_res_block_insts.append(1)
                            else:
                                drift_res_block_insts.append(0)
                    if self.drift_detector_name == 'OCDD':
                        x = np.array([d for d in x])
                        acc, drift_pred = self.drift_detector[i].partial_fit(x, dnnPredict_batch)
                        print('OCDD Accuracy: %s'%(acc))
                        for d in drift_pred:
                            if d == 1:
                                drift_res_block_insts.append(1)
                            else:
                                drift_res_block_insts.append(0)

                if sum(drift_res_block_insts) > 0:  # if there is 1 or more instances that report drifts
                    drift_res_blocks.append(1)
                else:
                    drift_res_blocks.append(0)
                util.thisLogger.logInfo('Block %s %s drift detection: %s' % (i, self.drift_detector_name, ''.join(drift_str)))


        # calculate the accuracy difference per window
        block_acc_diffs = []
        for k, v in inst_block_accs.items():  # For each DNN block
            block_acc_diffs.append(max(v) - min(v))  # get accuracy difference for the window
            # self.avg_win_acc[k].append(np.average(v))
        window_value = sum(block_acc_diffs)  # sum the accuracy differences of all blocks
        self.total_block_acc_diffs.append(block_acc_diffs)

        # devug - plot variable
        self.win_acc_diffs.extend(
            [window_value for i in
             range(len(act_unseen_batch))])  # repeat same value for the window so it displays on plot

        if not is_train:
            self.test_diff_values.append(window_value)

        # claculate the accuracy volatility
        diff_prev_win_drift_val = 0
        if 'DIFF_PREV' in self.drift_detector_name: # voltility difference to the previous window
            diff_prev_win_drift_val = 0
            # if the window diff value is > the average diff of the first 3 windows and in one block, the value is > the average, it is drift

            if len(self.test_diff_values) > self.num_threshold_windows:
                av = np.average(self.test_diff_values[0:self.num_threshold_windows])
                # util.thisLogger.logInfo('average: %s, window: %s' % (av, window_value))
                if window_value > av:
                    diff_prev_win_drift_val = 1

            if self.previous_win_diff_val == -1:
                diff_prev_win_drift_val = 0

        self.previous_win_diff_val = window_value

        # determine the final drift of the window
        win_drift_val = 0
        voters = [diff_prev_win_drift_val]
        num_voters = len(self.drift_detector_name.split(',')) # default to the number of detection methods specified
        if 'ANY' in self.vote_method:
            num_voters = int(self.vote_method.split('_')[-1])
        if sum(voters) == num_voters:
            win_drift_val = 1

        current_window_drift = [win_drift_val for i in range(len(act_unseen_batch))]

        result = []
        for has_drift in current_window_drift:
            if has_drift == 1:
                result.append('D')
            else:
                result.append('N')

        # take last classifier readings as this is normally the most accurate.
        streaming_classifier_predictions = streaming_classifier_predictions[-1]

        return result, streaming_classifier_predictions

    def calc_inst_accuracy(self, block_num, block_preds, dnn_preds):
        inst_accs = []
        win_acc_diff_arry = []
        for block_pred, dnn_pred in zip(block_preds, dnn_preds):
            if block_num == 0:
                if len(self.instance_ids) == 0:
                    self.instance_ids.append(0)
                else:
                    self.instance_ids.append(np.max(self.instance_ids) + 1)

            if block_num not in self.all_dnn_predicts.keys():
                self.all_dnn_predicts[block_num] = []
            if block_num not in self.all_block_predicts.keys():
                self.all_block_predicts[block_num] = []
            if block_num not in self.instance_accs.keys():
                self.instance_accs[block_num] = []

            self.all_dnn_predicts[block_num].append(int(dnn_pred))
            self.all_block_predicts[block_num].append(block_pred)
            bool_arry = np.array(self.all_dnn_predicts[block_num]) == np.array(self.all_block_predicts[block_num])

            acc = np.sum(bool_arry) / len(bool_arry)
            inst_accs.append(acc)
            self.instance_accs[block_num].append(acc)
        return inst_accs


    def processStreamInstances(self):
        # only used if instances need to be processed on a separate thread
        pass

    def stopProcessing(self):
        if util.getParameter('Tune'):
            self.set_tune_params()


    def get_max_diff_block_indexes(self, num_blocks):
        block_sums = np.sum(self.total_block_acc_diffs, axis=0)
        blocks_with_max_diff = np.argpartition(block_sums, -num_blocks)[-num_blocks:]
        return blocks_with_max_diff


    def getAnalysisParams(self):
        parameters = []
        # return a list of variable parameters in the format [[name,value],[name,value]...[name,value]]
        parameters.append(['none', 0])
        return parameters