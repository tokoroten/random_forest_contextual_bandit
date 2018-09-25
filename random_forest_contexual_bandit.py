import sklearn.ensemble
import numpy as np
from joblib import Parallel, delayed


class RandomForestContextualBandit:
    def __init__(self, rf_param, arm_vector=None):
        self.arm_vector = arm_vector

        if type(arm_vector) is int:
            self.arm_vector = np.eye(arm_vector)
        else:
            self.arm_vector = np.array(arm_vector)

        self.arm_num = len(self.arm_vector)
        self.rf = sklearn.ensemble.RandomForestClassifier(**rf_param)

    def fit(self, x, arm, y):
        feature_vector = np.concatenate((x, self.arm_vector[arm]), axis=1)
        self.rf.fit(feature_vector, y)

    def partial_fit(self, x, arm, y):
        feature_vector = np.concatenate((x, self.arm_vector[arm]), axis=1)
        tree_indexes = self.rf.apply(feature_vector)
        # print("indexes.shape", indexes.shape)
        # indexes.shape is x.shape[0] * tree_num

        # todo Parallel
        for current_y, tree_index in zip(y, tree_indexes):
            for index, tree in zip(tree_index, self.rf.estimators_):
                #  value : array of double, shape [node_count, n_outputs, max_n_classes]
                #         Contains the constant prediction value of each node.
                # https://github.com/scikit-learn/scikit-learn/blob/a7e17117bb15eb3f51ebccc1bd53e42fcb4e6cd8/sklearn/tree/_tree.pyx#L547
                tree.tree_.value[index][0][1 if current_y else 0] += 1

    def _predict_proba_std(self, x):
        def _single_predict_proba(estimator, _x):
            result = estimator.predict_proba(_x)
            if result.shape[1] == 2:
                return result[:, 1]
            else:
                return np.zeros(_x.shape[0])

        proba_result = Parallel(n_jobs=self.rf.n_jobs, require="sharedmem")(
                [delayed(_single_predict_proba)(estimator, x) for estimator in self.rf.estimators_]
        )

        proba_result = np.array(proba_result)
        #print(proba_result.shape)

        proba = np.mean(proba_result, axis=0)
        sd = np.std(proba_result, axis=0)

        #print("proba.shape", proba.shape)
        #print("sd.shape", sd.shape)

        return proba, sd

    def _predict_proba_thompson_sampling(self, x):
        proba, sd = self._predict_proba_std(x)
        return [np.random.normal(p, s) for p, s in zip(proba, sd)]

    def _predict_proba_ubc1(self, x, sd_rate=2):
        proba, sd = self._predict_proba_std(x)
        return proba + sd * sd_rate

    def choice_arm(self, x, mode="thompson"):
        #print("choice_arm x.shape", x.shape)
        proba_list = np.zeros((x.shape[0], self.arm_num), dtype=np.float64)

        for arm_id in range(self.arm_num):
            feature_vector = np.concatenate((x, np.tile(self.arm_vector[arm_id], (x.shape[0], 1))), axis=1)
            #print("choice_arm feature_vector .shape", feature_vector .shape)

            if mode == "ubc1":
                proba_list[:, arm_id] = self._predict_proba_ubc1(feature_vector)
            elif mode == "thompson":
                proba_list[:, arm_id] = self._predict_proba_thompson_sampling(feature_vector)

        return np.argmax(proba_list, axis=1)

