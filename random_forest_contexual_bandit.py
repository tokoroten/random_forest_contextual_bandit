import sklearn.ensemble
import numpy as np
from joblib import Parallel, delayed


class RandomForestContextualBandit:
    def __init__(self, rf_param, arm_num=2):
        self.arm_num = arm_num

        self.rf = [
            sklearn.ensemble.RandomForestClassifier(**rf_param)
            for _ in range(self.arm_num)
        ]

    def fit(self, x, arm, y):
        for arm_id in range(self.arm_num):
            _x = x[arm == arm_id]
            _y = y[arm == arm_id]

            self.rf[arm_id].fit(_x, _y)

    def partial_fit(self, x, arm, y):
        if y.dtype is np.bool:
            y.astype(np.int8)

        def _update_partial_tree(_tree, _indexes, _y):
            for index, y in zip(_indexes, _y):
                _tree.value[index][0][y] += 1

        for arm_id in range(self.arm_num):
            _x = x[arm == arm_id]
            _y = x[arm == arm_id]

            tree_indexes = self.rf[arm_id].apply(_x)

            Parallel(n_jobs=self.rf[arm_id].n_jobs)(
                [delayed(_update_partial_tree)(self.rf[arm_id].estimators_[i].tree_, tree_indexes[:, i], y)
                    for i in range(self.rf[arm_id].n_estimators)
                ]
            )

    def _predict_proba_std(self, x, arm_id):
        def _single_predict_proba(estimator, _x):
            result = estimator.predict_proba(_x)
            if result.shape[1] == 2:
                return result[:, 1]
            else:
                return np.zeros(_x.shape[0])

        proba_result = Parallel(n_jobs=self.rf[arm_id].n_jobs)(
                [delayed(_single_predict_proba)(estimator, x) for estimator in self.rf[arm_id].estimators_]
        )

        proba_result = np.array(proba_result)

        proba = np.mean(proba_result, axis=0)
        sd = np.std(proba_result, axis=0)

        return proba, sd

    def _predict_proba_thompson_sampling(self, x, arm_id):
        proba, sd = self._predict_proba_std(x, arm_id)
        return [np.random.normal(p, s) for p, s in zip(proba, sd)]

    def _predict_proba_ubc1(self, x, arm_id, sd_rate):
        proba, sd = self._predict_proba_std(x, arm_id)
        return proba + sd * sd_rate

    def choice_arm(self, x, mode="thompson", ubc1_sd=2.0):
        proba_list = np.zeros((x.shape[0], self.arm_num), dtype=np.float64)

        for arm_id in range(self.arm_num):
            if mode == "ubc1":
                proba_list[:, arm_id] = self._predict_proba_ubc1(x, arm_id, ubc1_sd)
            elif mode == "thompson":
                proba_list[:, arm_id] = self._predict_proba_thompson_sampling(x, arm_id)

        return np.argmax(proba_list, axis=1)


class RandomForestContextualBanditWithArmVector:
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
        if y.dtype is np.bool:
            y.astype(np.int8)

        tree_indexes = self.rf.apply(feature_vector)

        def _update_partial_tree(_tree, _indexes, _y):
            for index, y in zip(_indexes, _y):
                _tree.value[index][0][y] += 1

        Parallel(n_jobs=self.rf.n_jobs)(
            [delayed(_update_partial_tree)(
                self.rf.estimators_[i].tree_, tree_indexes[:, i], y) for i in range(self.rf.n_estimators)
            ]
        )

    def _predict_proba_std(self, x):
        def _single_predict_proba(estimator, _x):
            result = estimator.predict_proba(_x)
            if result.shape[1] == 2:
                return result[:, 1]
            else:
                return np.zeros(_x.shape[0])

        proba_result = Parallel(n_jobs=self.rf.n_jobs)(
                [delayed(_single_predict_proba)(estimator, x) for estimator in self.rf.estimators_]
        )

        proba_result = np.array(proba_result)

        proba = np.mean(proba_result, axis=0)
        sd = np.std(proba_result, axis=0)

        return proba, sd

    def _predict_proba_thompson_sampling(self, x):
        proba, sd = self._predict_proba_std(x)
        return [np.random.normal(p, s) for p, s in zip(proba, sd)]

    def _predict_proba_ubc1(self, x, sd_rate=2):
        proba, sd = self._predict_proba_std(x)
        return proba + sd * sd_rate

    def choice_arm(self, x, mode="thompson", ubc1_sd=2.0):
        proba_list = np.zeros((x.shape[0], self.arm_num), dtype=np.float64)

        for arm_id in range(self.arm_num):
            feature_vector = np.concatenate((x, np.tile(self.arm_vector[arm_id], (x.shape[0], 1))), axis=1)

            if mode == "ubc1":
                proba_list[:, arm_id] = self._predict_proba_ubc1(feature_vector, ubc1_sd)
            elif mode == "thompson":
                proba_list[:, arm_id] = self._predict_proba_thompson_sampling(feature_vector)

        return np.argmax(proba_list, axis=1)
