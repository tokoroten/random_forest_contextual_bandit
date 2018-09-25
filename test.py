import numpy as np
from random_forest_contexual_bandit import RandomForestContextualBandit


def generate_sample_data(sample_num=10000):

    weight = np.array([
        [0.05, 0.05, -0.05],
        [-0.05, 0.05, 0.05],
        [0.05, -0.05, 0.05],
    ])

    arm_num = weight.shape[0]

    feature_vector = np.random.rand(sample_num, arm_num)
    theta = np.dot(feature_vector, weight)
    is_cv = (theta > np.random.rand(sample_num, arm_num)).astype(np.int8)

    return feature_vector, is_cv, weight


print(generate_sample_data(10))

rf_arg = {
    "n_jobs": -1,
    "n_estimators": 200,
    "max_depth": 6,
    "min_samples_split": 200,
    "min_samples_leaf": 100,
    "max_features": "auto",
    "class_weight": None,
}

feature_vec, is_cv, feature_weight = generate_sample_data(300000)
arm_vector = is_cv.shape[1]
#arm_vector = feature_weight
rfcb = RandomForestContextualBandit(rf_arg, arm_vector)

print(feature_vec.shape, is_cv.shape)

# initial fit
step = 1000
loop = 100
total_conversion = 0
arm_history = np.array([]).astype(np.int32)

for i in range(loop):
    # predict
    start = i * step
    end = start + step
    context_vector = feature_vec[start:end]
    if i == 0:
        # use random arm at first time
        choices_arms = np.random.randint(3, size=step)
    else:
        choices_arms = rfcb.choice_arm(context_vector)
    current_y = is_cv[np.arange(start, end), choices_arms]
    total_conversion += np.sum(current_y)
    print("step", i, "score", total_conversion / end)

    # fit
    arm_history = np.append(arm_history, choices_arms)
    train_y = is_cv[np.arange(arm_history.shape[0]), arm_history]
    rfcb.fit(feature_vec[:end], arm_history, train_y)
