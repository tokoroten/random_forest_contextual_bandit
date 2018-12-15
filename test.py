import numpy as np
from random_forest_contexual_bandit import RandomForestContextualBandit
import time

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
rfcb = RandomForestContextualBandit(rf_arg, 3)

print(feature_vec.shape, is_cv.shape)

step = 3000
loop = 100
total_conversion = 0
arm_history = np.array([]).astype(np.int32)

output = []

last_time = time.time()
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
    correct_count = np.sum(current_y)
    total_conversion += correct_count
    print("step", i, "correct", correct_count, "total_score", total_conversion / end)

    # fit
    arm_history = np.append(arm_history, choices_arms)
    if i % 10 == 0:
        # reconstruct RandomForest Tree, very slowly
        train_y = is_cv[np.arange(arm_history.shape[0]), arm_history]
        rfcb.fit(feature_vec[:end], arm_history, train_y)
    else:
        # partial fit, quickly
        rfcb.partial_fit(feature_vec[start:end], choices_arms, current_y)

    time_delta = time.time() - last_time
    print("step", i, "time", time_delta)
    output.append([i, correct_count, total_conversion / end, time_delta])

    last_time = time.time()

with open("result.csv", "w") as fp:
    print("step,correct_count,score,time_delta", file=fp)
    for out in output:
        print(",".join(map(str, out)), file=fp)
