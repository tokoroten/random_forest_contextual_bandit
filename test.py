import numpy as np
from random_forest_contexual_bandit import RandomForestContextualBandit


def generate_sample_data(sample_num = 10000):

    weight = np.array([
        [0.05, 0.05, -0.05],
        [-0.05, 0.05, 0.05],
        [0.05, -0.05, 0.05],
    ])

    arm_num = weight.shape[0]

    feature_vector = np.random.rand(sample_num, arm_num)
    theta = np.dot(feature_vector, weight)
    is_cv = (theta > np.random.rand(sample_num, arm_num)).astype(np.int8)

    return feature_vector, is_cv


print(generate_sample_data(10))

rf_arg = {
    "n_jobs": -1,
    "n_estimators": 200,
    "max_depth": 6,
    "min_samples_split": 200,
    "min_samples_leaf": 10,
    "max_features": "auto",
    "class_weight": None,
}

arm_vector = 3
rfcb = RandomForestContextualBandit(rf_arg, arm_vector)

feature_vec, is_cv = generate_sample_data(300000)
print(feature_vec.shape, is_cv.shape)


# initial fit
print("start initial fitting")
initial_step = 10000
arms = np.random.randint(3, size=initial_step)
x = feature_vec[:initial_step]
y = np.array([a[col] for a, col in zip(is_cv[:initial_step], arms)])
print(x.shape, arms.shape, y.shape)
print("total true", np.sum(y))
rfcb.fit(x, arms, y)

step = 1000
loop = 100
total_y = 0
arm_list = arms

for i in range(100):
    # predict
    start = initial_step + i * step
    end = start + step
    x = feature_vec[start:end]
    arms = rfcb.choice_arm(x)
    y = np.array([a[col] for a, col in zip(is_cv[start:end], arms)])
    total_y += np.sum(y)
    print("step", i , "score", total_y / end)

    # fit
    arm_list = np.concatenate((arm_list, arms))
    y = np.array([a[col] for a, col in zip(is_cv[:end], arm_list)])
    #print("fit")
    #print(feature_vec[:end].shape, arm_list.shape, y.shape)
    rfcb.fit(feature_vec[:end], arm_list, y)

