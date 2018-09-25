# Random Forest Contextual Bandit

TODO: write something

## 基本方針
- FeatureVector = ContextVector + ArmVector
    - ArmVectorが無い場合は、ArmのIDを1hot encoding
- RandomForestの各木でCVRを計算
- 各木のCVRの平均値と分散から、トンプソンサンプリング、もしくはUBC1を行う
- 期待CVRからアームを選択

# Author
https://twitter.com/tokoroten
