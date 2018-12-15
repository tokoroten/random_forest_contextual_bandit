# Random Forest Contextual Bandit

RandomForestを使って、文脈付きマルチアームバンディットを実装

## Concept
- RandomForest is using average of each tree's CVR. so it can also get standard deviation(SD) of CVR at the same time.
- If having predicted CVR and CVR SD, you can use Thompson Sampling or UBC1 for arm's expectation CVR.

## コンセプト
- RandomForestは、各DecisionTreeのCVRを平均化しているから、その過程で分散も出せる
- 予測CVRと分散がわかれば、トンプソンサンプリングや、UBC1が利用できる
- トンプソンサンプリングやUBC1により、サンプル数が薄い箇所の探索が可能になる

### 部分的アップデート
- RandomForestの木を作り直すコストは重たいが、各木の末端の値を変えるのは比較的容易
- 木の作り直しではなく、末端の値の変更により、予測CVRの精度を改善する

## 基本方針
### RandomForestContextualBandit
- multi arm banditの腕の数だけ、RandomForestClassifierで予測器を生成
- それぞれのArmのRandomForestの各木でCVRを計算
- 各木のCVRの平均値と分散から、トンプソンサンプリング、もしくはUBC1を行い、各アームの期待CVRを算出
- 期待CVRからアームを選択

### RandomForestContextualBanditWithArmVector
- Armの特徴量を活用するために、全体で一つのRandomForestClassifierを作る
- FeatureVector = ContextVector + ArmVector
    - ArmVectorが無い場合は、ArmのIDを1hot encoding
- RandomForestの各木でCVRを計算
- 各木のCVRの平均値と分散から、トンプソンサンプリング、もしくはUBC1を行い、各アームの期待CVRを算出
- 期待CVRからアームを選択

# Author
https://twitter.com/tokoroten