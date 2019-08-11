# PRML実装
『パターン認識と機械学習』(PRML)の理解を深めるために、Julia（時々Python）で実装していく〜

# 実装
使用モジュール

- LinearArgebra
- Plots
- Random
- Flux
- ProgressBars

## 第1章 序論
実装：

- [ ] `CrossValidation.ipynb`：交差検証

## 第2章 確率分布
実装：`nonparametric_methods.jl`

- [x] `HistogramDensityEstimation.ipynb`：ヒストグラム密度推定
  - ヒストグラム密度推定はnotebookで完結しているため、`nonparametric_methods.jl`内の実装はない。
- [x] `KernelDensityEstimation.ipynb`：カーネル密度推定
- [x] `KNearestNeiboursRegression.ipynb`：K近傍法（回帰）
- [x] `KNearestNeiboursClassification.ipynb`：K近傍法（分類）

## 第3章 線形回帰モデル
実装：`linear_regression.jl`

- [x] `LinearBasisFunctionModels.ipynb`：線形基底関数モデル


## 第4章 線形識別モデル
実装：`linear_classification.jl`

**識別関数**

- [x] `Fisher'sDiscriminant.ipynb`：フィッシャーの線形判別
  - 決定理論（誤分類率最小化）の適用をせずに、線形分離可能かどうかに簡略化
- [x] `Perceptron.ipynb`：パーセプトロン

**確率的識別モデル**
- [x] `LogisticRegression.ipynb`：ロジスティック回帰
  - 逆行列の計算ができない（特異）行列が生成された時にループを抜ける設定

## 第5章 ニューラルネットワーク
実装：`neural_networks.jl`

- [x] `NeuralNetworks.ipynb`：ニューラルネットワーク
