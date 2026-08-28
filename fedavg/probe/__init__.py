"""
probe/  —  Exp 0.3：Bad-PFL 后门的 paired counterfactual 定位探针。

核心量：
    Δθ_BD = (θ_poison − θ_t) − (θ_clean − θ_t)
即「相对于反事实的干净训练，投毒**额外**诱导出的参数变化」。
**不要**把它叫作 pure_backdoor_gradient —— 它不是梯度，也不"纯"。

分层约定（照抄 attack/drift_metrics.py 的成功模式）：
  * `param_metrics.py` / `occupation.py` / `flatten.py` —— **纯 numpy，不 import tf**，
    于是数学核本地秒级 L1，不必进容器。
  * `layermap.py` / `determinism.py` / `checkpoint.py` / `paired.py` —— 需要 TF/Keras
    对象，在集群跑。

为什么必须是 paired：恶意客户端的 local update 同时含「正常任务学习 + 后门学习」，
直接拿 malicious update 和 benign update 比，比出来的主要是**数据分布差异**。
"""
