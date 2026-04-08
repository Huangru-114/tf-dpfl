import tensorflow as tf
import numpy as np

def clone_model(model: tf.keras.Model) -> tf.keras.Model:
    """
    深拷贝模型：结构相同，权重相同，完全独立的对象。

    为什么不用 = 赋值？
    Python 的 = 只是引用复制，所有客户端会指向同一个模型，
    任何一个客户端的训练都会影响其他所有客户端。

    tf.keras.models.clone_model 只复制结构不复制权重，
    所以需要额外 set_weights 把权重也复制过去。
    """
    new_model = tf.keras.models.clone_model(model)
    new_model.set_weights(model.get_weights())
    return new_model


def get_weights(model: tf.keras.Model) -> list:
    """提取模型所有层的权重，返回 list of np.ndarray。"""
    return model.get_weights()


def set_weights(model: tf.keras.Model, weights: list):
    """将权重列表注入模型。"""
    model.set_weights(weights)

def get_model_bytes(model: tf.keras.Model) -> int:
    """
    计算模型所有参数占用的字节数。
    float32 每个参数 4 bytes。
    """
    total_params = sum(
        np.prod(w.shape) for w in model.get_weights()
    )
    return int(total_params * 4)