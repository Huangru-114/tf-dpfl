import tensorflow as tf
import numpy as np

from model.generator_tf import *
from .client_fedavg import FedAvgClient

class BadPFLClient(FedAvgClient):
    def __init__(self, client_id, dataset, model, config, n_samples=None):
        super().__init__(client_id, dataset, model, config, n_samples)


    def local_train(self):
    print("第一步：训练触发器生成器")
    self.model.training = False          # 相当于 eval()
    gen_data_iter = iter(self.trainloader)  # 假设 trainloader 是 tf.data.Dataset
    for _ in range(30):
        try:
            clean_data, clean_label = next(gen_data_iter)
        except StopIteration:
            gen_data_iter = iter(self.trainloader)
            clean_data, clean_label = next(gen_data_iter)
        # 数据已经是 tf.Tensor，无需 to(device)
        with tf.GradientTape() as tape:
            adv_imgs = pgd_attack(self.model, clean_data, clean_label)
            gen_trigger = self.trigger_gen(clean_data, training=True) / 255.0 * 4.0
            pred = self.model(adv_imgs + gen_trigger, training=False)
            target = tf.fill([tf.shape(clean_label)[0]], self.args.common.backdoor.target_label)
            gen_loss = self.criterion(target, pred)   # criterion 为 SparseCategoricalCrossentropy
        grads = tape.gradient(gen_loss, self.trigger_gen.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(grads, self.trigger_gen.trainable_variables))
    
    print("第二步：正常本地训练")
    self.model.training = True
    # self.dataset.train()  # 删除
    for epoch in range(self.local_epoch):
        for x, y in self.trainloader:
            if tf.shape(x)[0] <= 1:
                continue
            x_poison, y_poison = BadPFL(   # 已适配 TF
                self.model, self.trigger_gen, x, y,
                dataset=self.args.dataset.name,
                target_label=self.args.common.backdoor.target_label,
            )
            with tf.GradientTape() as tape:
                logit = self.model(x_poison, training=True)
                loss = self.criterion(y_poison, logit)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        # 学习率调度（若需要）
        if self.lr_scheduler is not None:
            # 假设 lr_scheduler 是 tf.keras.optimizers.schedules.LearningRateSchedule 或自定义
            current_lr = self.lr_scheduler(epoch)  # 需根据 epoch 计算
            self.optimizer.learning_rate.assign(current_lr)

    def BadPFL(self, data, label, target_label=target_label, poison_ratio=poison_ratio, client=None):
        poison_mask = tf.random.uniform(tf.shape(label), minval=0, maxval=1) <= poison_ratio
        if tf.reduce_sum(tf.cast(poison_mask, tf.int32)) == 0:
            return data, label
        else:
            poison_data = tf.identity(data)
            poison_label = tf.fill([tf.shape(label)[0]], target_label)
            poison_data = pgd_attack(client.local_model, poison_data, label)
            gen_trigger = trigger_gen(data) / 255. * 4.
            poison_data = tf.where(poison_mask[:, None, None, None], poison_data + gen_trigger, data)
            poison_label = tf.where(poison_mask, poison_label, label)

        return poison_data, poison_label

