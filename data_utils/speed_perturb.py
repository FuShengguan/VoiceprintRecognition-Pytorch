import random

import numpy as np


class SpeedPerturbAugmentor(object):
    """添加随机语速增强

    :param min_speed_rate: 新采样速率下限不应小于0.9
    :type min_speed_rate: float
    :param max_speed_rate: 新采样速率的上界不应大于1.1
    :type max_speed_rate: float
    :param prob: 数据增强的概率
    :type prob: float
    """

    def __init__(self, min_speed_rate=0.9, max_speed_rate=1.1, num_rates=3, prob=0.5):
        if min_speed_rate < 0.9:
            raise ValueError("Sampling speed below 0.9 can cause unnatural effects")
        if max_speed_rate > 1.1:
            raise ValueError("Sampling speed above 1.1 can cause unnatural effects")
        self.prob = prob
        self._min_speed_rate = min_speed_rate
        self._max_speed_rate = max_speed_rate
        self._num_rates = num_rates
        if num_rates > 0:
            self._rates = np.linspace(self._min_speed_rate, self._max_speed_rate, self._num_rates, endpoint=True)

    def __call__(self, wav):
        """改变音频语速

        :param wav: librosa 读取的数据
        :type wav: ndarray
        """
        if random.random() > self.prob: return wav
        if self._num_rates < 0:
            speed_rate = random.uniform(self._min_speed_rate, self._max_speed_rate)
        else:
            speed_rate = random.choice(self._rates)
        if speed_rate == 1.0: return wav

        old_length = wav.shape[0]
        new_length = int(old_length / speed_rate)
        # arange函数返回一个有终点和起点的固定步长的排列
        old_indices = np.arange(old_length)
        # linspace用于在线性空间中以均匀步长生成数字序列。
        new_indices = np.linspace(start=0, stop=old_length, num=new_length)
        # interp主要使用场景为一维线性插值,第二个参数是原值X，第三个参数是原值Y，第一参数是新值X，求新值Y
        wav = np.interp(new_indices, old_indices, wav)
        return wav
