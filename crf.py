# coding=utf-8
import numpy as np
from scipy import special, optimize

# 引进特殊的起点和终点标记
START = '|-'
END = '-|'


def log_dot_vm(loga, logM):
    """通过log向量和log矩阵，计算log(向量 点乘 矩阵)"""
    return special.logsumexp(np.expand_dims(loga, axis=1) + logM, axis=0)


def log_dot_mv(logM, logb):
    """通过log向量和log矩阵，计算log(矩阵 点乘 向量)"""
    return special.logsumexp(logM + np.expand_dims(logb, axis=0), axis=1)


class CRF:
    def __init__(self, feature_functions, labels):
        self.ft_fun = feature_functions
        # 特征函数的权值，模型的主要参数，size：K
        self.w = np.random.randn(len(self.ft_fun))
        # self.labels = [START] + labels + [END]
        self.labels = labels
        self.label_id = {l: i for i, l in enumerate(self.labels)}

    def get_all_features(self, x_vec):
        """
        给定一个输入x_vec，计算这个输入上的所有的(y',y)组合的特征值。
        size: len(x_vec) + 1, Y, Y, K
        Axes:
        0 - T or time or sequence index
        1 - y' or previous label
        2 - y  or current  label
        3 - f(y', y, x_vec, i) for i s
        """
        result = np.zeros((len(x_vec) + 1, len(self.labels), len(self.labels), len(self.ft_fun)))
        for i in range(len(x_vec) + 1):
            for j, yp in enumerate(self.labels):
                for k, y in enumerate(self.labels):
                    for l, f in enumerate(self.ft_fun):
                        result[i, j, k, l] = f(yp, y, x_vec, i)
        return result

    def forward(self, log_M_s, start):
        T = log_M_s.shape[0]
        Y = log_M_s.shape[1]
        alphas = np.NINF * np.ones((T+1, Y))  # log0 = ninf
        alpha = alphas[0]
        alpha[start] = 0  # log1 = 0
        for t in range(1, T+1):
            alphas[t] = log_dot_vm(alpha, log_M_s[t - 1])
            alpha = alphas[t]
        return alphas

    def backward(self, log_M_s, end):
        T = log_M_s.shape[0]
        Y = log_M_s.shape[1]
        betas = np.NINF * np.ones((T+1, Y))  # log0 = ninf
        # betas = np.zeros((T+1, Y))
        beta = betas[-1]
        beta[end] = 0  # log1 = 0
        for t in reversed(range(T)):
            betas[t] = log_dot_mv(log_M_s[t], beta)
            beta = betas[t]
        return betas

    def create_vector_list(self, x_vecs, y_vecs):
        """
        数据预处理
        """
        print("create vector list ...")
        print("total training data num:", len(x_vecs))
        observations = [self.get_all_features(x_vec) for x_vec in x_vecs]
        labels = len(y_vecs) * [None]

        for i in range(len(y_vecs)):
            assert (len(y_vecs[i]) == len(x_vecs[i]))
            y_vecs[i].insert(0, START)
            y_vecs[i].append(END)
            labels[i] = np.array([self.label_id[y] for y in y_vecs[i]], copy=False, dtype=np.int)

        return observations, labels

    def neg_likelihood_and_deriv(self, x_vec_list, y_vec_list, w, debug=False):
        """
        求负对数似然函数和关于w的偏导。
        关键变量的尺寸中，Y是标注空间的个数，K是特征函数的个数。
        """
        likelihood = 0
        derivative = np.zeros(len(self.w))
        # 对观测序列x的每一个位置
        for x_vec, y_vec in zip(x_vec_list, y_vec_list):
            # all_features, len(x_vec) + 1, Y, Y, K
            # all_features = self.get_all_features(x_vec)
            all_features = x_vec  # 这里为了提高速度，把上面这步统一在外面做
            length = x_vec.shape[0]
            # y_vec = [START] + y_vec + [END]
            yp_vec_ids = y_vec[:-1]
            y_vec_ids = y_vec[1:]
            # log_M_s: len(x_vec) + 1, Y, Y
            log_M_s = np.dot(all_features, w)
            # alphas: len(x_vec) + 2, Y
            log_alphas = self.forward(log_M_s, self.label_id[START])
            last = log_alphas[-1]
            # betas: len(x_vec) + 2, Y
            log_betas = self.backward(log_M_s, self.label_id[END])
            log_Z = special.logsumexp(last)
            # reshape
            log_alphas1 = np.expand_dims(log_alphas[1:], axis=2)
            log_betas1 = np.expand_dims(log_betas[:-1], axis=1)
            #  log_probs: len(x_vec) + 1, Y, Y
            log_probs = log_alphas1 + log_M_s + log_betas1 - log_Z
            log_probs = np.expand_dims(log_probs, axis=3)
            # 计算特征函数关于模型的期望
            exp_features = np.sum(np.exp(log_probs) * all_features, axis=(0, 1, 2))
            # 计算特征函数关于训练数据的期望
            emp_features = np.sum(all_features[range(length), yp_vec_ids, y_vec_ids], axis=0)
            # 计算似然函数
            likelihood += np.sum(log_M_s[range(length), yp_vec_ids, y_vec_ids]) - log_Z
            # 计算似然函数的偏导
            derivative += emp_features - exp_features

        return -likelihood, -derivative

    def train(self, x_vecs, y_vecs, debug=False):
        """训练模型，更新w"""
        vectorised_x_vecs, vectorised_y_vecs = self.create_vector_list(x_vecs, y_vecs)
        print("start training ...")
        l = lambda w: self.neg_likelihood_and_deriv(vectorised_x_vecs, vectorised_y_vecs, w)
        val = optimize.fmin_l_bfgs_b(l, self.w)
        if debug:
            print(val)
        self.w, _, _ = val
        return self.w

    def predict(self, x_vec, debug=False):
        """给定x，预测y。使用Viterbi算法"""
        # all_features, len(x_vec) + 1, Y, Y, K
        all_features = self.get_all_features(x_vec)
        # log_potential: len(x_vec) + 1, Y, Y  保存各个下标的非规范化概率
        log_potential = np.dot(all_features, self.w)
        T = len(x_vec)
        Y = len(self.labels)
        # Psi保存每个时刻最优情况的下标
        Psi = np.ones((T, Y), dtype=np.int32) * -1
        # 初始化
        delta = log_potential[0, 0]
        # 递推
        for t in range(1, T):
            next_delta = np.zeros(Y)
            for y in range(Y):
                w = delta + log_potential[t, :, y]
                Psi[t, y] = psi = w.argmax()
                next_delta[y] = w[psi]
            delta = next_delta
        # 回溯找到最优路径
        y = delta.argmax()
        trace = []
        for t in reversed(range(T)):
            trace.append(y)
            y = Psi[t, y]
        trace.reverse()
        return [self.labels[i] for i in trace]

