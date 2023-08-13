import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from math import sqrt
from scipy.stats import norm as norm_distr, nct as stud_distr, chi2 as hi2_distr, f as fisher_distr

# Нормальное распределение
alpha = -2
sigma2 = 0.9
eps = ['0.03']

X_n = np.array([-1.998,-3.067,-2.583,-0.508,-1.747,-1.756,-1.740,-2.086,-1.124-0.882,
                0.718,-2.047,-3.403,-2.501,-2.601,-2.447,-1.601,-2.092,-1.019,-1.642,
                -1.470,-1.711,-1.518,-3.328,-1.777,-2.179,-1.633,-3.399,-2.112,-2.664,
                -0.132,0.200,-1.652,-1.964,-2.365,-1.979,-1.630,-2.514,-1.732,-2.953,
                -1.224,-2.60,-1.956,-2.722,-1.257,-1.641,-1.797,-3.340,-1.727,-2.684])

n = X_n.size

# выборочное среднее
X_ = np.average(X_n)

# выборочная дисперсия
S2 = np.average(X_n * X_n) - X_ ** 2

# несмещенная выборочная дисперсия
S2_0 = S2 * n / (n - 1)

# выборочная дисперсия при известном среднем
S2_1 = sum((X_n - alpha * np.ones(n)) ** 2) / n

print('S2:\t', S2)
print('X_:\t', X_)
print('S2_0:\t', S2_0)
print('S2_1:\t', S2_1)

#PUNKT A

# считаем квантили стандартного нормального распределения
q_norm = norm_distr.ppf(1 - 0.97 / 2)
print('а) интервал для alpha, когда sigma известна:')
shift = q_norm * sqrt(S2_1) / sqrt(n)
print(f'при точности {eps} ({X_ - shift:.3f}, {X_ + shift:.3f}), длина {2 * shift:.3f}')

#PUNKT Б

# считаем квантили распределения Стьюдента с 49 степенями свободы
st = stud_distr(df=n-1, nc=0)
q_stud = st.ppf(1 - 0.03 / 2)
print('б) интервал для alpha, когда sigma не известна:')
shift = q_stud * sqrt(S2_0) / sqrt(n)
print(f'при точности {eps} ({X_ - shift:.3f}, {X_ + shift:.3f}), длина {2 * shift:.3f}')

#PUNKT В

# считаем квантили распределения Хи-квадрат с 50 степенями свободы
ch = hi2_distr(df=n)
q_ch50 = [(ch.ppf(0.03 / 2), ch.ppf(1 - 0.03 / 2)),]
print('в) интервал для sigma, когда alpha известна:')
L = n*S2_1/q_ch50[0][1]
R = n*S2_1/q_ch50[0][0]
print(f'при точности {eps} ({L:.3f}, {R:.3f}), длина {R - L:.3f}')

#PUNKT Г

# считаем квантили распределения Хи-квадрат с 49 степенями свободы (с точностью 0.03)
ch = hi2_distr(df=n-1)
q_ch49 = [(ch.ppf(0.03 / 2), ch.ppf(1 - 0.03 / 2))]
print('г) интервал для sigma, когда alpha не известна:')
L = (n-1)*S2_0/q_ch49[0][1]
R = (n-1)*S2_0/q_ch49[0][0]
print(f'при точности {eps} ({L:.3f}, {R:.3f}), длина {R - L:.3f}')

