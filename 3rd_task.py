import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from math import sqrt
from scipy.stats import norm as norm_distr, nct as stud_distr, chi2 as hi2_distr, f as fisher_distr

alpha = -2
sigma2 = 0.9
eps = 0.03

X = np.array([-1.998,-3.067,-2.583,-0.508,-1.747,-1.756,-1.740,-2.086,-1.124-0.882,
                0.718,-2.047,-3.403,-2.501,-2.601,-2.447,-1.601,-2.092,-1.019,-1.642])
Y = np.array([-1.470,-1.711,-1.518,-3.328,-1.777,-2.179,-1.633,-3.399,-2.112,-2.664,
                -0.132,0.200,-1.652,-1.964,-2.365,-1.979,-1.630,-2.514,-1.732,-2.953,
                -1.224,-2.60,-1.956,-2.722,-1.257,-1.641,-1.797,-3.340,-1.727,-2.684])

# размеры выборок
n = X.size
m = Y.size

# выборочные средние
X_ =  np.average(X)
Y_ =  np.average(Y)
print(f'X_={X_}')
print(f'Y_={Y_}\n')

# выборочные дисперсии
S_X = sum((X - X_*np.ones(n))**2)/(n - 1)
S_Y = sum((Y - Y_*np.ones(m))**2)/(m - 1)
print(f'S_X={S_X}')
print(f'S_Y={S_Y}')

# критерий Стьюдента
T = (X_-Y_) * sqrt((m + n - 2)/(1/m + 1/n)) / sqrt(n*S_X+m*S_Y)
if abs(T) < stud_distr(df=n+m-2,nc=0).ppf(1-eps/2):
  print('Выборки имеют одинаковые alpha, при известных sigma2')
else:
  print('Выборки имеют разные alpha, при известных sigma2')
print(f'stud_stat={T}')

# критерий Фишера

f = fisher_distr(dfn=n-1,dfd=m-1)

f_stat = n*S_X/(m*S_Y)
print(f'f_stat={f_stat}')
if f.ppf(eps/2) < f_stat < f.ppf(1-eps/2):
  print('Выборки имеют одинаковые sigma2')
else:
  print('Выборки имеют разные sigma2')