import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import seaborn as sns
import warnings
from scipy.stats import norm as norm_distr, nct as stud_distr, chi2 as hi2_distr, f as fisher_distr

eps = 0.03
X_u = np.array([0.210,0.338,0.442,0.962,0.152,0.242,0.687,0.521,0.133,0.756
                 ,0.540,0.478,0.808,0.683,0.469,0.198,0.844,0.348,0.756,0.142,
                0.571,0.262,0.018,0.425,0.442,0.861,0.068,0.050,0.910,0.352])
n = X_u.size
X_u_sorted = np.sort(X_u)

#строим графики эмпирического и истинного распределений
plt.figure(figsize=(9, 9))
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
tmp = np.where(X_u_sorted <= X_u_sorted)[0]/n
plt.hlines(0, -1, X_u_sorted[0], linewidth=1.5, color='blue')
plt.hlines(1, X_u_sorted[n-1], 2, linewidth=1.5, color='blue')
for j in range(n - 1): plt.hlines(tmp[j + 1], X_u_sorted[j], X_u_sorted[j + 1], linewidth=1.5, color='blue')
plt.plot([-1, 0, 1, 2], [0, 0, 1, 1], linewidth=1.5, color='green')
plt.grid(True)

# находим и отмечаем статистическое расстояние между функциями распределений
D_tmp = max(max(abs(tmp - X_u_sorted)), abs(X_u_sorted[0]), abs(1-X_u_sorted[n-1]))
for elem, i in zip(tmp, range(len(tmp))):
  if abs(X_u_sorted[i]-elem) == D_tmp:
    plt.plot([X_u_sorted[i], X_u_sorted[i]], [X_u_sorted[i], tmp[i+1]], linewidth=1.5, color='red', marker='x', ls='--')
    plt.show()
    print(f'D достигается в точке X_u_sorted[{i}]={tmp[i+1]-X_u_sorted[i]}')
    print(f'D = {tmp[i+1]-X_u_sorted[i]:.3f}')
    print(f'D*sqrt(n) = {(tmp[i+1]-X_u_sorted[i])*sqrt(n):.3f}')


def do_work(k: int) -> bool:
  # считаем попадания в каждый отрезок
  nu = [len([x for x in X_u if i / k <= x <= (i + 1) / k]) for i in range(0, k)]

  # считаем величину, имеющую распределение Хи-квадрта (с k-1 степенью свободы)
  ro = sum([((nu[i] - n / k) ** 2) / (n / k) for i in range(k)])

  tmp_sum = 0

  # sns.displot(X_u, bins = [i/k for i in range(k+1)])
  for i in range(0, k):
    plt.hlines(nu[i] / (30 / k), i / k, (i + 1) / k, color='green')
    tmp_sum += nu[i] / (30 / k) * (1 / k)
  # plt.hlines(1/k, 0, 1)
  if k == 5:
    plt.show()
  else:
    plt.clf()

  C = hi2_distr(df=k - 1).ppf(1 - eps)
  print(f'ro={ro}')
  print(f'C={C}')
  # print(f'{tmp_sum:.5f}')
  return True if ro < C else False


for k in range(3, 10):
  print(f'k = {k}: ' + ('' if do_work(k) else 'не ') + 'равномерное', end='\n\n')