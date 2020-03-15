import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def find_polyder_roots(polynomial, left_cutoff, right_cutoff, om):
    polyder = np.polyder(polynomial, om)
    roots = np.real(np.roots(polyder)[np.imag(np.roots(polyder)) == 0])
    roots_in_bounds = roots[np.where(np.logical_and(roots >= left_cutoff, roots <= right_cutoff))]
    return roots_in_bounds


df = pd.read_table('/home/elwira/Pulpit/M10/FinapresAP.dat', sep="\s+", header=None)
x = np.array(df[0]).astype(float)
y = np.array(df[1]).astype(float)
oom = 17

p = find_peaks(y,height=150, distance=500)
indexs = p[0]
aray = []

for i in range(indexs.size-1):
    b = np.argmin(y[indexs[i]:indexs[i+1]])
    aray.append(indexs[i]+b)

polyfits = []
for i in range(len(aray)-1):
    hp = int(np.floor((aray[i+1]-aray[i])/2))+aray[i]
    lhp = int(np.floor((hp-aray[i])/5))+aray[i]
    rhp = int(np.ceil((aray[i+1]-hp)/2))+hp
    polyfits.append(np.poly1d(np.polyfit(x[aray[i]:hp] - x[aray[i]], y[aray[i]:hp], oom)))
    polyfits.append(np.poly1d(np.polyfit(x[lhp:rhp] - x[lhp], y[lhp:rhp], oom)))
    polyfits.append(np.poly1d(np.polyfit(x[hp:aray[i+1]] - x[hp], y[hp:aray[i+1]], oom)))

plt.plot(x, y)
for i in range(len(aray)-1):
    hp = int(np.floor((aray[i+1]-aray[i])/2))+aray[i]
    lhp = int(np.floor((hp-aray[i])/5))+aray[i]
    rhp = int(np.ceil((aray[i+1]-hp)/2))+hp
    xp1 = np.linspace(x[aray[i]], x[hp], 10000)
    yp1 = polyfits[3*i](xp1 - x[aray[i]])
    plt.figure(1)
    plt.plot(xp1, yp1, c='r')
    xp2 = np.linspace(x[lhp], x[rhp], 10000)
    yp2 = polyfits[(3*i)+1](xp2 - x[lhp])
    plt.plot(xp2, yp2, c='g')
    xp3 = np.linspace(x[hp], x[aray[i+1]], 10000)
    yp3 = polyfits[(3 * i) + 2](xp3 - x[hp])
    plt.plot(xp3, yp3, c='orange')

    plt.figure(2)
    plt.plot(xp1, np.polyder(polyfits[3*i], 1)(xp1 - x[aray[i]]), c='r')
    plt.plot(xp2, np.polyder(polyfits[(3*i)+1], 1)(xp2 - x[lhp]), c='g')
    plt.plot(xp3, np.polyder(polyfits[(3 * i) + 2], 1)(xp3 - x[hp]), c='orange')

    cutoff = 0.15
    m = (x[aray[i+1]] - x[aray[i]])/10
    l = (x[aray[i+1]] - x[aray[i]])/2
    lc1 = 0.15*(x[aray[i+1]] - x[aray[i]])/2
    rc1 = m*1.1
    lc3 = 0.25*(x[aray[i+1]] - x[aray[i]])/2
    rc3 = 0.85*(x[aray[i+1]] - x[aray[i]])/2
    xz1 = find_polyder_roots(polyfits[3*i], lc1, rc1, 1)
    xz2 = find_polyder_roots(polyfits[3*i+1], rc1-m, lc3+l-m, 1)
    xz3 = find_polyder_roots(polyfits[3*i+2], lc3, rc3, 1)
    yz1 = np.polyder(polyfits[3*i], 1)(xz1)
    yz2 = np.polyder(polyfits[3*i+1], 1)(xz2)
    yz3 = np.polyder(polyfits[3*i+2], 1)(xz3)
    xz = np.concatenate([xz1+x[aray[i]], xz2+x[lhp], xz3+x[hp]])
    yz = np.concatenate([yz1, yz2, yz3])
    plt.scatter(xz, yz, c='purple')

    plt.figure(1)
    yz1 = polyfits[3 * i](xz1)
    yz2 = polyfits[3 * i + 1](xz2)
    yz3 = polyfits[3 * i + 2](xz3)
    yz = np.concatenate([yz1, yz2, yz3])
    plt.scatter(xz, yz, c='purple')

    plt.figure(3)
    plt.plot(xp1, np.polyder(polyfits[3 * i], 2)(xp1 - x[aray[i]]), c='r')
    plt.plot(xp2, np.polyder(polyfits[(3 * i) + 1], 2)(xp2 - x[lhp]), c='g')
    plt.plot(xp3, np.polyder(polyfits[(3 * i) + 2], 2)(xp3 - x[hp]), c='orange')
    pxz1 = find_polyder_roots(polyfits[3 * i], lc1, rc1, 2)
    pxz2 = find_polyder_roots(polyfits[3 * i + 1], rc1 - m, lc3 + l - m, 2)
    pxz3 = find_polyder_roots(polyfits[3 * i + 2], lc3, rc3, 2)
    pyz1 = np.polyder(polyfits[3 * i], 2)(pxz1)
    pyz2 = np.polyder(polyfits[3 * i + 1], 2)(pxz2)
    pyz3 = np.polyder(polyfits[3 * i + 2], 2)(pxz3)
    pxz = np.concatenate([pxz1 + x[aray[i]], pxz2 + x[lhp], pxz3 + x[hp]])
    pyz = np.concatenate([pyz1, pyz2, pyz3])
    plt.scatter(pxz, pyz, c='black')
    plt.figure(1)
    pyz1 = polyfits[3 * i](pxz1)
    pyz2 = polyfits[3 * i + 1](pxz2)
    pyz3 = polyfits[3 * i + 2](pxz3)
    pyz = np.concatenate([pyz1, pyz2, pyz3])
    plt.scatter(pxz, pyz, c='black')
    plt.xlabel("time [s]")
    plt.ylabel("Pressure [mm Hg]")

plt.figure(4)
plt.plot(x,y)
plt.xlabel("time [s]")
plt.ylabel("Pressure [mm Hg]")
plt.show()
