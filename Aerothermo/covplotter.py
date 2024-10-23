import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

def plot_ellipses(series,ax,clr):
    if ax == '':
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect([1,1,1])
    for i, data in enumerate(series):
        cov = [data[-9:-6],data[-6:-3],data[-3:]]
        centre =  data[:3]
        # find the rotation matrix and radii of the axes
        U, s, rotation = np.linalg.svd(cov)
        radii = 1.0/np.sqrt(s)
        # now carry on with EOL's answer
        u = np.linspace(0.0, 2.0 * np.pi, 100)
        v = np.linspace(0.0, np.pi, 100)
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
        for i in range(len(x)):
            for j in range(len(x)):
                [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + centre
        ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color=clr, alpha=0.2)
    return ax
data =pd.read_csv('utstats_1.csv').to_numpy()
ax=plot_ellipses(data,'','r')
data =pd.read_csv('utstats_2.csv').to_numpy()
ax=plot_ellipses(data,ax,'b')
plt.show()