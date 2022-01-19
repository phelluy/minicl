from matplotlib.pyplot import *
from math import *
import numpy as np

with open("plotpy.dat", "r") as f:
    contenu = f.read().split("\n\n")
    # print(contenu)


x = contenu[0].split()
nx = len(x) - 1
x = np.array([float(x[i]) for i in range(nx+1)])
# print(x)
y = contenu[1].split()
ny = len(y) -1
y = np.array([float(y[i]) for i in range(ny+1)])
# print(y)
x , y = np.meshgrid(x,y)
z = contenu[2].split()
nz = len(z)
z = np.array([float(z[i]) for i in range(nz)]).reshape((ny+1,nx+1))
# print(z)
# plot(xi, unum, color="blue")
# plot(xi, uex, color="red")
# def quit_figure(event):
#     if event.key == 'q':
#         close(event.canvas.figure)
# cid = gcf().canvas.mpl_connect('key_press_event', quit_figure)


fig, ax = subplots()
ax.axis('equal')
cs = ax.pcolormesh(x,y,z,cmap = 'jet')
# cs = ax.countourf(x,y,z,100,cmap = 'jet')
cbar = fig.colorbar(cs)

print("press \'q\' to quit...");
show()