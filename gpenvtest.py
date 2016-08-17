import numpy as np
import matplotlib.pyplot as plt

import matplotlib.animation as animation

g = 9.80665

ag = np.array((0, -g))

cor = 0.95

xlim = (0, 30)
ylim = (0, 20)

xy = np.array((3.0, 18.0))
v = np.array((0.2, 0.3))

delta_t = 0.001

fig = plt.figure()
fig.canvas.toolbar.hide()
ax = fig.add_subplot(111, autoscale_on=False, xlim=xlim, ylim=ylim)
#ax.grid()



gravitypoint1 = np.array((5.0, 5.0))
gravitypoint2 = np.array((8.0, 5.0))

def onclick(event):
    print event.button
    if event.xdata != None and event.ydata != None:
        if event.button==1:
            gravitypoint1[0]=event.xdata
            gravitypoint1[1] =event.ydata
        else:
            gravitypoint2[0]=event.xdata
            gravitypoint2[1] =event.ydata
cid = fig.canvas.mpl_connect('button_press_event', onclick)

dot1, = ax.plot([], [], 'o', color='green', markersize=20)
dot2, = ax.plot([], [], 'x', color='green', markersize=20)
line1, = ax.plot([], [])
dot3, = ax.plot([], [], 'x', color='green', markersize=20)
line2, = ax.plot([], [])

def init():
    return []


def animate(t):
    # t is time in seconds
    global xy, v

    if xy[0] <= xlim[0]:
        # hit the left wall, reflect x component
        v[0] = cor * np.abs(v[0])

    elif xy[0] >= xlim[1]:
        v[0] = - cor * np.abs(v[0])

    if xy[1] <= ylim[0]:
        v[1] = cor * np.abs(v[1])

    elif xy[1] >= ylim[1]:
        v[1] = - cor * np.abs(v[1])

    # delta t is 0.1
    delta_v = delta_t * ag
    v += delta_v

    v += (gravitypoint1 - xy) / 400
    v += (gravitypoint2 - xy) / 400
    v*=0.99
    xy += v

    xy[0] = np.clip(xy[0], xlim[0], xlim[1])
    xy[1] = np.clip(xy[1], ylim[0], ylim[1])

    points1=[ [xy[0],gravitypoint1[0]] , [xy[1],gravitypoint1[1]] ]
    line1.set_data(points1)
    points2=[ [xy[0],gravitypoint2[0]] , [xy[1],gravitypoint2[1]] ]
    line2.set_data(points2)

    dot1.set_data(xy)
    dot2.set_data(gravitypoint1)
    dot3.set_data(gravitypoint2)

    return [line1,line2,dot1,dot2,dot3]

ani = animation.FuncAnimation(fig, animate, np.arange(0, 100, delta_t), init_func=init, interval=10, blit=True)


plt.show()