# Implements the Gradient Descent algorithm
# and visualizes the minimization progress
# of the target function.

import math
import numpy as np
from random import random
import matplotlib.pyplot as plt

def unpack(seq, n=2):
    ''' Unpacks n-tuples to mimic Python 3 star
    operator functionality.'''
    for row in seq:
        yield [e for e in seq]


def fun1(theta):
    '''evaluates f(a, b) = sin(theta_0) + cos^2(theta_1)'''
    return math.sin(theta[0]) + math.cos(theta[1])**2


def fun2(theta):
    '''evaluates f(a) = 0.05a^2 * 0.5cos(a), a function
    with several local minima (depending on interval)'''
    return (0.05 * (theta**2) * 0.5 * math.cos(theta))


def nabla1(theta):
    ''' Returns the gradient vector of fun1.'''
    return np.array((math.cos(theta[0]), \
                     -2 * math.sin(theta[1]) * math.cos(theta[1])))


def nabla2(theta):
    ''' Returns the gradient vector of fun2.'''
    return theta * (0.05 * math.cos(theta) - 0.025 * theta * math.sin(theta))



def grad(func, nabla, theta, eta, eps):
    ''' Performs gradient descent on a function func,
    given the derivative function nabla, an initial guess
    theta, a learning rate / step size of eta and a maximum
    error eps.

    Returns the list of parameters that were chosen in each
    iteration.'''

    diff = 1e10 # difference to previous iteration
    thetas = [] # list of parameter steps
    iteration = 0
    max_iters = 100

    while(diff > eps and iteration <= max_iters):
        fval = func(theta)  # evaluate function at current theta
        theta -= eta * nabla(theta) # take step into direction of negative gradient
        thetas.append( np.ravel(unpack(theta, len(theta)).next()) ) # save params for each iteration
        diff = abs(fval - (func(theta))) # check if convergence reached
        iteration += 1
        #print str(diff) + " > " + str(eps)

    return thetas



# choose random initial theta
init = np.array((random(), random()))

# prepare plot
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
#ax.set_title("Gradient Descent optimization")

# do gradient descent optimization on sin(a) + cos^2(b)
thetas = grad(fun1, nabla1, theta=init, eta=0.1, eps=1e-5)
thetas = np.array(thetas)

# calculate fun1(a, b) for "all" a, b
gridsize = 100
levels = 15
X, Y = np.meshgrid(np.linspace(-2, 2, gridsize), np.linspace(-2, 4, gridsize))
Z = np.vectorize(lambda a, b: math.sin(a) + math.cos(b)**2)(X, Y)

# plot contours of fun1 and route of the
# optimization process
conts = ax0.contour(X, Y, Z, levels, cmap=plt.cm.gnuplot, alpha=0.45)
ax0.clabel(conts, conts.levels[::3], fontsize="6", colors="k")
ax0.plot(thetas[:,0], thetas[:,1])
#ax0.scatter(thetas[::3,0,], thetas[::3,1], c="steelblue", marker="x")

# plot start and arrowhead to show direction
pos = (thetas[0,0], thetas[0,1])
circle = plt.Circle(pos, 0.060, color="steelblue")
ax0.add_artist(circle)
pos = (thetas[-1,0], thetas[-1,1])
ax0.set_xlabel("a")
ax0.set_ylabel("b")
ax0.annotate("", (pos[0]-0.05, pos[1]), pos, arrowprops=dict(fc="steelblue", width=0.1, edgecolor="steelblue", headlength=8, headwidth=8))


# do gradient descent on fun2 to show the
# algorithm behavior when local minima
# exist, depending on the initialization.
guesses = [6.8, 13.0, 25.0]
colors = ["#ffbbee", "#def1f9", "#ff6666"]

# mark starting points
for g in guesses:
    ax1.annotate("a=" + str(g), (g-3, fun2(g)+1), (g-3, fun2(g)+1))

for index, guess in enumerate(guesses):
    init = np.atleast_2d(np.array((guess)))
    thetas = grad(fun2, nabla2, theta=init, eta=0.05, eps=1e-5)

    # calculate fun2(a) values over grid
    gridsize = 200
    X = np.ravel(np.meshgrid(np.linspace(0, 25, gridsize)))
    Y = [fun2(x) for x in X]
    Yt = np.ravel([fun2(x) for x in thetas])

    # plot fun2
    if index is 0:
        ax1.plot(X, Y)

    # plot route of the optimization process
    # depending on init point
    ax1.plot(thetas, Yt, c=colors[index], linewidth=3.0)

    # add start and end point cues
    #pos = (thetas[0], fun2(thetas[0]))
    #circle = plt.Circle(pos, 0.3, color=colors[index])
    #ax1.add_artist(circle)

    # use last 10% of steps to determine direction of arrow
    fromtheta = len(thetas) - int(len(thetas) * 0.5)

    pos_from = (thetas[fromtheta], fun2(thetas[fromtheta]))
    pos_to = (thetas[-1], fun2(thetas[-1]))
    ax1.annotate("", pos_to, pos_from, arrowprops=dict(fc=colors[index], edgecolor=colors[index], headlength=5, headwidth=5))

ax1.set_xlim(0, 30)
ax1.set_ylim(-15, 18)
ax1.set_xlabel("a")
ax1.set_ylabel("g(a)")

fig.subplots_adjust(top=0.88, bottom=0.11, left=0.12, right=0.90, hspace=0.20, wspace=0.35)
plt.show()
