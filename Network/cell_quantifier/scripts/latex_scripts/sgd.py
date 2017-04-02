# Implements Stochastic Gradient Descent (SGD) algorithm
# and SGD with momentum.

import math
import numpy as np
from random import random
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import ticker


def unpack(seq, n=2):
    ''' Unpacks n-tuples to mimic Python 3 star
    operator functionality.'''
    for row in seq:
        yield [e for e in seq]


def fun(x, theta):
    '''evaluates the error function f(x) = 1/2(yi - m*xi + b)^2 '''
    m = theta[0]
    b = theta[1]
    xi = x[0]
    yi = x[1]

    return  0.5 * (yi - m*xi + b)**2


def nabla(x, theta):
    ''' Returns the (batch) gradient of fun().'''

    # determine batch size based on size of x
    batchsize = x.shape[0]
    batch_grad = np.array((0., 0.)) # accumulated gradient

    # variable weights
    m = theta[0]
    b = theta[1]

    # find batch gradient
    for n in range(batchsize):
        xi = x[n,0]
        yi = x[n,1]
        batch_grad[0] += xi * (b + m*xi - yi)
        batch_grad[1] += b + m*xi - yi

    batch_grad /= batchsize # average over batch


    return batch_grad


def sgd(func, x, nabla, guess, eta, eps, batch=1, momentum=False):
    ''' Performs SGD on a function func using a batch vector x,
    the derivative function nabla, an initial guess
    theta, a learning rate / step size of eta and a maximum
    error eps.

    Returns the list of parameters that were chosen in each
    iteration.'''

    error = 1e10 # init difference to previous iteration
    thetas = [] # list of parameter steps
    iteration = 0
    max_iters = 1000
    theta = guess.copy() # copy by value to keep same inits for both SGD passes


    if momentum:
        v = np.array((0., 0.))
        gamma = 0.85

    while(error > eps and iteration <= max_iters):

        xi = np.atleast_2d(x[np.random.randint(0, x.shape[0], size=(batch)), :]) # choose random datapoints from x

        # use momentum in SGD update
        if momentum == True:
            v = gamma * v + eta * nabla(xi, theta)
            theta -= v

        # barebones SGD
        else:
            theta -= eta * nabla(xi, theta) # take step into direction of negative gradient


        thetas.append( np.ravel(unpack(theta, len(theta)).next()) ) # save params for each iteration

        # convergence check
        error = 0.
        for n in range(x.shape[0]):
            xi = x[n, 0]
            yi = x[n, 1]
            m = theta[0]
            b = theta[1]
            error += 0.5 * (yi - m*xi + b)**2

        if(iteration % 1 == 0):
            if momentum:
                print "MOMENTUM: " + str(error) + " for m=" + str(theta[0]) + ", b=" + str(theta[1])
            else:
                print "VANILLA: " + str(error) + " for m=" + str(theta[0]) + ", b=" + str(theta[1])

        iteration += 1

    return thetas


# show a comparison of normal SGD vs momentum SGD
mguess = 463.924
bguess = 339.625
init = np.array((mguess, bguess))

print init

fig = plt.figure()

# create noisy linear dataset to fit
setsize = 50
data = np.zeros((setsize, 2))

for i in range(setsize):
    data[i, :] = i, i + random()

# normalize data
data /= np.max(data)


# get thetas without and with momentum
thetas = sgd(fun, data, nabla, guess=init, eta=0.1, eps=200, batch=1, momentum=False)
thetas = np.array(thetas)

print init

thetas2 = sgd(fun, data, nabla, guess=init, eta=0.1, eps=200, batch=1, momentum=True)
thetas2 = np.array(thetas2)


xi = data[0,0]
yi = data[0,1]

for index, th in enumerate((thetas, thetas2)):

    gridsize = 100
    levels = 1000

    X, Y = np.meshgrid(np.linspace(-500, 500, gridsize), np.linspace(-500, 500, gridsize))

    # calculate sum of errors for data for "all" possible lines
    Z = np.zeros((len(X), len(Y)))
    for d in range(len(data)):
        xi = data[d, 0]
        yi = data[d, 1]
        Z =+ np.vectorize(lambda m, b: 0.5 * ((m*xi + b))**2)(X, Y)

    ax = fig.add_subplot(1, 2, index+1)

    conts = ax.contourf(X, Y, Z, levels, alpha=1, locator=ticker.LogLocator(), cmap=plt.cm.plasma, vmin=80, vmax=10e5)

    #ax.clabel(conts, conts.levels[::20], fontsize="6", colors="k")
    ax.plot(th[:,0], th[:,1], color="white")

    if index == 1:
        #ax.yaxis.set_visible("False")
        ax.get_yaxis().set_ticklabels([])

    pos = (th[0,0], th[0,1])
    circle = plt.Circle(pos, 1, color="steelblue")
    ax.add_artist(circle)
    pos = (th[-1,0], th[-1,1])

    ax.set_xlabel("a")
    if index == 0:
        ax.set_ylabel("b")

    ax.set_xlim(-100, 300)
    ax.set_ylim(-200, 150)

# add colorbar for both plots without changing their size
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(conts, cax=cbar_ax)

#plt.tight_layout()
plt.show()



# Show line (sgd and momentum)
fig, ax = plt.subplots(ncols=1, nrows=1, sharey=True)
ax.set_xlabel("X")
ax.set_ylabel("Y")

for xi in range(data.shape[0]):
        ax.scatter(data[xi,0], data[xi,1], c="#ff8c00")


x0 = -10000000
y0 = thetas[-1, 0] * data[0,0] + thetas[-1, 1]
x1 = 10000000
y1 =  thetas[-1, 0] * data[-1,0] + thetas[-1, 1]

l = mlines.Line2D([x0,x1], [y0,y1], c="k")
ax.add_line(l)

x0 = -10000000
y0 = thetas2[-1, 0] * data[0,0] + thetas2[-1, 1]
x1 = 10000000
y1 =  thetas2[-1, 0] * data[-1,0] + thetas2[-1, 1]

l = mlines.Line2D([x0,x1], [y0,y1], c="b")
ax.add_line(l)

plt.show()
