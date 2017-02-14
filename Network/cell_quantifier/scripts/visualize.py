
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import timeit
import scipy.misc
from PIL import Image
import caffe


# start training
caffe.set_device(0)
caffe.set_mode_gpu()
print "Loading Solver... "
solver = caffe.get_solver("unet_solver.prototxt")
if len(sys.argv) > 1:
    if sys.argv[1].endswith(".solverstate"):
        solver.restore(sys.argv[1])

print "Training ..."
#f1, (ax1, ax2) = plt.subplots(1, 2)
#ax1.set_title("input")
#ax1.axis('off')
#ax2.set_title("output")
#ax2.axis('off')
#f2, ax3 = plt.subplots(1,1)
#ax3.axis('off')
#f3, ax4 = plt.subplots(1,1)
solver.step(1)

start_time = timeit.default_timer()

for step in range(500):
    #print solver.net.blobs['data'].data[0, 0]
    #_min, _max = solver.net.blobs['data'].data[0].min(), solver.net.blobs['data'].data[0].max()
    #ax1.imshow(solver.net.blobs['data'].data[0, 0], cmap='gray', vmin=_min, vmax=_max)

    #_min, _max = solver.net.blobs['score'].data[0].min(), solver.net.blobs['score'].data[0].max()
    #ax2.imshow(solver.net.blobs['score'].data[0, 1], cmap='gray', vmin=_min, vmax=_max)

    #visualize_weights(solver.net, 'conv_u0d-score', 4, ax3, 'filters/u0d-score'+str(step)+'.png')
    #visualize_weights(solver.net, 'conv_d0a-b', 4, ax4, 'filters/d0a-b'+str(step)+'.png')
    #visualize_weights(solver.net, 'conv_u0c-d', 4, ax4, 'filters/conv_u0c-d'+str(step)+'.png')

    #plt.show(block=False)
    #plt.pause(0.1)

    # extract feature (not weights!) layer and write to file
    #print solver.net.blobs['data'].data.shape
    features_out = solver.net.blobs['visualize_out'].data[0,:,:,:] # plt needs WxHxC

    #print "Label probs: " + "Background: " + format(features_out[170,150,0] * 100, '.5f') + "% " \
    #                      + "R: " + format(features_out[170,150,1] * 100, '.5f') + "% " \
    #                      + "G: " + format(features_out[170,150,2] * 100, '.5f') + "% " \
    #                      + "B: " + format(features_out[170,150,3] * 100, '.5f') + "%"

    features_out = features_out[1:4,:,:] # omit BG probability layer - pixel will become dark if other probabilities are low
    print features_out.shape
    minval = features_out.min()
    maxval = features_out.max()
    scipy.misc.toimage(features_out, cmin=minval, cmax=maxval).save("./visualizations/visualize_out_" + str(step) + ".png")
    #plt.axis('off')
    #plt.imshow(features_out[:,:], interpolation='none', cmap="gray") # render plot
    #plt.savefig("./visualizations/visualize_out_" + str(step) + ".png", dpi=96) # save plot

    solver.step(1)

    s_passed = timeit.default_timer() - start_time

    #print "RUNNING FOR: " + str(int(s_passed) / 60) + "m"

#visualize_weights(solver.net, 'conv1', 4, ax3)

print 'Visualization Done'
