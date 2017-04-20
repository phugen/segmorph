# Trains a network on a GPU, given the path to its prototxt,
# the GPU index and the optional maximum number of iterations.
# A previous training session can be resumed by supplying
# an optional path to a solver state file.

import sys
import os

# Google Logging params.
# Need to be set before importing Caffe!
os.environ["GLOG_log_dir"] = "." # where to save native log. NOTE: Needs new (December '16+) PyCaffe version to work!
os.environ["GLOG_stderrthreshold"] = "INFO" # log everything, regardless of severity level
os.environ["GLOG_alsologtostderr"]= "1" # in addition to logfile, show output on the console!

import caffe



if len(sys.argv) < 4:
    print "Usage: python train_network.py solverpath deviceno numiters [statepath]"
    exit(-1)

solverpath = sys.argv[1]
device_no = int(sys.argv[2])
numiters = int(sys.argv[3])

statepath = None
if len(sys.argv) == 5:
    statepath = sys.argv[4]

# set GPU to run on
caffe.set_device(device_no)
caffe.set_mode_gpu()

# load solver and network definition
solver = caffe.get_solver(solverpath)

# resume training if solverstate given
if statepath is not None:
    solver.restore(statepath)

# Train numiters steps
for step in range(numiters):
    solver.step(1)
