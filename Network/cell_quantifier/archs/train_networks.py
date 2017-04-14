# This script reads a list of paths to network solver files
# and trains them using specified GPU and the data specified in their prototxt input layer.

import os
import sys
import glob

# Google Logging params.
# Need to be set before importing Caffe!
os.environ['GLOG_log_dir'] = "." # NOTE: where to save native log. Needs new PyCaffe version to work!
os.environ['GLOG_logtostderr']= "0"

import caffe

# Trains a network on a GPU, given the path to its prototxt,
# the GPU index and the optional maximum number of iterations.
# A previous training session can be resumed by supplying
# an optional path to a solver state file.
def train(solverpath, device_no, numiters=1, statepath=None):

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



if len(sys.argv) < 3:
    print "Usage: python train_networks.py network_list device_id"
    print exit(-1)





# get params
network_list = sys.argv[1]
device = int(sys.argv[2])

# Get paths to all networks.
with open(network_list) as f:
    lines = f.readlines()
    nets = [x.strip("\r\n") for x in lines]


netnames = [] # save names of solvers for saving logs later

# Train all networks.
for net in nets:

    # ignore comments in file
    if not net.startswith("#"):
        print "Starting training for " + net + " on GPU " + str(device) + "!"
        ret = train(net, device, numiters=100000) # NOTE <-------------------------- Change #iters here
        print "Finished training for " + net + " !"
        print ""

        netnames.append(os.path.basename(net)[:-9] + ".log") # save solver filename for later




# Split concatenated native Caffe log file
# into smaller log files
logname = glob.glob("pycaffe.*.*.log.INFO.*")[0]

with open(logname, "r") as log:

    # get number of lines in log file
    for i, l in enumerate(log):
        pass
    numlines = i+1

    # find lines at which new logs begin
    breaklines = []

    log.seek(0)
    for l, line in enumerate(log):
        if "Initializing solver from parameters" in line:
            if l is not 4: # first four lines in new log file are header information
                breaklines.append(l)
    breaklines.append(numlines)


    log.seek(0)
    loglines = list(log)

    # split logs and save
    fromline = 4
    toline = numlines
    if breaklines is not []:
        toline = breaklines[0]

    for l in range(len(breaklines) - 1):
        with open("logs/" + netnames[l], "w") as newlog:
            newlog.writelines(loglines[fromline:toline])
        fromline = toline
        toline = breaklines[l + 1]

    # write last log
    with open("logs/" + netnames[len(breaklines) - 1], "w") as newlog:
        newlog.writelines(loglines[fromline:toline])

print ""
print "Training for all networks done. Exiting ..."
