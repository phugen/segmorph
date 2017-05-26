# This script reads a list of paths to network solver files
# and trains them using specified GPU and the data specified in their prototxt input layer.
#
# The structure of the network path file is expected to be
# one network solver path per line, i.e.
#
# path/to/network1.prototxt\r\n
# path/to/network2.prototxt\r\n
# ...

import os
import sys
import subprocess
import glob
import copy

if len(sys.argv) < 4:
    print "Usage: python start_training.py network_list device_id numiters"
    print exit(-1)


# get params
network_list = sys.argv[1]
device = int(sys.argv[2])
numiters = int(sys.argv[3])

# Get paths to all networks.
with open(network_list) as f:
    lines = f.readlines()
    nets = []
    logpaths = []

    for x in lines:
        # ignore comments in file
        if not x.startswith("#"):
            nets.append(x.strip("\r\n").split(",")[0])
            logpaths.append(x.strip("\r\n").split(",")[1])


# Train all networks.
for netno,net in enumerate(nets):

    # ignore comments in file
    if not net.startswith("#"):

        print "Starting training for " + net + " on GPU " + str(device) + "!"

        # write logs to both stdout and logfile
        logname = logpaths[netno] #os.path.basename(net)[:-9] + ".log"

        with open("logs/" + logname, "w") as log:

            # start training.
            # only redirect stderr to logfile to avoid deadlock
            # that can happen when two or more pipes are redirected!
            proc = subprocess.Popen(["python", "-u", "train_network.py", str(net), str(device), str(numiters)],\
                                    bufsize=1, stderr=subprocess.PIPE)

            # show logging output on stdout and write it to file
            for line in iter(proc.stderr.readline, b''):
                print line, # NOTE: the comma prevents duplicate newlines (softspace hack)
                log.write(line)

            proc.wait() # wait for training to finish before starting next one

        print "Finished training for " + net + " !"
        print ""


# Move native log to another folder
# If something goes wrong, we can always
# check the raw log this way.
nativelog = glob.glob("pycaffe.*.*.log.INFO.*")[0]
os.rename(nativelog, "rawlogs/" + os.path.basename(nativelog))
