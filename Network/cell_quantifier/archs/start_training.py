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
    nets = [x.strip("\r\n") for x in lines]


# Train all networks.
for net in nets:

    # ignore comments in file
    if not net.startswith("#"):
        print "Starting training for " + net + " on GPU " + str(device) + "!"

        # write to stdout/stderr and log file simultaneously
        logname = os.path.basename(net)[:-9] + ".log"

        # NOTE: change iters here if needed!
        proc = subprocess.Popen(["python", "train_network.py", str(net), str(device), str(numiters)],\
                                bufsize=1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        # write logs to both stdout and logfile
        with proc.stdout, open("logs/" + logname, "w") as log:
            for line in iter(proc.stdout.readline, b''):
                print line, # NOTE: the comma prevents duplicate newlines (softspace hack)
                log.write(line)
        proc.wait()


        print "Finished training for " + net + " !"
        print ""


# Move native log to another folder
# If something goes wrong, we can always
# check the raw log this way.
nativelog = glob.glob("pycaffe.*.*.log.INFO.*")[0]
os.rename(nativelog, "rawlogs/" + os.path.basename(nativelog))
