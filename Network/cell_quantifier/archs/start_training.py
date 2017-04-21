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
import tempfile

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

        # avoid deadlock that can be caused by using subprocess.PIPE for
        # temporary storage of subprocess outputs
        # (see https://thraxil.org/users/anders/posts/2008/03/13/Subprocess-Hanging-PIPE-is-your-enemy/ )
        tmp = tempfile.TemporaryFile()

        # start training
        proc = subprocess.Popen(["python", "-u", "train_network.py", str(net), str(device), str(numiters)],\
                                bufsize=1, stdout=tmp, stderr=tmp)

        # write logs to both stdout and logfile
        logname = os.path.basename(net)[:-9] + ".log"

        with open("logs/" + logname, "w") as log:
            for line in iter(proc.stdout.readline, ''):
                print line, # NOTE: the comma prevents duplicate newlines (softspace hack)
                log.write(line)
        print "COMMUNICATE:" + str(proc.communicate("n\n")[0]) # wait for child process to finish


        print "Finished training for " + net + " !"
        print ""


# Move native log to another folder
# If something goes wrong, we can always
# check the raw log this way.
nativelog = glob.glob("pycaffe.*.*.log.INFO.*")[0]
os.rename(nativelog, "rawlogs/" + os.path.basename(nativelog))
