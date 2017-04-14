# A wrapper for train_networks.py that is
# needed because Glog doesn't relinquish control
# over logging files until the script is done.
# Therefore, this just runs train_networks.py as a subprocess.
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

if len(sys.argv) < 3:
    print "Usage: python start_training.py network_list device_id"
    print exit(-1)

network_list = sys.argv[1]
device = sys.argv[2]

# run training script proper.
subprocess.call(["python", "train_networks.py", network_list, device])

# Move native log to another folder
# so train_networks.py can be called again without problems.
# If something goes wrong, we can always
# check the raw log this way.
logname = glob.glob("pycaffe.*.*.log.INFO.*")[0]
os.rename(logname, "rawlogs/" + os.path.basename(logname))
