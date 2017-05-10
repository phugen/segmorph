# Finds the best snapshots for a
# number of networks. For details, see
# find_best_snapshot.py .
#
# The configuration file is expected to have a format of
# device_id,path/to/deploy.prototxt,path/to/validation_hdf5/,path/to/snapshots/,use_fmeasure,test_interval\n
#
# The last parameter can be either "yes" or "no" and overrides the normal Cross-Entropy loss calculation.

from find_best_snapshot import find_snapshot
import sys

if len(sys.argv) < 2:
    print "Too few arguments!"
    print "Usage: python find_best_snapshot.py batchsize log/output/path/ path/to/config_file.txt"
    sys.exit(-1)

batchsize = int(sys.argv[1])
if batchsize < 1:
    print "Batchsize was " + str(batchsize) + " but needs to be >= 1!"
    exit(-1)

log_path = sys.argv[2]

print "Finding best snapshots for networks..."
with open(sys.argv[3], "r") as f:
    for l in f:

        # ignore comments
        if l.startswith("#"):
            continue

        # get settings from config
        split = l.strip("\r\n").split(",")

        device_id = int(split[0])
        modelfile = split[1]
        validation_path = split[2]
        snapshot_path = split[3]
        use_fmeasure = bool(split[4])
        interval = int(split[5])

        # get best snapshot and save in log_path
        ret = find_snapshot(device_id, batchsize, modelfile, validation_path, snapshot_path, log_path, use_fmeasure, interval)
        if ret == -1:
            exit(-1)

print "Done! Saved all logs to " + log_path + "."
