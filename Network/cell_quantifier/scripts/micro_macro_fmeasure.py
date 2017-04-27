# Calculates the Micro and Macro F-Measure by reading
# a CSV file of format
#
# Image number,TP,FP,FN NOTE: header included, is skipped!

import sys


def split_line(line):
    ''' Split a line into a list of statistics.'''
    return line.split(",")[1:] # throw away image number

def micro_fmeasure(csvfile):
    ''' Calculates the Micro F-Measure. '''

    # micro stats over entire dataset
    tpsum = 0
    fpsum = 0
    fnsum = 0

    eps = 1e-15 # prevent division by zero

    for line in csvfile:

        # skip comments/header
        if line.startswith("#") or line.startswith("\n"):
            continue

        split = split_line(line)

        # cumulate stats
        tpsum += int(split[0])
        fpsum += int(split[1])
        fnsum += int(split[2])

    # calculate micro precision and recall
    PR_mu = tpsum / float(tpsum + fpsum + eps)
    RC_mu = tpsum / float(tpsum + fnsum + eps)

    # calculate micro F-Measure
    return 2. * ((PR_mu * RC_mu) / (PR_mu + RC_mu))


def macro_fmeasure(csvfile):
    ''' Calculates the Macro F-Measure. '''

    n = 0. # for averaging
    eps = 1e-15 # prevent division by zero

    # "running" macro stats
    PR_M = 0.
    RC_M = 0.

    for line in csvfile:

        # skip comments/header
        if line.startswith("#") or line.startswith("\n"):
            continue

        split = split_line(line)

        # update running macro precision and recall
        PR_M += float(split[0]) / (float(split[0]) + float(split[1]) + eps)
        RC_M += float(split[0]) / (float(split[0]) + float(split[2]) + eps)

        # keep track of number of samples
        n += 1


    # average over entire dataset
    PR_M /= n
    RC_M /= n


    # calculate macro F-Measure
    return 2. * ((PR_M * RC_M) / (PR_M + RC_M))



if len(sys.argv) < 2:
    print "Usage: micro_macro_fmeasure.py csv_path"
    exit(-1)

csv_path = sys.argv[1]

with open(csv_path, "r") as csv:
    print "Micro F-Measure: " + repr(micro_fmeasure(csv))
    csv.seek(0) # reset file
    print "Macro F-Measure: " + repr(macro_fmeasure(csv))
