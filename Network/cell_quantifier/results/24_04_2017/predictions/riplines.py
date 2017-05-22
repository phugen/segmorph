# Splits a block stat file into <numclasses> files
# and saves them to the same directory.

import sys

if len(sys.argv) < 3:
    print "Usage: python riplines.py numclasses input_file"
    exit(-1)

numclasses = int(sys.argv[1])
input_file = sys.argv[2]

for c in range(numclasses):

    lines = []
    n = 0

    # rip lines concerning current class from block file
    with open(input_file, "r") as stats:
        for line in stats:

            # skip header, comments and newlines
            if line.startswith("#") or line == "\n":
                continue

            # rip correct lines for current class
            if n % numclasses == c:
                lines.append(line.strip("\n"))

            n += 1

    # write file containing only values of current class
    outname = input_file[:-4] + "_class" + str(c) + ".txt"
    with open(outname, "w+") as out:
        print >> out, "# Stats for class " + str(c)
        for line in lines:
            print >> out, line
