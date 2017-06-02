# Get the average time needed to classify an entire image
# from the log file containing results of the
# multiple passes on the same image.

filename = "benchmark_4.txt"
numiters = 100

cum = 0.
with open(filename, "r") as f:
    for line in f:
        cum += float(line)

cum /= float(numiters)

print cum
