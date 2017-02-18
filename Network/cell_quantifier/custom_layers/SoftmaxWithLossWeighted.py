import caffe
import numpy as np

# Defines a Caffe loss layer that works the same as a normal Softmax loss layer,
# but weights each pixel's impact according to a HxW weight map, supplied by a third bottom blob.
class SoftmaxWithLossWeighted(caffe.Layer):

    # check if input blob number matches
    def setup(self, bottom, top):
        if len(bottom) != 3:
            raise Exception("SoftmaxWithLossWeighted needs three inputs: Predictions, Labels, Weights!")



    def reshape(self, bottom, top):

        # check if input blob dimensions match each other (bottom[0] is RGB = 3 channels)
        if bottom[0].count == (3 * bottom[1].count) == (3 * bottom[2].count):
            pass

        else:
            print ""
            #print "MISMATCHED DIMENSIONS WARNING! " + str(bottom[0].count) + " == " \
            #                                        + str(3 * bottom[1].count) + " == " \
            #                                        + str(3 * bottom[2].count)
            #print "Prediction shape:" + str(bottom[0].data.shape)
            #print "Labels shape:" + str(bottom[1].data.shape)
            #print "Weight map shape:" + str(bottom[2].data.shape) + "\n"

        #    raise Exception("the dimension of inputs should match")

        # For passing probabilities to backward pass
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)

        # loss output is the weighted loss, a positive scalar
        top[0].reshape(1)


    # Forward pass (bottom -> top):
    # What loss does the network currently generate
    # given the current weights?
    def forward(self, bottom, top):

        preds = bottom[0]
        labels = bottom[1].data.astype(np.uint16)
        weightmap = bottom[2]
        weightsum = 0
        loss = 0

        batchsize = bottom[0].data.shape[0]
        channel_num = np.max(bottom[0].data.shape[1]) # actually the number of classes in our case.
        ysize = bottom[0].data.shape[2]
        xsize = bottom[0].data.shape[3]

        # softmax squashing with numerical stability
        # because of C = -max(e^x)
        # see: http://cs231n.github.io/linear-classify/#softmax
        #maxpreds = preds.data[:, range(0, channel_num)] - np.max(preds.data[:, range(0, channel_num)])
        #squashed = np.exp(maxpreds) / np.sum(np.exp(maxpreds[:,z]) for z in range(0, channel_num))
        squashed = bottom[0].data

        # Cross-entropy loss function (CE):
        # CE(label, prediction) = 1/n \sum_{x} -ln(prediction chance of true label(x))
        #
        # In our use case, only one label is correct, which has a true probability of 1.0, while all others
        # have a true probability of 0. The label input lists only the one true integer label for each pixel,
        # so only the calculated probability of this true label matters. All the other terms are dropped in
        # the cross-entropy loss sum anyway, which reduces the formula to -log(softmax(true_label_index)).
        correct_squashed = np.zeros((batchsize, ysize, xsize))

        for bt in range(0, batchsize):
            for i in range (0, ysize):
                for j in range(0, xsize):
                    label_val = labels[bt, i, j]
                    weightsum += weightmap.data[bt, i, j]
                    correct_squashed[bt, i, j] = max(squashed[bt, label_val, i, j], np.nextafter(0,1)) # prevent NaN
                    # correct_squashed[bt, i, j] = max(squashed[bt, label_val, i, j] * weightmap.data[bt, i, j], np.nextafter(0,1))

        loss = np.sum(-np.log(correct_squashed))


        print "BG: " + str(squashed[0,0,81,47]*100) + "% " + \
              "R: " + str(squashed[0,1,81,47]*100) + "% " + \
              "G: " + str(squashed[0,2,81,47]*100) + "% " + \
              "B: " + str(squashed[0,3,81,47]*100) + "%"


        # save softmax probablities for backwards pass
        self.diff[...] = squashed

        # define layer output as average loss over all pixels in the batch
        top[0].data[0] = loss / (batchsize * ysize * xsize)



    # Backward pass (top -> bottom):
    # How to change the weights to guarantee smaller loss? (gradient computation)
    def backward(self, top, propagate_down, bottom):

        batchsize = bottom[0].data.shape[0]
        channel_num = bottom[0].data.shape[1]
        ysize = bottom[0].data.shape[2]
        xsize = bottom[0].data.shape[3]
        labels = bottom[1].data

        # Calculate gradients for all samples in the batch
        for bt in range(0, batchsize):

            weightsum = 0

            for i in range (0, ysize):
                for j in range (0, xsize):

                    # get index of correct label
                    label_index = labels[bt, i, j].astype(np.uint8)

                    # encode correct class as one-hot vector
                    one_hot = np.zeros(channel_num)
                    one_hot[label_index] = 1

                    # set gradient dCE/dProbs = pred_prob_i - true_prob_i
                    # i.e. the partial derivatives with respect to the inputs
                    # (here, the inputs are the class scores)
                    bottom[0].diff[bt, range(0, channel_num), i, j] = self.diff[bt, range(0, channel_num), i, j] \
                                                                      - one_hot[range(0, channel_num)]

                    # weigh pixel using the weight map
                    #weight = bottom[2].data[bt, i, j]
                    #bottom[0].diff[bt, label_index, i, j] *= weight

                    if(i == 81 and j == 47):
                        #print "preds: " + str(preds.data[bt, range(0, channel_num), i, j])
                        #print "diff(label 3): " + str(bottom[0].diff[bt, range(0, channel_num), i, j])

                        mingrad_index = bottom[0].diff[bt, range(0, channel_num), i, j].argmin()

                        if mingrad_index == 0:
                            print "Gradient direction: BG"
                        elif mingrad_index == 1:
                            print "Gradient direction: Red"
                        elif mingrad_index == 2:
                            print "Gradient direction: Green"
                        else:
                            print "Gradient direction: Blue"

                    # collect sum of weights for gradient normalization
                    # should prevent problems with having to change learning params every time
                    # weights are changed
                    #weightsum += weight

            # gradient normalization
            bottom[0].diff[...] /= batchsize
            bottom[0].diff[]
            #bottom[0].diff[bt, :, :, :] /= weightsum
