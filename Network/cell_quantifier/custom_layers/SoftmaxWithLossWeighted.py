import caffe
import numpy as np

# Defines a Caffe loss layer that works the same as a normal Softmax loss layer,
# but weights each pixel's impact according to a HxW weight map, supplied by a third bottom blob.
class SoftmaxWithLossWeighted(caffe.Layer):

    # check if input blob number matches
    def setup(self, bottom, top):
        if len(bottom) != 3:
            raise Exception("SoftmaxWithLossWeighted needs three inputs: Predictions, Labels, Weights!")

        self.inputs = np.zeros(bottom[0].data.shape, dtype=np.float32)



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

        # loss output is the weighted loss, a positive scalar
        top[0].reshape(1)


    # Forward pass (bottom -> top):
    # What loss does the network currently generate
    # given the current weights?
    def forward(self, bottom, top):
        preds = bottom[0]
        labels = bottom[1].data.astype(np.uint8)
        weightmap = bottom[2]
        weightsum = 0
        loss = 0

        batchsize = bottom[0].data.shape[0]
        channel_num = np.max(bottom[0].data.shape[1]) # actually the number of classes in our case.
        ysize = bottom[0].data.shape[2]
        xsize = bottom[0].data.shape[3]

        # save probability scores for use in backwards pass
        self.inputs = preds.data

        # Calculate loss over all pixels in all samples of the batch
        for bt in range(0, batchsize):
            for i in range(0, ysize):
                for j in range(0, xsize):

                    # softmax squashing with numerical stability
                    # because of C = -max(e^x)
                    # see: http://cs231n.github.io/linear-classify/#softmax
                    maxpreds = preds.data[bt, range(0, channel_num), i, j] - np.max(preds.data[bt, range(0, channel_num), i, j])
                    squashed = np.exp(maxpreds) / np.sum(np.exp(maxpreds[z]) for z in range(0, channel_num))

                    # Cross-entropy loss function (CE):
                    # CE(label, prediction) = 1/n \sum_{x} -ln(prediction chance of true label(x))
                    #
                    # In our use case, only one label is correct, which has a true probability of 1.0, while all others
                    # have a true probability of 0. The label input lists only the one true integer label for each pixel,
                    # so only the calculated probability of this true label matters. All the other terms are dropped in
                    # the cross-entropy loss sum anyway, which reduces the formula to -log(softmax(true_label_index)).
                    label_val = labels[bt, i, j]
                    loss += -np.log(squashed[label_val])# * weightmap.data[bt, i, j]


                    if(i == 200 and j == 0):
                        print "preds: " + str(preds.data[bt, range(0, channel_num), i, j])
                        #print "maxpreds: " + str(maxpreds)
                        #print "squashed: " + str(squashed)
                        #print "real label: " + str(label_val)
                        #print "diff//forward: " + str(self.diff[bt, range(0, channel_num), i, j])


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

                    # derivative of cross-entropy is =
                    one_hot = np.zeros(channel_num)
                    one_hot[label_index] = 1

                    # set gradient dCE/dProbs = pred_prob_i - true_prob_i
                    # i.e. the partial derivatives with respect to the inputs
                    # (here, the inputs are the class scores)
                    bottom[0].diff[bt, range(0, channel_num), i, j] = self.inputs[bt, range(0, channel_num), i, j] - one_hot[range(0, channel_num)]

                    # weigh pixel using the weight map
                    weight = bottom[2].data[bt, i, j]
                    bottom[0].diff[bt, label_index, i, j]# *= weight

                    if(i == 200 and j == 0):
                        print "diff: " + str(bottom[0].diff[bt, range(0, channel_num), i, j])

                    # collect sum of weights for gradient normalization
                    # should prevent problems with having to change learning params every time
                    # weights are changed
                    weightsum += weight

            # gradient normalization
            bottom[0].diff[bt, :, :, :] /= ysize * xsize
            #bottom[0].diff[bt, :, :, :] /= weightsum
