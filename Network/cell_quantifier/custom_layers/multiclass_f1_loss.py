import caffe
import numpy as np

class F1Loss(caffe.Layer):
    """
    F1-score loss function for multi-class classification in neural networks.
    """
    numLabels = 0    
    oneHotLabels = None  


    def setup(self, bottom, top):
        # two inputs necessary: softmax output and ground truth
        if len(bottom) != 2:
            raise Exception("Two inputs required: softmax output and ground truth (labels)")

    def reshape(self, bottom, top):
        # bottom[0] is softmax out (i.e., contains class probabilities for each class), bottom[1] is assumed to contain (integer) labels
        numLabels = bottom[0].data.shape[1]

        # convert labels to one-hot encoding
        # labels are chosen as [0 .. n-1] where n is the number of classes 
        tmpOneHotLabels = np.eye(numLabels)[bottom[1].data.astype(int)]
        
        # check input dimensions match
        if bottom[0].data.size != tmpOneHotLabels.size:
            print("Number of Batches: " + str(bottom[0].data.shape[0]))
            print("Number of Labels: " + str(numLabels))
            print("Classifier Output Shape: " + str(bottom[0].data.shape))
            print("One-hot encoded Label Shape: " + str(tmpOneHotLabels.shape))
            raise Exception("the dimension of inputs should match")

        # loss output is a scalar for each class and an additional scalar (the combined F1-loss for all classes)
        # TODO: compute both version of combined F1-score (i.e., micro and macro)? 
        #top[0].reshape(numLabels + 1)
        top[0].reshape(numLabels)

    def forward(self, bottom, top):
        # bottom[0] is softmax out (i.e., contains class probabilities for each class), bottom[1] is assumed to contain (integer) labels
        numBatches = bottom[0].data.shape[0] 
        self.numLabels = bottom[0].data.shape[1]
        # calculate one-hot encoding for labels (see reshape() - we have to do this here, since reshape() is not called every iteration)
        self.oneHotLabels = np.eye(self.numLabels)[bottom[1].data.astype(int)]

        # this gives us the current net results as integer labels in a two-dim array [batch, pixels]
        result = np.reshape(np.squeeze(np.argmax(bottom[0].data[...],axis=1)),[numBatches,bottom[0].data.shape[2]])
        # create the one-hot encoding for this 
        oneHotResult = np.eye(self.numLabels)[result.astype(dtype=np.uint32)]

        # score is computed for each class separately and then in a combind fashion
        f1scores = np.zeros([self.numLabels, numBatches],dtype=np.float32)
        
        # we need the values for each batch for each class
        union = np.zeros([numBatches],dtype=np.float32)
        intersection = np.zeros([numBatches],dtype=np.float32)

        for c in range(0, self.numLabels):
            # get the binary results just for this class
            classResult = np.reshape(np.squeeze(oneHotResult[:, :, c]), [numBatches,bottom[0].data.shape[2]]).astype(dtype=np.float32)
            # get the binary labels just for this class
            classLabels = np.reshape(np.squeeze(self.oneHotLabels[:, :, c]), [numBatches,bottom[0].data.shape[2]]).astype(dtype=np.float32)

            # loop over batches
            for i in range(0,numBatches):
                # compute F1-score
                union[i]=(np.sum(classResult[i,:]) + np.sum(classLabels[i,:]))
                intersection[i]=(np.sum(classResult[i,:] * classLabels[i,:]))

                f1scores[c][i] = 2 * intersection[i] / (union[i]+0.00001)

            # average over batches
            top[0].data[c]=np.average(f1scores[c, :])

        # average over classes, TODO: compute actual combined F-measure...
        #top[0].data[self.numLabels] = np.average(top[0].data[0:self.numLabels])

    def backward(self, top, propagate_down, bottom): 
        for btm in [0]:
            numBatches = bottom[btm].data.shape[0]
            # we need the values for each batch for each class
            union = np.zeros([numBatches],dtype=np.float32)
            intersection = np.zeros([numBatches],dtype=np.float32)

            bottom[btm].diff[...] = np.zeros(bottom[btm].diff.shape, dtype=np.float32)
            
            # iterate over classes            
            for c in range(0, self.numLabels):
                # get class probabilities
                currentClassProb = np.reshape(np.squeeze(bottom[btm].data[:,c,:]), [numBatches, bottom[0].data.shape[2]]).astype(dtype=np.float32)
                classLabels = np.reshape(np.squeeze(self.oneHotLabels[:, :, c]), [numBatches,bottom[0].data.shape[2]]).astype(dtype=np.float32)


                # iterate over batches
                for i in range(0, bottom[btm].diff.shape[0]):
                    union[i]=(np.sum(currentClassProb[i,:]) + np.sum(classLabels[i,:]))
                    intersection[i]=(np.sum(currentClassProb[i,:] * classLabels[i,:]))

                    denom = (union[i]) ** 2 + 0.00001
                    partials = 2.0 * (classLabels[i, :] * union[i] - intersection[i]) / (denom * numBatches)

                    #print("Class " + str(c) + ":")
                    #print(currentClassProb[i,55*244+85])
                    #print(classLabels[i,55*244+85])
                    #print(partials[55*244+85])
                    #print("---------------------")

                    bottom[btm].diff[i, c, :] -= partials
