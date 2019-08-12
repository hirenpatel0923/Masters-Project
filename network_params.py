class NetworkParams():
    def __init__(self):
        self.z_dim = 128
        self.origImageWidth = 75
        self.origImageHeight = 75
        self.modifiedWidth = 64
        self.modifiedHeight = 64
        self.numChannels = 3
        self.batch_size = 128
        self.shuffle = True
        self.dim = (self.modifiedHeight, self.modifiedWidth)
    


