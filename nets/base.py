class BaseNet(object):

    def __init__(self):
        return

    def input(self):
        raise NotImplementedError

    def inference(self):
        raise NotImplementedError

    def loss(self):
        raise NotImplementedError

    def solver(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
