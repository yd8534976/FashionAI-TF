class BaseModel(object):

    def __init__(self):
        pass

    def build_inputs(self):
        raise NotImplementedError

    def build_inference(self):
        raise NotImplementedError

    def build_loss(self):
        raise NotImplementedError

    def build_solver(self):
        raise NotImplementedError

    def train_op(self):
        raise NotImplementedError
