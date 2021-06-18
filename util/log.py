import pickle

class logger:
    def __init__(self, name, save_path):
        self.reward = []
        self.delay = []
        self.rtf = [] # relative traffic flow
        self.acb = [] # average cross block
        self.nci = [] # number of cars in the intersection

        self.name = name
        self.save_path = save_path

    def save(self):
        pass