class foo:
    def __init__(self):
        self.a = None
        self.b = None
    
    @classmethod
    def load(cls, data):
        model = cls()
        model.__dict__.update(data)
        return model

class boo(foo):
    def __init__(self):
        super(boo, self).__init__()

data = {'a':1,'b':2,'c':'hi'}

model = boo.load(data)
print(vars(model))