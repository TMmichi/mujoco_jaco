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

data1 = {'a':1,'b':2,'c':'hi'}
data2 = {'a':1,'b':2,'c':'hello'}

model = boo.load(data1)
print(vars(model))
model = boo.load(data2)
print(vars(model))