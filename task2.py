import numpy


# QUADRATIC SAMPLE POINTS
def quadratic_generator(point_number=100, alpha=1, beta=1, bias=0):
    x = numpy.random.randint(-100, 100, point_number)
    y = []
    for xi in x:
        y.append(alpha*xi**2+beta*xi+bias)
    return x, y


# UPDATE THE VALUE OF W AND BIAS
def update(x, y, w, b, eta):
    ud_w = 0
    ud_b = 0
    for i in range(len(x)):
        ud_w += x[i]*(y[i]-w*x[i]-b)/len(x)
        ud_b += (y[i]-w*x[i]-b)/len(x)
    return w+eta*ud_w, b+eta*ud_b


def loss(y, pred):
    loss = 0
    for i in range(len(y)):
        loss += (y[i]-pred[i])**2
    return loss


