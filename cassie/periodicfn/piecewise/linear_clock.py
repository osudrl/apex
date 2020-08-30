import numpy as np

# There probably already is a decorator for this somewhere in numpy... seems like a no-brainer
def vectorized_func(func):
    vfunc = np.vectorize(func, doc=f'Vectorized `{func.__name__}`')
    return vfunc

# TODO: Replace flip with some sort of phase-offset
@vectorized_func
def linear_piecewise_clock(phi, ratio=0.5, alpha=0.05, flip=False):
    # domain for period is x: [0,1]
    beta = 0.0
    if flip:
        beta = 0.5
    phi = np.fmod(phi + beta, 1)

    saturation = alpha * (ratio / 2 - 1e-3)
    slope = 1 / ((ratio / 2) - saturation)

    if phi < saturation:
        return 1.0
    elif phi < ratio/2:
        return 1 - slope * (phi - saturation)
    elif phi < 1 - ratio/2:
        return 0.0
    elif phi < 1 - saturation:
        return 1 + slope * (phi + saturation - 1)
    else:
        return 1.0

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ratio = 0.5
    alpha = 0.05

    xs = np.array(list(range(5000))) / 1000
    ys = linear_piecewise_clock(xs, ratio=ratio, alpha=alpha)
    plt.plot(xs, ys)
    plt.show()
