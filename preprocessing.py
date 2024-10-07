import numpy as np


class NoNormalization:
    def __init__(self):
        pass

    def __mul__(self, x):
        return x

    def __rmul__(self, x):
        return x


class PeriodicNormalization:
    def __init__(self, x_max):
        self.x_max = x_max

    def __mul__(self, x):
        x = 2 * np.pi * x / self.x_max
        x_sin = np.sin(x)
        x_cos = np.cos(x)
        return np.array([(x_sin+1)/2.0, (x_cos+1)/2.0])

    def __rmul__(self, x):
        x = 2 * np.pi * x / self.x_max
        x_sin = np.sin(x)
        x_cos = np.cos(x)
        return np.array([(x_sin+1)/2.0, (x_cos+1)/2.0])


class OneHotEncoding:
    def __init__(self, classes):
        self.classes = classes

    def __mul__(self, x):
        identity_mat = np.eye(len(self.classes))
        return identity_mat[np.array(self.classes) == x][0]

    def __rmul__(self, x):
        identity_mat = np.eye(len(self.classes))
        return identity_mat[np.array(self.classes) == x][0]


class Normalize:
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def __mul__(self, x):
        if self.x_min == self.x_max:
            return 0
        else:
            return (x - self.x_min)/(self.x_max - self.x_min)

    def __rmul__(self, x):
        if self.x_min == self.x_max:
            return 0
        else:
            return (x - self.x_min)/(self.x_max - self.x_min)


class RemoveFeature:
    def __init__(self):
        pass

    def __mul__(self, x):
        return None

    def __rmul__(self, x):
        return None
