import numpy as np


class NoNormalization:
    def __init__(self):
        pass

    def __mul__(self, x):
        return np.array(object=[x], dtype=np.float32)

    def __rmul__(self, x):
        return np.array(object=[x], dtype=np.float32)

    def __truediv__(self, x):
        return x


class PeriodicNormalization:
    def __init__(self, x_max):
        self.x_max = x_max

    def __mul__(self, x):
        x = 2 * np.pi * x / self.x_max
        x_sin = np.sin(x)
        x_cos = np.cos(x)
        return np.array(object=[(x_sin + 1) / 2.0, (x_cos + 1) / 2.0], dtype=np.float32)

    def __rmul__(self, x):
        x = 2 * np.pi * x / self.x_max
        x_sin = np.sin(x)
        x_cos = np.cos(x)
        return np.array(object=[(x_sin + 1) / 2.0, (x_cos + 1) / 2.0], dtype=np.float32)

    def __truediv__(self, x):
        sin_val = 2 * x[0] - 1
        cos_val = 2 * x[1] - 1
        angle = np.arctan2(sin_val, cos_val)
        return int(np.round((angle * self.x_max / (2 * np.pi)), 0))


class OneHotEncoding:
    def __init__(self, classes):
        self.classes = classes

    def __mul__(self, x):
        identity_mat = np.eye(len(self.classes), dtype=np.float32)
        return identity_mat[np.array(self.classes) == x][0]

    def __rmul__(self, x):
        identity_mat = np.eye(len(self.classes), dtype=np.float32)
        return identity_mat[np.array(self.classes) == x][0]

    def __truediv__(self, x):
        idx = np.argmax(x)
        return self.classes[idx]


class MinMax:
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def __mul__(self, x):
        if self.x_min == self.x_max:
            return np.array(object=[0.], dtype=np.float32)
        else:
            return np.array(object=[(x - self.x_min) / (self.x_max - self.x_min)], dtype=np.float32)

    def __rmul__(self, x):
        if self.x_min == self.x_max:
            return np.array(object=[0.], dtype=np.float32)
        else:
            return np.array(object=[(x - self.x_min) / (self.x_max - self.x_min)], dtype=np.float32)

    def __truediv__(self, x):
        return x[0] * (self.x_max - self.x_min) + self.x_min


class Scale:
    def __init__(self, x_max):
        self.x_max = x_max

    def __mul__(self, x):
        if self.x_max == 0:
            return np.array(object=[0.], dtype=np.float32)
        else:
            return np.array(object=[x / self.x_max], dtype=np.float32)

    def __rmul__(self, x):
        if self.x_max == 0:
            return np.array(object=[0.], dtype=np.float32)
        else:
            return np.array(object=[x / self.x_max], dtype=np.float32)

    def __truediv__(self, x):
        return x[0] * self.x_max


class RemoveFeature:
    def __init__(self):
        pass

    def __mul__(self, x):
        return None

    def __rmul__(self, x):
        return None

    def __truediv__(self, x):
        return 0.
