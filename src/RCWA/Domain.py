import numpy as np


class Domain:
    def __init__(self):
        pass

    def set_domain(self, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        self.period_x = x_max - x_min
        self.period_y = y_max - y_min

    def set_period_centered(self, period_x, period_y):
        self.set_domain(-period_x/2, period_x/2, -period_y/2, period_y/2)

    def set_period(self, period_x, period_y):
        self.set_domain(0, period_x, 0, period_y)

    def get_coordinate(self, nx, ny):
        x = (np.arange(nx)+0.5) * self.period_x / nx + self.x_min
        y = (np.arange(ny)+0.5) * self.period_y / ny + self.y_min
        x, y = np.meshgrid(x, y)
        return x, y
