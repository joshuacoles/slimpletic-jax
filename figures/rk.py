import numpy as np


##########################################
# Class for 4th order Runge-Kutta method #
##########################################

class RungeKutta4(object):

    def __init__(self):
        self._b = np.array([1. / 6., 1. / 3., 1. / 3., 1. / 6.])
        self._k = [[], [], [], []]

    def _iter(self, tn, yn_list, f, h):
        self._k[0] = f(tn, yn_list)
        self._k[1] = f(tn + h / 2., yn_list + h / 2. * self._k[0])
        self._k[2] = f(tn + h / 2., yn_list + h / 2. * self._k[1])
        self._k[3] = f(tn + h, yn_list + h * self._k[2])
        return yn_list + h * np.sum(self._b[ii] * self._k[ii] for ii in range(4))

    def integrate(self, q0_list, v0_list, t, f):
        y0_list = np.hstack([q0_list, v0_list])
        h = t[1] - t[0]
        ans = np.zeros((t.size, len(y0_list)))
        ans[0, :] = y0_list
        for ii, tt in enumerate(t[:-1]):
            ans[ii + 1, :] = self._iter(tt, ans[ii], f, h)
        out = ans.T
        return out[:len(q0_list)], out[len(q0_list):]

    def __call__(self, q0, v0, t, f):
        return self.integrate(q0, v0, t, f)


##########################################
# Class for 2nd order Runge-Kutta method #
##########################################

class RungeKutta2(object):

    def __init__(self):
        self._b = np.array([0., 1.])
        self._k = [[], []]

    def _iter(self, tn, yn_list, f, h):
        self._k[0] = f(tn, yn_list)
        self._k[1] = f(tn + h / 2., yn_list + h / 2. * self._k[0])
        return yn_list + h * np.sum(self._b[ii] * self._k[ii] for ii in range(2))

    def integrate(self, q0_list, v0_list, t, f):
        y0_list = np.hstack([q0_list, v0_list])
        h = t[1] - t[0]
        ans = np.zeros((t.size, len(y0_list)))
        ans[0, :] = y0_list
        for ii, tt in enumerate(t[:-1]):
            ans[ii + 1, :] = self._iter(tt, ans[ii], f, h)
        # return ans.T
        out = ans.T
        return out[:len(q0_list)], out[len(q0_list):]

    def __call__(self, q0, v0, t, f):
        return self.integrate(q0, v0, t, f)
