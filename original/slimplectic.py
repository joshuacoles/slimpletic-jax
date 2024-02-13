import numpy as np
from .slimplectic_GGL import Gen_GGL_NC_VI_Map, Symbol, q_Generate_pm


###########################################
# Interface class for the nonconservative #
#         variational integrator          #
###########################################

class GalerkinGaussLobatto(object):

    def __init__(self, t, q_list, v_list, mod_list=False):
        """GalerkinGaussLobatto class:
        args: t : string for generating the sympy symbol the independent "time" variable
              q_list : list of strings for generating symbols for dof q
              v_list : list of strings for generating symbols for \dot{q}
              mod_list : list of values to mod the q values by for periodic variables
                         if not periodic, the value is set to False (default).

        methods: discretize : creates the variational integrator maps
                 integrate: applies the maps created by discretize
        """
        # Validate inputs
        assert type(t) is str, "String input required."
        assert len(q_list) == len(v_list), "Unequal number of coordinates and velocities."
        self._num_dof = len(q_list)
        assert type(q_list) is list, "List input required."
        assert type(v_list) is list, "List input required."

        for ii in range(len(q_list)):
            assert type(q_list[ii]) is str, "String input required."
            assert type(v_list[ii]) is str, "String input required."

        # Make sympy variables
        self.t = Symbol(t, real=True)
        self.q = [Symbol(qq, real=True) for qq in q_list]
        self.v = [Symbol(vv, real=True) for vv in v_list]

        # Double the sympy variables
        self.qp, self.qm = q_Generate_pm(self.q)
        self.vp, self.vm = q_Generate_pm(self.v)

        # keep track of which variables are periodic and need to be modded
        if mod_list:
            self.modlist = mod_list
        if not mod_list:
            self.modlist = []
            for q in q_list:
                self.modlist.append(False)

    def keys(self):
        return self.__dict__.keys()

    def discretize(self, L, K, order, method='explicit', verbose=False):
        """Generate the nonconservative variational integrator maps
        by setting the methods _qi_soln_map, _q_np1_map, _pi_np1_map, _qdot_n_map
        args: L : sympy expression for the conservative Lagrangian. Should be in
                  terms of the self.q and self.v variables
              K : sympy expression for the nonconservative potential. Should be in
                  terms of the self.qp/m and self.vp/m variables
              order: integer order (r) of the GGL method (r+2 is the total order of the method)
              method: string 'implicit' or 'explicit' evaluation (default explicit)
              verbose: Boolean. True to output mapping expressions (default False)
        output: none
        """
        self.order = order
        self.debug_escape_info = {}
        self._qi_soln_map, self._q_np1_map, self._pi_np1_map, self._qdot_n_map = Gen_GGL_NC_VI_Map(self.t, \
                                                                                                   self.q, self.qp,
                                                                                                   self.qm, \
                                                                                                   self.v, self.vp,
                                                                                                   self.vm, \
                                                                                                   L, K, order,
                                                                                                   method=method,
                                                                                                   verbose=verbose,
                                                                                                   debug_escape_info=self.debug_escape_info)

    def integrate(self, q0_list, pi0_list, t, dt=False, output_v=False, output_File=False, t_out=[False],
                  print_steps=1):
        """
        Numerical integration from given initial data
        args: q0_list: list of initial q values (floats)
              pi0_list: list of initial pi (nonconservative momentum) values (floats)
              t: list of t values (floats) over which to integrate the system.
              dt: defaults to the difference between the first two elements of the t array. Otherwise
                  the stepsize must be specified as a float.
              output_v: Boolean value, if True, output will be
                        q_list_soln.T, pi_list_soln.T, qdot_list_soln.T
                        defaults to False
              outputFile: filename for the output file. Defaults to False and no file is produced.
                          output is in csv format, with first line the column labels.
              t_out: sets the output t_array for the output_File. Defaults to same as t
              print_steps: outputs to file every print_steps steps. Defaults to 1.
        output: q_list_soln.T, pi_list_soln.T the integrated q and pi arrays at each time.
                unless output_v is True
        """
        # Check if total Lagrangian is discretized already
        if not hasattr(self, '_qi_soln_map'):
            raise AttributeError("Run `discretize` to discretize the total Lagrangian.")

        # Validate input
        assert type(q0_list) in [list, np.ndarray], "List or numpy array input required."
        assert type(pi0_list) in [list, np.ndarray], "List or numpy array input required."

        # Allocate memory for solutions
        t_len = t.size
        q_list_soln = np.zeros((t_len + 1, self._num_dof))
        pi_list_soln = np.zeros((t_len + 1, self._num_dof))
        qdot_list_soln = np.zeros((t_len + 1, self._num_dof))

        qi_reconstruction = np.zeros((t_len + 1, self._num_dof * (self.order + 2)))

        # Set initial data
        q_list_soln[0, :] = q0_list
        # mod the value of any periodic variables that have mod value specified
        # This prevents NC evolution error due to increasing roundoff error as
        # any cyclic periodic variables become large.
        for jj, mod in enumerate(self.modlist):
            if mod:
                q_list_soln[0, jj] = q_list_soln[0, jj] % mod
        pi_list_soln[0, :] = pi0_list

        # Perform the integration at fixed time steps
        if dt:
            ddt = dt
        else:
            ddt = t[1] - t[0]

        header = "t,"

        for ii in range(self._num_dof * (self.order + 2)):
            header += f"qi_sol_{ii},"

        print(header)

        for ii in range(1, t_len + 1):
            args = [q_list_soln[ii - 1], pi_list_soln[ii - 1], t[ii - 1], ddt]
            qi_sol = self._qi_soln_map(*args)
            qi_reconstruction[ii - 1] = qi_sol
            q_list_soln[ii] = self._q_np1_map(qi_sol, *args)

            # mod the value of any periodic variables that have mod value specified
            # This prevents NC evolution error due to increasing roundoff error as
            # any cyclic periodic variables become large.
            for jj, mod in enumerate(self.modlist):
                if mod:
                    q_list_soln[ii, jj] = q_list_soln[ii, jj] % mod

            pi_list_soln[ii] = self._pi_np1_map(qi_sol, *args)

            # Print row
            print(f"{t[ii - 1]},", end="")

            print()

            if output_v:
                qdot_list_soln[ii - 1] = self._qdot_n_map(qi_sol, *args)

        # Return the numerical solutions
        if output_v:
            return q_list_soln[:-1].T, pi_list_soln[:-1].T, qdot_list_soln[:-1].T

        return q_list_soln[:-1].T, pi_list_soln[:-1].T, qi_reconstruction

    def __call__(self, qi_list, vi_list, t):
        return self.integrate(qi_list, vi_list, t)


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
