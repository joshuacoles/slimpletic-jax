from sympy import Symbol

import original.slimplectic_GGL as ggl
import numpy as np

from slimpletic.helpers import floatify1


class GalerkinGaussLobatto:
    def __init__(
            self,
            t: str,
            q_list: list[str],
            v_list: list[str],
            mod_list: list[bool] = False
    ):
        """
        GalerkinGaussLobatto class:
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
        assert type(q_list) is list, "List input required."
        assert type(v_list) is list, "List input required."

        for ii in range(len(q_list)):
            assert type(q_list[ii]) is str, "String input required."
            assert type(v_list[ii]) is str, "String input required."

        self.degrees_of_freedom = len(q_list)

        # Make sympy variables
        self.t = ggl.Symbol(t, real=True)
        self.q = [ggl.Symbol(qq, real=True) for qq in q_list]
        self.v = [ggl.Symbol(vv, real=True) for vv in v_list]

        # Double the sympy variables
        self.qp, self.qm = ggl.generate_pm_symbols(self.q)
        self.vp, self.vm = ggl.generate_pm_symbols(self.v)
        self.modlist = mod_list or []

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
        self._qi_soln_map, self._q_np1_map, self.compute_pi_next_inner, self._qdot_n_map, self.symbol_map = ggl.Gen_GGL_NC_VI_Map(
            self.t,
            self.q,
            self.qp,
            self.qm,
            self.v,
            self.vp,
            self.vm,
            L, K, order,
        )

    def integrate(
            self,
            q0: np.ndarray,
            pi0: np.ndarray,
            t,
            dt,
    ):
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
        # Allocate memory for solutions
        t_len = t.size
        q_list_soln = np.zeros((t_len + 1, self.degrees_of_freedom))
        pi_list_soln = np.zeros((t_len + 1, self.degrees_of_freedom))
        qdot_list_soln = np.zeros((t_len + 1, self.degrees_of_freedom))

        # Set initial data
        q_list_soln[0, :] = self.mod_values(q0)
        pi_list_soln[0, :] = pi0

        for ii in range(1, t_len + 1):
            q_next, pi_next, v_current = self.compute_next(
                previous_state=(q_list_soln[ii - 1], pi_list_soln[ii - 1]),
                t_value=t[ii - 1],
                dt=dt
            )

            q_list_soln[ii] = q_next
            pi_list_soln[ii] = pi_next
            qdot_list_soln[ii - 1] = v_current

        # Return the numerical solutions
        return q_list_soln[:-1], pi_list_soln[:-1], qdot_list_soln[:-1]

    def lagrangian(self, q, v, t: float):
        l_expr = self.symbol_map['L']

        return float(l_expr.subs([
            *zip(self.q, q),
            *zip(self.v, v),
            (self.t, t)
        ]))

    def qi_subs_table(self, qi_values):
        qi_table = self.symbol_map['q_Table']
        subs = []

        for i in range(len(qi_values)):
            for dof in range(self.degrees_of_freedom):
                subs.append((qi_table[dof][i], qi_values[i][dof]))

        return subs

    def lagrangian_d(self, qi_vec, t0, dt):
        ld_expr = self.symbol_map['Ld']
        subs = [
            (self.symbol_map['dt'], dt),
            *self.qi_subs_table(qi_vec)
        ]

        return float(ld_expr.subs(subs))

    def compute_qi_values(self, previous_q, previous_pi, t_value, dt):
        return self._qi_soln_map(previous_q, previous_pi, t_value, dt)

    def compute_pi_next(self, qi_values, t_value, dt):
        # TODO: Explict time dependence
        subs = [
            (self.symbol_map['dt'], dt),
            *self.qi_subs_table(qi_values)
        ]

        return floatify1([expr.subs(subs) for expr in self.symbol_map['pi_exprs']])

    def compute_next(
            self,
            previous_state,
            t_value,
            dt
    ):
        (previous_q, previous_pi) = previous_state
        args = [previous_q, previous_pi, t_value, dt]
        qi_values = self.compute_qi_values(previous_q, previous_pi, t_value, dt)

        q_next = self.mod_values(self._q_np1_map(qi_values, *args))
        pi_next = self.compute_pi_next_inner(qi_values, previous_q, previous_pi, t_value, dt)
        v_current = self._qdot_n_map(qi_values, *args)

        return q_next, pi_next, v_current

    def mod_values(self, values):
        # mod the value of any periodic variables that have mod value specified
        # This prevents NC evolution error due to increasing roundoff error as
        # any cyclic periodic variables become large.
        for jj, mod in enumerate(self.modlist):
            if mod:
                values[jj] = values[jj] % mod

        return values
