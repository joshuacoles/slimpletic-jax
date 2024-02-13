import original.slimplectic_GGL as ggl
import numpy as np


class GalerkinGaussLobatto:
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
        self.t = ggl.Symbol(t, real=True)
        self.q = [ggl.Symbol(qq, real=True) for qq in q_list]
        self.v = [ggl.Symbol(vv, real=True) for vv in v_list]

        # Double the sympy variables
        self.qp, self.qm = ggl.q_Generate_pm(self.q)
        self.vp, self.vm = ggl.q_Generate_pm(self.v)

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
        self._qi_soln_map, self._q_np1_map, self._pi_np1_map, self._qdot_n_map = ggl.Gen_GGL_NC_VI_Map(self.t, \
                                                                                                       self.q, self.qp,
                                                                                                       self.qm, \
                                                                                                       self.v, self.vp,
                                                                                                       self.vm, \
                                                                                                       L, K, order,
                                                                                                       method=method,
                                                                                                       verbose=verbose)

    def integrate(self, q0_list, pi0_list, t, dt=False, output_v=False, output_File=False, t_out=[False],
                  print_steps=1):
        """Numerical integration from given initial data
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

        # Open output file if necessary
        if output_File:
            outfile = open(output_File, "w")

        # Set initial data
        q_list_soln[0, :] = q0_list
        # mod the value of any periodic variables that have mod value specified
        # This prevents NC evolution error due to increasing roundoff error as
        # any cyclic periodic variables become large.
        for jj, mod in enumerate(self.modlist):
            if mod:
                q_list_soln[0, jj] = q_list_soln[0, jj] % mod
        pi_list_soln[0, :] = pi0_list

        if output_File:
            outstring = 'step'
            outstring += ', t'
            for dof in range(len(q_list_soln[0])):
                outstring += ', {:s}'.format(str(self.q[dof]))
            for dof in range(len(pi_list_soln[0])):
                outstring += ', pi_{:s}'.format(str(self.q[dof]))
            if output_v:
                for dof in range(len(q_list_soln[0])):
                    outstring += ', {:s}'.format(str(self.v[dof]))
            outfile.write(outstring + '\n')
            outfile.flush()

        # Perform the integration at fixed time steps
        if dt:
            ddt = dt
        else:
            ddt = t[1] - t[0]
        for ii in range(1, t_len + 1):
            args = [q_list_soln[ii - 1], pi_list_soln[ii - 1], t[ii - 1], ddt]
            qi_sol = self._qi_soln_map(*args)
            q_list_soln[ii] = self._q_np1_map(qi_sol, *args)
            # mod the value of any periodic variables that have mod value specified
            # This prevents NC evolution error due to increasing roundoff error as
            # any cyclic periodic variables become large.
            for jj, mod in enumerate(self.modlist):
                if mod:
                    q_list_soln[ii, jj] = q_list_soln[ii, jj] % mod
            pi_list_soln[ii] = self._pi_np1_map(qi_sol, *args)
            if output_v:
                qdot_list_soln[ii - 1] = self._qdot_n_map(qi_sol, *args)

            if output_File:
                if (ii - 1) % print_steps == 0:
                    outstring = '{:d}'.format(ii - 1)
                    if isinstance(t_out[0], float):
                        outstring += ', {:.15e}'.format(t_out[ii - 1])
                    else:
                        outstring += ', {:.15e}'.format(t[ii - 1])
                    for q_val in q_list_soln[ii - 1]:
                        outstring += ', {:.15e}'.format(q_val)
                    for pi_val in pi_list_soln[ii - 1]:
                        outstring += ', {:.15e}'.format(pi_val)
                    if output_v:
                        for qdot_val in qdot_list_soln[ii - 1]:
                            outstring += ', {:.15e}'.format(qdot_val)
                    outfile.write(outstring + '\n')
                    outfile.flush()
        # Return the numerical solutions
        if output_v:
            return q_list_soln[:-1].T, pi_list_soln[:-1].T, qdot_list_soln[:-1].T
        return q_list_soln[:-1].T, pi_list_soln[:-1].T

    def __call__(self, qi_list, vi_list, t):
        return self.integrate(qi_list, vi_list, t)
