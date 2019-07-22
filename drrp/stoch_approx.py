import gurobipy as gu
import numpy as np
import os
import platform
import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '' and platform.system() == 'Linux':
    mpl.use('Agg')  # Use non-interactive Agg backend on Linux server
import matplotlib.pyplot as plt
import pandas as pd
import time
import itertools
from scipy.io import savemat, loadmat
mpl.rcParams['font.family'] = 'serif'

gu_status_codes = {1: "Loaded", 2: "Optimal", 3: "Infeasible", 4: "Infeasible or unbounded",
                   5: "Unbounded", 6: "Cutoff", 7: "Iteration limit", 8: "Node limit",
                   9: "Time limit", 10: "Solution limit", 11: "Interrupted", 12: "Numerical issues",
                   13: "Suboptimal", 14: "In progress", 15: "User objective limit"}


# Stochastic approximation model
class SAModel(object):

    def __init__(self, nw, cost_params, alg_params, t_horizon, instance, label=''):
        self.T = t_horizon
        self.nw = nw
        self.cut_lists = []
        self.w_vars_added = [0] * self.T
        self.w_indices = []
        self.alg_params = alg_params
        self.cost_params = cost_params
        self.instance_no = instance

        # Optimization models
        self.m_s1, self.c_list_s1, self.x_s1, self.vstage_s1 = self.create_s1_model(cost_params)
        self.m_s1r = self.m_s1.relax()
        self.m_s1_warmstart = None
        self.set_s1_initial_state()

        self.m_s2, self.c_list_s2, self.x_s2 = self.create_s2_model(cost_params)
        self.m_s2r = self.m_s2.relax()
        self.set_s2_constr_rhs()

        # Store value function approximations (vfpp = VF pre-projection; vf = Value Function)
        self.vfpp = [np.zeros((self.T, self.nw.nr_regions, 2 * self.alg_params['max_dev']))]
        self.vf_proj_m, self.vf_proj_x = self.setup_unit_projection_model(concave=False,
                                                                          max_slope=25.0)
        self.vf = [self.project_vf(self.vfpp[0])]
        self.last_x1_sol = None
        # Dictionaries for cost estimates
        self.lb, self.ub, self.stats = {}, {}, {}
        self.ub_no_action, self.stats_no_action = None, None
        # Results dataframe
        self.results_df = pd.DataFrame(columns=['N', 'V', 'T', 'Inst.', 'k', 'Integer',
                                                'Cost', 'SR', 't1', 't2'])
        self.results_label = label

    def approx(self):
        """
        Computes an approximate solution to the stochastic DRRP by
        (i) Iteratively approximating the second-stage value function, representing the expected
            cost of unserved customers
        (ii) Using the approximate second-stage value function to generate a solution to the first
            stage
        :return: Nothing, outputs file results to directory ./output/
        """
        N, V, T, i = self.nw.nr_regions, self.nw.nr_vehicles, self.T, self.instance_no

        random_s1, nominal_s2 = self.alg_params['random_s1'], self.alg_params['nominal_s2']
        save_iter_models, max_dev = self.alg_params['save_iter_models'], self.alg_params['max_dev']
        eval_cost_k = self.alg_params['eval_cost_k']

        # (i) Iteratively approximate second-stage value function with a separable function

        n_iter = self.alg_params['n_iter']
        for k in range(1, n_iter + 1):
            print "k = %d:" % k
            if save_iter_models and not random_s1:
                print "Saving model..."
                self.m_s1.write('output/model_N%d_V%d_T%d_i%d_k%d' % (N, V, T, i, k) +
                                self.results_label + '.mps')

            # 1. Optimize vehicle routes and repositioning actions against current VFA, return xk
            if random_s1:
                s1_cost, cost_to_go, ts1 = 0.0, 0.0, 0.0
                s1_sol = np.random.random_integers(-max_dev, max_dev - 1, (N, T))
                s1_really_solved = False
            else:
                s1_cost, cost_to_go, s1_sol, ts1 = self.optimize_stage(stage=1)
                s1_really_solved = True
            self.last_x1_sol = s1_sol
            # self.print_s1_solution(s1_sol)
            # 2. Update initial conditions of S2 model
            self.set_s2_constr_rhs(s1_sol, really_solved_s1=s1_really_solved)
            # 3. Evaluate lower and upper bounds
            self.lb[k] = s1_cost
            eval_this_iteration = k in eval_cost_k or (N == 64 and V == 9)
            if eval_this_iteration:
                self.ub[k], self.stats[k] = self.estimate_costs(s1_cost - cost_to_go,
                                                                deterministic_s2=nominal_s2)
            # 4. Optimize journeys given xk, and return objective function value and multipliers
            self.resample_s2_demand(deterministic=nominal_s2)
            s2_cost, s2_sol, s2_lambdas, ts2 = self.optimize_stage(stage=2)
            # 5. Update value function using step size rule
            self.update_vf(s1_sol, s2_lambdas, k, verify=False)
            # 6. Log results for this iteration
            if eval_this_iteration:
                df_entry = {'N': np.rint(N), 'V': np.rint(V), 'T': np.rint(T), 'Inst.': np.rint(i),
                            'k': np.rint(k), 'Integer': 0,
                            'Cost': np.round(self.ub[k]['cost'], 6),
                            'SR': np.round(self.stats[k]['sr'] * 100, 6),
                            't1': np.round(ts1, 6), 't2': np.round(ts2, 6)}
            else:
                df_entry = {'N': np.rint(N), 'V': np.rint(V), 'T': np.rint(T), 'Inst.': np.rint(i),
                            'k': np.rint(k), 'Integer': 0,
                            't1': np.round(ts1, 6), 't2': np.round(ts2, 6)}
            self.results_df = self.results_df.append(df_entry, ignore_index=True)
            if not self.alg_params['random_s1'] or (self.alg_params['random_s1'] and
                                                    divmod(k, 10)[1] == 0):
                self.results_df.to_csv('output/stats_N%d_V%d_T%d_i%d' % (N, V, T, i) +
                                       self.results_label + '.csv', index=False)

        # Save final approximate VF for use in subsequent optimizations
        mat_fname = 'vfs/N%d_V%d_T%d_i%d_k%d' % (N, V, T, i, n_iter) + self.results_label + '.mat'
        savemat(mat_fname, {'vf': self.vf[-1]})

        # (ii) Generate a final first-stage solution and estimate resulting second-stage costs
        if self.alg_params['final_sol']:
            final_method = self.alg_params['final_sol_method']
            final_ts1, final_ub, final_stats = self.integer_solve(method=final_method)
            df_entry = {'N': np.rint(N), 'V': np.rint(V), 'T': np.rint(T), 'Inst.': np.rint(i),
                        'k': n_iter+1, 'Integer': final_method, 't1': np.round(final_ts1, 6),
                        'Cost': np.round(final_ub['cost'], 6),
                        'SR': np.round(final_stats['sr'] * 100, 6)}
            self.results_df = self.results_df.append(df_entry, ignore_index=True)
            self.results_df['N'] = self.results_df['N'].map(lambda x: "%d" % x)
            self.results_df['V'] = self.results_df['V'].map(lambda x: "%d" % x)
            self.results_df['T'] = self.results_df['T'].map(lambda x: "%d" % x)
            self.results_df['Inst.'] = self.results_df['Inst.'].map(lambda x: "%d" % x)
            self.results_df['k'] = self.results_df['k'].map(lambda x: "%d" % x)
            self.results_df.to_csv('output/stats_N%d_V%d_T%d_i%d' % (N, V, T, i) +
                                   self.results_label + '.csv', index=False)

    def optimize_stage(self, stage=1, mute=False):
        m = self.m_s1 if stage == 1 else self.m_s2
        x_list = self.x_s1 if stage == 1 else self.x_s2
        # if stage == 1 and self.m_s1_warmstart is not None:
        #     for i, v in enumerate(x_list):
        #         v.setAttr('VarHintVal', self.m_s1_warmstart[i])
        m.optimize()
        if m.status in [2, 9, 13]:
            # print "  Solved S%d in %.3f s, status %d." % (stage, m.runtime, m.status)
            cost = m.getObjective().getValue()
            opt_vec = [v.x for v in x_list]
            if not mute:
                print "  S%d solved in %.3f s." % (stage, m.runtime)
            if stage == 1:
                # self.m_s1_warmstart = opt_vec
                cost_to_go = np.sum([1.0 * v.x for v in x_list
                                     if v.getAttr("VarName")[:4] == "epi_"])
                return cost, cost_to_go, opt_vec, m.runtime
            else:
                lambdas = [c.Pi for c in self.c_list_s2[0]]  # Station dynamics multipliers
                if not mute:
                    print "  lambda stats: # non-zero: %d/%d. Max: %.3f. Mean: %.3f" % \
                        (np.count_nonzero(lambdas), len(lambdas), np.max(lambdas), np.mean(lambdas))
                pen_created_bikes = self.cost_params['created bike cost'] * \
                                    np.sum([v.x for v in x_list if v.varname[:4] == 'crea'])
                pen_lost_bikes = self.cost_params['lost bike cost'] * \
                                 np.sum([v.x for v in x_list if v.varname[:4] == 'lost'])
                return cost - pen_created_bikes - pen_lost_bikes, opt_vec, lambdas, m.runtime
        elif m.status in [3, 4]:
            print "Optimization status for S%d: %d (%s)!" % (stage, m.status,
                                                             gu_status_codes[m.status])
            m.computeIIS()
            print "Constraints forming Irreducible Inconsistent Subsystem:"
            for c in m.getConstrs():
                if c.getAttr('IISConstr') != 0:
                    print c.ConstrName
                    self.print_non_zero_coeffs([c], m)
            for v in m.getVars():
                if v.iislb != 0:
                    print v.varname + " LB of " + str(v.lb)
                if v.iisub != 0:
                    print v.varname + " UB of " + str(v.ub)
        else:
            print "Optimization status for S%d: %d (%s)! Exiting." % (stage, m.status,
                                                                      gu_status_codes[m.status])
        raise SystemExit

    def integer_only(self):
        N, V, T, i = self.nw.nr_regions, self.nw.nr_vehicles, self.T, self.instance_no
        n_iter = self.alg_params['n_iter']
        self.results_df = pd.read_csv('output/stats_N%d_V%d_T%d_i%d' % (N, V, T, i) +
                                      self.results_label + '.csv')
        sol_method = self.alg_params['final_sol_method']
        final_ts1, final_ub, final_stats = self.integer_solve(method=sol_method, load_vf=True)
        df_entry = {'N': np.rint(N), 'V': np.rint(V), 'T': np.rint(T), 'Inst.': np.rint(i),
                    'k': n_iter+1, 'Integer': sol_method, 't1': np.round(final_ts1, 6),
                    'Cost': np.round(final_ub['cost'], 6),
                    'SR': np.round(final_stats['sr'] * 100, 6)}
        self.results_df = self.results_df.append(df_entry, ignore_index=True)
        self.results_df['N'] = self.results_df['N'].map(lambda x: "%d" % x)
        self.results_df['V'] = self.results_df['V'].map(lambda x: "%d" % x)
        self.results_df['T'] = self.results_df['T'].map(lambda x: "%d" % x)
        self.results_df['Inst.'] = self.results_df['Inst.'].map(lambda x: "%d" % x)
        self.results_df['k'] = self.results_df['k'].map(lambda x: "%d" % x)
        self.results_df.to_csv('output/stats_N%d_V%d_T%d_i%d' % (N, V, T, i) +
                               self.results_label + '.csv')

    def integer_solve(self, method=None, load_vf=False):
        """
        Generate a final integer solution using the value function approximation derived so far.
        """
        print "Generating final S1 solution using '" + method + "' method..."
        if load_vf:
            N, V, T, i = self.nw.nr_regions, self.nw.nr_vehicles, self.T, self.instance_no
            n_it = self.alg_params['n_iter']
            mat_fname = 'vfs/N%d_V%d_T%d_i%d_k%d' % (N, V, T, i, n_it) + self.results_label + '.mat'
            self.load_s1_cost_to_go(mat_fname)
        if method == 'greedy_seq':
            # Compute RV actions one by one rather than sequentially, to reduce computational cost

            # 1. Get RV list and initial locations from nw object
            # For each RV in the list:
            #     2. Fix model RH sides to allow optimization only over this RV
            #     3. Optimize
            #     4. Modify constraint RH sides to reflect redeployment actions and route chosen
            #     5. Unfix as appropriate to allow next RV decisions to be made.
            # 6. Recover final actions z,y and associated cost.

            print "Method 'greedy_seq' not yet implemented. Exiting."
            raise SystemExit()
        elif method == 'random':
            print "Method 'random' not yet implemented. Exiting."
            raise SystemExit()
        elif method == 'exact':
            # Jointly optimize all actions exactly
            self.unrelax_s1()
            s1_cost, cost_to_go, s1_sol, ts1 = self.optimize_stage(stage=1)
            print "  Exact S1 cost: %.3f" % (s1_cost - cost_to_go)
            # self.unrelax_s1(relax_or_unrelax='relax')  # 'Re-relax' model
        else:
            print "Unrecognised method for final S1 solution: " + method + ". Exiting."
            raise SystemExit()
        self.set_s2_constr_rhs(s1_sol, really_solved_s1=True)
        ub, stats = self.estimate_costs(s1_cost - cost_to_go, deterministic_s2=False)
        return ts1, ub, stats

    def update_vf(self, s1_sol_in, lambdas_in, k, verify=False):
        # Generate gradient from lambdas and S1 solution
        vf_old = self.vf[-1]
        xi_k = self.generate_xi(vf_old, s1_sol_in, lambdas_in)
        # Generate new VF based on step size rule
        alpha = self.step_size(k)
        vf_new = vf_old + alpha * xi_k
        self.vfpp.append(vf_new)
        # Project new VF onto feasible set (convex functions) and append to list.
        vf_new_p = self.project_vf(vf_new)
        self.vf.append(vf_new_p)
        # Update S1 model to reflect new VF
        vf_constrs = self.c_list_s1[5]
        max_dev = self.alg_params['max_dev']
        T, N, ni = self.T, self.nw.nr_regions, 2 * self.alg_params['max_dev']
        for it, n, l in itertools.product(range(T), range(N), range(ni)):
            t = it + 1
            ev_index = it * self.vstage_s1 + N + (N * N) + (N * N) + 2 * N + n  # Epigraph variable
            yp_index = it * self.vstage_s1 + N + (N * N) + (N * N) + 2 * n  # Unload bikes variable
            ym_index = it * self.vstage_s1 + N + (N * N) + (N * N) + 2 * n + 1  # Load bikes var
            # assert self.x_s1[ev_index].getAttr("VarName") == 'epi_%d_%d' % (n, t)
            vf_constr_index = it * N * ni + n * ni + l  # Index of epigraph constraint
            # assert vf_constrs[vf_constr_index].getAttr("ConstrName") == 'vf_%d_%d_%d' % \
            #     (t, n, l - self.alg_params['max_dev'])
            self.m_s1.chgCoeff(vf_constrs[vf_constr_index], self.x_s1[ym_index], vf_new_p[it, n, l])
            self.m_s1.chgCoeff(vf_constrs[vf_constr_index], self.x_s1[yp_index], -vf_new_p[it, n, l])
            vf_constrs[vf_constr_index].setAttr("RHS", -1 * (np.sum(vf_new_p[it, n, :l])
                                                             + (max_dev - l) * vf_new_p[it, n, l]))
        self.m_s1.update()
        if verify and divmod(k, 1)[1] == 0:
            n, t = 3, 1
            it = t - 1
            ev_index = it * self.vstage_s1 + N + (N * N) + (N * N) + 2 * N + n  # Epigraph variable
            yp_index = it * self.vstage_s1 + N + (N * N) + (N * N) + 2 * n  # Unload bikes variable
            ym_index = it * self.vstage_s1 + N + (N * N) + (N * N) + 2 * n + 1  # Load bikes var
            m, c = [], []  # List for storing lines in the form y >= mx + c
            assert self.x_s1[yp_index].getAttr("VarName") == 'y_%d_%d+' % (n, t)
            assert self.x_s1[ym_index].getAttr("VarName") == 'y_%d_%d-' % (n, t)
            for l in range(ni):
                vf_constr_index = it * N * ni + n * ni + l
                assert vf_constrs[vf_constr_index].getAttr("ConstrName") == 'vf_%d_%d_%d' % \
                    (it + 1, n, l - max_dev)
                assert self.m_s1.getCoeff(vf_constrs[vf_constr_index], self.x_s1[ev_index]) == -1.
                # self.print_non_zero_coeffs([vf_constrs[vf_constr_index]], self.m_s1)
                m.append(self.m_s1.getCoeff(vf_constrs[vf_constr_index], self.x_s1[ym_index]))
                c.append(-1 * vf_constrs[vf_constr_index].getAttr("RHS"))

            plot_range = range(-max_dev, max_dev + 1)
            plt.figure()
            plt.subplot(211)
            vf_stored = self.vf[-1][it, n, :]
            nx, vc = len(vf_stored), np.cumsum(vf_stored)
            plt.plot(plot_range, np.hstack((np.array([-vc[max_dev]]), vc - vc[max_dev])), 'b')
            plt.ylim([-2, 10])
            plt.subplot(212)
            for l in range(ni):
                plt.plot(plot_range, [m[l] * x + c[l] - c[max_dev] for x in plot_range])
            plt.ylim([-2, 10])
            plt.show()

    def load_s1_cost_to_go(self, fname):
        """
        Load cost-to-go data from file fname and update model epigraph variables for the cost-to-go
        to reflect this.
        :param fname: .mat file including relative location. Loads as dict, with key ['vf']
        :return: Nothing; updates S1 model in place.
        """
        vf_new_p = loadmat(fname)['vf']
        vf_constrs = self.c_list_s1[5]
        max_dev = self.alg_params['max_dev']
        T, N, ni = self.T, self.nw.nr_regions, 2 * self.alg_params['max_dev']
        for it, n, l in itertools.product(range(T), range(N), range(ni)):
            t = it + 1
            ev_index = it * self.vstage_s1 + N + (N * N) + (N * N) + 2 * N + n  # Epigraph variable
            yp_index = it * self.vstage_s1 + N + (N * N) + (N * N) + 2 * n  # Unload bikes variable
            ym_index = it * self.vstage_s1 + N + (N * N) + (N * N) + 2 * n + 1  # Load bikes var
            assert self.x_s1[ev_index].getAttr("VarName") == 'epi_%d_%d' % (n, t)
            vf_constr_index = it * N * ni + n * ni + l  # Index of epigraph constraint
            assert vf_constrs[vf_constr_index].getAttr("ConstrName") == 'vf_%d_%d_%d' % \
                (t, n, l - self.alg_params['max_dev'])
            self.m_s1.chgCoeff(vf_constrs[vf_constr_index], self.x_s1[ym_index], vf_new_p[it, n, l])
            self.m_s1.chgCoeff(vf_constrs[vf_constr_index], self.x_s1[yp_index],
                               -vf_new_p[it, n, l])
            vf_constrs[vf_constr_index].setAttr("RHS", -1 * (np.sum(vf_new_p[it, n, :l])
                                                             + (max_dev - l) * vf_new_p[it, n, l]))
        self.m_s1.update()

    def create_s1_model(self, cost_params, print_stats=True):
        N, V, K, T = self.nw.nr_regions, self.nw.nr_vehicles, self.nw.nr_lag_steps, self.T
        C_s, C_v = self.nw.C_s, self.nw.C_v
        assert C_v.tolist().count(C_v[0]) == len(C_v)  # All vehicles have the same capacity
        vmc = cost_params['vehicle movt cost']
        ypluscost, yminuscost = cost_params['load cost'], cost_params['unload cost']

        ds_0, max_dev = self.nw.ds_0, self.alg_params['max_dev']

        m = gu.Model("DRRP central flow")
        m.params.outputflag = 0 if N < 100 else 1
        m.params.mipgap = 5e-3
        m.params.threads = 2
        m.params.optimalitytol = 1e-3
        # m.params.mipgapabs = 5e-4
        m.params.timelimit = 1200
        m.params.method = 2
        m.params.crossover = 0
        x, vars_per_stage = [], 0

        t1 = time.time()
        # Define constraints in one-stage problem
        st_dyns = []  # Station fill level dynamics
        for t in range(1, T + 1):
            for n in range(N):
                st_dyns.append(m.addConstr(name='st_dyn_%d_%d' % (t, n),
                                           lhs=0, rhs=0, sense=gu.GRB.EQUAL))

        flow_dyns = []  # Bike-on-vehicle flow conservation
        b_flow_caps = []  # Bike-on-vehicle flow limits
        for t in range(1, T + 1):
            for i in range(N):
                flow_dyns.append(m.addConstr(name='flow_dyns_%d_%d' % (t, i),
                                             lhs=0, rhs=0, sense=gu.GRB.EQUAL))
                for j in range(N):
                    b_flow_caps.append(m.addConstr(name='flow_cap_%d_%d_%d' % (t, i, j),
                                                   lhs=0, rhs=0, sense=gu.GRB.LESS_EQUAL))

        z_dyns = []  # Truck movement consistency constraints
        # vehicle_flow_lbs = []  # Vehicle consistency (redundant)
        for t in range(1, T + 1):
            for n in range(N):
                z_dyns.append(m.addConstr(name='z_dyn_%d_%d' % (t, n), lhs=0, rhs=0,
                                          sense=gu.GRB.EQUAL))
            # vehicle_flow_lbs.append(m.addConstr(name='vfl_%d' % t, lhs=0, rhs=V,
            #                                     sense=gu.GRB.EQUAL))
        # Index of time t+1, station n: (tN + n)

        # Epigraph constraints for value function for load/unload sum for time t and station n
        vfs, lb, ub = [], -self.alg_params['max_dev'], self.alg_params['max_dev']
        for t in range(1, T + 1):
            for n in range(N):
                for l in range(lb, ub):
                    vfs.append(m.addConstr(name='vf_%d_%d_%d' % (t, n, l), lhs=0, rhs=0,
                                           sense=gu.GRB.LESS_EQUAL))
        n_vf = ub - lb  # Number of segments in each station's second stage VF

        m.update()

        if self.alg_params['relax_s1'] == "All z":
            t_relax_dict = {t: gu.GRB.CONTINUOUS for t in range(1, T + 1)}
        elif self.alg_params['relax_s1'] == "No z":
            t_relax_dict = {t: gu.GRB.INTEGER for t in range(1, T + 1)}
        elif not isinstance(self.alg_params['relax_s1'], bool) and \
                isinstance(self.alg_params['relax_s1'], int) and \
                self.alg_params['relax_s1'] <= T:
            t_relax_dict = {t: gu.GRB.INTEGER if t <= self.alg_params['relax_s1']
                            else gu.GRB.CONTINUOUS
                            for t in range(1, T + 1)}
        else:
            print "Cannot use self.alg_params['relax_s1'] value " + repr(self.alg_params['relax_s1'])
            print "Legal values: 'All z', 'No z', or integer 0 to T (indicating last integer step)."
            raise SystemExit()

        # Create variables, and add their coefficients to the constraints defined above
        for t in range(1, T + 1):
            it = t - 1
            for n in range(N):  # Station fill levels d_s
                x.append(m.addVar(name="ds_%d_%d" % (n, t),
                                  vtype=gu.GRB.CONTINUOUS,
                                  lb=ds_0[n] - t * max_dev, ub=ds_0[n] + t * max_dev, obj=0.))
                vars_per_stage += 1 if it == 0 else 0
                m.chgCoeff(st_dyns[it * N + n], x[-1], 1)
                if t < T:
                    m.chgCoeff(st_dyns[(it + 1) * N + n], x[-1], -1)

            for i in range(N):  # Bike flows on path from i to j
                for j in range(N):
                    x.append(m.addVar(name="dv_%d_%d_%d" % (i, j, t),
                                      vtype=gu.GRB.CONTINUOUS, lb=0, ub=np.sum(C_v), obj=0.))
                    vars_per_stage += 1 if it == 0 else 0
                    m.chgCoeff(flow_dyns[it * N + i], x[-1], 1)
                    m.chgCoeff(b_flow_caps[it * N * N + i * N + j], x[-1], 1)
                    if t < T:
                        m.chgCoeff(flow_dyns[(it + 1) * N + j], x[-1], -1)

            for i in range(N):  # Vehicle movements z
                for j in range(N):
                    z_obj_coeff = 0 if i == j else vmc
                    zlb = 0 if (t > 3 and i == j == 0) else 0
                    x.append(m.addVar(name="z_%d_%d_%d" % (i, j, t),
                                      vtype=t_relax_dict[t],
                                      obj=z_obj_coeff,
                                      lb=zlb, ub=self.nw.adjacency_matrix[i, j] * V))
                    # x[-1].setAttr('BranchPriority', T - it)  # Branch on earlier time steps first
                    vars_per_stage += 1 if it == 0 else 0
                    m.chgCoeff(z_dyns[it * N + i], x[-1], 1)
                    m.chgCoeff(b_flow_caps[it * N * N + i * N + j], x[-1], -C_v[0])
                    # m.chgCoeff(vehicle_flow_lbs[it], x[-1], 1)
                    if t < T:
                        m.chgCoeff(z_dyns[(it + 1) * N + j], x[-1], -1)

            for n in range(N):  # Load/unload actions y+/y-
                yplb = 0 if (n == 2 and t == 2) else 0
                x.append(m.addVar(name="y_%d_%d+" % (n, t),
                                  vtype=t_relax_dict[t],
                                  lb=yplb, ub=np.sum(C_v), obj=ypluscost))
                vars_per_stage += 1 if it == 0 else 0
                m.chgCoeff(st_dyns[it * N + n], x[-1], 1)  # Load from station...
                m.chgCoeff(flow_dyns[it * N + n], x[-1], -1)  # ... on to vehicle
                ymub = np.sum(C_v) if (n == 2 and t == 2) else np.sum(C_v)
                x.append(m.addVar(name="y_%d_%d-" % (n, t),
                                  vtype=t_relax_dict[t],
                                  lb=0, ub=ymub, obj=yminuscost))
                vars_per_stage += 1 if it == 0 else 0
                m.chgCoeff(st_dyns[it * N + n], x[-1], -1)  # Unload onto station...
                m.chgCoeff(flow_dyns[it * N + n], x[-1], 1)  # ... from vehicle

            for n in range(N):  # Epigraph variables for value of new configuration
                x.append(m.addVar(name='epi_%d_%d' % (n, t), vtype=gu.GRB.CONTINUOUS,
                                  lb=-gu.GRB.INFINITY, ub=gu.GRB.INFINITY, obj=1.))
                vars_per_stage += 1 if it == 0 else 0
                for l in range(n_vf):
                    m.chgCoeff(vfs[it * (N * n_vf) + n * n_vf + l], x[-1], -1)
                    # Constraint form: v - epi <= 0

        m.update()
        t2 = time.time()
        print "Stage 1 model created in %.3f s." % (t2 - t1)
        if self.alg_params['relax_s1'] == 'All z':
            print "  Using linear relaxation of S1 model."
        elif self.alg_params['relax_s1'] == 'No z':
            print "  Using full integer model for S1."
        elif isinstance(self.alg_params['relax_s1'], int):
            print "  Relaxing S1 z variables after t = %d." % self.alg_params['relax_s1']

        if print_stats:
            n_continuous = len([v for v in m.getVars() if v.vtype == gu.GRB.CONTINUOUS])
            n_binary = len([v for v in m.getVars() if v.vtype == gu.GRB.BINARY])
            n_integer = len([v for v in m.getVars() if v.vtype == gu.GRB.INTEGER])
            n_constraints = len(m.getConstrs())
            print "  %d vars (%d cont, %d binary, %d int), %d constraints." % \
                  (len(x), n_continuous, n_binary, n_integer, n_constraints)

        return m, [st_dyns, b_flow_caps, z_dyns, flow_dyns, None, vfs], x, vars_per_stage

    def create_s2_model(self, cost_params, print_stats=True):
        N, V, K, T = self.nw.nr_regions, self.nw.nr_vehicles, self.nw.nr_lag_steps, self.T
        C_s, C_v = self.nw.C_s, self.nw.C_v  # Station and vehicle capacities
        assert C_v.tolist().count(C_v[0]) == len(C_v)  # All vehicles have the same capacity
        ldc, vmc = cost_params['lost demand cost'], cost_params['vehicle movt cost']
        lost_bike_cost = cost_params['created bike cost']
        created_bike_cost = cost_params['created bike cost']
        ldc_spread_low = cost_params['lost demand cost spread low']
        ldc_spread_high = cost_params['lost demand cost spread high']

        m = gu.Model("Customer flow")
        m.params.outputflag = 0
        m.params.optimalitytol = 1e-3
        x = []

        t1 = time.time()
        # Define constraints in one-stage problem
        st_dyns = []  # Station fill level dynamics
        for t in range(1, T + 1):
            for n in range(N):
                st_dyns.append(m.addConstr(name='st_dyn_%d_%d' % (t, n),
                                           lhs=0, rhs=0, sense=gu.GRB.EQUAL))

        wd_constrs = []  # Constraints that limit customer movements to available capacity
        for t in range(1, T + 1):
            for i in range(N):
                for j in range(N):
                    for k in range(K + 1):
                        wd_constrs.append(m.addConstr(name='wd_%d_%d_%d_%d' % (t, i, j, k),
                                                      lhs=0, rhs=0, sense=gu.GRB.LESS_EQUAL))
        # Index of time t+1, origin i, destination j, lag k: (tN^2(K+1) + iN(K+1) + j(K + 1) + k)

        q_constrs = []  # Constraints that propagate queues of customers due to arrive at stations
        for t in range(1, T + 1):
            for n in range(N):
                for k in range(K):
                    q_constrs.append(m.addConstr(name='q_constr_%d_%d_%d' % (t, n, k + 1),
                                                 lhs=0, rhs=0, sense=gu.GRB.EQUAL))
        # Index of time t+1, station n, lag k: (tNK + nK + (k-1))

        zx_constrs = []  # Constraints that constrain previous state to be equal to its duplicate
        nzx = N + (N * K)
        for t in range(1, T + 1):
            for c in range(nzx):
                zx_constrs.append(m.addConstr(name='zx_constr_%d_%d' % (t, c), lhs=0, rhs=0,
                                              sense=gu.GRB.EQUAL))

        m.update()
        all_demand_lost_cost = 0.  # Keep a running total of cost if all demand were unsatisfied

        # Create variables, and add their coefficients to the constraints defined above
        idx = 0
        for t in range(1, T + 1):
            it = t - 1
            for n in range(N):  # Station fill levels d_s
                x.append(m.addVar(name="ds_%d_%d" % (n, t),
                                  vtype=gu.GRB.CONTINUOUS, lb=0, ub=C_s[n], obj=0.))
                m.chgCoeff(st_dyns[it * N + n], x[-1], 1)
                if t < T:
                    m.chgCoeff(zx_constrs[(it + 1) * nzx + n], x[-1], -1)
                idx += 1

            for i in range(N):
                for j in range(N):
                    for k in [0]:  # Customer actions w
                        dem = int(self.nw.F[t][k][i, j])
                        for d in range(dem):
                            value = np.random.uniform(low=ldc_spread_low * ldc,
                                                      high=ldc_spread_high * ldc)  # Trip value
                            x.append(m.addVar(name="w_%d_%d_%d_%d_%d" % (i, j, k, t, d),
                                              vtype=gu.GRB.CONTINUOUS, lb=0, ub=1, obj=-value))
                            all_demand_lost_cost += value  # Since coefficients are -ve
                            if i != j:
                                m.chgCoeff(st_dyns[it * N + i], x[-1], 1)
                                m.chgCoeff(st_dyns[it * N + j], x[-1], -1)
                            m.chgCoeff(wd_constrs[it * (N * N * (K + 1)) +
                                                  i * (N * (K + 1)) +
                                                  j * (K + 1) +
                                                  k], x[-1], 1)
                            self.w_indices.append(idx)
                            idx += 1
                        self.w_vars_added[it] += dem
                    for k in range(1, K + 1):  # Customer actions w
                        dem = int(self.nw.F[t][k][i, j])
                        for d in range(dem):
                            value = np.random.uniform(low=ldc_spread_low * ldc,
                                                      high=ldc_spread_high * ldc)  # Trip value
                            x.append(m.addVar(name="w_%d_%d_%d_%d_%d" % (i, j, k, t, d),
                                              vtype=gu.GRB.CONTINUOUS, lb=0, ub=1, obj=-value))
                            all_demand_lost_cost += value
                            m.chgCoeff(st_dyns[it * N + i], x[-1], 1)
                            m.chgCoeff(wd_constrs[it * (N * N * (K + 1)) +
                                                  i * (N * (K + 1)) + j * (K + 1) + k], x[-1], 1)
                            m.chgCoeff(q_constrs[it * N * K + j * K + (k - 1)], x[-1], -1)
                            self.w_indices.append(idx)
                            idx += 1
                        self.w_vars_added[it] += dem

            for n in range(N):  # Queues of customers due to arrive at stations, q
                for k in range(1, K + 1):
                    x.append(m.addVar(name='q_%d_%d_%d' % (n, k, t), vtype=gu.GRB.CONTINUOUS, lb=0,
                                      ub=self.nw.nr_bikes, obj=0.))
                    m.chgCoeff(q_constrs[it * N * K + n * K + (k - 1)], x[-1], 1)
                    if t < T:
                        m.chgCoeff(zx_constrs[(it + 1) * nzx + N + n * K + (k - 1)], x[-1], -1)
                    idx += 1

            for n in range(N):
                x.append(m.addVar(name='lost_bike_%d_%d' % (n, t),
                                  vtype=gu.GRB.CONTINUOUS, obj=lost_bike_cost, lb=0.))
                m.chgCoeff(st_dyns[it * N + n], x[-1], 1)
                idx += 1
            for n in range(N):
                x.append(m.addVar(name='created_bike_%d_%d' % (n, t),
                                  vtype=gu.GRB.CONTINUOUS, obj=created_bike_cost, lb=0.))
                m.chgCoeff(st_dyns[it * N + n], x[-1], -1)
                idx += 1

            # Variables for duplicating the previous state
            for c in range(N):  # Duplicates of previous ds variables
                i = c
                x.append(m.addVar(name='zx_ds_%d_%d' % (i, t),
                                  vtype=gu.GRB.CONTINUOUS, lb=0, ub=C_s[i], obj=0.))
                m.chgCoeff(zx_constrs[it * nzx + c], x[-1], 1)
                m.chgCoeff(st_dyns[it * N + i], x[-1], -1)
                idx += 1
                den = np.sum([np.sum(self.nw.F[t][k][i, :]) for k in range(K + 1)])
                if den > 0:
                    for k in range(K + 1):
                        for j in range(N):
                            ds_coeff = float(self.nw.F[t][k][i, j]) / den
                            m.chgCoeff(wd_constrs[it * N * N * (K + 1) +
                                                  i * (N * (K + 1)) + j * (K + 1) + k],
                                       x[-1], -ds_coeff)

            for c in range(N, N + (N * K)):  # Duplicates of previous q vars
                posn = c - N
                n, k = divmod(posn, K)
                x.append(m.addVar(name='zx_q_%d_%d_%d' % (n, k, t),
                                  vtype=gu.GRB.CONTINUOUS, lb=0, ub=self.nw.nr_bikes, obj=0.))
                m.chgCoeff(zx_constrs[it * nzx + c], x[-1], 1)
                if k == 0:
                    m.chgCoeff(st_dyns[it * N + n], x[-1], -1)
                else:
                    m.chgCoeff(q_constrs[it * N * K + n * K + (k - 1)], x[-1], -1)
                idx += 1

        # Constant offset to objective function
        m.setAttr('ObjCon', all_demand_lost_cost)
        # Remove all wd_constrs entries (which enforce proportional sharing of demand)
        for c in wd_constrs:
            m.remove(c)
        m.update()
        t2 = time.time()

        print "Stage 2 model created in %.3f s." % (t2 - t1)
        if print_stats:
            n_continuous = len([v for v in m.getVars() if v.vtype == gu.GRB.CONTINUOUS])
            n_binary = len([v for v in m.getVars() if v.vtype == gu.GRB.BINARY])
            n_integer = len([v for v in m.getVars() if v.vtype == gu.GRB.INTEGER])
            n_constraints = len(m.getConstrs())
            print "  %d vars (%d cont, %d binary, %d int), %d constraints." % \
                  (len(x), n_continuous, n_binary, n_integer, n_constraints)
            print "  Initialized the model with a demand of %d customers." % len(self.w_indices)

        return m, [st_dyns, wd_constrs, q_constrs, zx_constrs], x

    def resample_s2_costs(self):
        cost_params = self.cost_params
        ldc = cost_params['lost demand cost']
        ldc_spread_low = cost_params['lost demand cost spread low']
        ldc_spread_high = cost_params['lost demand cost spread high']
        print "  There are %d w-variables in the model." % len(self.w_indices)
        for idx in self.w_indices:
            value = np.random.uniform(low=ldc_spread_low * ldc,
                                      high=ldc_spread_high * ldc)  # Trip value
            # assert self.x_s2[idx].getAttr('VarName')[:2] == "w_"  # Check we're looking at a w var
            self.x_s2[idx].setAttr('obj', -value)
        self.m_s2.update()

    def resample_s2_demand(self, deterministic=False, mute=False):
        N, V, K, T = self.nw.nr_regions, self.nw.nr_vehicles, self.nw.nr_lag_steps, self.T
        cost_params = self.cost_params
        ldc = cost_params['lost demand cost']
        ldc_spread_low = cost_params['lost demand cost spread low']
        ldc_spread_high = cost_params['lost demand cost spread high']
        t1 = time.time()
        for idx in self.w_indices[::-1]:  # Step back through list deleting x from model and list
            # assert self.x_s2[idx].getAttr('VarName')[:2] == "w_", self.x_s2[idx].getAttr('VarName')  # Check we're looking at a w var
            self.m_s2.remove(self.x_s2[idx])
            del self.x_s2[idx]
        self.m_s2.update()

        self.w_indices, idx = [], len(self.m_s2.getVars())
        self.w_vars_added = [0] * self.T
        all_demand_lost_cost = 0
        m, x = self.m_s2, self.x_s2,
        st_dyns, wd_constrs, q_constrs = self.c_list_s2[0], self.c_list_s2[1], self.c_list_s2[2]
        # Re-generate new demand
        for t in range(1, T + 1):
            it = t - 1
            for i in range(N):
                for j in range(N):
                    for k in [0]:  # Customer actions w
                        if deterministic:
                            dem = int(np.rint(self.nw.F[t][k][i, j]))
                        else:
                            dem = np.random.poisson(self.nw.F[t][k][i, j])
                        for d in range(dem):
                            if deterministic:
                                value = 0.5 * (ldc_spread_low + ldc_spread_high)
                            else:
                                value = np.random.uniform(low=ldc_spread_low * ldc,
                                                          high=ldc_spread_high * ldc)  # Trip value
                            x.append(m.addVar(name="w_%d_%d_%d_%d_%d" % (i, j, k, t, d),
                                              vtype=gu.GRB.CONTINUOUS, lb=0, ub=1, obj=-value))
                            all_demand_lost_cost += value  # Since coefficients are -ve
                            if i != j:
                                m.chgCoeff(st_dyns[it * N + i], x[-1], 1)
                                m.chgCoeff(st_dyns[it * N + j], x[-1], -1)
                            # m.chgCoeff(wd_constrs[it * (N * N * (K + 1)) +
                            #                       i * (N * (K + 1)) +
                            #                       j * (K + 1) +
                            #                       k], x[-1], 1)
                            self.w_indices.append(idx)
                            idx += 1
                        self.w_vars_added[it] += dem
                    for k in range(1, K + 1):  # Customer actions w
                        if deterministic:
                            dem = int(np.rint(self.nw.F[t][k][i, j]))
                        else:
                            dem = np.random.poisson(self.nw.F[t][k][i, j])
                        for d in range(dem):
                            value = np.random.uniform(low=ldc_spread_low * ldc,
                                                      high=ldc_spread_high * ldc)  # Trip value
                            x.append(m.addVar(name="w_%d_%d_%d_%d_%d" % (i, j, k, t, d),
                                              vtype=gu.GRB.CONTINUOUS, lb=0, ub=1, obj=-value))
                            all_demand_lost_cost += value
                            m.chgCoeff(st_dyns[it * N + i], x[-1], 1)
                            # m.chgCoeff(wd_constrs[it * (N * N * (K + 1)) +
                            #                       i * (N * (K + 1)) + j * (K + 1) + k], x[-1], 1)
                            m.chgCoeff(q_constrs[it * N * K + j * K + (k - 1)], x[-1], -1)
                            self.w_indices.append(idx)
                            idx += 1
                        self.w_vars_added[it] += dem
        m.setAttr('ObjCon', all_demand_lost_cost)
        m.update()
        t2 = time.time()
        if not mute:
            print "  Resampled S2 demand in %.3f s: new demand is %d customers" % \
                  (t2 - t1, len(self.w_indices))

    def eval_no_action_cost(self):
        N, V, T, i = self.nw.nr_regions, self.nw.nr_vehicles, self.T, self.instance_no
        self.set_s2_constr_rhs(s1_sol_vec=None, mute=True)
        self.ub_no_action, self.stats_no_action = self.estimate_costs(first_stage_cost=0)
        print "Mean cost with no action is %.3f; mean service rate is %.1f%%." % \
              (self.ub_no_action['cost'], self.stats_no_action['sr'] * 100)
        df_entry = {'N': np.rint(N), 'V': np.rint(V), 'T': np.rint(T), 'Inst.': np.rint(i), 'k': 0,
                    'Cost': np.round(self.ub_no_action['cost'], 6),
                    'SR': np.round(self.stats_no_action['sr'] * 100, 6)}
        self.results_df = self.results_df.append(df_entry, ignore_index=True)
        self.results_df.to_csv('output/stats_N%d_V%d_T%d_i%d' % (N, V, T, i) +
                               self.results_label + '.csv')

    def estimate_costs(self, first_stage_cost, deterministic_s2=False):
        print "  Estimating S2 cost..."
        t1 = time.time()
        n_samples = self.alg_params['cost_eval_samples']
        cost_list = np.zeros((n_samples,), dtype=float)
        f_sum_list = np.zeros((n_samples,), dtype=float)
        uf_sum_list = np.zeros((n_samples,), dtype=float)
        f_value_list = np.zeros((n_samples,), dtype=float)
        uf_value_list = np.zeros((n_samples,), dtype=float)
        for i in range(n_samples):
            # Measure cost and solutions stats for each of the samples
            # np.random.seed(i)
            self.resample_s2_demand(deterministic=deterministic_s2, mute=False)
            cost_i, _, _, _ = self.optimize_stage(stage=2, mute=True)
            cost_list[i] = cost_i + first_stage_cost
            stats = self.s2_solution_stats()
            f_sum_list[i] = stats['Fulfilled demand']
            uf_sum_list[i] = stats['Unfulfilled demand']
            f_value_list[i] = stats['Fulfilled value']
            uf_value_list[i] = stats['Unfulfilled value']
        t2 = time.time()
        # Build dictionaries summarising cost and solution stats
        ub = {'cost': np.mean(cost_list), 'sd': np.sqrt(np.var(cost_list)),
              '5pc': np.percentile(cost_list, 5), '95pc': np.percentile(cost_list, 95)}
        stats_out = {'f_sum': np.sum(f_sum_list), 'uf_sum': np.sum(uf_sum_list),
                     'sr': np.sum(f_sum_list) / (np.sum(f_sum_list) + np.sum(uf_sum_list)),
                     'time': t2-t1}

        print "  MC evaluation of S2 cost (%d samples) took %.3f s." % (n_samples, t2 - t1)
        print "    Mean service rate: %.1f%%" % (stats_out['sr'] * 100.)

        return ub, stats_out

    def set_s1_initial_state(self):
        N, V, K = self.nw.nr_regions, self.nw.nr_vehicles, self.nw.nr_lag_steps

        vehicle_flow_mat = np.zeros((N, N), dtype=int)
        bike_on_vehicle_flow_mat = np.zeros((N, N), dtype=int)

        for loc, z0_val in enumerate(self.nw.z_0[:N * V]):
            v_location, vehicle_id = divmod(loc, V)
            vehicle_flow_mat[v_location, v_location] += z0_val
            bike_on_vehicle_flow_mat[v_location, v_location] += self.nw.dv_0[vehicle_id]

        # x0 = []
        # for n in range(N):  # ds
        #     x0.append(np.float(self.nw.ds_0[n]))  # Initial fill level of stations
        # for i in range(N):  # dv
        #     for j in range(N):
        #         x0.append(bike_on_vehicle_flow_mat[i, j])  # Bikes currently being transported
        # for i in range(N):
        #     for j in range(N):
        #         x0.append(vehicle_flow_mat[i, j])  # Number of vehicles just arrived from i to j

        #  Set RHS of constraint equating the duplicate x_t-1 (xz in the optimization vector) to x0
        m = self.m_s1
        st_dyns = self.c_list_s1[0]
        for n in range(N):
            st_dyns[n].setAttr('RHS', np.rint(self.nw.ds_0[n]))
        z_dyns = self.c_list_s1[2]
        for j in range(N):
            flow_to_j = 0
            for i in range(N):
                flow_to_j += vehicle_flow_mat[i, j]
            z_dyns[j].setAttr('RHS', flow_to_j)
        flow_dyns = self.c_list_s1[3]
        for j in range(N):
            flow_to_j = 0
            for i in range(N):
                flow_to_j += bike_on_vehicle_flow_mat[i, j]
            flow_dyns[j].setAttr('RHS', flow_to_j)

        # nzx = N + (N * N) + (N * N)  # ds, dv, z
        # zx_constrs = self.c_list_s1[4]
        # for c in range(nzx):
        #     if np.abs(np.rint(x0[c]) - x0[c]) > 1e-4:  # Check for non-integer x0
        #         print "x_l[%d]:" % c, x0[c]
        #         raise SystemExit()
        #     zx_constrs[c].setAttr('RHS', np.rint(x0[c]))
        m.update()

    def set_s2_constr_rhs(self, s1_sol_vec=None, mute=True, really_solved_s1=False):
        """Set RHS of constraints in S2 model to reflect (i) initial state of system, and (ii) the
        effect of S1 y decisions. Note that for (i) this only updates constraints to reflect initial
        fill level of stations, and not queues of customers en route to stations (variables q).

        :param s1_sol_vec: Stage 1 decision vector
        :param mute: Hide printed description of the effect of the Stage 1 y-decisions
        :param really_solved_s1: True if S1 sol'n was really optimized and not just randomly chosen
        :return: nothing (updates model in place)
        """
        T, N, V, K = self.T, self.nw.nr_regions, self.nw.nr_vehicles, self.nw.nr_lag_steps
        m = self.m_s2
        st_dyn_constrs = self.c_list_s2[0]
        if s1_sol_vec is not None:
            if self.alg_params['random_s1'] and not really_solved_s1:
                n_x1_per_t = None
            else:
                assert divmod(len(s1_sol_vec), T)[1] == 0  # Length of opt vec should be a mult of T
                n_x1_per_t = len(s1_sol_vec) / T
                assert n_x1_per_t == self.vstage_s1
            for t in range(1, T + 1):
                it = t - 1
                for n in range(N):
                    if self.alg_params['random_s1'] and not really_solved_s1:
                        delta_y = s1_sol_vec[n, it]  # s1_sol_vec is a N*T matrix in this case
                    else:
                        yplus_index = it * n_x1_per_t + (N + 2 * N * N) + (2 * n)
                        yminus_index = it * n_x1_per_t + (N + 2 * N * N) + (2 * n + 1)
                        delta_y = s1_sol_vec[yminus_index] - s1_sol_vec[yplus_index]
                    if delta_y != 0 and not mute:
                        print "    Delta y at t = %d, n = %d: %d" % (t, n, delta_y)
                    if t > 1:  # Update to account for S1 decision to deposit/remove bikes
                        st_dyn_constrs[it * N + n].setAttr("RHS", delta_y)
                    else:  # Set initial condition accounting for initial number of bikes present
                        st_dyn_constrs[n].setAttr("RHS", delta_y + self.nw.ds_0[n])
        else:  # No information about depositing or removing bikes, so just set initial fill level
            for t in range(1, T + 1):
                it = t - 1
                for n in range(N):
                    if t > 1:  # Update to account for S1 decision to deposit/remove bikes
                        st_dyn_constrs[it * N + n].setAttr("RHS", 0)
                    else:  # Set initial condition accounting for initial number of bikes present
                        st_dyn_constrs[n].setAttr("RHS", 0 + self.nw.ds_0[n])
        m.update()

    def step_size(self, iteration):
        if self.alg_params['ss_rule'] == '1/k':
            return min(1.0, self.alg_params['1k_const'] / iteration)
        elif self.alg_params['ss_rule'] == 'const':
            return self.alg_params['ss_const']
        elif self.alg_params['ss_rule'] == 'PRT':  # As suggested by Powell-Ruszczynksi-Topaloglu.
            return 20.0 / (40.0 + iteration)
        else:
            print "Unrecognised step size rule: " + self.alg_params['ss_rule']
            raise SystemExit()

    def project_vf(self, vf_in):
        assert len(vf_in.shape) == 3, "Value function array is %d-dimensional!" % len(vf_in.shape)
        (nt, nr, nx) = vf_in.shape
        n_projs = nt * nr  # Number of value functions to project
        t1 = time.time()
        solver_time = 0.
        vf_out = np.zeros((nt, nr, nx), dtype=float)
        for t in range(nt):
            for r in range(nr):
                vf_out[t, r, :], runtime = self.unit_vf_projection(vf_in[t, r, :])
                solver_time += runtime
        t2 = time.time()
        # print "  Took %.3f s for %d VF projections (mean: %.1f us of which %.1f us solving)." % \
        #       (t2 - t1, n_projs, (t2 - t1)/n_projs * 1e6, solver_time/n_projs * 1e6)
        return vf_out

    def unit_vf_projection(self, v_in):
        m = self.vf_proj_m
        for i, zi in enumerate(v_in):
            self.vf_proj_x[i].setAttr("Obj", -zi)
        m.optimize()
        if m.status in [2, 9, 11, 12]:
            # print "Solved in %.1f us, status %d" % (m.runtime * 1e6, m.status)
            if m.status == 12:
                print "Warning: Found numerical issues (status 12)"
            v_proj = [v.X for v in self.vf_proj_x]
            return v_proj, m.runtime
        else:
            print "Projection failed! Status: %d" % m.status
            raise SystemExit()

    def setup_unit_projection_model(self, concave=False, max_slope=gu.GRB.INFINITY):
        m = gu.Model("Projection")
        m.params.outputflag = 0
        m.params.optimalitytol = 1e-6

        x = []
        qe = gu.QuadExpr()
        constr_sense = gu.GRB.LESS_EQUAL if concave else gu.GRB.GREATER_EQUAL
        for i in range(-self.alg_params['max_dev'], self.alg_params['max_dev']):
            x.append(m.addVar(name="v_%d" % i, lb=-max_slope, ub=max_slope,
                              vtype=gu.GRB.CONTINUOUS, obj=0.))
            m.update()
            qe += 0.5 * x[-1] * x[-1]
            if i > -self.alg_params['max_dev']:
                m.addConstr(x[-1] - x[-2], constr_sense, 0)
        m.setObjective(qe)
        return m, x

    def generate_xi(self, vf_old, s1_x, lambdas):
        """Generate gradient direction based on lagrange multipliers of second stage problem

        :param vf_old: Old VF
        :param s1_x: S1 decision vector
        :param lambdas: Vector of Lagrange multipliers
        :return: xi
        """
        T, N, max_dev = self.T, self.nw.nr_regions, self.alg_params['max_dev']
        xi = np.zeros((T, N, 2 * max_dev))

        if self.alg_params['random_s1']:  # Random S1 decision
            for it, n in itertools.product(range(T), range(N)):
                grad_posn = int(s1_x[n, it]) + max_dev
                if 0 <= grad_posn <= 2 * max_dev - 1:
                    xi[it, n, grad_posn] = lambdas[it * N + n] - vf_old[it, n, grad_posn]
                else:
                    print "t = %d, n = %d: Wanted to modify gradient at pos'n %d (out of bounds)!" \
                        % (it + 1, n, grad_posn)
        else:  # Optimization-based solution to S1
            nx1_per_t = len(s1_x) / T
            assert nx1_per_t == self.vstage_s1
            for it, n in itertools.product(range(T), range(N)):
                # Fill level implied by S1 solution, which doesn't account for customer actions
                y_minus_nt = s1_x[it * nx1_per_t + N + (N * N) + (N * N) + 2 * n + 1]
                y_plus_nt = s1_x[it * nx1_per_t + N + (N * N) + (N * N) + 2 * n]
                # Determine at which integer breakpoint to modify VF approximation for this t, n
                grad_posn = int(y_minus_nt - y_plus_nt) + max_dev
                if 0 <= grad_posn <= 2 * max_dev - 1:
                    xi[it, n, grad_posn] = lambdas[it * N + n] - vf_old[it, n, grad_posn]
                else:
                    print "t = %d, n = %d: Wanted to modify gradient at pos'n %d (out of bounds)!" \
                        % (it + 1, n, grad_posn)
        return xi

    def print_s1_solution(self, opt_vec):
        N, V, K, T = self.nw.nr_regions, self.nw.nr_vehicles, self.nw.nr_lag_steps, self.T

        opt_vec_series = [opt_vec[(t * self.vstage_s1):(t+1) * self.vstage_s1] for t in range(T)]
        assert np.sum(len(ov) for ov in opt_vec_series) == len(opt_vec)

        def all_integer_valued(list_in):
            for v in list_in:
                if np.abs(v - np.rint(v)) > 1e-4:
                    return False
            return True

        print "Solution stats:"

        # Load/unload actions
        loaded_or_unloaded_something = False
        y_index = N + (N * N) + (N * N)
        for it, vec in enumerate(opt_vec_series):
            t = it + 1
            for n in range(N):
                yplus, yminus = vec[y_index + 2 * n], vec[y_index + 2 * n + 1]
                if yplus > 0:
                    if not loaded_or_unloaded_something:
                        print "  Bike load/unload actions:"
                    loaded_or_unloaded_something = True
                    print "    t = %d, loaded %d bikes onto vehicles at station %d" % \
                          (t, yplus - yminus, n)
                elif yminus > 0:
                    if not loaded_or_unloaded_something:
                        print "  Bike load/unload actions:"
                    loaded_or_unloaded_something = True
                    print "    t = %d, unloaded %d bikes from vehicles at station %d" % \
                          (t, yminus - yplus, n)
        if not loaded_or_unloaded_something:
            print "  No bikes loaded or unloaded from vehicles."

        # Station fill levels: same for both 'flow' and 'vehicles' model form.
        if loaded_or_unloaded_something:
            print "  Station fill levels before accounting for customer movements:"
            station_fill_trajs = [[] for _ in range(N)]
            for n in range(N):
                for vec in opt_vec_series:
                    station_fill_trajs[n].append(vec[n])
                print "    Station %d:" % n, [int(self.nw.ds_0[n])], \
                    [int(a) for a in station_fill_trajs[n]]

        # Vehicle flows:
        z_index = N + (N * N)
        a_vehicle_moved = False
        for it, vec in enumerate(opt_vec_series):
            t = it + 1
            for i in range(N):
                for j in range(N):
                    flow = vec[z_index + i * N + j]
                    if flow > 0 and all_integer_valued([flow]):
                        if i == j:
                            pass
                            # print "    t = %d, staying at %d: %d" % (t, i, flow)
                        else:
                            if not a_vehicle_moved:
                                print "  Vehicle movements:"
                            a_vehicle_moved = True
                            print "    t = %d, %d to %d: %d" % (t, i, j, flow)
                    elif flow > 0:
                        if i == j:
                            pass
                            # print "    t = %d, staying at %d: %.2f" % (t, i, flow)
                        else:
                            if not a_vehicle_moved:
                                print "  Vehicle movements:"
                            a_vehicle_moved = True
                            print "    t = %d, %d to %d: %.2f" % (t, i, j, flow)
        if not a_vehicle_moved:
            print "  No vehicle movements."

        # Bike flows
        if loaded_or_unloaded_something:
            dv_index = N
            print "  Nonzero bike flows:"
            for it, vec in enumerate(opt_vec_series):
                t = it + 1
                for i in range(N):
                    for j in range(N):
                        flow = vec[dv_index + i * N + j]
                        if flow > 1e-4 and all_integer_valued([flow]):
                            if i == j:
                                print "    t = %d, staying at %d: %d" % (t, i, flow)
                            else:
                                print "    t = %d, %d to %d: %d" % (t, i, j, flow)
                        elif flow > 1e-4:
                            if i == j:
                                print "    t = %d, staying at %d: %.2f" % (t, i, flow)
                            else:
                                print "    t = %d, %d to %d: %.2f" % (t, i, j, flow)

    def print_s2_solution(self, opt_vec_series):
        N, V, K = self.nw.nr_regions, self.nw.nr_vehicles, self.nw.nr_lag_steps

        assert len(opt_vec_series) == self.T

        def all_integer_valued(list_in):
            for v in list_in:
                if np.abs(v - np.rint(v)) > 1e-4:
                    return False
            return True

        print "Solution stats, '" + self.model_form + "' form:"

        # Unfulfilled customer demand: same for 'flow' and 'vehicles' except for vector index.
        print "  Fulfilled demand:"
        w_index = N + (N * N) if self.model_form == 'flow' else N + V
        f_sum = 0
        for it, vec in enumerate(opt_vec_series):
            for i in range(N):
                for j in range(N):
                    for k in range(K + 1):
                        t = it + 1
                        w = vec[w_index + i * (N * (K + 1)) + j * (K + 1) + k]
                        f = self.nw.F[t][k][i, j]
                        if np.abs(f - w) <= 1e-5 and w >= 1e-5:
                            # if all_integer_valued([f, w]):
                            #     print "    w=%d at t = %d, %d -> %d (k = %d)" % (w, t, i, j, k)
                            # else:
                            #     print "    w=%.2f at t = %d, %d -> %d (k = %d)" % (w, t, i, j, k)
                            f_sum += w
        print "    Total:", f_sum
        print "  Lost demand:"
        uf_sum = 0
        for it, vec in enumerate(opt_vec_series):
            for i in range(N):
                for j in range(N):
                    for k in range(K + 1):
                        t = it + 1
                        w = vec[w_index + i * (N * (K + 1)) + j * (K + 1) + k]
                        f = self.nw.F[t][k][i, j]
                        if np.abs(f - w) > 1e-5:
                            if all_integer_valued([f, w]):
                                print "    %d (f=%d, w=%d) at t = %d, %d -> %d (k = %d)" % \
                                      (f - w, f, w, t, i, j, k)
                            else:
                                print "    %.2f (f=%.2f, w=%.2f) at t = %d, %d -> %d (k = %d)" % \
                                      (f - w, f, w, t, i, j, k)
                            uf_sum += f - w
        print "    Total:", uf_sum

        # Station fill levels: same for both 'flow' and 'vehicles' model form.
        print "  Station fill levels:"
        station_fill_trajs = [[] for _ in range(N)]
        for n in range(N):
            for vec in opt_vec_series:
                station_fill_trajs[n].append(vec[n])
            print "    Station %d:" % n, [int(self.x0[n])], \
                [int(a) for a in station_fill_trajs[n]]

        # Vehicle flows:
        print "  Vehicle movements:"
        z_index = N + (N * N) + (K+1) * N*N if self.model_form == 'flow' else N + V + (K+1) * N*N
        for it, vec in enumerate(opt_vec_series):
            t = it + 1
            if self.model_form == 'flow':
                for i in range(N):
                    for j in range(N):
                        flow = vec[z_index + i * N + j]
                        if flow > 0 and all_integer_valued([flow]):
                            if i == j:
                                print "    t = %d, staying at %d: %d" % (t, i, flow)
                            else:
                                print "    t = %d, %d to %d: %d" % (t, i, j, flow)
                        elif flow > 0:
                            if i == j:
                                print "    t = %d, staying at %d: %.2f" % (t, i, flow)
                            else:
                                print "    t = %d, %d to %d: %.2f" % (t, i, j, flow)
            else:
                for v in range(V):
                    for i in range(N):
                        for j in range(N):
                            flow = vec[z_index + i * N * V + j * V + v]
                            if flow > 0 and all_integer_valued([flow]):
                                if i == j:
                                    print "    v%d, t = %d, staying at %d: %d" % (v, t, i, flow)
                                else:
                                    print "    v%d, t = %d, %d to %d: %d" % (v, t, i, j, flow)
                            elif flow > 0:
                                if i == j:
                                    print "    v%d, t = %d, staying at %d: %.2f" % (v, t, i, flow)
                                else:
                                    print "    v%d, t = %d, %d to %d: %.2f" % (v, t, i, j, flow)
        # Load/unload actions
        if self.model_form == 'flow':
            y_index = N + N * N + N * N * (K + 1) + N * N
            print "  Nonzero load/unload actions:"
            for it, vec in enumerate(opt_vec_series):
                t = it + 1
                for n in range(N):
                    yplus, yminus = vec[y_index + 2 * n], vec[y_index + 2 * n + 1]
                    if yplus > 0:
                        print "    t = %d, loaded %d bikes onto vehicles at station %d" % \
                              (t, yplus - yminus, n)
                    elif yminus > 0:
                        print "    t = %d, unloaded %d bikes from vehicles at station %d" % \
                              (t, yminus - yplus, n)
        else:
            y_index = N + V + N * N * (K + 1) + N * N * V
            print "  Nonzero load/unload actions:"
            for it, vec in enumerate(opt_vec_series):
                t = it + 1
                for v in range(V):
                    for n in range(N):
                        yplus = vec[y_index + 2 * n * V + 2 * v]
                        yminus = vec[y_index + 2 * n * V + 2 * v + 1]
                        if yplus > 0:
                            print "    t = %d, v%d loaded %d bikes onto vehicles at station %d" % \
                                  (t, v, yplus - yminus, n)
                        elif yminus > 0:
                            print "    t = %d, v%d unloaded %d bikes from vehicles at station %d" % \
                                  (t, v, yminus - yplus, n)

        # Passenger queues
        q_index = N + V + N * N * (K + 1) + N * N * V + 2 * N * V \
            if self.model_form == 'vehicles' else N + N * N + N * N * (K + 1) + N * N + 2 * N
        q_trajs = [[[] for _ in range(N)] for k in range(K)]
        for it, vec in enumerate(opt_vec_series):
            t = it + 1
            for n in range(N):
                for k in range(K):
                    q_trajs[k][n].append(vec[q_index + n * K + k])
        for k in range(1, K + 1):
            print "  Station queues, k = %d:" % k
            for n in range(N):
                if all_integer_valued(q_trajs[k-1][n]):
                    print "    Station %d:" % n, [int(v) for v in q_trajs[k-1][n]]
                else:
                    print "    Station %d:" % n, q_trajs[k-1][n]

        # Bike and vehicle flows
        dv_index = N
        if self.model_form == 'flow':
            print "  Nonzero bike flows:"
            for it, vec in enumerate(opt_vec_series):
                t = it + 1
                for i in range(N):
                    for j in range(N):
                        flow = vec[dv_index + i * N + j]
                        if flow > 1e-4 and all_integer_valued([flow]):
                            if i == j:
                                print "    t = %d, staying at %d: %d" % (t, i, flow)
                            else:
                                print "    t = %d, %d to %d: %d" % (t, i, j, flow)
                        elif flow > 1e-4:
                            if i == j:
                                print "    t = %d, staying at %d: %.2f" % (t, i, flow)
                            else:
                                print "    t = %d, %d to %d: %.2f" % (t, i, j, flow)
        else:
            dv_trajs = [[] for v in range(V)]
            for it, vec in enumerate(opt_vec_series):
                t = it + 1
                for v in range(V):
                    dv_trajs[v].append(vec[dv_index + v])
            print "  Bikes carried by vehicles:"
            for v in range(V):
                if all_integer_valued(dv_trajs[v]):
                    print "    v%d:" % v, [int(x) for x in dv_trajs[v]]
                else:
                    print "    v%d:" % v, dv_trajs[v]

    def s2_solution_stats(self):
        opt_vec = self.x_s2
        stats_dict = {}

        # Unfulfilled customer demand
        total_demand = len(self.w_indices)
        f_sum = np.sum([opt_vec[idx].X for idx in self.w_indices])
        uf_sum = total_demand - f_sum
        f_value = np.sum([opt_vec[idx].obj for idx in self.w_indices if opt_vec[idx].X == 1])
        uf_value = np.sum([opt_vec[idx].obj for idx in self.w_indices if opt_vec[idx].X == 0])
        stats_dict['Fulfilled demand'], stats_dict['Unfulfilled demand'] = f_sum, uf_sum
        stats_dict['Service rate'] = 100.0 * float(f_sum) / float(f_sum + uf_sum)
        stats_dict['Fulfilled value'], stats_dict['Unfulfilled value'] = f_value, uf_value

        return stats_dict

    def plot_cost_estimates(self, suffix=''):
        lb, ub, stats = self.lb, self.ub, self.stats
        lbx, lby = [], []
        for (k, v) in lb.iteritems():
            lbx.append(int(k))
            lby.append(float(v))
        ubx, uby, ub5pc, ub95pc = [], [], [], []
        for (k, v) in ub.iteritems():
            ubx.append(int(k))
            uby.append(float(v['cost']))
            ub5pc.append(float(v['5pc']))
            ub95pc.append(float(v['95pc']))
        lbx, lby, ubx, uby = np.array(lbx), np.array(lby), np.array(ubx), np.array(uby)
        ub5pc, ub95pc = np.array(ub5pc), np.array(ub95pc)
        lbx_order = np.argsort(lbx)
        lbx, lby = lbx[lbx_order], lby[lbx_order]
        ubx_order = np.argsort(ubx)
        ubx, uby, ub5pc, ub95pc = ubx[ubx_order], uby[ubx_order], ub5pc[ubx_order], ub95pc[ubx_order]
        plt.figure()
        plt.subplot(211)
        # plt.plot(lbx, lby, 'r')
        plt.plot(ubx, uby, 'b')
        plt.plot(ubx, ub5pc, 'b--')
        plt.plot(ubx, ub95pc, 'b--')
        if self.ub_no_action is not None:
            ubnay, ubna5pc, ubna95pc = self.ub_no_action['cost'], self.ub_no_action['5pc'], \
                self.ub_no_action['95pc']
            plt.plot([np.min(ubx), np.max(ubx)], [ubnay, ubnay], 'g')
            plt.plot([np.min(ubx), np.max(ubx)], [ubna5pc, ubna5pc], 'g--')
            plt.plot([np.min(ubx), np.max(ubx)], [ubna95pc, ubna95pc], 'g--')
        plt.xlim([np.min(ubx), np.max(ubx)])
        plt.xlabel('Iteration $k$')
        plt.ylabel('Cost estimate')
        plt.title('%d stations, %d vehicles' % (self.nw.nr_regions, self.nw.nr_vehicles))

        stx, stsr = [], []
        for (k, v) in stats.iteritems():
            stx.append(int(k))
            stsr.append(float(v['sr']))
        stx, stsr = np.array(stx), np.array(stsr)
        stx_order = np.argsort(stx)
        stx, stsr = stx[stx_order], stsr[stx_order]
        plt.subplot(212)
        plt.plot(stx, stsr * 100., 'b')
        if self.stats_no_action is not None:
            plt.plot([np.min(stx), np.max(stx)],
                     [self.stats_no_action['sr'] * 100., self.stats_no_action['sr'] * 100.], 'g')
        plt.xlim([np.min(ubx), np.max(ubx)])
        plt.xlabel('Iteration $k$')
        plt.ylabel('Average service rate')
        plt.title('Service rate')
        plt.tight_layout()
        plt.savefig('output/sr_N%d_V%d_T%d' % (self.nw.nr_regions, self.nw.nr_vehicles, self.T)
                    + suffix + '.pdf')

    def plot_vf_evolution(self, t_in=None, n_in=None, mode='all_tn', suffix=''):

        k_max = len(self.vfpp)
        plot_every = self.alg_params['plot_every']

        ds_0, max_dev = self.nw.ds_0, self.alg_params['max_dev']

        if mode == 'all_tn':
            print "Plotting value function for all stations and time steps..."
            T, N = self.vfpp[0].shape[0], self.vfpp[0].shape[1]
            plt.figure(figsize=(30, 2.5 * N))

            # Determine y-axis scale
            vpcs = np.cumsum(self.vf[-1], 2)  # Cumulative sum of projected value function
            vpcs -= vpcs[:, :, max_dev - 1:max_dev]  # Treatment of axis 2 ensures broadcast works
            max_v, min_v = min(np.max(vpcs), 20), max(np.min(vpcs), -20)
            min_v_plot, max_v_plot = min_v - 0.02 * (max_v - min_v), max_v + 0.02 * (max_v - min_v)

            for (i, it) in itertools.product(range(N), range(T)):
                # vfppe = [np.squeeze(v[t_in, n_in, :]) for v in self.vfpp]
                vfe = [np.squeeze(v[it, i, :]) for v in self.vf]  # VF evolution by iteration
                plt.subplot(N, T, i * T + (it+1))
                for k, sv in enumerate(vfe):
                    assert len(sv) == 2 * max_dev
                    vpc = np.cumsum(sv)
                    if k_max < 20 or divmod(k, plot_every)[1] == 0:
                        redeploys = [x for x in range(-max_dev, max_dev + 1)]
                        val_rel_to_no_action = np.hstack((np.array([0]), vpc)) - vpc[max_dev - 1]
                        plt.plot(redeploys, val_rel_to_no_action, 'b', alpha=0.2)
                        plt.ylim([min_v_plot, max_v_plot])
                    if k == k_max - 1:  # Last VF recorded
                        redeploys = [x for x in range(-max_dev, max_dev + 1)]
                        val_rel_to_no_action = np.hstack((np.array([0]), vpc)) - vpc[max_dev - 1]
                        plt.plot(redeploys, val_rel_to_no_action, 'k')
                        plt.ylim([min_v_plot, max_v_plot])

                        # Add a marker showing the redeploy action (y- minus y+)
                        if self.last_x1_sol is not None:
                            s1_action = self.last_x1_sol[it * self.vstage_s1 +
                                                         N + (N * N) + (N * N) + 2 * i + 1] - \
                                        self.last_x1_sol[it * self.vstage_s1 +
                                                         N + (N * N) + (N * N) + 2 * i]
                            plt.scatter([s1_action], [0],
                                        s=60 if s1_action == 0 else 80,
                                        c='r' if s1_action == 0 else 'g',
                                        marker='d')  # More prominent marker if loaded/unloaded
                plt.text(0, max_v_plot - 0.1 * (max_v_plot - min_v_plot),
                         '$i = %d$, $t = %d$' % (i + 1, it + 1), ha='center', family='serif')
                if i == 0:
                    plt.title('t = %d' % (it + 1))
                if it == 0:
                    plt.ylabel('i = %d' % (i + 1))
                if i == N - 1:
                    plt.xlabel('$y_i^{-,t} - y_i^{+,t}$')
            plt.tight_layout()
            plt.savefig('output/vf__N%d_V%d_T%d_i%d' % (self.nw.nr_regions, self.nw.nr_vehicles,
                                                        self.T, self.instance_no) + suffix + '.pdf')
            plt.close()
            print "  Done."
        else:
            assert t_in is not None and n_in is not None
            vfppe = [np.squeeze(v[t_in, n_in, :]) for v in self.vfpp]
            vfe = [np.squeeze(v[t_in, n_in, :]) for v in self.vf]
            plt.figure()
            for k, v in enumerate(vfppe):
                nx = len(v)
                vc = np.cumsum(v)
                if divmod(k, plot_every)[1] == 0:
                    plt.plot(range(nx + 1), np.hstack((np.array([0]), vc)), 'g',
                             alpha=(k + 1.0) / k_max)
            plt.savefig('output/vfpp_%d_%d.pdf' % (t_in, n_in))
            plt.close()
            plt.figure()
            for k, v in enumerate(vfe):
                nx = len(v)
                vpc = np.cumsum(v)
                if divmod(k, plot_every)[1] == 0:
                    plt.plot(range(nx + 1), np.hstack((np.array([0]), vpc)), 'b',
                             alpha=(k + 1.0) / k_max)
            plt.savefig('output/vf_%d_%d.pdf' % (t_in, n_in))
            plt.close()

    def unrelax_s1(self, relax_or_unrelax='unrelax'):
        """
        Un-relaxes a relaxed S1 model by setting routing and redeployment decisions to integers. Can
        also relax an integer model by doing the reverse.
        :param relax_or_unrelax: Can only take the values 'unrelax' or 'relax'
        :return: Nothing
        """
        t1 = time.time()
        if relax_or_unrelax == 'unrelax':
            for x in self.x_s1:
                if x.varname[0] in ['y', 'z']:
                    x.vtype = gu.GRB.INTEGER
        elif relax_or_unrelax == 'relax':
            print "Warning: re-relaxation of S1 model not properly implemented!"
            for x in self.x_s1:
                if x.varname[0] in ['y', 'z']:
                    x.vtype = gu.GRB.CONTINUOUS
        else:
            print "Unrecognised relax or unrelax option, " + relax_or_unrelax
            raise SystemExit()
        self.m_s1.update()
        t2 = time.time()
        if relax_or_unrelax == 'unrelax':
            print "Unrelaxed S1 model in %.3f s." % (t2 - t1)
        else:
            print "(Re-)relaxed S1 model in %.3f s." % (t2 - t1)

    @staticmethod
    def print_non_zero_coeffs(c_list, m):
        """Print coefficients of constraints appearing in model m.

        :param c_list: List of Gurobi constraint objects
        :param m: Gurobi model
        :return: Nothing. Prints output.
        """
        for constr_to_test in c_list:
            print "Coefficients for constraint", constr_to_test.getAttr('ConstrName') + ":"
            for var in m.getVars():
                varname = var.getAttr('VarName')
                if m.getCoeff(constr_to_test, var) != 0:
                    print " ", varname, m.getCoeff(constr_to_test, var)
            print "  RHS:", constr_to_test.getAttr('RHS')
            print "  Sense:", constr_to_test.sense

    @staticmethod
    def print_constr_membership(var_name, m):
        var = m.getVarByName(var_name)
        for c in m.getConstrs():
            if m.getCoeff(c, var) != 0:
                SAModel.print_non_zero_coeffs([c], m)

    @staticmethod
    def plot_v_and_vp(v, vp, label=''):
        plt.figure()
        plt.subplot(211)
        assert len(v) == len(vp)
        nx = len(v)
        vc = np.cumsum(v)
        vpc = np.cumsum(vp)
        plt.plot(range(nx+1), np.hstack((np.array([0]), vc)), 'g')
        plt.plot(range(nx+1), np.hstack((np.array([0]), vpc)), 'b')
        plt.subplot(212)
        plt.plot(range(nx), v, 'g')
        plt.plot(range(nx), vp, 'b')
        plt.savefig('output/vfp' + label + '.pdf')
