from openmdao.api import ExplicitComponent
import numpy as np

class LMTD_CALC(ExplicitComponent):
    """ Calculate output temperatures"""

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int,
                             desc='Number of nodes to be evaluated')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('T_h_in', val=np.ones(nn), units='degK', desc='hot side input temperature')
        self.add_input('T_h_out', val=np.ones(nn), units='degK', desc='hot side output temperature')
        self.add_input('T_c_in', val=np.ones(nn), units='degK', desc='cold side input temperature')
        self.add_input('T_c_out', val=np.ones(nn), units='degK', desc='cold side output temperature')

        self.add_output('dT_lm', val=np.ones(nn), units='degK', desc='Log mean temperature difference')

        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='dT_lm', wrt=['*'], rows=arange, cols=arange)

    @staticmethod
    def _log_mean_and_partials(dT1, dT2):
        """Evaluate the log mean and its derivatives, including equal differences."""
        difference = dT1 - dT2
        total = dT1 + dT2
        near_equal = np.abs(difference) < 1.e-3 * np.abs(total)
        mean = np.empty_like(dT1)
        first = np.empty_like(dT1)
        second = np.empty_like(dT1)

        # z / atanh(z) has a regular even series at z = 0. Keeping the
        # z**6 term also gives stable derivatives near equal nonzero differences.
        z = difference[near_equal] / total[near_equal]
        z2 = z * z
        ratio = 1 - z2 * (1./3 + z2 * (4./45 + z2 * 44./945))
        derivative = -z * (2./3 + z2 * (16./45 + z2 * 264./945))
        mean[near_equal] = 0.5 * total[near_equal] * ratio
        first[near_equal] = 0.5 * (ratio + (1 - z) * derivative)
        second[near_equal] = 0.5 * (ratio - (1 + z) * derivative)

        other = ~near_equal
        logarithm = np.log(dT1[other] / dT2[other])
        mean[other] = difference[other] / logarithm
        first[other] = 1 / logarithm - difference[other] / (dT1[other] * logarithm**2)
        second[other] = difference[other] / (dT2[other] * logarithm**2) - 1 / logarithm
        return mean, first, second

    def compute(self, inputs, outputs):
        dT1 = inputs['T_h_in'] - inputs['T_c_out']
        dT2 = inputs['T_h_out'] - inputs['T_c_in']
        outputs['dT_lm'] = self._log_mean_and_partials(dT1, dT2)[0]

    def compute_partials(self, inputs, J):
        dT1 = inputs['T_h_in'] - inputs['T_c_out']
        dT2 = inputs['T_h_out'] - inputs['T_c_in']
        _, first, second = self._log_mean_and_partials(dT1, dT2)
        J['dT_lm', 'T_h_in'] = first
        J['dT_lm', 'T_h_out'] = second
        J['dT_lm', 'T_c_in'] = -second
        J['dT_lm', 'T_c_out'] = -first


if __name__ == "__main__":

    import time
    from openmdao.api import Problem, IndepVarComp

    prob = Problem()
    Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    # Flow properties
    Vars.add_output('T_h_in', val=422, units='degK', desc='hot side input temperature')
    Vars.add_output('T_h_out', val=400, units='degK', desc='hot side input temperature')
    Vars.add_output('T_c_in', val=317, units='degK', desc='cold side input temperature')
    Vars.add_output('T_c_out', val=401, units='degK', desc='cold side input temperature')

    prob.model.add_subsystem('prop_calc', LMTD_CALC(),
        promotes_inputs=['*'])


    # Blk.set_check_partial_options(wrt='C_c', step_calc='rel')
    prob.setup(force_alloc_complex=True)
    prob.run_model()
    prob.check_partials(compact_print=True, method='cs')
    #
    print('dT_lm = ', prob.get_val('prop_calc.dT_lm')[0])
