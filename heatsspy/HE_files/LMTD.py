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

    def compute(self, inputs, outputs):
        dT1 = inputs['T_h_in'] - inputs['T_c_out']
        dT2 = inputs['T_h_out'] - inputs['T_c_in']
        outputs['dT_lm'] = (dT1 - dT2)/np.log(dT1/dT2)

    def compute_partials(self, inputs, J):
        T_h_in = inputs['T_h_in']
        T_h_out = inputs['T_h_out']
        T_c_in = inputs['T_c_in']
        T_c_out = inputs['T_c_out']

        TP = T_h_in + T_c_in - T_h_out - T_c_out
        LR = np.log((T_h_in-T_c_out)/(T_h_out-T_c_in))
        LR2 = np.log((T_h_in-T_c_out)/(T_h_out-T_c_in))**2

        J['dT_lm','T_c_in'] = 1/LR - TP/((T_h_out - T_c_in)*LR2)
        J['dT_lm','T_h_in'] = 1/LR - TP/((T_h_in - T_c_out)*LR2)
        J['dT_lm','T_c_out'] = TP/((T_h_in - T_c_out)*LR2) - 1/LR
        J['dT_lm','T_h_out'] = TP/((T_h_out - T_c_in)*LR2) - 1/LR


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
