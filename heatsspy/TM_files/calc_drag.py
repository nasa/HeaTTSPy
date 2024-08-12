from openmdao.api import ExplicitComponent, Group
import numpy as np

class calc_drag(ExplicitComponent):
    """ Estimate heat exchanger drag via coefficient of drag"""
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int,
                             desc='Number of nodes to be evaluated')
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('W', val=np.ones(nn), units='kg/s', desc='mass flow')
        self.add_input('v', val=np.ones(nn), units='m/s', desc='free stream velocity')
        self.add_output('Fd', val=np.ones(nn), units='N', desc='ram drag')
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='Fd', wrt=['v','W'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        outputs['Fd'] = inputs['v']*inputs['W']

    def compute_partials(self, inputs, J):
        J['Fd','v']= inputs['W']
        J['Fd','W']= inputs['v']


if __name__ == "__main__":

    import time
    from openmdao.api import Problem, IndepVarComp

    prob = Problem()
    Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    # Flow properties
    Vars.add_output('W',val=25,units='kg/s',desc='fluid mass flow')
    Vars.add_output('v', val=0.5, units='m/s', desc='velocity')
    Blk1 = prob.model.add_subsystem('Drag', calc_drag(),
        promotes_inputs=['W','v'])

    # Blk.set_check_partial_options(wrt='*', step_calc='rel')
    prob.setup()
    prob.run_model()
    prob.check_partials(compact_print=True)
    #
    print('Fd = '+str(prob['Drag.Fd'][0]))
