from openmdao.api import ExplicitComponent
import numpy as np

class HE_out_q(ExplicitComponent):
    """ Calculate actual heat transfer rate"""
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int,
                             desc='Number of nodes to be evaluated')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('effect', val=np.ones(nn), desc='cooler effectiveness')
        self.add_input('q_max', val=np.ones(nn), units='W', desc='max possible heat transfer rate')

        self.add_output('q', val=np.ones(nn), units='W', desc='actual heat transfer rate')

        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='q', wrt=['effect','q_max'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        effect  = inputs['effect']
        q_max  = inputs['q_max']

        outputs['q'] = effect * q_max

    def compute_partials(self, inputs, J):
        nn = self.options['num_nodes']
        effect  = inputs['effect']
        q_max  = inputs['q_max']

        J['q','effect'] = q_max
        J['q','q_max'] = effect


class HE_out_q_max(ExplicitComponent):
    """ Calculate maximum heat transfer rate"""

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int,desc='Number of nodes to be evaluated')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('C_min', val=np.ones(nn), units='W/degK', desc='minimum capacity rate')
        self.add_input('T1', val=np.ones(nn), units='degK', desc='cold side input temperature')
        self.add_input('T2', val=np.ones(nn), units='degK', desc='hot side input temperature')

        self.add_output('q_max', val=np.ones(nn), units='W', desc='max possible heat transfer rate')

        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='q_max', wrt=['C_min','T1','T2'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']

        C_min  = inputs['C_min']
        T1 = inputs['T1']
        T2 = inputs['T2']

        # Kays and London Eqn: 2-6
        outputs['q_max'] = q_max = C_min*(T2 - T1)

    def compute_partials(self, inputs, J):
        nn = self.options['num_nodes']

        C_min  = inputs['C_min']
        T1 = inputs['T1']
        T2 = inputs['T2']

        J['q_max','C_min'] = (T2 - T1)
        J['q_max','T1'] = - C_min
        J['q_max','T2'] = C_min


if __name__ == "__main__":

    import time
    from openmdao.api import Problem, IndepVarComp
    nn = 2
    prob = Problem()
    Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    # Flow properties
    Vars.add_output('C_min', val=[4880, 1000], units='Btu/h/degR', desc='capacity rate of hot side')
    Vars.add_output('T1', val=[60, 80], units='degF', desc='cold side input temperature')
    Vars.add_output('T2', val=[260, 300], units='degF', desc='hot side input temperature')

    Vars.add_output('effect', val=[0.90, 0.8], desc='cooler effectiveness')

    Blk1 = prob.model.add_subsystem('prop_calc1', HE_out_q_max(num_nodes=nn),
        promotes_inputs=['*'], promotes_outputs=['*'])
    Blk2 = prob.model.add_subsystem('prop_calc2', HE_out_q(num_nodes=nn),
        promotes=['*'])

    # Blk.set_check_partial_options(wrt='delta', step_calc='rel')
    prob.setup(force_alloc_complex=True)
    prob.run_model()
    prob.check_partials(compact_print=True, method='cs')
    #
    print('q_max = '+str(prob['prop_calc1.q_max'][0]))
    print('q = '+str(prob['prop_calc2.q'][0]))
