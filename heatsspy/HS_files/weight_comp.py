from openmdao.api import ExplicitComponent
import numpy as np

class weight_calc(ExplicitComponent):
    ''' Define weight of heat exchanger'''
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('rho_sink', default=2700, desc='density of aluminum')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('Vol',
                      val=np.ones(nn),
                      desc='Heat sink volume',
                      units='m**3')
        self.add_output('Wt',
                        val=np.ones(nn),
                        desc='Weight of Heat Exchanger',
                        units = 'kg')

        self.declare_partials(of='Wt', wrt=['Vol'])

    def compute(self, inputs, outputs):
        outputs['Wt'] = inputs['Vol']*self.options['rho_sink']

    def compute_partials(self, inputs, J):
        J['Wt', 'Vol']  = self.options['rho_sink']


if __name__ == "__main__":

    import time
    from openmdao.api import Problem, Group, IndepVarComp, ExplicitComponent

    prob = Problem()
    Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])

    Vars.add_output('Vol',
                  val=0.001,
                  desc='Heat sink fin height',
                  units='m**3')

    Blk = prob.model.add_subsystem('Wt_calc',weight_calc(num_nodes=1),
        promotes_inputs=['*'])

    prob.setup()

    prob.run_model()
    prob.check_partials(compact_print=True)
    print('Wt ='+str(prob['Wt_calc.Wt'][0]))
