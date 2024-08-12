from openmdao.api import ExplicitComponent
import numpy as np

class Dh_calc(ExplicitComponent):
    ''' Define hydraulic diameter of heat exchanger'''
    def initialize(self):
        self.options.declare('num_nodes', types=int)
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('Ht',
                      val=np.ones(nn),
                      desc='Heat sink fin height',
                      units='m')
        self.add_input('Sp',
                      val=np.ones(nn),
                      desc='Heat sink optimal spacing',
                      units='m')

        self.add_output('Dh',
                        val= np.ones(nn),
                        desc='Hydraulic diameter',
                        units = 'm')

        #self.declare_partials(of='Dh', wrt='Ht')
        #self.declare_partials(of='Dh', wrt='Sp')
        self.declare_partials(of='*',wrt='*',method = 'cs')
    def compute(self, inputs, outputs):
        Ht = inputs['Ht']
        Sp = inputs['Sp']

        outputs['Dh'] = 2* Ht*Sp/(Ht+Sp)

    #def compute_partials(self, inputs, J):
    #    Ht = inputs['Ht']
    #    Sp = inputs['Sp']
    #
    #    J['Dh', 'Ht'] = 2* Sp**2/(Ht+Sp)**2
    #    J['Dh', 'Sp'] = 2* Ht**2/(Ht+Sp)**2

if __name__ == "__main__":

    import time
    from openmdao.api import Problem, Group, IndepVarComp, ExplicitComponent

    prob = Problem()
    Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    # Flow properties
    Vars.add_output('Ht', 20, units='mm')
    Vars.add_output('Sp', 0.001, units='m')

    prob.model.add_subsystem('Dh_calc',Dh_calc(num_nodes=1),
        promotes_inputs=['*'])

    prob.setup()

    prob.run_model()
    prob.check_partials(compact_print=True)
    print('Dh ='+str(prob['Dh_calc.Dh'][0]))
