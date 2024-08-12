from openmdao.api import ExplicitComponent
import numpy as np

class Vin_calc(ExplicitComponent):
    ''' Define approach velocity of heat exchanger'''
    def initialize(self):
        self.options.declare('num_nodes', types=int)
    def setup(self):
        nn = self.options['num_nodes']
        # Inputs
        self.add_input('W',
                       val=np.ones(nn),
                       desc='Design air flow',
                       units='kg/s')
        self.add_input('rho',
                       val=np.ones(nn),
                       desc='Density of air',
                       units='kg/m**3')
        self.add_input('Ht',
                       val=np.ones(nn),
                       desc='Heat sink fin height',
                       units='m')
        self.add_input('Wth',
                       val=np.ones(nn),
                       desc='Heat sink total width',
                       units='m')
        self.add_output('V',
                       val= np.ones(nn),
                       desc='Velocity of airflow entering heat sink',
                       units = 'm/s')

        self.declare_partials(of='V', wrt='W')
        self.declare_partials(of='V', wrt='rho')
        self.declare_partials(of='V', wrt='Ht')
        self.declare_partials(of='V', wrt='Wth')

    def compute(self, inputs, outputs):
        W = inputs['W']
        rho = inputs['rho']
        Ht = inputs['Ht']
        Wth = inputs['Wth']

        outputs['V'] = W/(rho*Ht*Wth)

    def compute_partials(self, inputs, J):
        W = inputs['W']
        rho = inputs['rho']
        Ht = inputs['Ht']
        Wth = inputs['Wth']

        J['V', 'W']    = 1/(rho*Ht*Wth)
        J['V', 'rho']  = - W/(rho**2*Ht*Wth)
        J['V', 'Ht']   = - W/(rho*Ht**2*Wth)
        J['V', 'Wth']  = - W/(rho*Ht*Wth**2)


if __name__ == "__main__":

    import time
    from openmdao.api import Problem, Group, IndepVarComp, ExplicitComponent

    prob = Problem()
    Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    # Flow properties
    Vars.add_output('W',
                   val=0.0415314,
                   desc='Design air flow',
                   units='kg/s')
    Vars.add_output('rho',
                   val=1.1,
                   desc='Density of air',
                   units='kg/m**3')
    Vars.add_output('Ht',
                   val=20,
                   desc='Heat sink fin height',
                   units='mm')
    Vars.add_output('Wth',
                   val=0.19,
                   desc='Heat sink total width',
                   units='m')


    Blk = prob.model.add_subsystem('Vin_calc',Vin_calc(num_nodes=1),
        promotes_inputs=['*'])
    Blk.set_check_partial_options(wrt=['Ht','Wth'], step_calc='rel')
    prob.setup()

    prob.run_model()
    prob.check_partials(compact_print=True)
    print('Vin ='+str(prob['Vin_calc.V'][0]))
