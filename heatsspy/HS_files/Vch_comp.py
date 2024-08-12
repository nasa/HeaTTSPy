from openmdao.api import ExplicitComponent
import numpy as np

class Vch_calc(ExplicitComponent):
    ''' Define channel velocity of heat exchanger'''
    def initialize(self):
        self.options.declare('num_nodes', types=int)
    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('W_flow',
                       val= np.ones(nn),
                       desc='mass flow of airflow entering heat sink',
                       units = 'kg/s')
        self.add_input('rho_air',
                       val=np.ones(nn),
                       desc='air density',
                       units='kg/(m**3)')
        self.add_input('Sp',
                      val=np.ones(nn),
                      desc='Heat sink optimal spacing',
                      units='m')
        self.add_input('N_fins',
                      val=np.ones(nn),
                      desc='Number of heat sink fins')
        self.add_input('Ht',
                      val=np.ones(nn),
                      units='m',
                      desc='height of heat sink fins')

        self.add_output('Vch',
                        val= np.ones(nn),
                        desc='Channel velocity',
                        units = 'm/s')
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='Vch', wrt='*', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        W_flow = inputs['W_flow']
        rho_air = inputs['rho_air']
        Sp = inputs['Sp']
        N_fins = inputs['N_fins']
        Ht = inputs['Ht']

        A = Ht*(N_fins-1)*Sp
        outputs['Vch'] = W_flow/rho_air/A

    def compute_partials(self, inputs, J):
        W_flow = inputs['W_flow']
        rho_air = inputs['rho_air']
        Sp = inputs['Sp']
        N_fins = inputs['N_fins']
        Ht = inputs['Ht']

        A = Ht*(N_fins-1)*Sp
        dA_dHt = (N_fins-1)*Sp
        dA_dN_fins = Ht*Sp
        dA_dSp= Ht*(N_fins-1)

        J['Vch', 'W_flow'] = 1/rho_air/A
        J['Vch', 'rho_air'] = -W_flow/rho_air**2/A
        J['Vch', 'Sp']     = -W_flow/rho_air/A**2*dA_dSp
        J['Vch', 'N_fins'] = -W_flow/rho_air/A**2*dA_dN_fins
        J['Vch', 'Ht'] = -W_flow/rho_air/A**2*dA_dHt


