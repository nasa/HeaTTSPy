from openmdao.api import ExplicitComponent
import numpy as np

class Re_calc(ExplicitComponent):
    ''' Define Reynolds number'''
    def initialize(self):
        self.options.declare('num_nodes', types=int)
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('Vch',
                        val= np.ones(nn),
                        desc='Channel velocity',
                        units = 'm/s')
        self.add_input('char_length',
                        val= np.ones(nn),
                        desc='characteristic length',
                        units = 'm')
        self.add_input('mu',
                        val=np.ones(nn),
                        desc='dynamic viscosity of air',
                        units='Pa*s')
        self.add_input('rho_air',
                        val=np.ones(nn),
                        desc='density of air',
                        units='kg/m**3')

        self.add_output('Re',
                        val= np.ones(nn),
                        desc='Reynolds number of channel')

        self.declare_partials(of='Re', wrt='Vch')
        self.declare_partials(of='Re', wrt='char_length')
        self.declare_partials(of='Re', wrt='mu')
        self.declare_partials(of='Re', wrt='rho_air')

    def compute(self, inputs, outputs):
        Vch = inputs['Vch']
        char_length = inputs['char_length']
        v = inputs['mu']/inputs['rho_air']

        #outputs['Re'] = Vch * char_length / v
        outputs['Re'] = Vch * char_length / v
        

    def compute_partials(self, inputs, J):
        Vch = inputs['Vch']
        char_length = inputs['char_length']
        v = inputs['mu']/inputs['rho_air']
        dv_drho = -inputs['mu']/inputs['rho_air']**2
        dv_dmu = 1/inputs['rho_air']

        J['Re', 'Vch'] =  char_length / v
        J['Re', 'char_length']   = Vch / v
        J['Re', 'mu']    = - Vch * char_length / v**2 * dv_dmu
        J['Re', 'rho_air']    = - Vch * char_length / v**2 * dv_drho


if __name__ == "__main__":

    import time
    from openmdao.api import Problem, Group, IndepVarComp, ExplicitComponent

    prob = Problem()
    Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    # Flow properties
    Vars.add_output('Vch',
                    val= 16.37,
                    desc='Channel velocity',
                    units = 'm/s')
    Vars.add_output('char_length',
                    val= 0.00303,
                    desc='Hydraulic diameter',
                    units = 'm')
    Vars.add_output('mu',
                    val= 20.92e-6,
                    desc='Kinematic viscosity of air',
                    units='Pa*s')


    Blk = prob.model.add_subsystem('Re_calc',Re_calc(num_nodes=1),
        promotes_inputs=['*'])
    prob.setup(force_alloc_complex=True)

    prob.run_model()
    prob.check_partials(compact_print=True, method='cs')
    #print('Re ='+str(prob['Re_calc.Re'][0]))
    print('Re ='+str(Blk.get_val('Re_calc.Re')))