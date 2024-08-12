from openmdao.api import ExplicitComponent
import numpy as np

class Kc_calc(ExplicitComponent):
    ''' Define contraction loss coefficient'''
    def initialize(self):
        #self.options.declare('a', default=  0.42    , desc='emperical constant')
        self.options.declare('num_nodes', types=int)
    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('Dh',
                        val= 0*np.zeros(nn),
                        desc='Channel Hydraulic Diameter',
                        units = 'm')
        self.add_input('Dh_far',
                       val=0*np.zeros(nn),
                       desc='Far Inlet Hydrualic Diameter',
                       units='m')


        self.add_output('Kc',
                        val=0*np.zeros(nn),
                        desc='contraction loss coefficient')

        #self.declare_partials(of='Kc', wrt='t_fin')
        #self.declare_partials(of='Kc', wrt='Wth')
        #self.declare_partials(of='Kc', wrt='N_fins')

        self.declare_partials(of='*',wrt='*',method = 'cs')
    def compute(self, inputs, outputs):
        Dh = inputs['Dh']
        Dh_far = inputs['Dh_far']

        outputs['Kc'] = 0.42*( (1 - (Dh/Dh_far)**2) )

    '''
    def compute_partials(self, inputs, J):
        Th = inputs['t_fin']
        Wth = inputs['Wth']
        N_fins = inputs['N_fins']
        a = self.options['a']

        J['Kc', 't_fin']      = - 2*a*N_fins*(N_fins*Th-Wth)/Wth**2
        J['Kc', 'Wth']     =   2*a*N_fins*Th*(N_fins*Th-Wth)/Wth**3
        J['Kc', 'N_fins']  = - 2*a*Th*(N_fins*Th-Wth)/Wth**2
    '''

if __name__ == "__main__":

    import time
    from openmdao.api import Problem, Group, IndepVarComp, ExplicitComponent

    prob = Problem()
    Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    # Flow properties
    Vars.add_output('Dh',
                    val= 0.0019047619047619048,
                    desc='Far Inlet Hydrualic Diameter',
                    units = 'm')
    Vars.add_output('Dh_far',
                   val=0.011428571428571429,
                   desc='Far Inlet Hydrualic Diameter',
                   units='m')



    Blk = prob.model.add_subsystem('Kc_calc',Kc_calc(num_nodes=1),
        promotes_inputs=['*'])
    #Blk.set_check_partial_options(wrt='t_fin', step_calc='rel')
    prob.setup()

    prob.run_model()
    prob.check_partials(compact_print=True)
    print('Kc ='+str(prob['Kc_calc.Kc'][0]))
