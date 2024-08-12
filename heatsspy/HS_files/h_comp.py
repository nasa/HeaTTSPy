from openmdao.api import ExplicitComponent
import numpy as np

class Pr_calc(ExplicitComponent):
    ''' Define Prandtl number'''
    def initialize(self):
        self.options.declare('num_nodes', types=int)
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('mu_air',
                        val= np.ones(nn),
                        desc='dynamic viscosity of air',
                        units = 'Pa*s')
        self.add_input('cp_air',
                        val= np.ones(nn),
                        desc='specific heat of air',
                        units = 'J/(kg*K)')
        self.add_input('k_air',
                        val= np.ones(nn),
                        desc='thermal conductivity of air',
                        units='W/(m*K)')

        self.add_output('Pr',
                        val= np.ones(nn),
                        desc='Prandtl number of channel')

        arange = np.arange(nn)
        self.declare_partials(of='Pr', wrt='*', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        mu_air, cp_air, k_air = inputs.values()

        outputs['Pr'] = mu_air * cp_air / k_air

    def compute_partials(self, inputs, J):
        mu_air, cp_air, k_air = inputs.values()

        J['Pr', 'mu_air'] = cp_air / k_air
        J['Pr', 'cp_air'] = mu_air / k_air
        J['Pr', 'k_air'] = - mu_air * cp_air / k_air**2

class h_calc(ExplicitComponent):
    ''' Calculate Convective heat transfer and Nusselt number,
        as defined by Teerstra et al. and Nellis'''
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('h_calc_method', values=('teerstra', 'nellis'), default='teerstra')
        self.options.declare('a', default=  1     , desc='emperical constant for nellis correlation 1')
        self.options.declare('b', default= -2.042 , desc='emperical constant for nellis correlation 2')
        self.options.declare('c', default=  3.085 , desc='emperical constant for nellis correlation 3')
        self.options.declare('d', default= -2.477 , desc='emperical constant for nellis correlation 4')
        self.options.declare('m', default=  8.325 , desc='emperical multiplier for nellis correlation')

    def setup(self):
        nn = self.options['num_nodes']
        h_calc_method = self.options['h_calc_method']

        if h_calc_method == 'teerstra':
            self.add_input('Re',
                          val=1*np.ones(nn),
                          desc='Reynolds number',
                          units=None)
            self.add_input('Pr',
                          val=1.0*np.ones(nn),
                          desc='Prandtl number',
                          units=None)
            self.add_input('k_sink',
                            val=1*np.ones(nn),
                            desc='thermal conductivity of fins',
                            units='W/(m*K)')
            self.add_input('Sp',
                          val=1*np.ones(nn),
                          desc='Heat sink optimal spacing',
                          units='m')
            self.add_input('L',
                            val=np.ones(nn),
                            units='m',
                            desc='length of heat sink in direction of airflow')
            self.add_input('k_air',
                            val=np.ones(nn),
                            units='W/(m*K)',
                            desc='Thermal conductivity of the heat air')
            self.add_input('Ht',
                            val=np.ones(nn),
                            units='m',
                            desc='height of the heat sink fins')
            self.add_input('t_fin',
                            val = np.ones(nn),
                            units='m',
                            desc='thickness of heat sink fin')

        if h_calc_method == 'nellis':

            self.add_input('Ht',
                          val=1*np.ones(nn),
                          desc='Heat sink fin height',
                          units='m')
            self.add_input('k_sink',
                          val=0.02682*np.ones(nn),
                          desc='Conduction coefficient',
                          units='W/(m*K)')
            self.add_input('Dh',
                            val=1*np.ones(nn),
                            desc='hydraulic diameter',
                            units='m')
            self.add_input('Sp',
                          val=1*np.ones(nn),
                          desc='Heat sink optimal spacing',
                          units='m')
            self.add_input('k_air',
                            val=np.ones(nn),
                            units='W/(m*K)',
                            desc='Thermal conductivity of the heat air')

        self.add_output('Nu',
                        val=1*np.ones(nn),units=None,
                        desc='Nusselt number for heat exchanger')
        self.add_output('h',
                        val=1*np.ones(nn),
                        desc='Convective heat transfer coefficient of heat exchanger',
                        units = 'W/(m**2*K)')

        arange = np.arange(self.options['num_nodes'])

        self.declare_partials(of='Nu', wrt='*', rows=arange, cols=arange, method='cs')
        self.declare_partials(of='h', wrt='*', rows=arange, cols=arange, method='cs')

    def compute(self, inputs, outputs):

        h_calc_method = self.options['h_calc_method']

        if h_calc_method == 'teerstra':
            Re, Pr, k_sink, Sp, L, k_air, Ht, t_fin = inputs.values()

            #Re_star = Re * Sp / L
            Re_b = Re * Sp / L
            
            '''
            Nu_i = ((Re_star*Pr/2)**-3 + (0.664*Re_star**0.5*Pr**(1/3)*(1+(3.65/(Re_star**0.5)))**0.5)**-3)**-(1/3)

            
            # This is wrong VVVV this is the efficiency parameter, not NU
            Nu_b = outputs['Nu'] = np.tanh((2*Nu_i * (k_air/k_sink)*(Ht/Sp)*(Ht/t_fin)*((t_fin/L)+1))**0.5)/ \
                                    (2*Nu_i * (k_air/k_sink)*(Ht/Sp)*(Ht/t_fin)*((t_fin/L)+1))**0.5
            
            eta_b = outputs['Nu'] = np.tanh((2*Nu_i * (k_air/k_sink)*(Ht/Sp)*(Ht/t_fin)*((t_fin/L)+1))**0.5)/ \
                        (2*Nu_i * (k_air/k_sink)*(Ht/Sp)*(Ht/t_fin)*((t_fin/L)+1))**0.5
            
            Nu_lb = eta_b*Nu_i
            '''
            Nu_i_p1 = (Re_b*Pr/2)**(-3)

            Nu_i_p2 = 0.664*(np.sqrt(Re_b))*(Pr**(1/3))

            Nu_i_p3 = np.sqrt(1 + (3.65/np.sqrt(Re_b)))

            Nu_i = ( Nu_i_p1 + ( (Nu_i_p2*Nu_i_p3)**(-3)) )**(-1/3)

            eta_dummy = np.sqrt(2*Nu_i*k_air*Ht*Ht*(1 + (t_fin/L))/(k_sink*Sp*t_fin))

            eta = np.tanh(eta_dummy)/eta_dummy

            Nu_bar_teerstra = eta*Nu_i


            outputs['Nu'] = Nu_bar_teerstra
            #outputs['h'] = k_sink*Nu_b/Sp
            outputs['h'] = k_air*Nu_bar_teerstra/Sp

        elif h_calc_method == 'nellis':
            a = self.options['a']
            b = self.options['b']
            c = self.options['c']
            d = self.options['d']
            m = self.options['m']
            Sp = inputs['Sp']
            Ht = inputs['Ht']
            #k = inputs['k_sink']        # Wrong! This should be k_air
            k = inputs['k_air']        
            Dh = inputs['Dh']

            # NOte, check back on how to apply an average Nusselt number
            Nu = outputs['Nu'] = m*(a + b*Sp/Ht + c*(Sp/Ht)**2 + d*(Sp/Ht)**3)  
            #outputs['h'] = k*Nu/Dh
            outputs['h'] = k*Nu/Dh
            #print('k_check =', k)

    def compute_partials(self, inputs, J):

        h_calc_method = self.options['h_calc_method']

        if h_calc_method == 'nellis':
            a = self.options['a']
            b = self.options['b']
            c = self.options['c']
            d = self.options['d']
            m = self.options['m']
            Sp = inputs['Sp']
            Ht = inputs['Ht']
            #k = inputs['k_sink']    # Wrong! This should be k_air
            k = inputs['k_air']    
            Dh = inputs['Dh']

            
            J['Nu', 'Sp']  =    m*(b/Ht + 2*c*Sp/Ht**2 + 3*d*Sp**2/Ht**3)
            J['Nu', 'Ht']  = -  m*(b*Sp/Ht**2 + 2*c*Sp**2/Ht**3 + 3*d*Sp**3/Ht**4)
            J['h' , 'Sp']  =  k*m*(b/Ht + 2*c*Sp/Ht**2 + 3*d*Sp**2/Ht**3)/Dh
            J['h' , 'Ht']  = -k*m*(b*Sp/Ht**2 + 2*c*Sp**2/Ht**3 + 3*d*Sp**3/Ht**4)/Dh
            #J['h' , 'k_sink']   =  m*(a + b*Sp/Ht + c*(Sp/Ht)**2 + d*(Sp/Ht)**3)/Dh
            J['h' , 'k_air']   =  m*(a + b*Sp/Ht + c*(Sp/Ht)**2 + d*(Sp/Ht)**3)/Dh
            J['h' , 'Dh'] = -k*m*(a + b*Sp/Ht + c*(Sp/Ht)**2 + d*(Sp/Ht)**3)/Dh**2

if __name__ == "__main__":

    import time
    from openmdao.api import Problem, Group, IndepVarComp, ExplicitComponent

    prob = Problem()
    Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])

    h_calc_method = 'teerstra'
    # Flow properties


    if h_calc_method == 'nellis':
        Vars.add_output('Dh',
                        val=0.005642970701794231,
                        desc='hydraulic diameter',
                        units='m')

    Vars.add_output('Re',
                    val=2371.3792623631184,
                    desc='Reynolds number',
                    units=None)
    Vars.add_output('Pr',
                    val=0.7,
                    desc='Prandtl number',
                    units=None)
    Vars.add_output('k_sink',
                    val=218,
                    desc='thermal conductivity of fins',
                    units='W/(m*K)')
    Vars.add_output('Sp',
                  val=0.00303,
                  desc='Heat sink optimal spacing',
                  units='m')

    Vars.add_output('L',
                    val=0.3886,
                    units='m',
                    desc='length of heat sink in direction of airflow')
    Vars.add_output('k_air',
                  val=0.03,
                  desc='Conduction coefficient',
                  units='W/(m*K)')
    Vars.add_output('Ht',
                    val=0.041,
                    units='m',
                    desc='height of the heat sink fins')
    Vars.add_output('t_fin',
                    val = 0.001,
                    units='m',
                    desc='thickness of heat sink fin')


    Blk = prob.model.add_subsystem('h_calc',h_calc(num_nodes=1,h_calc_method = h_calc_method),
        promotes_inputs=['*'])

    Blk.set_check_partial_options(wrt='*', step_calc='rel')
    prob.setup()

    prob.run_model()
    prob.check_partials(compact_print=True)
    print('Nu ='+str(prob['h_calc.Nu'][0]))
    #print('Dh ='+str(prob['h_calc.Dh'][0]))
    print('k_air ='+str(prob['h_calc.k_air'][0]))
    print('h ='+str(prob['h_calc.h'][0]))
