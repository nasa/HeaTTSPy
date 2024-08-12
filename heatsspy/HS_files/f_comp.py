from openmdao.api import ExplicitComponent
import numpy as np


class f_calc(ExplicitComponent):
    ''' Define fully developed laminar friction factor'''
    def initialize(self):
        '''
        self.options.declare('a', default=  1    , desc='emperical constant 1')
        self.options.declare('b', default= -1.3553 , desc='emperical constant 2')
        self.options.declare('c', default=  1.9467 , desc='emperical constant 3')
        self.options.declare('d', default= -1.7012 , desc='emperical constant 4')
        self.options.declare('e', default=  0.9564 , desc='emperical constant 5')
        self.options.declare('g', default= - 0.2537 , desc='emperical constant 6')
        '''
        self.options.declare('num_nodes', types=int)
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('Ht',
                      val=0*np.zeros(nn),
                      desc='Heat sink fin height',
                      units='m')
        self.add_input('Sp',
                      val=0*np.zeros(nn),
                      desc='Heat sink optimal spacing',
                      units='m')
        self.add_input('Re_Dh',
                        val= 0*np.zeros(nn),
                        desc='Reynolds number of channel based on Hydraulic Diameter')
        self.add_input('Dh',
                        val= 0*np.zeros(nn),
                        desc='Hydrualic Diameter',
                        units = 'm')
        self.add_output('f',
                        val=0*np.zeros(nn),
                        desc='fully developed laminar friction factor')

        #self.declare_partials(of='f', wrt='Ht')
        #self.declare_partials(of='f', wrt='Sp')
        #self.declare_partials(of='f', wrt='Re')
        self.declare_partials(of = '*', wrt='*', method='cs')

    def compute(self, inputs, outputs):
        '''
        a = self.options['a']
        b = self.options['b']
        c = self.options['c']
        d = self.options['d']
        e = self.options['e']
        g = self.options['g']
        Ht = inputs['Ht']
        Sp = inputs['Sp']
        Re = inputs['Re']

        B = Sp/Ht
        
        outputs['f'] = (a + b*B + c*B**2 + d*B**3 + e*B**4 + g*B**5)/Re
        '''
        Sp = inputs['Sp']
        Ht = inputs['Ht']
        Re_Dh = inputs['Re_Dh']
        
        AR_channel = Sp/Ht
        #print(AR_channel)

        Dh = 2*Sp*Ht/(Sp + Ht)

        if Re_Dh < 2300:
            
            #Laminar
            
            # Laminar, Nellis Heat Transfer, pg. 653, equation (5 - 61)

            # Determine the fully developed friction factor
            f_fd_p1 = 1 - (1.3553*AR_channel) + (1.9467*(AR_channel**2)) - (1.7012*(AR_channel**3)) + (0.9564*(AR_channel**4)) - (0.2537*(AR_channel**5))
            f_fd = 96*f_fd_p1/Re_Dh       # Fully Developed, laminar friction factor, square duct, units = N/A
            #print(f_fd)
            '''
            # Determine the non-dimensional length parameter
            L_plus = L_channel/(Dh*Re_Dh)

            # Now determine the average (aka apparent) friction factor. This includes the developing region
            f_avg_p1 = 3.44/(np.sqrt(L_plus))
            f_avg_p2 = (1.25/(4*L_plus)) + (f_fd_hyd_lam*Re_Dh/4) - (3.44/np.sqrt(L_plus))
            f_avg_p3 = 1 + (0.00021/(L_plus**2))

            # Put intermediate calculations together
            f_fd = (4/Re_Dh)*(f_avg_p1 + (f_avg_p2/f_avg_p3))
            '''
        else:

            # Turbulent, Nellis Heat Transfer, pg. 654, equation (5-65)
            eta_rough = 0       # Roughness

            f_avg_p1 = 2*eta_rough/(7.54*Dh)
            f_avg_p2 = 5.02/Re_Dh
            f_avg_p3 = (2*eta_rough/(7.54*Dh)) + (13/Re_Dh)

            f_avg_p4 = f_avg_p2*np.log10(f_avg_p3)
            f_avg_p5 = f_avg_p1 - f_avg_p4

            # Put intermediate calculations together
            f_fd = ( -2.0*np.log10(f_avg_p5) )**(-2)

        outputs['f'] = f_fd
    '''
    def compute_partials(self, inputs, J):
        a = self.options['a']
        b = self.options['b']
        c = self.options['c']
        d = self.options['d']
        e = self.options['e']
        g = self.options['g']
        Ht = inputs['Ht']
        Sp = inputs['Sp']
        Re = inputs['Re']

        J['f', 'Ht']  = - (b*Sp/Ht**2 + 2*c*Sp**2/Ht**3 + 3*d*Sp**3/Ht**4 + 4*e*Sp**4/Ht**5 + 5*g*Sp**5/Ht**6)/Re
        J['f', 'Sp']   = (b/Ht + 2*c*Sp/Ht**2 + 3*d*Sp**2/Ht**3 + 4*e*Sp**3/Ht**4 + 5*g*Sp**4/Ht**5)/Re
        J['f', 'Re']   = - (a + b*Sp/Ht + c*(Sp/Ht)**2 + d*(Sp/Ht)**3 + e*(Sp/Ht)**4 + g*(Sp/Ht)**5)/Re**2
    '''

if __name__ == "__main__":

    import time
    from openmdao.api import Problem, Group, IndepVarComp, ExplicitComponent

    prob = Problem()
    Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    
    
    Vars.add_output('Ht', 20, units='mm')
    Vars.add_output('Sp', 0.001, units='m')
    #Laminar
    #Vars.add_output('Re_Dh', 1043.18)
    #Turbulent
    Vars.add_output('Re_Dh', 2503.6325)
    Vars.add_output('Dh', 0.001904, units = 'm')


    prob.model.add_subsystem('f_calc',f_calc(num_nodes=1),
        promotes_inputs=['*'])

    prob.setup()

    prob.run_model()
    prob.check_partials(compact_print=True)
    print('f_calc ='+str(prob['f_calc.f'][0]))
