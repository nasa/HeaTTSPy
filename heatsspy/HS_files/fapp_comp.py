from openmdao.api import ExplicitComponent
import numpy as np

class fapp_calc(ExplicitComponent):
    ''' Define apparent friction factor'''
    def initialize(self):
        self.options.declare('num_nodes', types=int)
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('f',
                        val=np.ones(nn),
                        desc='fully developed laminar friction factor')
        self.add_input('Re_Dh',
                        val= np.ones(nn),
                        desc='Reynolds number of channel based on hydraulic diameter')
        #self.add_input('xp',
        #                val=np.ones(nn),
        #                desc='Hydrodynamic entry length ratio')
        self.add_input('L_channel',
                        val=np.ones(nn),
                        desc='Channel Length',
                        units = 'm')
        self.add_input('Dh',
                        val= 0*np.zeros(nn),
                        desc='Hydrualic Diameter',
                        units = 'm')

        self.add_output('fapp',
                        val=np.ones(nn),
                        desc='apparent friction factor')

        #self.declare_partials(of='fapp', wrt='f')
        #self.declare_partials(of='fapp', wrt='Re')
        #self.declare_partials(of='fapp', wrt='xp')
        self.declare_partials(of = '*', wrt='*', method='cs')
    def compute(self, inputs, outputs):
        f_fd = inputs['f']
        Re_Dh = inputs['Re_Dh']
        L_channel = inputs['L_channel']
        Dh = inputs['Dh']
        #xp = inputs['xp']
        
        if Re_Dh < 2300:

            #print('Laminar')
            #Laminar
            # Determine the non-dimensional length parameter
            L_plus = L_channel/(Dh*Re_Dh)

            # Now determine the average (aka apparent) friction factor. This includes the developing region
            f_avg_p1 = 3.44/(np.sqrt(L_plus))
            f_avg_p2 = (1.25/(4*L_plus)) + (f_fd*Re_Dh/4) - (3.44/np.sqrt(L_plus))
            f_avg_p3 = 1 + (0.00021/(L_plus**2))


            
            # Put intermediate calculations together
            f_avg = (4/Re_Dh)*(f_avg_p1 + (f_avg_p2/f_avg_p3))

            #print('L_plus=',L_plus)
            #print('f_avg_p1 =', f_avg_p1)
            #print('f_avg_p2 =', f_avg_p2)
            #print('f_avg_p3 =', f_avg_p3)
            #print('f_avg =', f_avg)
                    

        else:
            #print('Turbulent')
            # Turbulent
            f_avg = f_fd*(1 + ( (Dh/L_channel)**0.7) )

        #outputs['fapp'] =fapp = ((3.44/xp**0.5)**2 + (f*Re)**2)**0.5/Re
        outputs['fapp'] = f_avg
    
    '''
    def compute_partials(self, inputs, J):
        f = inputs['f']
        Re = inputs['Re']
        xp = inputs['xp']

        J['fapp', 'f']  = f*Re/((f*Re)**2 + 3.44**2/xp)**0.5
        J['fapp', 'Re']  = - 3.44**2/(Re**2*xp*((f*Re)**2 + 3.44**2/xp)**0.5)
        J['fapp', 'xp']  = - 5.9168/(Re*xp**2*((f*Re)**2 + 3.44**2/xp)**0.5)

    '''
if __name__ == "__main__":

    import time
    from openmdao.api import Problem, Group, IndepVarComp, ExplicitComponent

    prob = Problem()
    Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    # Flow properties
    Vars.add_output('L_channel', 0.025, units = 'm' )
    Vars.add_output('Dh',0.0019047619047619048, units = 'm')

     #Laminar
    #Vars.add_output('f', 0.08621896679036387)
    #Vars.add_output('Re_Dh', 1043.180209381171)

    #Turb
    Vars.add_output('f', 0.04569413232366966)
    Vars.add_output('Re_Dh', 2503.6325025148094)


    #Vars.add_output('xp', 0.014632171)

    prob.model.add_subsystem('fa_calc',fapp_calc(num_nodes=1),
        promotes_inputs=['*'])

    prob.setup()

    prob.run_model()
    prob.check_partials(compact_print=True)
    print('fa_calc ='+str(prob['fa_calc.fapp'][0]))
