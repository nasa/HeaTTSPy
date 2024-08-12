from openmdao.api import ExplicitComponent
import numpy as np

class dP_calc(ExplicitComponent):
    ''' Define delta pressure
        Source: https://www.electronics-cooling.com/2015/12/calculation-corner-estimating-parallel-plate-fin-heat-sink-thermal-resistance/'''
    def initialize(self):
        self.options.declare('num_nodes', types=int)
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('fapp', val=np.ones(nn), desc='apparent friction factor')

        self.add_input('Kc', val=np.ones(nn), desc='contraction loss coefficient')
        self.add_input('Ke', val=np.ones(nn), desc='expansion loss coefficient')
        self.add_input('Lng', val=np.ones(nn), desc='Heat sink length', units='m')
        self.add_input('P', val= 2*np.ones(nn), desc='input pressure', units='Pa')
        self.add_input('rho_air', val=1*np.ones(nn), desc='Density of air', units='kg/m**3')
        self.add_input('Vch', val= 2*np.ones(nn), desc='Channel velocity', units = 'm/s')
        self.add_input('Dh', val=np.ones(nn), desc='Heat sink Channel Hydraulic Diameter', units='m')

        self.add_output('dP',
                        val=0*np.ones(nn),
                        desc='Pressure loss',
                        units='Pa')
        self.add_output('dPqP',
                        val=0*np.ones(nn),
                        desc='pressure change/initial pressure')
        
        arange = np.arange(self.options['num_nodes'])
        #self.declare_partials(of='dP', wrt=['fapp','Ht','Kc','Ke','Lng','N_fins','rho_air','Sp','Vch','Wth'], rows=arange, cols=arange)
        #self.declare_partials(of='dPqP', wrt=['fapp','Ht','Kc','Ke','Lng','N_fins','rho_air','Sp','Vch','Wth','P'], rows=arange, cols=arange)
        self.declare_partials(of='*',wrt='*', method = 'cs')

    def compute(self, inputs, outputs):
        fapp = inputs['fapp']

        Kc = inputs['Kc']
        Ke = inputs['Ke']
        Lng = inputs['Lng']
        rho = inputs['rho_air']
        Vch = inputs['Vch']
        Dh = inputs['Dh']
        

        P = inputs['P']

        P_dynamic = 0.5*rho*(Vch**2)
        outputs['dP'] = P_dynamic*( (fapp*Lng/Dh) + Kc + Ke)
        outputs['dPqP'] = outputs['dP']/P
    '''
    def compute_partials(self, inputs, J):
        fapp = inputs['fapp']
        Ht = inputs['Ht']
        Kc = inputs['Kc']
        Ke = inputs['Ke']
        Lng = inputs['Lng']
        N_fins = inputs['N_fins']
        rho = inputs['rho_air']
        Sp = inputs['Sp']
        Wth = inputs['Wth']
        Vch = inputs['Vch']
        P = inputs['P']

        J['dP', 'fapp']   =   0.5*N_fins*Lng*rho*Vch**2*(2*Ht+Sp)/Ht/Wth
        J['dP', 'Ht']     =  -0.5*fapp*Lng*N_fins*rho*Sp*Vch**2/Ht**2/Wth
        J['dP', 'Kc']     =   0.5*rho*Vch**2
        J['dP', 'Ke']     =   0.5*rho*Vch**2
        J['dP', 'Lng']    =   0.5*fapp*N_fins*rho*Vch**2*(2*Ht+Sp)/Ht/Wth
        J['dP', 'N_fins'] =   0.5*fapp*Lng*rho*Vch**2*(2*Ht+Sp)/Ht/Wth
        J['dP', 'rho_air']    =   (fapp*N_fins*(2*Ht*Lng+Sp*Lng)/(Ht*Wth)+Kc+Ke)*(0.5*Vch**2)
        J['dP', 'Sp']     =   0.5*fapp*N_fins*Lng*rho*Vch**2/Ht/Wth
        J['dP', 'Wth']    =  -0.5*fapp*N_fins*Lng*rho*Vch**2*(2*Ht+Sp)/Ht/Wth**2
        J['dP', 'Vch']    =   (fapp*N_fins*(2*Ht*Lng+Sp*Lng)/(Ht*Wth)+Kc+Ke)*(rho*Vch)

        J['dPqP', 'fapp']   =   0.5*N_fins*Lng*rho*Vch**2*(2*Ht+Sp)/Ht/Wth/P
        J['dPqP', 'Ht']     =  -0.5*fapp*Lng*N_fins*rho*Sp*Vch**2/Ht**2/Wth/P
        J['dPqP', 'Kc']     =   0.5*rho*Vch**2/P
        J['dPqP', 'Ke']     =   0.5*rho*Vch**2/P
        J['dPqP', 'Lng']    =   0.5*fapp*N_fins*rho*Vch**2*(2*Ht+Sp)/Ht/Wth/P
        J['dPqP', 'N_fins'] =   0.5*fapp*Lng*rho*Vch**2*(2*Ht+Sp)/Ht/Wth/P
        J['dPqP', 'rho_air']    =   (fapp*N_fins*(2*Ht*Lng+Sp*Lng)/(Ht*Wth)+Kc+Ke)*(0.5*Vch**2)/P
        J['dPqP', 'Sp']     =   0.5*fapp*N_fins*Lng*rho*Vch**2/Ht/Wth/P
        J['dPqP', 'Wth']    =  -0.5*fapp*N_fins*Lng*rho*Vch**2*(2*Ht+Sp)/Ht/Wth**2/P
        J['dPqP', 'Vch']    =   (fapp*N_fins*(2*Ht*Lng+Sp*Lng)/(Ht*Wth)+Kc+Ke)*(rho*Vch)/P
        J['dPqP', 'P']    =  - (fapp*N_fins*(2*Ht*Lng+Sp*Lng)/(Ht*Wth)+Kc+Ke)*(0.5*rho*Vch**2)/P**2

    '''
if __name__ == "__main__":

    import time
    from openmdao.api import Problem, Group, IndepVarComp, ExplicitComponent

    prob = Problem()
    Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    # Flow properties
    Vars.add_output('fapp', val=0.09812309781136098, desc='apparent friction factor')
    Vars.add_output('Kc', val=0.4083333333333333, desc='contraction loss coefficient')
    Vars.add_output('Ke', val=0.945216049382716, desc='expansion loss coefficient')
    Vars.add_output('Lng', val=0.025, desc='Heat sink length', units='m')

    Vars.add_output('P', val=1e5, desc='pressure', units='Pa')
    Vars.add_output('rho_air', val=1.225, desc='Density of air', units='kg/m**3')

    Vars.add_output('Vch', val= 14.40, desc='Channel velocity', units = 'm/s')
    Vars.add_output('Dh', val= 0.0019047619047619048, desc='Channel Hydraulic Diameter', units = 'm')

    prob.model.add_subsystem('dP_calc',dP_calc(num_nodes=1),
        promotes_inputs=['*'])

    prob.setup()

    prob.run_model()
    prob.check_partials(compact_print=True)
    print('dP ='+str(prob['dP_calc.dP'][0]))
    print('dPqP ='+str(prob['dP_calc.dPqP'][0]))
