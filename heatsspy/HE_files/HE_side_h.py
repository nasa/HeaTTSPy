from openmdao.api import ExplicitComponent
import numpy as np
import openmdao.api as om

class HE_side_h_tubes(ExplicitComponent):
    """ Calculate convection coefficienct via Nusselt number"""

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int, desc='Number of nodes to be evaluated')
        #Re_trans is an aggressive guess (does not perfectly line up with equations chosen)
        self.options.declare('Re_trans', default=2300, desc='Reynolds transition number')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('k', val=np.ones(nn), units='W/m/degK', desc='thermal conductivity')
        self.add_input('Re', val=np.ones(nn), desc='Reynolds number')
        self.add_input('r_h', val=np.ones(nn), units='m', desc='hydraulic radius')
        self.add_input('Pr', val=np.ones(nn), desc='Prandtl number')

        self.add_output('Nu', val=np.ones(nn), desc='Nusselt number',lower = 1e-5)
        self.add_output('h', val=np.ones(nn), units='W/m**2/degK', desc='thermal convection',lower = 1e-5)
        self.add_output('f', val=np.ones(nn), units=None, desc='friction factor',lower = 1e-5)

        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='Nu', wrt=['Re','Pr'], rows=arange, cols=arange)
        self.declare_partials(of='h', wrt=['Re','Pr','k','r_h'], rows=arange, cols=arange)
        self.declare_partials(of='f', wrt=['Re'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        Re_trans = self.options['Re_trans']
        r_h  = inputs['r_h']
        k  = inputs['k']
        Re  = inputs['Re']
        Pr  = inputs['Pr']
        Nu = np.ones(nn, dtype=complex) # initialize Nu variable
        #rough approximations for tube flow
        # Laminar flow
        Lam = np.where(Re < Re_trans)
        outputs['Nu'][Lam] = Nu[Lam] = 3.66 # Incropera eqn 8.55
        outputs['f'][Lam]  = 64*Re[Lam]**-1.0 # Incropera eqn 8.19
        # Turbulent flow
        Turb = np.where(Re >= Re_trans)
        outputs['Nu'][Turb] = Nu[Turb] = 0.023*Re[Turb]**0.8*Pr[Turb]**0.4 # Incropera eqn 8.60, Dittus-Boelter eqn
        outputs['f'][Turb]  = 0.046*Re[Turb]**-0.2 # Kays and London Fig 6.6, note: Incropera eqn 8.20b is 0.184*Re^-0.2

        outputs['h'] = Nu*k/4/r_h

    def compute_partials(self, inputs, J):
        nn= self.options['num_nodes']
        Re_trans = self.options['Re_trans']
        r_h  = inputs['r_h']
        k  = inputs['k']
        Re  = inputs['Re']
        Pr  = inputs['Pr']

        Lam = np.where(Re < Re_trans)
        J['Nu','Re'][Lam] = 0
        J['Nu','Pr'][Lam] = 0

        J['h','Re'][Lam] = 0
        J['h','Pr'][Lam] = 0
        J['h','k'][Lam] = 3.66/4/r_h[Lam]
        J['h','r_h'][Lam] = - 3.66*k[Lam]/4/r_h[Lam]**2

        J['f','Re'][Lam] = -64*Re[Lam]**-2.0

        Turb = np.where(Re >= Re_trans)
        J['Nu','Re'][Turb] = 0.023*0.8*Re[Turb]**-0.2*Pr[Turb]**0.4
        J['Nu','Pr'][Turb] = 0.023*0.4*Re[Turb]**0.8*Pr[Turb]**-0.6

        J['h','Re'][Turb] = 0.023*0.8*Re[Turb]**-0.2*Pr[Turb]**0.4*k[Turb]/4/r_h[Turb]
        J['h','Pr'][Turb] = 0.023*0.4*Re[Turb]**0.8*Pr[Turb]**-0.6*k[Turb]/4/r_h[Turb]
        J['h','k'][Turb] = 0.023*Re[Turb]**0.8*Pr[Turb]**0.4/4/r_h[Turb]
        J['h','r_h'][Turb] = - 0.023*Re[Turb]**0.8*Pr[Turb]**0.4*k[Turb]/4/r_h[Turb]**2

        J['f','Re'][Turb] = -0.2*0.046*Re[Turb]**-1.2

class HE_side_h_fit(ExplicitComponent):
    """ Calculate convection coefficienct via table lookups of f and St (curve fit)"""

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int, desc='Number of nodes to be evaluated')
        self.options.declare('hex_def',default='hex_props',desc='heat exchanger definition')
        self.options.declare('side_number',default='' ,desc= 'side number for lookup reference')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('Cp', val=np.ones(nn), units='J/kg/degK', desc='specific heat with constant pressure')
        self.add_input('G', val=np.ones(nn), units='kg/(s*m**2)', desc='flow stream mass velocity')
        self.add_input('Pr', val=np.ones(nn), desc='Prandtl number')
        self.add_input('Re', val=np.ones(nn), desc='Reynolds number')

        self.add_output('h', val=np.ones(nn), units='W/m**2/degK', desc='convection coefficient',lower = 1e-5)
        self.add_output('f', val=np.ones(nn), desc='friction factor',lower = 1e-5)
        self.add_output('St', val=np.ones(nn), desc='Stanton number',lower = 1e-5)

        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='h', wrt=['Re','Pr','G','Cp'], rows=arange, cols=arange)
        self.declare_partials(of='f', wrt=['Re'], rows=arange, cols=arange)
        self.declare_partials(of='St', wrt=['Re','Pr'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        Cp  = inputs['Cp']
        G  = inputs['G']
        Pr  = inputs['Pr']
        Re = inputs['Re']
        hex_def = self.options['hex_def']
        side_number = self.options['side_number']

        j = np.ones(nn, dtype=float)

        Re_ok = np.where(Re>1)
        hex_def.get_j(Re[Re_ok],side_number)
        hex_def.get_f(Re[Re_ok],side_number)
        j[Re_ok] = getattr(self.options['hex_def'], 'j_'+str(self.options['side_number']))
        outputs['f'][Re_ok] = getattr(self.options['hex_def'], 'f'+str(self.options['side_number']))
        Re_low = np.where(Re<=1)
        j[Re_low] = -9999*Re[Re_low]+1e4
        outputs['f'][Re_low] = -9999*Re[Re_low]+1e4
        # if self.pathname == 'traj.phases.phase0.rhs_all.TMS_dynamic.TMS_dyn.tmsEo.AOCE.h1_calc':
        if np.any(Pr < 0) or np.any(np.isnan(Pr)):
            print(self.pathname)
            raise om.AnalysisError('Raise AnalysisError in HE_side_h.py: Pr < 0')

        outputs['St'] = St = j/Pr**(2/3)
        outputs['h']= h = St*G*Cp

    def compute_partials(self, inputs, J):
        nn = self.options['num_nodes']
        Cp  = inputs['Cp']
        G  = inputs['G']
        Pr  = inputs['Pr']
        Re = inputs['Re']
        hex_def = self.options['hex_def']
        side_number = self.options['side_number']

        j = np.ones(nn, dtype=complex)
        dj = np.ones(nn, dtype=complex)

        Re_ok = np.where(Re>1)
        hex_def.get_j(Re[Re_ok],side_number)
        hex_def.get_j_partials(Re[Re_ok],side_number)
        hex_def.get_f_partials(Re[Re_ok],side_number)

        j[Re_ok] = getattr(self.options['hex_def'], 'j_'+str(self.options['side_number']))
        dj[Re_ok] = getattr(self.options['hex_def'], 'dj_dRe'+str(self.options['side_number']))
        Re_low = np.where(Re<=1)
        j[Re_low] = -9999*Re[Re_low]+1e4
        dj[Re_low] = -9999

        J['f','Re'][Re_ok] = getattr(self.options['hex_def'], 'df_dRe'+str(self.options['side_number']))
        J['f','Re'][Re_low] = -9999

        J['St','Re'] =   dj*Pr**(-2/3)
        J['St','Pr'] = - (2/3)*j*Pr**(-5/3)

        J['h','Re'] =   dj*Pr**(-2/3)*G*Cp
        J['h','Pr'] = - (2/3)*j*Pr**(-5/3)*G*Cp
        J['h','Cp'] =   j*Pr**(-2/3)*G
        J['h','G'] =    j*Pr**(-2/3)*Cp

if __name__ == "__main__":

    import time
    from openmdao.api import Problem, IndepVarComp
    from heatsspy.include.HexParams_Regenerator import hex_params_regenerator
    hex_def = hex_params_regenerator()

    nn = 2
    prob = Problem()
    Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    # Flow properties
    Vars.add_output('G1', val=np.array([26.18,26.18]), units='kg/(s*m**2)', desc='flow stream mass velocity')
    Vars.add_output('Re1', val=np.array([1760,5080]), desc='Reynolds number')
    Vars.add_output('Pr1', val=np.array([0.666,0.666]), desc='Prandtl number')
    Vars.add_output('Cp1', val=np.array([1050,1050]), units='J/kg/degK', desc='specific heat with constant pressure')

    Vars.add_output('G2', val=np.array([13.36,13.36]), units='kg/(s*m**2)', desc='flow stream mass velocity')
    Vars.add_output('Re2', val=np.array([1760,5080]), desc='Reynolds number')
    Vars.add_output('Pr2', val=np.array([0.670,0.670]), desc='Prandtl number')
    Vars.add_output('Cp2', val=np.array([1084,1084]), units='J/kg/degK', desc='specific heat with constant pressure')

    Vars.add_output('Re3', val=np.array([1760,5080]), desc='Reynolds number')
    Vars.add_output('Pr3', val=np.array([0.670,0.670]), desc='Prandtl number')
    Vars.add_output('r_h3', val=0.00625*np.ones(nn), units='m', desc='pipe diameter')
    Vars.add_output('k3', val=np.array([0.12,0.12]), units='W/m/degK', desc='thermal conductivity')

    Blk1 = prob.model.add_subsystem('prop_calc1', HE_side_h_fit(num_nodes=nn, hex_def=hex_def, side_number =1),
        promotes_inputs=[('G','G1'),('Pr','Pr1'),('Re','Re1'),('Cp','Cp1')])
    Blk2 = prob.model.add_subsystem('prop_calc2', HE_side_h_fit(num_nodes=nn, hex_def=hex_def, side_number =2),
        promotes_inputs=[('G','G2'),('Pr','Pr2'),('Re','Re2'),('Cp','Cp2')])
    Blk3 = prob.model.add_subsystem('prop_calc3', HE_side_h_tubes(num_nodes=nn),
            promotes_inputs=[('Pr','Pr3'),('Re','Re3'),('r_h','r_h3'),('k','k3')])

    # Blk2.set_check_partial_options(wrt=['r_h','Re'], step_calc='rel')
    prob.setup(force_alloc_complex=True)
    prob.run_model()
    prob.check_partials(compact_print=True, method='cs')
    #
    print('nn = '+str(np.size(prob['prop_calc1.h'])))
    print('h1 = '+str(prob['prop_calc1.h'][0]))
    print('f1 = '+str(prob['prop_calc1.f'][0]))
    print('St1 = '+str(prob['prop_calc1.St'][0]))
    print('h2 = '+str(prob['prop_calc2.h'][0]))
    print('f2 = '+str(prob['prop_calc2.f'][0]))
    print('St2 = '+str(prob['prop_calc2.St'][0]))
    print('h3a = '+str(prob['prop_calc3.h'][0]))
    print('f3a = '+str(prob['prop_calc3.f'][0]))
    print('Nu3a = '+str(prob['prop_calc3.Nu'][0]))
    print('h3b = '+str(prob['prop_calc3.h'][1]))
    print('f3b = '+str(prob['prop_calc3.f'][1]))
    print('Nu3b = '+str(prob['prop_calc3.Nu'][1]))
