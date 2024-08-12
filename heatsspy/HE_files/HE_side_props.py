from openmdao.api import ExplicitComponent
import numpy as np

class HE_side_Pr_v(ExplicitComponent):
    """ Calculate Exchanger flow stream Prahl number and kinematic visosity"""
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int, desc='Number of nodes to be evaluated')
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('Cp', val=np.ones(nn), units='J/kg/degK', desc='specific heat with constant pressure')
        self.add_input('k', val=np.ones(nn), units='W/m/degK', desc='thermal conductivity')
        self.add_input('mu', val=np.ones(nn), units='Pa*s', desc='viscosity')
        self.add_input('rho', val=np.ones(nn), units='kg/m**3', desc='fluid density')


        self.add_output('Pr', val=np.ones(nn), desc='Prandtl number')
        self.add_output('v', val=np.ones(nn), units='m**2/s', desc='kinematic viscoisty')

        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='Pr', wrt=['Cp','k','mu'], rows=arange, cols=arange)
        self.declare_partials(of='v', wrt=['rho','mu'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        Cp = inputs['Cp']
        k = inputs['k']
        mu = inputs['mu']
        rho = inputs['rho']

        outputs['Pr'] = mu*Cp/k
        outputs['v'] = mu/rho
        # print(self.pathname)
        # print('mu = ',mu,' Cp = ' ,Cp,' k = ', k)

    def compute_partials(self, inputs, J):
        Cp = inputs['Cp']
        k = inputs['k']
        mu = inputs['mu']
        rho= inputs['rho']

        J['Pr','mu'] = Cp/k
        J['Pr','Cp'] = mu/k
        J['Pr','k'] = - mu*Cp/k**2

        J['v','mu'] = 1/rho
        J['v','rho'] = -mu/rho**2


class HE_side_G_Re(ExplicitComponent):
    """ Calculate Exchanger flow stream mass velocity and Reynolds number"""

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int, desc='Number of nodes to be evaluated')
        self.options.declare('hex_def',default='hex_props',desc='heat exchanger definition')
        self.options.declare('side_number',default='' ,desc= 'side number for lookup reference')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('W', val=np.ones(nn), units="kg/s", desc='Mass flow input')
        self.add_input('Afr', val=np.ones(nn), units='m**2', desc='Frontal Area')
        self.add_input('mu', val=np.ones(nn), units='Pa*s', desc='viscosity')

        self.add_output('G', val=np.ones(nn), units='kg/(s*m**2)', desc='flow stream mass velocity',lower = 1e-5)
        self.add_output('Re', val=np.ones(nn), desc='Reynolds number',lower = 1)

        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='G', wrt=['W', 'Afr'], rows=arange, cols=arange)
        self.declare_partials(of='Re', wrt=['W','mu', 'Afr'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        hex_def = self.options['hex_def']
        side_number = self.options['side_number']
        sigma = getattr(self.options['hex_def'], 'sigma'+str(self.options['side_number']))
        r_h = getattr(self.options['hex_def'], 'r_h'+str(self.options['side_number']))

        W  = inputs['W']
        Afr = inputs['Afr']
        mu = inputs['mu']
        outputs['G'] = W/(sigma*Afr) # Kays and London, eqn 2-30b
        outputs['Re'] = 4*r_h*W/(sigma*Afr*mu)

    def compute_partials(self, inputs, J):
        hex_def = self.options['hex_def']
        side_number = self.options['side_number']
        sigma = getattr(self.options['hex_def'], 'sigma'+str(self.options['side_number']))
        r_h = getattr(self.options['hex_def'], 'r_h'+str(self.options['side_number']))

        W  = inputs['W']
        Afr = inputs['Afr']
        mu = inputs['mu']

        J['G','W'] = 1/Afr/sigma
        J['G','Afr'] = - W/Afr**2/sigma

        J['Re','W'] = 4*r_h/(sigma*Afr*mu)
        J['Re','Afr'] = - 4*r_h*W/sigma/Afr**2/mu
        J['Re','mu'] = - 4*r_h*W/sigma/Afr/mu**2


class HE_side_props(ExplicitComponent):
    """ Calculate Exchanger flow stream mass velocity, Reynolds number, Prahl number, and kinematic visosity"""

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int, desc='Number of nodes to be evaluated')
        self.options.declare('hex_def',default='hex_props',desc='heat exchanger definition')
        self.options.declare('side_number',default='' ,desc= 'side number for lookup reference')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('W', val=np.ones(nn), units="kg/s", desc='Mass flow input')
        self.add_input('Afr', val=np.ones(nn), units='m**2', desc='Frontal Area')
        self.add_input('rho', val=np.ones(nn), units='kg/m**3', desc='fluid density')
        self.add_input('mu', val=np.ones(nn), units='Pa*s', desc='viscosity')
        self.add_input('Cp', val=np.ones(nn), units='J/kg/degK', desc='specific heat with constant pressure')
        self.add_input('k', val=np.ones(nn), units='W/m/degK', desc='thermal conductivity')

        self.add_output('G', val=np.ones(nn), units='kg/(s*m**2)', desc='flow stream mass velocity',lower = 1e-5)
        self.add_output('Re', val=np.ones(nn), desc='Reynolds number',lower = 1)
        self.add_output('Pr', val=np.ones(nn), desc='Prandtl number',lower = 1e-5)
        self.add_output('v', val=np.ones(nn), units='m**2/s', desc='kinematic viscoisty',lower = 1e-9)

        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='G', wrt=['W', 'Afr'], rows=arange, cols=arange)
        self.declare_partials(of='Re', wrt=['W','mu', 'Afr'], rows=arange, cols=arange)
        self.declare_partials(of='Pr', wrt=['Cp','k','mu'], rows=arange, cols=arange)
        self.declare_partials(of='v', wrt=['rho','mu'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        hex_def = self.options['hex_def']
        side_number = self.options['side_number']
        sigma = getattr(self.options['hex_def'], 'sigma'+str(self.options['side_number']))
        r_h = getattr(self.options['hex_def'], 'r_h'+str(self.options['side_number']))

        W  = inputs['W']
        Afr = inputs['Afr']
        mu = inputs['mu']
        rho = inputs['rho']
        Cp = inputs['Cp']
        k = inputs['k']
        outputs['G'] = W/(sigma*Afr) # Kays and London, eqn 2-30b
        outputs['Re'] = 4*r_h*W/(sigma*Afr*mu)
        outputs['Pr'] = mu*Cp/k
        outputs['v'] = mu/rho

    def compute_partials(self, inputs, J):
        hex_def = self.options['hex_def']
        side_number = self.options['side_number']
        sigma = getattr(self.options['hex_def'], 'sigma'+str(self.options['side_number']))
        r_h = getattr(self.options['hex_def'], 'r_h'+str(self.options['side_number']))

        W  = inputs['W']
        Afr = inputs['Afr']
        mu = inputs['mu']
        rho = inputs['rho']
        Cp = inputs['Cp']
        k = inputs['k']

        J['G','W'] = 1/Afr/sigma
        J['G','Afr'] = - W/Afr**2/sigma

        J['Re','W'] = 4*r_h/(sigma*Afr*mu)
        J['Re','Afr'] = - 4*r_h*W/sigma/Afr**2/mu
        J['Re','mu'] = - 4*r_h*W/sigma/Afr/mu**2

        J['Pr','mu'] = Cp/k
        J['Pr','Cp'] = mu/k
        J['Pr','k'] = - mu*Cp/k**2

        J['v','mu'] = 1/rho
        J['v','rho'] = -mu/rho**2


if __name__ == "__main__":

    import time
    from openmdao.api import Problem, IndepVarComp
    from heatsspy.include.HexParams_Regenerator import hex_params_regenerator
    hex_def = hex_params_regenerator()

    prob = Problem()
    Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    # Flow properties
    Vars.add_output('W', val=24.3, units="kg/s", desc='Mass flow input')
    Vars.add_output('Afr', val=2.09, units='m**2', desc='Frontal Area')
    Vars.add_output('rho', val=7.04, units='kg/m**3', desc='density')
    Vars.add_output('mu', val=2.85e-5, units='Pa*s', desc='viscosity')
    Vars.add_output('k', val=0.043, units='W/m/degK', desc='thermal conductivity')
    Vars.add_output('Cp', val=1016, units='J/kg/degK', desc='specific heat with constant pressure')

    Blk1 = prob.model.add_subsystem('prop_calc_Pr_v', HE_side_Pr_v(),
        promotes_inputs=['rho','mu','k','Cp'])

    Blk2 = prob.model.add_subsystem('prop_calc_G_Re', HE_side_G_Re(hex_def=hex_def, side_number=1),
        promotes_inputs=['W','Afr','mu'])

    prob.setup(force_alloc_complex=True)
    prob.run_model()
    prob.check_partials(compact_print=True, method='cs')
    #
    print('G = '+str(prob['prop_calc_G_Re.G'][0]))
    print('Re = '+str(prob['prop_calc_G_Re.Re'][0]))
    print('Pr = '+str(prob['prop_calc_Pr_v.Pr'][0]))
    print('v = '+str(prob['prop_calc_Pr_v.v'][0]))
