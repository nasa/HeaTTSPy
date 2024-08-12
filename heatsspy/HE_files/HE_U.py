from openmdao.api import ExplicitComponent
import numpy as np
class HE_fineff(ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int, desc='Number of nodes to be evaluated')
        self.options.declare('hex_def',default='hex_props', desc='heat exchanger definition')
        self.options.declare('side_number',default='', desc= 'side number for lookup reference')

    def setup(self):
        nn=self.options['num_nodes']
        self.add_input('h',val=np.ones(nn), units='W/m**2/degK',desc='convection coefficient')
        self.add_output('n_0', val=np.ones(nn), desc='fin effectiveness')

        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='n_0', wrt=['h'], rows=arange, cols=arange)

    def compute(self,inputs,outputs):
        hex_def = self.options['hex_def']
        k  = hex_def.k_material
        h_ok = np.where(inputs['h']>0)
        h_low = np.where(inputs['h']<=0)
        h  = inputs['h']
        AfqA  = getattr(self.options['hex_def'], 'AfqA'+str(self.options['side_number']))
        b  = getattr(self.options['hex_def'], 'b'+str(self.options['side_number']))
        delta = getattr(self.options['hex_def'], 'delta'+str(self.options['side_number']))
        outputs['n_0'][h_ok] = 1 - AfqA*(1 - np.tanh(np.sqrt(2*h[h_ok]/k/delta)*b/2)/(np.sqrt(2*h[h_ok]/k/delta)*b/2))
        outputs['n_0'][h_low] = 1 - AfqA*(1 - 2*h[h_low]/k/delta*b/2)

    def compute_partials(self, inputs, J):
        hex_def = self.options['hex_def']
        k  = hex_def.k_material
        h_ok = np.where(inputs['h']>0)[0]
        h_low = np.where(inputs['h']<=0)[0]
        h_array = inputs['h']
        AfqA  = getattr(self.options['hex_def'], 'AfqA'+str(self.options['side_number']))
        b  = getattr(self.options['hex_def'], 'b'+str(self.options['side_number']))
        delta = getattr(self.options['hex_def'], 'delta'+str(self.options['side_number']))
        if len(h_ok)>0:
            h = h_array[h_ok]
            Bterm = b*np.sqrt(h/delta/k)/np.sqrt(2)
            J['n_0','h'][h_ok] = - AfqA *(np.tanh(Bterm)/(np.sqrt(2)*b*(delta*k*(h/delta/k)**(3/2)))-1/np.cosh(Bterm)**2/2/h )

        if len(h_low)>0:
            h = h_array[h_low]
            J['n_0','h'][h_low] = - AfqA*(- 2/k/delta*b/2)


class HE_U(ExplicitComponent):
    """ Calculate coefficient of heat transfer, and heat capacity rates"""

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int, desc='Number of nodes to be evaluated')
        self.options.declare('hex_def',default='hex_props',desc='heat exchanger definition')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('h2', val=np.ones(nn), units='W/m**2/degK', desc='side 2 convection coefficient')
        self.add_input('n2', val=np.ones(nn), desc='side 2 temperature effectiveness or overall surface efficiency')
        self.add_input('h1', val=np.ones(nn), units='W/m**2/degK', desc='side 1 convection coefficient')
        self.add_input('n1', val=np.ones(nn), desc='side 1 temperature effectiveness or overall surface efficiency')

        self.add_output('U2', val=np.ones(nn), units='W/m**2/degK', desc='hot side coefficient of heat transfer',lower = 1e-5)
        self.add_output('U1', val=np.ones(nn), units='W/m**2/degK', desc='cold side coefficient of heat transfer',lower = 1e-5)

        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='U2', wrt=['h2','n2','h1','n1'], rows=arange, cols=arange)
        self.declare_partials(of='U1', wrt=['h2','n2','h1','n1'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        hex_def = self.options['hex_def']
        h2  = inputs['h2']
        n2  = inputs['n2']
        alpha2  = hex_def.alpha2
        h1  = inputs['h1']
        n1  = inputs['n1']
        alpha1  = hex_def.alpha1

        outputs['U2'] = (alpha1*n2*h2*n1*h1)/(alpha2*n2*h2 + alpha1*n1*h1) # Kays and London Eqn 2.2, neglected wall Resistance
        outputs['U1'] = (alpha2*n2*h2*n1*h1)/(n2*h2*alpha2 + n1*h1*alpha1) # Kays and London Eqn 2.2, neglected wall Resistance

    def compute_partials(self, inputs, J):
        hex_def = self.options['hex_def']
        h2  = inputs['h2']
        n2  = inputs['n2']
        alpha2  = hex_def.alpha2
        h1  = inputs['h1']
        n1  = inputs['n1']
        alpha1  = hex_def.alpha1

        J['U2','n2']     = (alpha1**2*h2*n1**2*h1**2)/(alpha2*n2*h2 + alpha1*n1*h1)**2
        J['U2','h2']     = (alpha1**2*n2*n1**2*h1**2)/(alpha2*n2*h2 + alpha1*n1*h1)**2
        J['U2','n1'] = (alpha2*alpha1*n2**2*h2**2*h1)/(alpha2*n2*h2 + alpha1*n1*h1)**2
        J['U2','h1'] = (alpha2*alpha1*n2**2*h2**2*n1)/(alpha2*n2*h2 + alpha1*n1*h1)**2

        J['U1','n2']     = (alpha1*alpha2*h2*n1**2*h1**2)/(n2*h2*alpha2 + n1*h1*alpha1)**2
        J['U1','h2']     = (alpha1*alpha2*n2*n1**2*h1**2)/(n2*h2*alpha2 + n1*h1*alpha1)**2
        J['U1','n1'] = (alpha2**2*n2**2*h2**2*h1)/(n2*h2*alpha2 + n1*h1*alpha1)**2
        J['U1','h1'] = (alpha2**2*n2**2*h2**2*n1)/(n2*h2*alpha2 + n1*h1*alpha1)**2

if __name__ == "__main__":

    import time
    from openmdao.api import Problem, IndepVarComp

    from heatsspy.include.HexParams_Regenerator import hex_params_regenerator
    hex_def = hex_params_regenerator()

    prob = Problem()
    Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])

    Vars.add_output('h2', val=15, units='Btu/(h*ft**2*degR)', desc='hot side convection coefficient')
    Vars.add_output('n2', val=0.887, desc='hot side temperature effectiveness or overall surface efficiency')

    Vars.add_output('h1', val=46.1, units='Btu/(h*ft**2*degR)', desc='cold side convection coefficient')
    Vars.add_output('n1', val=0.786, desc='cold side temperature effectiveness or overall surface efficiency')

    Vars.add_output('h',val=262,units='W/(degK*m**2)' ) # Btu/h/ft/ft/degC

    Blk = prob.model.add_subsystem('prop_calc', HE_U(hex_def=hex_def),
        promotes_inputs=['*'])
    Blk2 = prob.model.add_subsystem('FE',HE_fineff(hex_def=hex_def, side_number=1),
        promotes_inputs=['h'],
        promotes_outputs= ['n_0'])

    # Blk2.set_check_partial_options(wrt='delta', step_calc='rel')
    prob.setup(force_alloc_complex=True)
    prob.run_model()
    prob.check_partials(compact_print=True,method='cs')
    #
    print('U2 = '+str(prob['prop_calc.U2'][0]))
    print('U1 = '+str(prob['prop_calc.U1'][0]))
    print('n_0 = '+str(prob['n_0'][0]))
