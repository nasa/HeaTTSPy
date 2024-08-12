from openmdao.api import ExplicitComponent
import numpy as np

class HE_side_out_P(ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int, desc='Number of nodes to be evaluated')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('dPqP', val=np.ones(nn), units=None, desc='change in pressure / input pressure')
        self.add_input('P_in', val=np.ones(nn), units='Pa' ,desc='input pressure')

        self.add_output('dP', val=np.ones(nn), units='Pa', desc='change in pressure',lower = 1)
        self.add_output('P_out', val=np.ones(nn), units='Pa', desc='output pressure',lower = 1)

        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of=['dP','P_out'], wrt=['P_in','dPqP'], rows=arange, cols=arange)

    def compute(self,inputs,outputs):
        outputs['dP']=inputs['dPqP']*inputs['P_in']
        outputs['P_out'] = inputs['P_in']*(1 - inputs['dPqP'])

    def compute_partials(self, inputs, J):
        J['dP','dPqP'] = inputs['P_in']
        J['dP','P_in'] = inputs['dPqP']

        J['P_out','dPqP'] = - inputs['P_in']
        J['P_out','P_in'] = 1 - inputs['dPqP']


class HE_side_out_dP(ExplicitComponent):
    """ Calculate output pressure"""

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int, desc='Number of nodes to be evaluated')
        self.options.declare('rho_const_EN', default=False, types=bool)
        self.options.declare('hex_def',default='hex_props',desc='heat exchanger definition')
        self.options.declare('side_number',default='' ,desc= 'side number for lookup reference')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('f', val=np.ones(nn), desc='friction factor')
        self.add_input('G', val=np.ones(nn), units='lbm/(h*ft**2)', desc='flow stream mass velocity')
        self.add_input('L', val=np.ones(nn), units='ft', desc='flow length')
        self.add_input('P_in', val=np.ones(nn), units='psi', desc='input pressure')
        if self.options['rho_const_EN']:
        	#Note: rho_in = rho_out for incompressible fluids
        	self.add_input('rho', val=np.ones(nn), units='lbm/ft**3', desc='constant fluid density')
        else:
	        self.add_input('rho_in', val=np.ones(nn), units='lbm/ft**3', desc='input fluid density')
	        self.add_input('rho_out', val=np.ones(nn), units='lbm/ft**3', desc='output fluid density')

        self.add_output('dP', val=np.ones(nn), units='psi', desc='change in pressure',lower = 1)
        self.add_output('dPqP', val=np.ones(nn), units=None, desc='change in pressure / input pressure',lower = 1e-5)
        self.add_output('P_out', val=np.ones(nn), units='psi', desc='output pressure',lower = 1)

        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of=['dP'], wrt=['f','G', 'L'], rows=arange, cols=arange)
        self.declare_partials(of=['dPqP','P_out'], wrt=['f','G','P_in', 'L'], rows=arange, cols=arange)
        if self.options['rho_const_EN']:
            self.declare_partials(of=['dP'], wrt=['rho'], rows=arange, cols=arange)
            self.declare_partials(of=['dPqP','P_out'], wrt=['rho'], rows=arange, cols=arange)
        else:
            self.declare_partials(of=['dP'], wrt=['rho_in','rho_out'], rows=arange, cols=arange)
            self.declare_partials(of=['dPqP','P_out'], wrt=['rho_in','rho_out'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        hex_def=self.options['hex_def']
        K_c = getattr(self.options['hex_def'], 'K_c'+str(self.options['side_number']))
        K_e = getattr(self.options['hex_def'], 'K_e'+str(self.options['side_number']))
        f  = inputs['f']
        G  = inputs['G']
        L  = inputs['L']
        P_in  = inputs['P_in']
        r_h  = getattr(self.options['hex_def'], 'r_h'+str(self.options['side_number'])) * 3.28084
        sigma  = getattr(self.options['hex_def'], 'sigma'+str(self.options['side_number']))

        if self.options['rho_const_EN']:
            rho_in  = rho_out =  inputs['rho']
        else:
            rho_in  = inputs['rho_in']
            rho_out  = inputs['rho_out']
        # Kays and London Eqn: 2-26
        # dPqP = Velocity term per P_in * [ entrance effect + flow acceleration + core friction - exit effect]
        # Note: when K_e and K_c are zero and rho_in = rho_out the equation simplifies greatly to utilize flow friction component only
        rpr = rho_in/rho_out
        outputs['dPqP'] = dPqP = (G/3600)**2/(rho_in*2*P_in*144*32.174) * ((K_c+1-sigma**2) + 2*(rpr-1) + f*L/r_h*rho_in*(1/rho_in + 1/rho_out)/2 - (1-sigma**2-K_e)*rpr)
        outputs['dP'] = dP = dPqP*P_in
        outputs['P_out'] = P_in - dP
        # if 'AOCE.dP2_calc' in self.pathname:
        # if np.any(outputs['dPqP']<1):
        # print(self.pathname, ' : dPqP = ', outputs['dPqP'])
        # print(self.pathname, ' : P = ', P_in)
        # print(self.pathname, ' : G = ', G)
        # print(self.pathname, ' : rho = ', rho_in)

    def compute_partials(self, inputs, J):
        hex_def=self.options['hex_def']
        K_c = getattr(self.options['hex_def'], 'K_c'+str(self.options['side_number']))
        K_e = getattr(self.options['hex_def'], 'K_e'+str(self.options['side_number']))
        f  = inputs['f']
        G  = inputs['G']
        L  = inputs['L']
        P_in  = inputs['P_in']
        r_h  = getattr(self.options['hex_def'], 'r_h'+str(self.options['side_number'])) * 3.28084
        sigma  = getattr(self.options['hex_def'], 'sigma'+str(self.options['side_number']))

        C = (1/3600)**2/2/144/32.174 # for unit conversion

        if self.options['rho_const_EN']:
            rho = inputs['rho']
        else:
            rho_in  = inputs['rho_in']
            rho_out  = inputs['rho_out']
            rpr = rho_in/rho_out
        # C*G**2/rho_in * ((K_c+1-sigma**2) + 2*(rpr-1) + f*L/r_h*rho_in*(1/rho_in + 1/rho_out)/2 - (1-sigma**2-K_e)*rpr)
        # C*G**2/rho * ((K_c+1-sigma**2) + f*L/r_h - (1-sigma**2-K_e))
        if self.options['rho_const_EN']:
            J['dP','f'] = dPqdf = C*G**2/rho * L/r_h
            J['dP','G'] = dPqdG = 2*C*G/rho * ((K_c+1-sigma**2) + f*L/r_h - (1-sigma**2-K_e))
            J['dP','L'] = dPqdL = C*G**2/rho * f/r_h
            J['dP','rho'] = dPqdrho = - C*G**2/rho**2 * ((K_c+1-sigma**2) + f*L/r_h - (1-sigma**2-K_e))
        else:
            J['dP','f'] = dPqdf = C*G**2/rho_in * (L/r_h*rho_in*(1/rho_in + 1/rho_out)/2)
            J['dP','G'] = dPqdG = 2*C*G/rho_in * ((K_c+1-sigma**2) + 2*(rpr-1) + f*L/r_h*rho_in*(1/rho_in + 1/rho_out)/2 - (1-sigma**2-K_e)*rpr)
            J['dP','L'] = dPqdL = C*G**2/rho_in * (f/r_h*rho_in*(1/rho_in + 1/rho_out)/2)
            J['dP','rho_in'] = dPqdrho_in = C*G**2 * (-(K_c+1-sigma**2)/rho_in**2 + 2/rho_in**2 + f*L/r_h*(-1/rho_in**2)/2)

            J['dP','rho_out'] = dPqdrho_out = C*G**2/rho_in * (2*(-rho_in/rho_out**2) + f*L/r_h*rho_in*(- 1/rho_out**2)/2 + (1-sigma**2-K_e)*rho_in/rho_out**2)

        J['dPqP','f']  = dPqdf/P_in
        J['dPqP','G'] = dPqdG/P_in
        J['dPqP','L'] = dPqdL/P_in

        if self.options['rho_const_EN']:
            J['dPqP','P_in'] = - C*G**2/rho * ((K_c+1-sigma**2) + f*L/r_h - (1-sigma**2-K_e))/P_in**2
            J['dPqP','rho'] = dPqdrho/P_in
        else:
            J['dPqP','P_in'] = - C*G**2/rho_in * ((K_c+1-sigma**2) + 2*(rpr-1) + f*L/r_h*rho_in*(1/rho_in + 1/rho_out)/2 - (1-sigma**2-K_e)*rpr)/P_in**2
            J['dPqP','rho_in'] = dPqdrho_in/P_in
            J['dPqP','rho_out'] = dPqdrho_out/P_in

        J['P_out','f']  = - dPqdf
        J['P_out','G'] = - dPqdG
        J['P_out','L'] = - dPqdL
        J['P_out','P_in'] = 1
        if self.options['rho_const_EN']:
            J['P_out','rho'] = - dPqdrho
        else:
            J['P_out','rho_in'] = - dPqdrho_in
            J['P_out','rho_out'] = - dPqdrho_out


if __name__ == "__main__":

    import time
    from openmdao.api import Problem, IndepVarComp
    from heatsspy.include.HexParams_PlateFin import hex_params_platefin
    from heatsspy.include.HexParams_Regenerator import hex_params_regenerator
    # PF_def = hex_params_platefin()
    PF_def = hex_params_regenerator()
    prob = Problem()
    nn = 2
    Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    # Flow properties
    Vars.add_output('f', val=[0.0375, 0.0155], desc='friction factor')
    Vars.add_output('G', val=[19300, 9850], units='lbm/(h*ft**2)', desc='flow stream mass velocity')
    Vars.add_output('L', val=[6.0, 3.0], units='ft', desc='flow length')
    Vars.add_output('P_in', val=[132, 14.9], units='psi', desc='input pressure')
    Vars.add_output('rho_in', val=[0.438596, 0.0315], units='lbm/ft**3', desc='input fluid density')
    Vars.add_output('rho_out', val=[0.3003, 0.04237], units='lbm/ft**3', desc='output fluid density')

    Vars.add_output('dPqP', val=[0.01, 0.01], units=None, desc='change in pressure/ pressure')

    Vars.add_output('rho', val=[4, 4], units='kg/m**3', desc='constant fluid density')
    # Vars.add_output('rho', val=997, units='kg/m**3', desc='viscosity')

    Blk = prob.model.add_subsystem('prop_calc', HE_side_out_dP(num_nodes=nn, hex_def=PF_def, side_number=1),
        promotes_inputs=['*'])

    Blk = prob.model.add_subsystem('prop_calc2', HE_side_out_dP(rho_const_EN=True, num_nodes=nn, hex_def=PF_def, side_number=1),
        promotes_inputs=['*'])

    prob.model.add_subsystem('PoutCalc', HE_side_out_P(num_nodes=nn),
        promotes_inputs=['*'])

    prob.setup(force_alloc_complex=True)
    prob.run_model()
    # prob.check_partials(compact_print=True, method='cs')
    #
    print('dP = '+str(prob['prop_calc.dP']))
    print('dPqP = '+str(100*prob['prop_calc.dPqP']))
    print('P_out = '+str(prob['prop_calc.P_out']))

    print('P_out2 = '+str(prob['prop_calc2.P_out']))

    print('dP = '+str(prob['PoutCalc.dP']))
    print('P_out = '+str(prob['PoutCalc.P_out']))
