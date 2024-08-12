from openmdao.api import ExplicitComponent, Group
import numpy as np

class HE_Wt_sp(ExplicitComponent):
    ''' Estimate heat exchanger weight based on specific power'''
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int,desc='Number of nodes to be evaluated')
        self.options.declare('specific_power', default=0.5, desc='heat exchanger specific power (rejected power / weight), kW/kg')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('q', val=np.ones(nn), units='kW', desc='heat transfer rate')
        self.add_output('Wt', val=np.ones(nn), units='kg', desc='heat exchanger weight')

        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='Wt', wrt=['q'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        specific_power = self.options['specific_power']
        q = inputs['q']
        outputs['Wt'] = np.abs(q)/specific_power

    def compute_partials(self, inputs, J):
        specific_power = self.options['specific_power']
        q = inputs['q']
        J['Wt','q'] = q/specific_power/np.abs(q)


class HE_Wt(ExplicitComponent):
    """ Estimate heat exchanger weight based on heat exchanger size"""
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int,desc='Number of nodes to be evaluated')
        self.options.declare('dim',default=2, desc='flow dimension',values=(1, 2))
        self.options.declare('hex_def',default='hex_props',desc='heat exchanger definition')

    def setup(self):
        nn = self.options['num_nodes']
        dim = self.options['dim']
        if dim ==1:
            self.add_input('Afr', val=np.ones(nn), units='m**2', desc='frontal area')
            self.add_input('L', val=np.ones(nn), units='m', desc='length')
            self.add_input('rho_cool', np.ones(nn), units='kg/m**3', desc='density of coolant1')
        else:
            self.add_input('Afr1', val=np.ones(nn), units='m**2', desc='side 1 frontal area')
            self.add_input('L1', val=np.ones(nn), units='m', desc='side 1 length')
            self.add_input('Afr2', val=np.ones(nn), units='m**2', desc='side 2 frontal area')
            self.add_input('L2', val=np.ones(nn), units='m', desc='side 2 length')
            self.add_input('rho_cool1', np.ones(nn), units='kg/m**3', desc='density of coolant1')
            self.add_input('rho_cool2', np.ones(nn), units='kg/m**3', desc='density of coolant2')

        self.add_output('Wt', val=np.ones(nn), units='kg', desc='heat exchanger weight',lower = 1e-2)
        arange = np.arange(self.options['num_nodes'])
        if dim ==1:
            self.declare_partials(of='Wt', wrt=['Afr','L','rho_cool'], rows=arange, cols=arange)
        else:
            self.declare_partials(of='Wt', wrt=['rho_cool1', 'rho_cool2','Afr1','L1','Afr2','L2', 'rho_cool1', 'rho_cool2'], rows=arange, cols=arange)


    def compute(self, inputs, outputs):
        dim = self.options['dim']
        hex_def = self.options['hex_def']
        rho = hex_def.rho_material

        if dim == 1:
            sigma  = hex_def.sigma
            Afr  = inputs['Afr']
            L = inputs['L']
            rho_cool = inputs['rho_cool']
            outputs['Wt'] = Afr*(1-sigma)*L*rho + Afr*sigma*L*rho_cool
        else:
            sigma1  = hex_def.sigma1
            Afr1  = inputs['Afr1']
            L1 = inputs['L1']
            sigma2  = hex_def.sigma2
            Afr2  = inputs['Afr2']
            L2 = inputs['L2']
            rho_cool1 = inputs['rho_cool1']
            rho_cool2 = inputs['rho_cool2']
            # Wt = (Volume side one - free flow Volume side two) * density
            outputs['Wt'] = (Afr1*(1-sigma1)*L1 - sigma2*Afr2*L2)*rho + (sigma1*Afr1*L1)*rho_cool1 \
                                + (sigma2*Afr2*L2)*rho_cool2

    def compute_partials(self, inputs, J):
        dim = self.options['dim']
        hex_def = self.options['hex_def']
        rho = hex_def.rho_material

        if dim == 1:
            sigma  = hex_def.sigma
            Afr  = inputs['Afr']
            L = inputs['L']
            rho_cool = inputs['rho_cool']
            J['Wt','Afr'] = (1-sigma)*L*rho + sigma*L*rho_cool
            J['Wt','L'] = Afr*(1-sigma)*rho + Afr*sigma*rho_cool
            J['Wt','rho_cool'] = Afr*sigma*L
        else:
            sigma1  = hex_def.sigma1
            Afr1  = inputs['Afr1']
            L1 = inputs['L1']
            sigma2  = hex_def.sigma2
            Afr2  = inputs['Afr2']
            L2 = inputs['L2']
            rho_cool1 = inputs['rho_cool1']
            rho_cool2 = inputs['rho_cool2']

            J['Wt','Afr1'] = (1-sigma1)*L1*rho + (sigma1*L1*rho_cool1)
            J['Wt','L1'] = Afr1*(1-sigma1)*rho + (sigma1*Afr1*rho_cool1)
            J['Wt','rho_cool1'] = (sigma1*Afr1*L1)

            J['Wt','Afr2'] = - sigma2*L2*rho + (sigma2*L2*rho_cool2)
            J['Wt','L2'] = - sigma2*Afr2*rho + (sigma2*Afr2*rho_cool2)
            J['Wt','rho_cool2'] = (sigma2*Afr2*L2)


if __name__ == "__main__":

    import time
    from openmdao.api import Problem, IndepVarComp
    from heatsspy.include.HexParams_Regenerator import hex_params_regenerator
    hex_def = hex_params_regenerator()

    prob = Problem()
    Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    # Flow properties
    Vars.add_output('Afr', val=1, units='m**2', desc='face area')
    Vars.add_output('Afr1', val=0.29463, units='m**2', desc='face area side one')
    Vars.add_output('Afr2', val=0.99125, units='m**2', desc='face area side two')
    Vars.add_output('L', val=2, units='m', desc='length')
    Vars.add_output('L1', val=1.625, units='m', desc='side 1 length')
    Vars.add_output('L2', val=0.4830, units='m', desc='side 2 length')
    Vars.add_output('rho_cool', val=997, units='kg/m**3', desc='side 1 length')
    Vars.add_output('rho_cool1', val=1000, units='kg/m**3', desc='side 1 length')
    Vars.add_output('rho_cool2', val=1100, units='kg/m**3', desc='side 2 length')
    Vars.add_output('q', val=1.1, units='kW', desc='heat rejected')

    # Blk1 = prob.model.add_subsystem('Wt_calc1', HE_Wt(dim=1, hex_def=),
    #     promotes_inputs=['sigma','Afr','L','rho_cool'])

    Blk2 = prob.model.add_subsystem('Wt_calc2', HE_Wt(dim=2, hex_def=hex_def),
        promotes_inputs=['Afr1','L1','rho_cool1','Afr2','L2','rho_cool2'])

    Blk3 = prob.model.add_subsystem('Wt_sp', HE_Wt_sp(),
        promotes_inputs=['q'])

    # Blk.set_check_partial_options(wrt='*', step_calc='rel')
    prob.setup(force_alloc_complex=True)
    prob.run_model()
    prob.check_partials(compact_print=True,method='fd')
    #
    # print('Wt1 = '+str(prob['Wt_calc1.Wt'][0]))
    print('Wt2 = '+str(prob['Wt_calc2.Wt'][0]))

    print('Wt3 = ', prob.get_val('Wt_sp.Wt')[0])
