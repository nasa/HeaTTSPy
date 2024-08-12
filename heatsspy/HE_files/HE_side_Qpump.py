from openmdao.api import ExplicitComponent, Group
import numpy as np

class HE_side_Qpump(ExplicitComponent):
    """ Estimate required pumping power"""
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int,
                             desc='Number of nodes to be evaluated')
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('W', val=np.ones(nn), units='kg/s', desc='mass flow')
        self.add_input('dP', val=np.ones(nn), units='Pa', desc='pressure rise')
        self.add_input('rho', val=np.ones(nn), units='kg/m**3', desc='fluid density')

        self.add_output('Qpump', val=np.ones(nn), units='J/s', desc='Required pumping power',lower = 1)

        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='Qpump', wrt=['W','dP','rho'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        W  = inputs['W']
        dP = inputs['dP']
        rho = inputs['rho']

        outputs['Qpump'] = W*dP/rho # Cengel eqn 10-62

    def compute_partials(self, inputs, J):
        W  = inputs['W']
        dP = inputs['dP']
        rho = inputs['rho']

        J['Qpump','W'] = dP/rho
        J['Qpump','dP'] = W/rho
        J['Qpump','rho'] = - W*dP/rho**2

class pump_weight_calc(ExplicitComponent):
    """ Estimate weight of the pump"""
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int,
                             desc='Number of nodes to be evaluated')
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('W', val=np.ones(nn), units='lbm/s', desc='mass flow')
        self.add_input('rho', val=np.ones(nn), units='lbm/inch**3', desc='fluid density')

        self.add_output('disp', val=np.ones(nn), units='inch**3/rev', desc='displacement',lower = 1e-5)
        self.add_output('weight', val=np.ones(nn), units='lb', desc='Pump weight',lower = 1e-5)

        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='disp', wrt=['W'], rows=arange, cols=arange)
        self.declare_partials(of='disp', wrt=['rho'], rows=arange, cols=arange)
        self.declare_partials(of='weight', wrt=['W'], rows=arange, cols=arange)
        self.declare_partials(of='weight', wrt=['rho'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        W  = inputs['W']
        rho = inputs['rho']

        V_flow = W / rho * 0.004329 #convert to gallons

        outputs['disp'] = 0.0092 * V_flow **1.3857;
        outputs['weight'] = 8.5942 * outputs['disp'] + 2.4229

    def compute_partials(self, inputs, J):
        W  = inputs['W']
        rho = inputs['rho']

        J['disp','W'] = 1.3857 * 0.0092 * (W/rho * 0.004329)**0.3857 * 0.004329 / rho
        J['disp','rho'] = 1.3857 * 0.0092 * (W/rho * 0.004329)**0.3857 * -0.004329 * W / rho**2
        J['weight','W'] = 8.5942 * 1.3857 * 0.0092 * (W/rho * 0.004329)**0.3857 * 0.004329 / rho
        J['weight','rho'] = 8.5942 * 1.3857 * 0.0092 * (W/rho * 0.004329)**0.3857 * -0.004329 \
                                * W / rho**2

class dP_calc(ExplicitComponent):
    """ Estimate weight of the pump"""
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int,
                             desc='Number of nodes to be evaluated')
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('Pout', val=np.ones(nn), units='Pa', desc='Pressure at pump exit')
        self.add_input('Pin', val=np.ones(nn), units='Pa', desc='Pressure at pump entrance')

        self.add_output('dP', val=np.ones(nn), units='Pa', desc='Pressure difference over pump')

        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='dP', wrt=['Pout'], rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='dP', wrt=['Pin'], rows=arange, cols=arange, val=-1.0)

    def compute(self, inputs, outputs):
        Pout = inputs['Pout']
        Pin = inputs['Pin']

        outputs['dP'] =  Pout - Pin


class HE_pump(Group):
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int,
                             desc='Number of nodes to be evaluated')
        self.options.declare('calc_dP', types=bool, default=False,
                             desc='if True, input P1 and P2, if False, input dP')
        self.options.declare('calc_weight_EN', types=bool, default=True,
                             desc='calculate weight of component')

    def setup(self):
        nn=self.options['num_nodes']

        if self.options['calc_dP']:
            self.add_subsystem('dP', dP_calc(num_nodes=nn),
            promotes_inputs=['Pin','Pout'],
            promotes_outputs=['dP'])

        self.add_subsystem('pump_power', HE_side_Qpump(num_nodes=nn),
            promotes_inputs=['W','dP','rho'],
            promotes_outputs=['Qpump'])

        if self.options['calc_weight_EN']:
            self.add_subsystem('pump_weight', pump_weight_calc(num_nodes=nn),
                                promotes_inputs=['W','rho'],
                                promotes_outputs=[('weight', 'weight_pump')])

if __name__ == "__main__":

    import time
    from openmdao.api import Problem, IndepVarComp

    prob = Problem()
    Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    # Flow properties

    calc_dP=True

    if calc_dP:
        Blk = prob.model.add_subsystem('Qp_calc', HE_pump(calc_dP=calc_dP),
        promotes_inputs=['W','Pout','Pin','rho'])

        Vars.add_output('Pout', val=5000, units='Pa', desc='pressure at pump exit')
        Vars.add_output('Pin', val=100, units='Pa', desc='pressure at pump entrance')

    else:
        Blk = prob.model.add_subsystem('Qp_calc', HE_pump(calc_dP=calc_dP),
        promotes_inputs=['W','dP','rho'])
        Vars.add_output('dP', val=5000, units='Pa', desc='pump dP')

    Vars.add_output('W', val=0.164, units='kg/s', desc='mass flow')
    Vars.add_output('rho', val=999, units='kg/m**3', desc='fluid density')



    # Blk.set_check_partial_options(wrt='*', step_calc='rel')
    prob.setup(force_alloc_complex=True)
    prob.run_model()
    prob.check_partials(compact_print=True,method='cs')
    #
    print('Qpump = '+str(prob['Qp_calc.Qpump'][0]))
    print('weight pump (lbs) = '+str(prob['Qp_calc.weight_pump'][0]))
