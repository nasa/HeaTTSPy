import numpy as np
from scipy.interpolate import Akima1DInterpolator as Interp

from openmdao.api import Group,ExplicitComponent

from heatsspy.api import PassThrough, FlowStart

altitude_vector = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16])*5000/3.2808
Ts_vector = np.array([536.51, 518.67, 500.84, 483.03, 465.22, 447.41, 429.62, 411.84, 394.06, 389.97, 389.97, 389.97, 389.97, 392.25, 397.69])*5/9
Ps_vector = np.array([17.554, 14.696, 12.228, 10.108, 8.297, 6.759, 5.461, 4.373, 3.468, 2.73, 2.149, 1.692, 1.049, 0.651, 0.406])*6894.757

Ts_interp = Interp(altitude_vector, Ts_vector) # akima spline fit
dTs_interp = Ts_interp.derivative()
Ps_interp = Interp(altitude_vector, Ps_vector) # akima spline fit
dPs_interp = Ps_interp.derivative()

class get_static(ExplicitComponent):
    ''' look up Ts and Ps based on dT and altitude'''
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int, desc='Number of nodes to be evaluated')
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('Alt', val=np.ones(nn), units='m', desc='altitude')
        self.add_input('dT', val=np.ones(nn), units='degC', desc='off standard temperature')
        self.add_output('Ts', val=np.ones(nn), units='degK', desc='static temperature')
        self.add_output('Ps', val=np.ones(nn), units='Pa', desc='static pressure')
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials('Ts', ['Alt','dT'], rows=arange, cols=arange)
        self.declare_partials('Ps', 'Alt', rows=arange, cols=arange)
    def compute(self,inputs,outputs):
        Alt=inputs['Alt']
        dT=inputs['dT']
        outputs['Ts']= Ts_interp(Alt) + dT
        outputs['Ps']= Ps_interp(Alt)
    def compute_partials(self, inputs, J):
        Alt=inputs['Alt']
        dT=inputs['dT']
        J['Ts','Alt'] = dTs_interp(Alt)
        J['Ts','dT'] = 1
        J['Ps','Alt'] = dPs_interp(Alt)

class calc_Pt(ExplicitComponent):
    ''' calculate total pressure'''
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int, desc='Number of nodes to be evaluated')
        self.options.declare('gamma',default=1.4)
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('MN', val=np.ones(nn), units=None, desc='mach number')
        self.add_input('Ps', val=np.ones(nn), units='Pa', desc='static pressure')
        self.add_output('Pt', val=np.ones(nn), units='Pa', desc='total pressure')
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials('Pt', ['MN','Ps'], rows=arange, cols=arange)
    def compute(self,inputs,outputs):
        gamma=self.options['gamma']
        outputs['Pt'] = inputs['Ps']/(1+(gamma-1)/2*inputs['MN']**2)**(-gamma/(gamma-1))
    def compute_partials(self, inputs, J):
        g=self.options['gamma']
        J['Pt','Ps'] = 1/(1+(g-1)/2*inputs['MN']**2)**(-g/(g-1))
        J['Pt','MN'] = g*inputs['MN']*(1 + 0.5*(g - 1)*inputs['MN']**2)**(1/(-1 + g))*inputs['Ps']


class calc_Tt(ExplicitComponent):
    ''' calculate total temperature'''
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int, desc='Number of nodes to be evaluated')
        self.options.declare('gamma',default=1.4)
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('MN', val=np.ones(nn), units=None, desc='mach number')
        self.add_input('Ts', val=np.ones(nn), units='degK', desc='static temperature')
        self.add_output('Tt', val=np.ones(nn), units='degK', desc='total temperature')
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials('Tt', ['MN','Ts'], rows=arange, cols=arange)
    def compute(self,inputs,outputs):
        gamma=self.options['gamma']
        outputs['Tt'] = inputs['Ts']/(1+(gamma-1)/2*inputs['MN']**2)**-1
    def compute_partials(self, inputs, J):
        g=self.options['gamma']
        J['Tt','Ts'] = 1/(1+(g-1)/2*inputs['MN']**2)**-1
        J['Tt','MN'] = (g-1)*inputs['MN']*inputs['Ts']


class calc_a(ExplicitComponent):
    ''' calculate speed of sound'''
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int, desc='Number of nodes to be evaluated')
        self.options.declare('gamma',default=1.4)
        self.options.declare('R',default=286.9) # J/kg K
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('Ts', val=np.ones(nn), units='degK', desc='static temperature')
        self.add_output('a', val=np.ones(nn), units='m/s', desc='speed of sound')
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials('a', ['Ts'], rows=arange, cols=arange)
    def compute(self,inputs,outputs):
        gam=self.options['gamma']
        R=self.options['R']
        Ts=inputs['Ts']
        outputs['a']=np.sqrt(gam*R*Ts)
    def compute_partials(self, inputs, J):
        gam=self.options['gamma']
        R=self.options['R']
        Ts=inputs['Ts']
        J['a','Ts'] = gam*R/2/np.sqrt(gam*R*Ts)


class calc_v(ExplicitComponent):
    ''' calculate velocity'''
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int,desc='Number of nodes to be evaluated')
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('MN', val=np.ones(nn), units=None, desc='Mach number')
        self.add_input('a', val=np.ones(nn), units='m/s', desc='speed of sound')
        self.add_output('v', val=np.ones(nn), units='m/s', desc='velocity')
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials('v', ['a','MN'], rows=arange, cols=arange)
    def compute(self,inputs,outputs):
        MN=inputs['MN']
        a=inputs['a']
        outputs['v']=MN*a
    def compute_partials(self, inputs, J):
        MN=inputs['MN']
        a=inputs['a']
        J['v','MN']=a
        J['v','a']=MN


class calc_Fd(ExplicitComponent):
    ''' calculate drag thrust'''
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int, desc='Number of nodes to be evaluated')
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('mdot', val=np.ones(nn), units='kg/s', desc='mass flow')
        self.add_input('v', val=np.ones(nn), units='m/s', desc='velocity')
        self.add_output('Fd',val=np.ones(nn),units='N',desc='gross thrust')
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials('Fd', ['v','mdot'], rows=arange, cols=arange)
    def compute(self,inputs,outputs):
        outputs['Fd']=inputs['mdot']*inputs['v']
    def compute_partials(self, inputs, J):
        J['Fd','v']=inputs['mdot']
        J['Fd','mdot']=inputs['v']


class isen_ambient(Group):
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int, desc='Number of nodes to be evaluated')
        self.options.declare('gamma', default=1.4, types=float, desc='specific heat ratio value')
        self.options.declare('R', default=286.9, desc='gas constant') # J/kg K
        self.options.declare('fluid', default='water', desc='fluid type')
        self.options.declare('unit_type', default='SI', desc='output unit type', values=('SI', 'ENG', 'IMP'))
        self.options.declare('thermo', default='file', desc='output unit type')
    def setup(self):
        nn = self.options['num_nodes']
        gamma = self.options['gamma']
        R = self.options['R']
        self.add_subsystem('Alt', PassThrough("input", "in", val=np.ones(nn), units='m'))
        self.add_subsystem('MN', PassThrough("input", "in", val=np.ones(nn), units=None))
        self.add_subsystem('dT', PassThrough("input", "in", val=np.ones(nn), units='degC'))
        self.add_subsystem('mdot', PassThrough("input", "in", val=np.ones(nn), units='kg/s'))

        self.add_subsystem('get_static', get_static(num_nodes=nn))
        self.connect('Alt.in','get_static.Alt')
        self.connect('dT.in','get_static.dT')

        self.add_subsystem('calc_Pt', calc_Pt(num_nodes=nn, gamma=gamma))
        self.connect('get_static.Ps','calc_Pt.Ps')
        self.connect('MN.in','calc_Pt.MN')

        self.add_subsystem('calc_Tt', calc_Tt(num_nodes=nn, gamma=gamma))
        self.connect('get_static.Ts','calc_Tt.Ts')
        self.connect('MN.in','calc_Tt.MN')

        self.add_subsystem('calc_a', calc_a(num_nodes=nn, gamma=gamma, R=R))
        self.connect('get_static.Ts','calc_a.Ts')

        self.add_subsystem('calc_v', calc_v(num_nodes=nn))
        self.connect('MN.in','calc_v.MN')
        self.connect('calc_a.a','calc_v.a')

        self.add_subsystem('calc_Fd', calc_Fd(num_nodes=nn))
        self.connect('mdot.in','calc_Fd.mdot')
        self.connect('calc_v.v','calc_Fd.v')

        self.add_subsystem('FS_outputs', FlowStart(unit_type=self.options['unit_type'], fluid=self.options['fluid'],
                                                      thermo=self.options['thermo'], num_nodes=nn),promotes_outputs=['*'])
        self.connect('calc_Pt.Pt', 'FS_outputs.P')
        self.connect('calc_Tt.Tt', 'FS_outputs.T')
        self.connect('mdot.in', 'FS_outputs.W')

        self.add_subsystem('Pamb', PassThrough("out", "output", val=np.ones(nn), units='Pa'))
        self.connect('get_static.Ps', 'Pamb.out')
        self.add_subsystem('Fd', PassThrough("out", "output", val=np.ones(nn), units='N'))
        self.connect('calc_Fd.Fd', 'Fd.out')


if __name__ == "__main__":

    import time
    from openmdao.api import Problem, Group, IndepVarComp, ExplicitComponent
    from openmdao.api import BalanceComp, DirectSolver, BoundsEnforceLS, NewtonSolver

    from heatsspy.include.props_air import air_props

    prob = Problem()

    model = prob.model
    newton = model.nonlinear_solver = NewtonSolver()
    newton.options['atol'] = 1e-6
    newton.options['rtol'] = 1e-10
    newton.options['iprint'] = -1
    newton.options['maxiter'] = 10
    newton.options['solve_subsystems'] = True
    newton.options['max_sub_solves'] = 1001
    newton.linesearch = BoundsEnforceLS()
    # newton.linesearch.options['maxiter'] = 1
    newton.linesearch.options['bound_enforcement'] = 'scalar'
    newton.linesearch.options['iprint'] = 2

    model.linear_solver = DirectSolver(assemble_jac=True)

    Vars =  prob.model.add_subsystem('Vars',IndepVarComp())
    # Flow properties
    Vars.add_output('MN',val=[0.0, 0.8],units=None)
    Vars.add_output('Alt',val=[0, 35000],units='ft')
    Vars.add_output('dT',val=[0, 0],units='degC') # this needs to be in deg C, note 27 dF = 15 degC
    Vars.add_output('mdot',val=[1, 2],units='kg/s')

    prob.model.add_subsystem('AE', isen_ambient(num_nodes=2, fluid=air_props()))
    prob.model.connect('Vars.Alt', 'AE.Alt.input')
    prob.model.connect('Vars.MN', 'AE.MN.input')
    prob.model.connect('Vars.dT', 'AE.dT.input')
    prob.model.connect('Vars.mdot', 'AE.mdot.input')
    prob.setup()

    # prob.setup(force_alloc_complex=True)
    prob.setup()
    prob.run_model()
    # prob.check_partials(compact_print=True,method='fd')
    print('Ts ='+str(prob.get_val('AE.get_static.Ts', units='degC')))
    print('Ps ='+str(prob.get_val('AE.get_static.Ps', units='Pa')))
    print('Tt ='+str(prob.get_val('AE.calc_Tt.Tt', units='degC')))
    print('Pt ='+str(prob.get_val('AE.Fl_O:tot:P', units='Pa')))
    print('velocity ='+str(prob.get_val('AE.calc_v.v', units='m/s')))
    print('Pamb ='+str(prob.get_val('AE.Pamb.output', units='Pa')))
