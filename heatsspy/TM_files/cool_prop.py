from openmdao.api import ExplicitComponent, Group, ExecComp
import numpy as np
try:
	import CoolProp.CoolProp as CP
except ImportError:
    CP = None

class cool_prop(ExplicitComponent):
    def initialize(self):
        self.options.declare('fluid', desc='fluid type')
        self.options.declare('mode',
                              desc='the input variable that defines the total properties',
                              default='T',
                              values=('T', 'S', 'H'))
        self.options.declare('num_nodes', default=1, types=int,
                             desc='Number of nodes to be evaluated')

    def setup(self):
        nn = self.options['num_nodes']
        mode = self.options['mode']
        fluid = self.options['fluid']

        self.add_input('Pset', val=np.ones(nn), units="Pa", desc='Pressure set')
        if mode == 'T':
            self.add_input('Tset', val=np.ones(nn), units='degK', desc='Temperature set')
        elif mode == 'S':
            self.add_input('Sset', val=np.ones(nn), units='J/kg/degK', desc='Entropy set')
        elif mode == 'H':
            self.add_input('Hset', val=np.ones(nn), units='J/kg', desc='Enthlapy set')

        self.add_output('P', val=np.ones(nn), units="Pa", desc='Pressure')
        self.add_output('T', val=np.ones(nn), units='degK', desc='Temperature')
        self.add_output('S', val=np.ones(nn), units='J/kg/degK', desc='Entropy')
        self.add_output('H', val=np.ones(nn), units='J/kg', desc='Enthlapy')
        self.add_output('rho', val=np.ones(nn), units='kg/m**3', desc='density')
        self.add_output('Cp', val=np.ones(nn), units='J/kg/degK', desc='specific heat with constant pressure')
        self.add_output('Cv', val=np.ones(nn), units='J/kg/degK', desc='specific heat with constant volume')
        self.add_output('mu', val=np.ones(nn), units='Pa*s', desc='viscosity')
        self.add_output('k', val=np.ones(nn), units='W/m/degK', desc='thermal conductivity')

        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='P', wrt='Pset', val=1.0, rows=arange, cols=arange)
        self.declare_partials(of=['rho','Cp','Cv'], wrt='Pset', rows=arange, cols=arange)
        self.declare_partials(of='mu', wrt='Pset', method='fd')
        self.declare_partials(of='k', wrt='Pset', method='fd')
        if mode == 'T':
            self.declare_partials(of='P', wrt='Tset', val =0.0, rows=arange, cols=arange)
            self.declare_partials(of='T', wrt='Pset', val =0.0, rows=arange, cols=arange)
            self.declare_partials(of='T', wrt='Tset', val =1.0, rows=arange, cols=arange)
            self.declare_partials(of=['S','H'], wrt=['Pset','Tset'], rows=arange, cols=arange)

        elif mode == 'S':
            self.declare_partials(of='P', wrt='Sset', val =0.0, rows=arange, cols=arange)
            self.declare_partials(of='S', wrt='Pset', val =0.0, rows=arange, cols=arange)
            self.declare_partials(of='S', wrt='Sset', val =1.0, rows=arange, cols=arange)
            self.declare_partials(of=['H','T'], wrt=['Pset','Sset'], rows=arange, cols=arange)

        elif mode == 'H':
            self.declare_partials(of='P', wrt='Hset', val =0.0, rows=arange, cols=arange)
            self.declare_partials(of='H', wrt='Pset', val =0.0, rows=arange, cols=arange)
            self.declare_partials(of='H', wrt='Hset', val =1.0, rows=arange, cols=arange)
            self.declare_partials(of=['T','S'], wrt=['Pset','Hset'], rows=arange, cols=arange)

        self.declare_partials(of=['rho','Cp','Cv'], wrt=f'{mode}set', rows=arange, cols=arange)
        # fd is used to compute partials, this should be looked at in the future
        self.declare_partials(of='mu', wrt=f'{mode}set', method='fd')
        self.declare_partials(of='k', wrt=f'{mode}set', method='fd')

    def compute(self, inputs, outputs):
        mode = self.options['mode']
        fluid = self.options['fluid']

        Pset  = inputs['Pset']
        if mode == 'T':
            Tset  = inputs['Tset']
        elif mode == 'S':
            Sset  = inputs['Sset']
        elif mode == 'H':
            Hset  = inputs['Hset']
        set_value = 0

        outputs['P'] = Pset
        if mode == 'T':
            Tset[Tset<280]=280
            Tset[Tset>450]=450
            # print(self.pathname,'T = ',Tset)
            outputs['T'] = set_value = Tset
            outputs['S'] = CP.PropsSI('Smass','P',Pset,'T',Tset,fluid)
            outputs['H'] = CP.PropsSI('Hmass','P',Pset,'T',Tset,fluid)
        elif mode == 'S':
            outputs['T'] = CP.PropsSI('T','P',Pset,'S',Sset,fluid)
            outputs['S'] = set_value = Sset
            outputs['H'] = CP.PropsSI('Hmass','P',Pset,'S',Sset,fluid)
        elif mode == 'H':
            outputs['T'] = CP.PropsSI('T','P',Pset,'H',Hset,fluid)
            outputs['S'] = CP.PropsSI('Smass','P',Pset,'H',Hset,fluid)
            outputs['H'] = set_value = Hset

        outputs['rho'] = CP.PropsSI('Dmass','P',Pset,f'{mode}',set_value,fluid)
        outputs['Cp'] = CP.PropsSI(f'd(Hmass)/d(T)|P','P',Pset,f'{mode}',set_value,fluid)
        outputs['Cv'] = CP.PropsSI(f'd(Umass)/d(T)|Dmass','P',Pset,f'{mode}',set_value,fluid)
        outputs['mu'] = CP.PropsSI('viscosity','P',Pset,f'{mode}',set_value,fluid)
        outputs['k'] = CP.PropsSI('conductivity','P',Pset,f'{mode}',set_value,fluid)

    def compute_partials(self, inputs, J):
        mode = self.options['mode']
        fluid = self.options['fluid']

        Pset = inputs['Pset']
        if mode == 'T':
            Tset = inputs['Tset']
        elif mode == 'S':
            Sset = inputs['Sset']
        elif mode == 'H':
            Hset = inputs['Hset']
        set_value = 0

        if mode == 'T':
            J['S','Pset'] = CP.PropsSI('d(Smass)/d(P)|T','P',Pset,'T',Tset,fluid)
            J['S','Tset'] = CP.PropsSI('d(Smass)/d(T)|P','P',Pset,'T',Tset,fluid)
            J['H','Pset'] = CP.PropsSI('d(Hmass)/d(P)|T','P',Pset,'T',Tset,fluid)
            J['H','Tset'] = CP.PropsSI('d(Hmass)/d(T)|P','P',Pset,'T',Tset,fluid)
            set_value = Tset
        elif mode == 'S':
            J['T','Pset'] = CP.PropsSI('d(T)/d(P)|S','P',Pset,'S',Sset,fluid)
            J['T','Sset'] = CP.PropsSI('d(T)/d(S)|P','P',Pset,'S',Sset,fluid)
            J['H','Pset'] = CP.PropsSI('d(Hmass)/d(P)|S','P',Pset,'S',Sset,fluid)
            J['H','Sset'] = CP.PropsSI('d(Hmass)/d(S)|P','P',Pset,'S',Sset,fluid)
            set_value = Sset
        elif mode == 'H':
            J['T','Pset'] = CP.PropsSI('d(T)/d(P)|H','P',Pset,'H',Hset,fluid)
            J['T','Hset'] = CP.PropsSI('d(T)/d(H)|P','P',Pset,'H',Hset,fluid)
            J['S','Pset'] = CP.PropsSI('d(Smass)/d(P)|H','P',Pset,'H',Hset,fluid)
            J['S','Hset'] = CP.PropsSI('d(Smass)/d(H)|P','P',Pset,'H',Hset,fluid)
            set_value = Hset

        J['rho','Pset'] = CP.PropsSI(f'd(Dmass)/d(P)|{mode}','P',Pset,f'{mode}',set_value,fluid)
        J['rho',f'{mode}set'] = CP.PropsSI(f'd(Dmass)/d({mode})|P','P',Pset,f'{mode}',set_value,fluid)
        J['Cp','Pset'] = CP.PropsSI(f'd(d(Hmass)/d(T)|P)/d(P)|{mode}','P',Pset,f'{mode}',set_value,fluid)
        J['Cp',f'{mode}set'] = CP.PropsSI(f'd(d(Hmass)/d(T)|P)/d({mode})|P','P',Pset,f'{mode}',set_value,fluid)
        J['Cv','Pset'] = CP.PropsSI(f'd(d(Umass)/d(T)|Dmass)/d(P)|{mode}','P',Pset,f'{mode}',set_value,fluid)
        J['Cv',f'{mode}set'] = CP.PropsSI(f'd(d(Umass)/d(T)|Dmass)/d({mode})|P','P',Pset,f'{mode}',set_value,fluid)
        # J['mu','Pset'] = CP.PropsSI(f'd(V)/d(P)|{mode}','P',Pset,f'{mode}',set_value,fluid)
        # J['mu',f'{mode}set'] = CP.PropsSI(f'd(V)/d({mode})|P','P',Pset,f'{mode}',set_value,fluid)
        # J['k','Pset'] = CP.PropsSI(f'd(conductivity)/d(P)|{mode}','P',Pset,f'{mode}',set_value,fluid)
        # J['k',f'{mode}set'] = CP.PropsSI(f'd(conductivity)/d({mode})|P','P',Pset,f'{mode}',set_value,fluid)


if __name__ == "__main__":

    import time
    from openmdao.api import Problem, Group, IndepVarComp, ExplicitComponent

    prob = Problem()
    Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    # Flow properties
    Vars.add_output('Pset', 2e5, units='Pa')
    Vars.add_output('Tset', 300, units='degK')
    Vars.add_output('Sset', 723, units='J/kg/degK')
    Vars.add_output('Hset', 1.3e5, units='J/kg')

    Blk = prob.model.add_subsystem('setTotal_T',cool_prop(mode='T', fluid = 'water'),promotes_inputs=['Tset','Pset'])
    Blk2 = prob.model.add_subsystem('setTotal_S',cool_prop(mode='S', fluid = 'water'),promotes_inputs=['Pset','Sset'])
    Blk3 = prob.model.add_subsystem('setTotal_H',cool_prop(mode='H', fluid = 'water'),promotes_inputs=['Pset','Hset'])

    # Blk.set_check_partial_options(wrt='*', step_calc='rel')
    # Blk2.set_check_partial_options(wrt='*', step_calc='rel')
    # Blk3.set_check_partial_options(wrt='*', step_calc='rel')

    prob.setup()
    prob.run_model()
    prob.check_partials(compact_print=True, method='fd')
    print('mode T')
    print('P = '+str(prob['setTotal_T.P'][0]))
    print('T = '+str(prob['setTotal_T.T'][0]))
    print('S = '+str(prob['setTotal_T.S'][0]))
    print('H = '+str(prob['setTotal_T.H'][0]))
    print('rho = '+str(prob['setTotal_T.rho'][0]))
    print('Cp = '+str(prob['setTotal_T.Cp'][0]))
    print('Cv = '+str(prob['setTotal_T.Cv'][0]))
    print('mu = '+str(prob['setTotal_T.mu'][0]))
    print('k = '+str(prob['setTotal_T.k'][0]))
    print(' ')
    print('mode S')
    print('P = '+str(prob['setTotal_S.P'][0]))
    print('T = '+str(prob['setTotal_S.T'][0]))
    print('S = '+str(prob['setTotal_S.S'][0]))
    print('H = '+str(prob['setTotal_S.H'][0]))
    print('rho = '+str(prob['setTotal_S.rho'][0]))
    print('Cp = '+str(prob['setTotal_S.Cp'][0]))
    print('Cv = '+str(prob['setTotal_S.Cv'][0]))
    print('mu = '+str(prob['setTotal_S.mu'][0]))
    print('k = '+str(prob['setTotal_S.k'][0]))
    print(' ')
    print('mode H')
    print('P = '+str(prob['setTotal_H.P'][0]))
    print('T = '+str(prob['setTotal_H.T'][0]))
    print('S = '+str(prob['setTotal_H.S'][0]))
    print('H = '+str(prob['setTotal_H.H'][0]))
    print('rho = '+str(prob['setTotal_H.rho'][0]))
    print('Cp = '+str(prob['setTotal_H.Cp'][0]))
    print('Cv = '+str(prob['setTotal_H.Cv'][0]))
    print('mu = '+str(prob['setTotal_H.mu'][0]))
    print('k = '+str(prob['setTotal_H.k'][0]))
    print(' ')
