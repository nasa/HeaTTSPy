import numpy as np
from scipy.interpolate import Akima1DInterpolator as Interp

from openmdao.api import Group, ExplicitComponent, BalanceComp
from openmdao.api import DirectSolver, NewtonSolver

class file_prop_lookup(ExplicitComponent):
    def initialize(self):
        self.options.declare('fluid_props',default='oil',desc='fluid properties class')
        self.options.declare('num_nodes', default=1, types=int, desc='Number of nodes to be evaluated')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('Tset',val=np.ones(nn),units='degK',desc='temperature')
        self.add_input('Pset',val=np.ones(nn), units='Pa', desc='pressure')

        self.add_output('h', val=np.ones(nn),units='J/kg', desc='Enthlapy',lower=1e-5)
        self.add_output('rho', val=np.ones(nn),units='kg/m**3', desc='density',lower=1e-5)
        self.add_output('Cp', val=np.ones(nn),units='J/kg/degK', desc='specific heat with constant pressure',lower=1e-5)
        self.add_output('mu', val=np.ones(nn),units='Pa*s', desc='dynamic viscosity',lower=1e-5)
        self.add_output('k', val=np.ones(nn),units='W/m/degK', desc='thermal conductivity',lower=1e-5)

        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of=['h','Cp','mu','k', 'rho'], wrt=['Tset', 'Pset'], rows=arange, cols=arange)

    def compute(self,inputs,outputs):
        fluid_props = self.options['fluid_props']
        fluid_props.get_parameters(inputs['Tset'],inputs['Pset'])
        outputs['Cp'] = fluid_props.Cp
        outputs['h'] = fluid_props.h
        outputs['k'] = fluid_props.k
        outputs['mu'] = fluid_props.mu
        outputs['rho'] = fluid_props.rho


    def compute_partials(self,inputs,J):
        fluid_props = self.options['fluid_props']
        fluid_props.get_partials(inputs['Tset'],inputs['Pset'])
        J['Cp','Tset'] = fluid_props.dCp_dT
        J['Cp','Pset'] = fluid_props.dCp_dP
        J['h','Tset'] = fluid_props.dh_dT
        J['h','Pset'] = fluid_props.dh_dP
        J['k','Tset'] = fluid_props.dk_dT
        J['k','Pset'] = fluid_props.dk_dP
        J['mu','Tset'] = fluid_props.dmu_dT
        J['mu','Pset'] = fluid_props.dmu_dP
        J['rho','Tset'] = fluid_props.drho_dT
        J['rho','Pset'] = fluid_props.drho_dP


class prop_lookup(Group):
    """
    This group looks up thermo properties based on a temperature (T) or enthlapy (h).
    """
    def initialize(self):
        self.options.declare('mode',
            desc='the input variable that defines the total properties',
            default='T',
            values=('T', 'h'))
        self.options.declare('fluid_props',default='oil',desc='fluid properties class')
        self.options.declare('num_nodes', default=1, types=int, desc='Number of nodes to be evaluated')

    def setup(self):
        nn = self.options['num_nodes']
        mode = self.options['mode']
        fluid_props = self.options['fluid_props']

        if mode=='T':
            pi_eqn = ['Tset','Pset']
        elif mode=='h':
            pi_eqn = [('Tset','Tbal'),'Pset']

        self.add_subsystem('file_prop_lookup',file_prop_lookup(num_nodes=nn,fluid_props=fluid_props),
            promotes_inputs=pi_eqn,
            promotes_outputs=['Cp','h','rho','mu','k'])

        if mode=='h':
            balance = self.add_subsystem('balance',BalanceComp(),
                promotes_inputs=[('lhs:Tbal','hset'),('rhs:Tbal','h')],
                promotes_outputs=['Tbal'])

            balance.add_balance('Tbal', val=300*np.ones(nn), units='degK', eq_units='J/kg')
            self.linear_solver = DirectSolver(assemble_jac=True)

            self.nonlinear_solver = NewtonSolver(maxiter=50, iprint=0, solve_subsystems=True)


if __name__ == "__main__":

    import time
    from openmdao.api import Problem, Group, IndepVarComp, ExplicitComponent
    from heatsspy.include.props_water import water_props
    fluid_props = water_props()
    prob = Problem()
    Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    # Flow properties
    nn = 2
    Vars.add_output('Pset', 1e5*np.ones(nn), units='Pa')
    Vars.add_output('Tset', 25*np.ones(nn), units='degC')
    Vars.add_output('Sset', 723*np.ones(nn), units='J/kg/degK')
    Vars.add_output('hset', 1e5*np.ones(nn), units='J/kg')

    prob.model.add_subsystem('prop_lookup',prop_lookup(mode='T',fluid_props=fluid_props,num_nodes=nn),
        promotes_inputs=['Tset','Pset'])
    prob.model.add_subsystem('prop_lookup2',prop_lookup(mode='h',fluid_props=fluid_props,num_nodes=nn),
        promotes_inputs=['hset','Pset'])

    prob.setup(force_alloc_complex=True)
    prob.run_model()
    prob.check_partials(compact_print=True,method='cs')

    print(fluid_props)
    print('h ='+str(prob['prop_lookup.h']))
    print('T ='+str(prob['Tset']))
    print('rho ='+str(prob['prop_lookup.rho']))
    print('Cp ='+str(prob['prop_lookup.Cp']))
    print('mu ='+str(prob['prop_lookup.mu']))
    print('k ='+str(prob['prop_lookup.k']))

    print('hset')
    print('h ='+str(prob['prop_lookup2.h'][0]))
    print('T ='+str(prob['prop_lookup2.Tbal'][0]))
    print('rho ='+str(prob['prop_lookup2.rho'][0]))
    print('Cp ='+str(prob['prop_lookup2.Cp'][0]))
    print('mu ='+str(prob['prop_lookup2.mu'][0]))
    print('k ='+str(prob['prop_lookup2.k'][0]))
