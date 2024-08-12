from openmdao.api import ExplicitComponent, Group, BalanceComp, IndepVarComp, NewtonSolver,\
                            DirectSolver
import numpy as np

class LinePressureDropDesign(ExplicitComponent):
    ''' Define design pressure drop'''
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # INPUTS
        self.add_input('Pin',
                       val=np.ones(nn),
                       desc='Pressure in line',
                       units='Pa')
        self.add_input('dPqP',
                       val=np.ones(nn),
                       desc='percent pressure loss',
                       units=None)

        # OUTPUTS
        self.add_output('dP_des',
                        val=np.ones(nn),
                        desc='pressure drop over the line',
                        units='Pa',lower = 1e-5)

        arange = np.arange(self.options['num_nodes'])

        self.declare_partials(of='dP_des', wrt='Pin', rows=arange, cols=arange)
        self.declare_partials(of='dP_des', wrt='dPqP', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        Pin = inputs['Pin']
        dPqP = inputs['dPqP']

        outputs['dP_des'] = Pin * dPqP

    def compute_partials(self, inputs, J):
        Pin = inputs['Pin']
        dPqP = inputs['dPqP']

        J['dP_des', 'Pin'] = dPqP
        J['dP_des', 'dPqP'] = Pin


class CoolantLineWeight(ExplicitComponent):
    ''' fluid weight within line '''
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
        self.options.declare('length_scaler', default=1, types=int)

    def setup(self):
        nn = self.options['num_nodes']
        L_scaler = self.options['length_scaler']

        self.add_input('L_fluid_line', val=np.ones(nn), desc='length of fluid line', units='m')
        self.add_input('rho', val=np.ones(nn), desc='fluid density', units='kg/m**3')
        self.add_input('D', val=np.ones(nn), desc='diameter', units='m')

        self.add_output('m_coolant', val=np.ones(nn), desc='mass of coolant', units='kg',lower = 1e-5)

        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='m_coolant', wrt=['L_fluid_line', 'rho', 'D'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        L_scaler = self.options['length_scaler']
        L_fluid_line = inputs['L_fluid_line'] * L_scaler
        rho = inputs['rho']
        D = inputs['D']
        outputs['m_coolant'] = L_fluid_line * np.pi * D**2 * rho /4

    def compute_partials(self, inputs, J):
        L_scaler = self.options['length_scaler']
        L_fluid_line = inputs['L_fluid_line'] * L_scaler
        rho = inputs['rho']
        D = inputs['D']

        J['m_coolant', 'L_fluid_line'] = L_scaler * np.pi * D**2/4 * rho
        J['m_coolant', 'D'] = L_fluid_line * np.pi * 2 * D/4 * rho
        J['m_coolant', 'rho'] = L_fluid_line * np.pi * D**2/4


class LinePressureDrop(ExplicitComponent):
    ''' line pressure drop'''
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
        self.options.declare('length_scaler', default=1, types=int)

    def setup(self):
        nn = self.options['num_nodes']
        L_scaler = self.options['length_scaler']

        # INPUTS
        self.add_input('mdot', val=np.ones(nn), desc='coolant mass flow', units='kg/s')
        self.add_input('v', val=np.ones(nn), desc='kinematic viscosity', units='m**2/s')
        self.add_input('L_fluid_line', val=np.ones(nn), desc='length of fluid line', units='m')
        self.add_input('D', val=np.ones(nn), desc='diameter', units='m')
        # OUTPUTS
        self.add_output('dP', val=np.ones(nn), desc='pressure drop over the line', units='Pa',lower = 1e-5)

        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='dP', wrt=['mdot', 'v', 'L_fluid_line', 'D'] , rows=arange, cols=arange)


    def compute(self, inputs, outputs):
        L_scaler = self.options['length_scaler']
        mdot = inputs['mdot']
        L_fluid_line = inputs['L_fluid_line'] * L_scaler
        v = inputs['v']
        D = inputs['D']

        outputs['dP'] = (128 * v * L_fluid_line * mdot) / (np.pi * D**4)

    def compute_partials(self, inputs, J):
        L_scaler = self.options['length_scaler']
        mdot = inputs['mdot']
        L_fluid_line = inputs['L_fluid_line'] * L_scaler
        v = inputs['v']
        D = inputs['D']

        J['dP', 'mdot'] = (128 * v * L_fluid_line) / (np.pi * D**4)
        J['dP', 'v'] = (128 * L_fluid_line * mdot) / (np.pi * D**4)
        J['dP', 'L_fluid_line'] = (128 * v * mdot * L_scaler) / (np.pi * D**4)
        J['dP', 'D'] = - 4 * (128 * v * L_fluid_line * mdot) / (np.pi * D**5)


class DiameterCalc(ExplicitComponent):
    ''' calcualte diameter based on dPdes'''
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
        self.options.declare('dPdes', default=2e5*0.01)
        self.options.declare('length_scaler', default=1, types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # INPUTS
        self.add_input('mdot', val=np.ones(nn), desc='coolant mass flow', units='kg/s')
        self.add_input('L_fluid_line', val=np.ones(nn), desc='length of fluid line', units='m')
        self.add_input('v', val=np.ones(nn), desc='kinematic viscosity', units='m**2/s')
        # OUTPUTS
        self.add_output('D', val=np.ones(nn), desc='diameter', units='m',lower = 1e-5)

        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='D', wrt=['mdot', 'L_fluid_line', 'v'] , rows=arange, cols=arange)


    def compute(self, inputs, outputs):
        dP = self.options['dPdes']
        L_scaler = self.options['length_scaler']
        v = inputs['v']
        mdot = inputs['mdot']
        L_fluid_line = inputs['L_fluid_line'] * L_scaler

        # outputs['dP'] = (128 * v * L_fluid_line * mdot) / (np.pi * D**4)
        outputs['D'] = ((128 * v * L_fluid_line * mdot)  / (np.pi * dP))**(1/4)

    def compute_partials(self, inputs, J):
        dP = self.options['dPdes']
        L_scaler = self.options['length_scaler']
        v = inputs['v']
        mdot = inputs['mdot']
        L_fluid_line = inputs['L_fluid_line'] * L_scaler

        J['D', 'L_fluid_line'] = (1/4) * (L_fluid_line/L_scaler)**(-3/4) * ((128*L_scaler * v *  mdot)  / (np.pi * dP))**(1/4)
        J['D', 'mdot'] = (1/4) * mdot **(-3/4) * ((128 * v *  L_fluid_line)  / (np.pi * dP))**(1/4)
        J['D', 'v'] = (1/4) * v **(-3/4) * ((128 * L_fluid_line * mdot)  / (np.pi * dP))**(1/4)


class CoolantLineWeightW(Group):
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
        self.options.declare('length_scaler', default=1, types=int)

    def setup(self):
        nn = self.options['num_nodes']
        L_scaler = self.options['length_scaler']

        self.add_subsystem('D_calc', DiameterCalc(num_nodes=nn,length_scaler = L_scaler),
            promotes_inputs = ['mdot','L_fluid_line','v'],
            promotes_outputs = ['D'])
        self.add_subsystem('oil_line_weight', CoolantLineWeight(length_scaler=2),
            promotes_inputs=['rho', 'L_fluid_line', 'D'],
            promotes_outputs=['m_coolant'])


class coolant_weight_group_dP(Group):

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
        self.options.declare('length_scaler', default=1, types=int)
        self.options.declare('dPqP_des', default=True, types=bool)

    def setup(self):
        nn = self.options['num_nodes']
        dPqP_des = self.options['dPqP_des']
        L_scaler = self.options['length_scaler']

        if dPqP_des:

            self.add_subsystem('dPqP', IndepVarComp('dPqP', val=0.01*np.ones(nn), units=None), promotes=['*'])

            self.add_subsystem('dP_des_comp', LinePressureDropDesign(num_nodes=nn),
                promotes_inputs=['Pin', 'dPqP'],
                promotes_outputs=['dP_des'])

            self.add_subsystem('dP_comp', LinePressureDrop(num_nodes=nn,
                length_scaler=L_scaler),
                promotes_inputs=['mdot', 'L_fluid_line', 'D', 'v'],
                promotes_outputs=['dP'])

            def D_guess_nonlinear(inputs, outputs, residuals):
                if residuals.get_norm() > 1e0 or outputs['D'].any() < 0.0001 or outputs['D'].any() > 100:
                    outputs['D'] = 0.01

            self.add_subsystem(name='balanceW',
                           subsys=BalanceComp(name='D', val=0.01*np.ones(nn), normalize=False,
                                              rhs_name='dP_des',lhs_name='dP', units='m',
                                              eq_units='Pa', lower = 0.0001, upper= 100.,
                                              guess_func = D_guess_nonlinear),
                           promotes=['*'])

            self.add_subsystem('wt_comp', CoolantLineWeight(num_nodes=nn,
                length_scaler=L_scaler),
                promotes_inputs=['rho', 'L_fluid_line', 'D'],
                promotes_outputs=['m_coolant'])

            # self.nonlinear_solver = NewtonSolver()
            # self.linear_solver = DirectSolver()

        else:
            self.add_subsystem('dP_comp', PressureDrop(num_nodes=nn,
                length_scaler=L_scaler),
                promotes_inputs=['mdot', 'L_fluid_line', 'D', 'v'],
                promotes_outputs=['dP'])

            self.add_subsystem('wt_comp', CoolantWeightComp(num_nodes=nn,
                length_scaler=L_scaler),
                promotes_inputs=['rho', 'L_fluid_line', 'D'],
                promotes_outputs=['m_coolant'])


if __name__ == "__main__":

    import time
    from openmdao.api import Problem, IndepVarComp, BoundsEnforceLS

    prob = Problem()
    Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    # Flow properties
    Vars.add_output('mdot', val=2.66819865, units='kg/s', desc='mass flow')
    Vars.add_output('v', val=16.68877099e-06, units='m**2/s', desc='kinematic viscosity')
    Vars.add_output('L_fluid_line', val=6.096/2, units='m', desc='length pipe')
    Vars.add_output('rho', val=836.08173859, units='kg/m**3', desc='density')
    # Vars.add_output('D', val=0.0687, units='m', desc='pipe diameter')
    Vars.add_output('Pin', val=200000, units='Pa', desc='starting oil pressure')

    weight = prob.model.add_subsystem('coolant_weight', coolant_weight_group_dP(num_nodes=1, length_scaler=2,
                                        dPqP_des=True), promotes=['*'])

    prob.model.add_subsystem('CLWW', CoolantLineWeightW(num_nodes=1, length_scaler=2),
                promotes_inputs=['rho','mdot','L_fluid_line','v'],
                promotes_outputs=[('m_coolant','m_coolant2')])

    newton = prob.model.nonlinear_solver = NewtonSolver(maxiter=30, atol=1e-6)
    newton.options['solve_subsystems'] = True
    newton.options['max_sub_solves'] = 500
    newton.options['iprint'] = 2
    newton.options['solve_subsystems']=True
    newton.linesearch = BoundsEnforceLS()
    prob.model.linear_solver = DirectSolver()

    prob.setup(force_alloc_complex=True,check='all')
    prob.run_model()
    prob.check_partials(compact_print=True, method='cs')
    print('m_coolant (kg): ', prob['m_coolant'])
    print('Diameter (m): ', prob['D'])
    print('dP (Pa): ', prob['dP'])
    print('m_coolant2 : ', prob['m_coolant2'])
    # prob.check_partials(compact_print=False)

    # from openmdao.api import view_model

    # view_model(prob)
