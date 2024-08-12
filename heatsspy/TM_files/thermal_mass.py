from openmdao.api import ExplicitComponent
import numpy as np

class thermal_mass(ExplicitComponent):
    ''' Define change in temperature'''
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        # Inputs
        self.add_input('Q_cool',val=np.ones(nn),desc='heat out of system',units='W')
        self.add_input('Q_in',val=np.ones(nn),desc='convection coefficient',units='W')
        self.add_input('Cp',val=np.ones(nn),desc='specific heat',units='J/(kg * K)')
        self.add_input('mass',val=np.ones(nn),desc='Mass',units='kg')

        self.add_output('Tdot',val=np.ones(nn),desc='dT',units='K/s')

        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='Tdot', wrt=['Q_cool','Q_in','Cp','mass'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        Q_cool = inputs['Q_cool']
        Q_in = inputs['Q_in']
        Cp = inputs['Cp']
        mass = inputs['mass']

        outputs['Tdot'] = (Q_in-Q_cool)/(mass*Cp)

    def compute_partials(self, inputs, J):
        Q_cool = inputs['Q_cool']
        Q_in = inputs['Q_in']
        Cp = inputs['Cp']
        mass = inputs['mass']

        J['Tdot', 'Q_cool'] = -1/(mass*Cp)
        J['Tdot', 'Q_in']   = 1/(mass*Cp)
        J['Tdot', 'Cp']   = (Q_cool-Q_in)/(mass*Cp**2)
        J['Tdot', 'mass']  = (Q_cool-Q_in)/(mass**2*Cp)


class temperature_from_heat(ExplicitComponent):
    ''' calculate surface temperature for neglected mass'''
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        # Inputs
        self.add_input('Cp', val=np.ones(nn), desc='specific heat of fluid', units='J/(kg * K)')
        self.add_input('effect', val=np.ones(nn), desc='heat exchange effectivness', units=None)
        self.add_input('Q', val=np.ones(nn), desc='heat out of system', units='W')
        self.add_input('T', val=np.ones(nn), desc='fluid temperature', units='degK')
        self.add_input('W', val=np.ones(nn), desc='mass flow', units='kg/s')

        self.add_output('Ts', val=np.ones(nn), desc='surface temperature', units='degK')

        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='Ts', wrt=['Q', 'W', 'Cp', 'effect', 'T'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        outputs['Ts'] = inputs['T'] + inputs['Q']/(inputs['effect']*inputs['W']*inputs['Cp'])

    def compute_partials(self, inputs, J):
        J['Ts', 'Cp'] = - inputs['Q']/(inputs['effect'] * inputs['W'] * inputs['Cp']**2)
        J['Ts', 'effect'] = - inputs['Q']/(inputs['effect']**2 * inputs['W'] * inputs['Cp'])
        J['Ts', 'Q'] = 1/(inputs['effect']*inputs['W']*inputs['Cp'])
        J['Ts', 'T'] = 1
        J['Ts', 'W'] = - inputs['Q']/(inputs['effect'] * inputs['W']**2 * inputs['Cp'])


if __name__ == "__main__":

    import time
    from openmdao.api import Problem, Group, IndepVarComp, ExplicitComponent

    prob = Problem()
    Vars =  prob.model.add_subsystem('Vars', IndepVarComp(), promotes_outputs=['*'])
    # Flow properties
    Vars.add_output('Q_cool', val=10, desc='heat out of system', units='W')
    Vars.add_output('Q_in', val=20, desc='heat into system', units='W')
    Vars.add_output('Cp', val=900, desc='specific heat', units='J/(kg * K)')
    Vars.add_output('mass', val=500, desc='Mass', units='kg')

    Vars.add_output('effect', val = 0.95, desc = 'effectivness', units = None)
    Vars.add_output('Q', val = 10, desc = 'heat out of system', units = 'kW')
    Vars.add_output('T', val = 310, desc = 'fluid temperature', units = 'degK')
    Vars.add_output('W', val = 1, desc = 'mass flow', units = 'kg/s')

    prob.model.add_subsystem('lump_calc', thermal_mass(num_nodes=1), promotes_inputs=['*'])

    prob.model.add_subsystem('temp_calc', temperature_from_heat(num_nodes=1), promotes_inputs=['*'])
    # Blk.set_check_partial_options(wrt=['Ht','Wth'], step_calc='rel')
    prob.setup(force_alloc_complex=True)

    prob.run_model()
    prob.check_partials(compact_print=True, method='cs')
    print('dT ='+str(prob['lump_calc.Tdot'][0]))
    print('Ts = ',prob['temp_calc.Ts'][0])
