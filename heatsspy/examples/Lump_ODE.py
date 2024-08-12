import numpy as np

from openmdao.api import  Group
from openmdao.api import Problem, Group, IndepVarComp, BalanceComp, ExecComp

from heatsspy.api import thermal_mass

class Lump_ODE(Group):
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)

    def setup(self):
        nn = self.options['num_nodes']
        ExVars=self.add_subsystem('ExVars',IndepVarComp() ,promotes_outputs=['*'])
        # from mass calcs
        ExVars.add_output('Cp', 900*np.ones(nn), units='J/(kg*K)')  # use constant for now, assume Al
        ExVars.add_output('mass', 5000*np.ones(nn), units='kg', desc='mass of lump') # 571
        # ExVars.add_output('Q_cool', np.zeros(nn), units='kW', desc='power exiting mass')
        # ExVars.add_output('Q_in', 0.1*np.ones(nn), units='kW', desc='power entering mass')


        self.add_subsystem('Test_mass',thermal_mass(num_nodes=nn),
            promotes_inputs = ['Q_cool','Q_in',
                               'Cp', 'mass'],
            promotes_outputs = ['Tdot'])

def RunTimeStep():
    p.run_model()
    print('Tsurface = ',p.model.get_val('RunVars.Tsurface')[0], 'Tdot = ', p.model.get_val('mass.Tdot')[0])
    p.set_val('RunVars.Tsurface', p.model.get_val('RunVars.Tsurface')-p.model.get_val('mass.Tdot'))

if __name__ == "__main__":

    p = Problem()
    RunVars=p.model.add_subsystem('RunVars',IndepVarComp())
    # from mass calcs
    RunVars.add_output('Tsurface', 500, units='degK', desc='surface temperature')
    RunVars.add_output('Tcool', 400, units='degK', desc='cooling temperature')
    RunVars.add_output('Q_in', 20, units='W', desc='power to be rejected')


    p.model.add_subsystem('Q_calc', ExecComp('Q_cool =  1000000.0 * (Tcool - Tsurface)',
                    Q_cool = {'val': 1.0 , 'units':'W'},
                    Tsurface = {'val': 1.0 , 'units':'degK'},
                    Tcool = {'val': 1.0 , 'units':'degK'}))
    p.model.connect('RunVars.Tsurface', 'Q_calc.Tsurface')
    p.model.connect('RunVars.Tcool', 'Q_calc.Tcool')

    p.model.add_subsystem('mass',Lump_ODE())
    p.model.connect('RunVars.Q_in', 'mass.Q_in')
    p.model.connect('Q_calc.Q_cool', 'mass.Q_cool')

    p.setup()
    RunTimeStep()
    RunTimeStep()
    RunTimeStep()
    RunTimeStep()
    RunTimeStep()
    RunTimeStep()
    RunTimeStep()
    RunTimeStep()
    RunTimeStep()
    RunTimeStep()
    RunTimeStep()
    RunTimeStep()
    RunTimeStep()
    RunTimeStep()
    RunTimeStep()
    RunTimeStep()
    RunTimeStep()
    RunTimeStep()
    RunTimeStep()
    RunTimeStep()
