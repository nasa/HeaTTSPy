import openmdao.api as om 
import numpy as np

class TotalResistanceComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):

        nn = self.options['num_nodes']

        self.add_input('R_th_contact',
                        val = 1.0*np.ones(nn),
                        units = 'K/W',
                        desc ='Thermal contact resistance of the contact with component')
        self.add_input('R_th_base_cond',
                        val = np.ones(nn),
                        units='K/W',
                        desc='thermal resistance of conduction through the heat sink base')
        self.add_input('R_th_base_conv',
                        val = np.ones(nn),
                        units='K/W',
                        desc='thermal resistance of the heat sink base exposed to flow through convection')
        self.add_input('R_th_fins',
                        val = np.ones(nn),
                        units='K/W',
                        desc='thermal resistance of the heat sink')

        self.add_output('R_th_tot',
                        val=np.ones(nn),
                        units='K/W',
                        desc='total thermal resistance of heat sink')

    def setup_partials(self):

        nn = self.options['num_nodes']
        arange = np.arange(nn)

        self.declare_partials(of='R_th_tot', wrt=['*'])

    def compute(self, inputs, outputs):

        R_th_contact, R_th_base_cond, R_th_base_conv, R_th_fins = inputs.values()

        outputs['R_th_tot'] = R_th_contact + R_th_base_cond + (R_th_base_conv**-1 + R_th_fins**-1)**-1

    def compute_partials(self, inputs, partials):
        R_th_contact, R_th_base_cond, R_th_base_conv, R_th_fins = inputs.values()

        partials['R_th_tot', 'R_th_contact'] = 1
        partials['R_th_tot', 'R_th_base_cond'] = 1
        partials['R_th_tot', 'R_th_base_conv'] = -(R_th_base_conv**-1 + R_th_fins**-1)**-2 * -R_th_base_conv**-2
        partials['R_th_tot', 'R_th_fins'] = -(R_th_base_conv**-1 + R_th_fins**-1)**-2 * -R_th_fins**-2