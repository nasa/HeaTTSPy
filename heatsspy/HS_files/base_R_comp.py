import openmdao.api as om 
import numpy as np

class BaseResistanceComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('ducted', types=bool, default=True)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('t_base',
                        val=np.ones(nn),
                        units='m',
                        desc='heat sink base thickness')
        self.add_input('L',
                        val=np.ones(nn),
                        units='m',
                        desc='Length parallel to flow')
        self.add_input('W',
                        val=np.ones(nn),
                        units='m',
                        desc='width perpendicular to flow')
        self.add_input('k_sink',
                        val=np.ones(nn),
                        units='W/(m*K)',
                        desc='heat sink thermal conductivity')
        self.add_input('h',
                        val=np.ones(nn),
                        units='W/(m**2*K)',
                        desc='heat transfer coefficient')
        self.add_input('N_fins',
                        val=np.ones(nn),
                        units=None,
                        desc='number of fins')
        self.add_input('A_c',
                        val=np.ones(nn), 
                        units='m**2',
                        desc='cross-sectional area of fin tip')
        self.add_input('R_th_cont_per_area',
                        val = 1E-3*np.ones(nn),
                        units='m**2*K/W',
                        desc='thermal resistance of the contact resistance per area')

        self.add_output('R_th_contact',
                        val = np.ones(nn),
                        units='K/W',
                        desc='thermal contact resistance')
        self.add_output('R_th_base_cond',
                        val = np.ones(nn),
                        units='K/W',
                        desc='thermal resistance of conduction through the heat sink base')
        self.add_output('R_th_base_conv',
                        val = np.ones(nn),
                        units='K/W',
                        desc='thermal resistance of the heat sink base exposed to flow through convection')

    def setup_partials(self):
        nn = self.options['num_nodes']
        arange = np.arange(nn)

        self.declare_partials(of='R_th_contact', wrt=['R_th_cont_per_area', 'L', 'W'], rows=arange, cols=arange)
        self.declare_partials(of='R_th_base_cond', wrt=['t_base', 'k_sink', 'L', 'W'], rows=arange, cols=arange)
        self.declare_partials(of='R_th_base_conv', wrt=['h', 'W', 'N_fins', 'A_c', 'L'], rows=arange, cols=arange)

    def compute(self, inputs, outputs): 
        t_base, L, W, k_sink, h, N_fins, A_c, R_th_cont_per_area = inputs.values()

        outputs['R_th_contact'] = R_th_cont_per_area / (L*W)
        outputs['R_th_base_cond'] = t_base / (k_sink*L*W)
        outputs['R_th_base_conv'] = 1/(h*(W*L-A_c*N_fins))

    def compute_partials(self, inputs, partials):
        t_base, L, W, k_sink, h, N_fins, A_c, R_th_cont_per_area = inputs.values()

        partials['R_th_contact', 'R_th_cont_per_area'] = 1/(W*L)
        partials['R_th_contact', 'L'] = -R_th_cont_per_area / (L**2*W)
        partials['R_th_contact', 'W'] = -R_th_cont_per_area / (L*W**2)

        partials['R_th_base_cond', 't_base'] = 1 / (k_sink*L*W)
        partials['R_th_base_cond', 'k_sink'] = -t_base / (k_sink**2*L*W)
        partials['R_th_base_cond', 'L'] = -t_base / (k_sink*L**2*W)
        partials['R_th_base_cond', 'W'] = -t_base / (k_sink*L*W**2)

        partials['R_th_base_conv', 'h'] = -1/(h**2*(W*L-A_c*N_fins))
        partials['R_th_base_conv', 'W'] = -1/(h*(W*L-A_c*N_fins))**2 * h*L
        partials['R_th_base_conv', 'N_fins'] = -1/(h*(W*L-A_c*N_fins))**2 * h*-A_c
        partials['R_th_base_conv', 'A_c'] = -1/(h*(W*L-A_c*N_fins))**2 * h*-N_fins
        partials['R_th_base_conv', 'L'] = -1/(h*(W*L-A_c*N_fins))**2 * h*W