import openmdao.api as om
import numpy as np

class FinResistanceComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('ducted', types=bool, default=True)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('N_fins',
                        val=np.ones(nn),
                        units=None,
                        desc='number of fins')
        self.add_input('Pm',
                        val=np.ones(nn),
                        units='m',
                        desc='fin perimeter')
        self.add_input('h',
                        val=np.ones(nn),
                        units='W/(m**2*K)',
                        desc='heat transfer coefficient')
        self.add_input('k_sink',
                        val=np.ones(nn),
                        units='W/(m*K)',
                        desc='heat sink thermal conductivity')
        self.add_input('A_c',
                        val=np.ones(nn),
                        units='m**2',
                        desc='cross-sectional area of fin tip')
        self.add_input('Ht',
                        val=np.ones(nn),
                        units='m',
                        desc='height of fins')

        self.add_output('R_th_fins',
                        val = np.ones(nn),
                        units='K/W',
                        desc='thermal resistance of the heat sink')

    def setup_partials(self):
        nn = self.options['num_nodes']
        arange = np.arange(nn)

        self.declare_partials(of='R_th_fins', wrt=['*'], rows=arange, cols=arange)

    def compute(self, inputs, outputs): 
        N_fins, P, h, k_sink, A_c, Ht = inputs.values()

        m = (h*P/(k_sink*A_c))**0.5

        if self.options['ducted']:
            R_fin = 1/(np.tanh(m*Ht)*(h*P*k_sink*A_c)**0.5)

        else:
            R_fin = (np.cosh(m*Ht) + (h/(m*k_sink))*np.sinh(m*Ht)) / \
                    (np.sinh(m*Ht) + (h/(m*k_sink))*np.cosh(m*Ht))

        outputs['R_th_fins'] = R_fin/N_fins

    def compute_partials(self, inputs, partials):
        N_fins, P, h, k_sink, A_c, Ht = inputs.values()

        m = (h*P/(k_sink*A_c))**0.5

        dm_dh = 0.5 * (h*P/(k_sink*A_c))**-0.5 * P/(k_sink*A_c)
        dm_dP = 0.5 * (h*P/(k_sink*A_c))**-0.5 * h/(k_sink*A_c)
        dm_dk_sink = 0.5 * (h*P/(k_sink*A_c))**-0.5 * -h*P/(k_sink**2*A_c)
        dm_dA_c = 0.5 * (h*P/(k_sink*A_c))**-0.5 * -h*P/(k_sink*A_c**2)

        if self.options['ducted']:

            R_fin = 1/(np.tanh(m*Ht)*(h*P*k_sink*A_c)**0.5)

            dR_dh = -(np.tanh(m*Ht)*(h*P*k_sink*A_c)**0.5)**-2 * (1/(np.cosh(m*Ht))**2 * Ht*dm_dh*(h*P*k_sink*A_c)**0.5 \
                    + np.tanh(m*Ht)*0.5*(h*P*k_sink*A_c)**-0.5*P*k_sink*A_c)
            dR_dP = -(np.tanh(m*Ht)*(h*P*k_sink*A_c)**0.5)**-2 * (1/(np.cosh(m*Ht))**2 * Ht*dm_dP*(h*P*k_sink*A_c)**0.5 \
                    + np.tanh(m*Ht)*0.5*(h*P*k_sink*A_c)**-0.5*h*k_sink*A_c)
            dR_dA_c = -(np.tanh(m*Ht)*(h*P*k_sink*A_c)**0.5)**-2 * (1/(np.cosh(m*Ht))**2 * Ht*dm_dA_c*(h*P*k_sink*A_c)**0.5 \
                    + np.tanh(m*Ht)*0.5*(h*P*k_sink*A_c)**-0.5*P*h*k_sink)
            dR_dk_sink = -(np.tanh(m*Ht)*(h*P*k_sink*A_c)**0.5)**-2 * (1/(np.cosh(m*Ht))**2 * Ht*dm_dk_sink*(h*P*k_sink*A_c)**0.5 \
                    + np.tanh(m*Ht)*0.5*(h*P*k_sink*A_c)**-0.5*P*h*A_c)
            dR_dHt = -(np.tanh(m*Ht)*(h*P*k_sink*A_c)**0.5)**-2 * (1/(np.cosh(m*Ht))**2 * m *(h*P*k_sink*A_c)**0.5)

        partials['R_th_fins', 'h'] = 1/N_fins * dR_dh
        partials['R_th_fins', 'Pm'] = 1/N_fins * dR_dP
        partials['R_th_fins', 'A_c'] = 1/N_fins * dR_dA_c
        partials['R_th_fins', 'k_sink'] = 1/N_fins * dR_dk_sink
        partials['R_th_fins', 'Ht'] = 1/N_fins * dR_dHt
        partials['R_th_fins', 'N_fins'] = -R_fin/N_fins**2 