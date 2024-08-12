import openmdao.api as om
import numpy as np

class FinArrayGeomComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('L',
                        val=np.ones(nn),
                        units='m',
                        desc='length of heat sink in direction of airflow')
        self.add_input('W',
                        val=np.ones(nn),
                        units='m',
                        desc='width of heat sink perpindicular to fins')
        self.add_input('Ht',
                        val=np.ones(nn),
                        units='m',
                        desc='height of fins')
        self.add_input('N_fins',
                        val=np.ones(nn),
                        units=None,
                        desc='number of fins')
        self.add_input('t_fin',
                        val=np.ones(nn),
                        units='m',
                        desc='fin thickness')
        self.add_input('t_base',
                        val=np.ones(nn),
                        units='m',
                        desc='fin thickness')

        self.add_output('A_c',
                        val=np.ones(nn),
                        units='m**2',
                        desc='cross-sectional area of fin tip')
        self.add_output('D_h',
                        val=np.ones(nn),
                        units='m',
                        desc='hydraulic fin space diameter')
        self.add_output('Pm',
                        val=np.ones(nn),
                        units='m',
                        desc='fin perimeter')
        self.add_output('Sp',
                        val=np.ones(nn),
                        units='m',
                        desc='space between fins')
        self.add_output('Vol',
                        val=np.ones(nn),
                        units='m**3',
                        desc='heat sink volume')

        arange = np.arange(nn)

        self.declare_partials(of='A_c', wrt=['t_fin', 'L'], rows=arange, cols=arange, method='cs')
        self.declare_partials(of='D_h', wrt=['Ht', 't_fin', 'W', 'N_fins'], rows=arange, cols=arange, method='cs')
        self.declare_partials(of='Pm', wrt=['t_fin', 'L'], rows=arange, cols=arange, method='cs')
        self.declare_partials(of='Sp', wrt=['t_fin', 'W', 'N_fins'], rows=arange, cols=arange, method='cs')
        self.declare_partials(of='Vol', wrt=['t_fin', 'W', 'N_fins', 'Ht', 'L', 't_base'], rows=arange, cols=arange, method='cs')

    def compute(self, inputs, outputs):
        L, W, Ht, N_fins, t, t_base = inputs.values()

        outputs['A_c'] = L * t
        #outputs['D_h'] =  4*Ht*t / (2*(t+Ht))       # This is wrong, should be 4*Ht*Sp/(2*(Ht + Sp))
        outputs['Pm'] = 2 * (L+t)

        Sp = (W-(t*N_fins))/(N_fins-1)
        outputs['Sp'] = Sp

        outputs['D_h'] =  4*Ht*Sp / (2*(Sp+Ht))
        outputs['Vol'] = L*W*(Ht+t_base) - (outputs['Sp']*(N_fins-1)*Ht*L)


    # def compute_partials(self, inputs, partials):
    #     L, W, Ht, N_fins, t, t_base = inputs.values()
    #
    #     Sp = (W-(t*N_fins))/(N_fins-1)
    #
    #     dSp_dt_fin = partials['Sp', 't_fin'] = -N_fins/(N_fins-1)
    #     dSp_dW = partials['Sp', 'W'] = 1/(N_fins-1)
    #     dSp_dN_fins = partials['Sp', 'N_fins'] = (-t*(N_fins-1) - (W-t*N_fins))/(N_fins-1)**2
    #
    #     partials['A_c', 'L'] = t
    #     partials['A_c', 't_fin'] = L
    #
    #     partials['D_h', 'Ht'] = (4*t*(2*(Sp+Ht)) - 2*4*Ht*Sp)/(2*(Sp+Ht))**2
    #     partials['D_h', 't_fins'] = (4*Ht*(2*(Sp+Ht)) - 2*4*Ht*Sp)/(2*(Sp+Ht))**2
    #     partials['D_h', 'W'] =
    #     partials['D_h', 'N_fins'] =
    #
    #
    #     partials['Pm', 't_fin'] = 2
    #     partials['Pm', 'L'] = 2
    #
    #     partials['Vol', 'L'] = W*(Ht+t_base) - Sp*(N_fins-1)*Ht
    #     partials['Vol', 'W'] = L*(Ht+t_base) - (N_fins-1)*Ht*L*dSp_dW
    #     partials['Vol', 'Ht'] = L*W - Sp*(N_fins-1)*L
    #     partials['Vol', 'N_fins'] = -dSp_dN_fins * (N_fins-1)*Ht*L - Sp*Ht*L
    #     partials['Vol', 't_fin'] = - (N_fins-1)*Ht*L*dSp_dt_fin
    #     partials['Vol', 't_base'] = L*W

if __name__ == "__main__":

    import time
    from openmdao.api import Problem, Group, IndepVarComp, ExplicitComponent

    prob = Problem()
    Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    # Flow properties
    Vars.add_output('L', 0.3886, units='m')
    Vars.add_output('W', 0.085, units='m')
    Vars.add_output('Ht', 0.041, units='m')
    Vars.add_output('N_fins', 43, units=None)
    Vars.add_output('t_fin', 0.001, units='m')
    Vars.add_output('t_base', 0.1, units='m')

    prob.model.add_subsystem('f_calc',FinArrayGeomComp(num_nodes=1),
        promotes_inputs=['*'])

    prob.setup()

    prob.run_model()
    prob.check_partials(compact_print=True)
    print('f_calc ='+str(prob['f_calc.f'][0]))
