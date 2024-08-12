from openmdao.api import ExplicitComponent
import numpy as np

class xp_calc(ExplicitComponent):
    ''' Define hydrodynamic entry length ratio'''
    def initialize(self):
        self.options.declare('num_nodes', types=int)
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('Lng',
                        val=0*np.zeros(nn),
                        desc='Heat sink total length',
                        units='m')
        self.add_input('Dh',
                        val= 0*np.zeros(nn),
                        desc='Hydraulic diameter',
                        units = 'm')
        self.add_input('Re',
                        val= 0*np.zeros(nn),
                        desc='Reynolds number of channel')

        self.add_output('xp',
                        val=0*np.zeros(nn),
                        desc='Hydrodynamic entry length ratio',lower=0.000001)

        self.declare_partials(of='xp', wrt='Lng')
        self.declare_partials(of='xp', wrt='Dh')
        self.declare_partials(of='xp', wrt='Re')

    def compute(self, inputs, outputs):
        Lng = inputs['Lng']
        Dh = inputs['Dh']
        Re = inputs['Re']

        outputs['xp'] =xp = Lng/(Dh*Re)
    def compute_partials(self, inputs, J):
        Lng = inputs['Lng']
        Dh = inputs['Dh']
        Re = inputs['Re']

        J['xp', 'Lng']  =  1/(Dh*Re)
        J['xp', 'Dh']   = - Lng/(Dh**2*Re)
        J['xp', 'Re']   = -  Lng/(Dh*Re**2)


if __name__ == "__main__":

    import time
    from openmdao.api import Problem, Group, IndepVarComp, ExplicitComponent

    prob = Problem()
    Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    # Flow properties
    Vars.add_output('Lng',
                    val=0.135,
                    desc='Heat sink total length',
                    units='m')
    Vars.add_output('Dh',
                    val= 0.0033436123953574687,
                    desc='Hydraulic diameter',
                    units = 'm')
    Vars.add_output('Re',
                    val= 2528.685,
                    desc='Reynolds number of channel')


    Blk = prob.model.add_subsystem('xp_calc',xp_calc(num_nodes=1),
        promotes_inputs=['*'])
    # Blk.set_check_partial_options(wrt=['Ht','Wth'], step_calc='rel')
    prob.setup()

    prob.run_model()
    prob.check_partials(compact_print=True)
    print('xp ='+str(prob['xp_calc.xp'][0]))
