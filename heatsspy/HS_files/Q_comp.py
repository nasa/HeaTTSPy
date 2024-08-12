from openmdao.api import ExplicitComponent
import numpy as np

class Q_calc(ExplicitComponent):
    ''' Define energy transfer from heat exchanger'''
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('Tb',
                       val=1*np.ones(nn),
                       desc='component temperature',
                       units='K')
        self.add_input('Tinf',
                       val= 100 * np.ones(nn),
                       desc='ambient temperature',
                       units='K')
        self.add_input('h',
                       val= 210 * np.ones(nn),
                       desc='convection coefficient',
                       units='W/(m**2 * K)')
        self.add_input('Aw',
                       val= .5 * np.ones(nn),
                       desc='wet convection coefficient',
                       units='m**2')
        # self.add_input('Ab',
        #                val= .5 * np.ones(nn),
        #                desc='wet convection coefficient',
        #                units='m**2')

        self.add_output('UA',
                        val= 1*np.ones(nn),
                        desc='overall heat transfer coefficient * Area',
                        units='W/degK')
        self.add_output('Rw',
                        val= 1*np.ones(nn),
                        desc='wet resistance',
                        units='degK/W')
        # self.add_output('Rb',
        #                 val= 1*np.ones(nn),
        #                 desc='base resistance',
        #                 units='degK/W')
        self.add_output('Q_cool',
                        val= 1*np.ones(nn),
                        desc='heat being rejected',
                        units='J/s')

        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='Rw', wrt=['Aw','h'], rows=arange, cols=arange)
        self.declare_partials(of='UA', wrt=['Aw','h'], rows=arange, cols=arange)
        self.declare_partials(of='Q_cool', wrt=['Aw','h','Tb','Tinf'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        Tb = inputs['Tb']
        Tinf = inputs['Tinf']
        h = inputs['h']
        Aw = inputs['Aw']

        # to add fidelity include Rb (base conductive resistance) and
        # an overall surface efficiency term based on fin efficiency Incorpera section 3.6.5
        outputs['Rw'] = Rw = 1/h/Aw
        outputs['UA'] = UA = 1/Rw
        
        outputs['Q_cool'] = UA*(Tb - Tinf) # positive Q is energy leaving heat exchanger

    def compute_partials(self, inputs, J):
        Tb = inputs['Tb']
        Tinf = inputs['Tinf']
        h = inputs['h']
        Aw = inputs['Aw']

        J['Rw','Aw'] = - 1/h/Aw**2
        J['Rw','h']  = - 1/h**2/Aw

        J['UA','Aw'] = h
        J['UA','h']  = Aw

        J['Q_cool','Aw']  = h*(Tb - Tinf)
        J['Q_cool','h']   = Aw*(Tb - Tinf)
        J['Q_cool','Tb']  = h*Aw
        J['Q_cool','Tinf']= - h*Aw


if __name__ == "__main__":

    import time
    from openmdao.api import Problem, Group, IndepVarComp, ExplicitComponent

    prob = Problem()
    Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    # Flow properties
    Vars.add_output('Tb',
                   val=450,
                   desc='component temperature',
                   units='K')
    Vars.add_output('Tinf',
                   val= 273,
                   desc='ambient temperature',
                   units='K')
    Vars.add_output('h',
                   val= 60,
                   desc='convection coefficient',
                   units='W/(m**2 * K)')
    Vars.add_output('Aw',
                   val= 0.447,
                   desc='wet convection coefficient',
                   units='m**2')


    Blk = prob.model.add_subsystem('Q_calc',Q_calc(num_nodes=1),
        promotes_inputs=['*'])
    # Blk.set_check_partial_options(wrt='Th', step_calc='rel')
    prob.setup()

    prob.run_model()
    prob.check_partials(compact_print=True)
    print('Rw ='+str(prob['Q_calc.Rw'][0]))
    print('UA ='+str(prob['Q_calc.UA'][0]))
    print('Q_cool ='+str(prob['Q_calc.Q_cool'][0]))
