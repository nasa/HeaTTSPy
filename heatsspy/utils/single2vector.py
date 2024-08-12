import numpy as np
from openmdao.api import ExplicitComponent

class single2vector(ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes',types=int)
        self.options.declare('units',default='kg/s')
    def setup(self):
        nn = self.options['num_nodes']
        units= self.options['units']
        # input thrusts and drags
        self.add_input('S',val=1.0,units=units,desc='single value')

        self.add_output('V',val=np.ones(nn),units=units,desc='vector value')

        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='V',wrt=['S'], val=1.0)

    def compute(self, inputs, outputs):
        outputs['V']=inputs['S']*np.ones(self.options['num_nodes'])

if __name__ == "__main__":

    import time
    from openmdao.api import Problem, IndepVarComp

    prob=Problem()
    Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    Vars.add_output('X', val=3, units='Pa')

    prob.model.add_subsystem('s2v',single2vector(num_nodes=5,units='Pa'),
        promotes_inputs=[('S','X')],promotes_outputs=['V'])
    prob.setup()
    prob.run_model()
    prob.check_partials(compact_print=True)
    print(str(prob['V']))
