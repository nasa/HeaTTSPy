from openmdao.api import ExplicitComponent, Group,IndepVarComp
import numpy as np

class HE_size(ExplicitComponent):
    """ Determine Exchanger physical characteristics """

    def initialize(self):
        self.options.declare('dim',default=2, desc='flow dimension',values=(1, 2))
        self.options.declare('num_nodes', default=1, types=int, desc='Number of nodes to be evaluated')

    def setup(self):
        dim = self.options['dim']
        nn = self.options['num_nodes']
        self.add_input('width', val=np.ones(nn), units='m', desc='Width of exchanger')
        if dim == 1:
            self.add_input('height', val=np.ones(nn), units='m', desc='height of exchanger')
            self.add_input('length', val=np.ones(nn), units='m', desc='length of exchanger')
            self.add_output('Afr', val=np.ones(nn), units='m**2', desc='Frontal Area')
        else:
            self.add_input('height1', val=np.ones(nn), units='m', desc='height of side 1, length of side 2')
            self.add_input('height2', val=np.ones(nn), units='m', desc='height of side 2, height of side 1')

            self.add_output('Afr1', val=np.ones(nn), units='m**2', desc='Frontal Area for side 1')
            self.add_output('length1', val=np.ones(nn), units='m', desc='length of side 1')
            self.add_output('Afr2', val=np.ones(nn), units='m**2', desc='Frontal Area for side 2')
            self.add_output('length2', val=np.ones(nn), units='m', desc='length of side 2')
        self.add_output('vol', val=np.ones(nn), units='m**3', desc='heat exchanger volume')

        arange = np.arange(self.options['num_nodes'])
        if dim ==1:
            self.declare_partials(of='Afr', wrt=['width','height'], rows=arange, cols=arange)
            self.declare_partials(of='vol', wrt=['height','length','width'], rows=arange, cols=arange)
        else:
            self.declare_partials(of='Afr1', wrt=['width','height1'], rows=arange, cols=arange)
            self.declare_partials(of='length1', wrt=['height2'], val=1.0, rows=arange, cols=arange)
            self.declare_partials(of='Afr2', wrt=['width','height2'], rows=arange, cols=arange)
            self.declare_partials(of='vol', wrt=['height1','height2','width'], rows=arange, cols=arange)
            self.declare_partials(of='length2', wrt=['height1'], val=1.0, rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        dim = self.options['dim']
        if dim ==1:
            outputs['Afr'] = inputs['width']*inputs['height']
            outputs['vol'] = inputs['height']*inputs['length']*inputs['width']
        else:
            outputs['Afr1'] = inputs['width']*inputs['height1']
            outputs['length1']  = inputs['height2']
            outputs['Afr2'] = inputs['width']*inputs['height2']
            outputs['vol']  = inputs['height1']*inputs['height2']*inputs['width']
            outputs['length2']  = inputs['height1']

    def compute_partials(self, inputs, J):
        dim = self.options['dim']
        if dim ==1:
            J['Afr','height'] = inputs['width']
            J['Afr','width'] = inputs['height']

            J['vol','height'] = inputs['width']*inputs['length']
            J['vol','width']  = inputs['height']*inputs['length']
            J['vol','length'] = inputs['height']*inputs['width']

        else:
            J['Afr1','height1'] = inputs['width']
            J['Afr1','width']   = inputs['height1']

            J['Afr2','height2'] = inputs['width']
            J['Afr2','width']   = inputs['height2']

            J['vol','height1'] = inputs['height2']*inputs['width']
            J['vol','height2'] = inputs['height1']*inputs['width']
            J['vol','width']   = inputs['height1']*inputs['height2']


if __name__ == "__main__":

    import time
    from openmdao.api import Problem, IndepVarComp

    prob = Problem()
    Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    # Flow properties
    Vars.add_output('height1', val=0.91, units="m", desc='height1')
    Vars.add_output('height2', val=1.83, units='m', desc='height2')
    Vars.add_output('width', val=2.29, units='m', desc='width')
    Vars.add_output('height', val=2.0, units="m", desc='height')
    Vars.add_output('length', val=1.5, units="m", desc='length')

    Blk1 = prob.model.add_subsystem('prop_calcA', HE_size(dim = 1),
        promotes_inputs=['height','width','length'])
    Blk2 = prob.model.add_subsystem('prop_calcB', HE_size(dim = 2),
        promotes_inputs=['height1','height2','width'])

    prob.setup(force_alloc_complex=True)
    prob.run_model()
    prob.check_partials(compact_print=True, method='cs')
    #
    print('Afr = '+str(prob['prop_calcA.Afr'][0]))
    print('vol = '+str(prob['prop_calcA.vol'][0]))

    print('Afr1 = '+str(prob['prop_calcB.Afr1'][0]))
    print('length1 = '+str(prob['prop_calcB.length1'][0]))
    print('Afr2 = '+str(prob['prop_calcB.Afr2'][0]))
    print('length2 = '+str(prob['prop_calcB.length2'][0]))
    print('vol = '+str(prob['prop_calcB.vol'][0]))
