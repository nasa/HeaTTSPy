import numpy as np
from openmdao.api import  ExplicitComponent

class GroupPrint(ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes',default= 1, types=int)
        self.options.declare('locs',default=())
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('print',val=np.ones(nn),units=None,desc='print me')
        self.add_output('foobar',val=np.ones(nn), units=None,desc='foobar')

    def compute(self, inputs, outputs):
        locs = self.options['locs']
        if len(locs) == 0:
            print(self.pathname)
            print(inputs['print'])
        else:
            for loc in locs:
                if loc in self.pathname:
                    print(self.pathname)
                    print(inputs['print'])

