import numpy as np
from openmdao.api import ExplicitComponent


class general_obj_fun(ExplicitComponent):
    ''' empierical formula for fuel spent,
    corroates Q,F, and Wt into an objective for optimization '''
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
        self.options.declare('s_Q',default=0.11) #0.11
        self.options.declare('s_F',default=0.14) #0.14
        self.options.declare('s_W',default=1) #1.0
        self.options.declare('DEFAULT', default=True)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('Qp', val=np.ones(nn), units='J/s', desc='Power required')
        self.add_input('Fn',val=np.ones(nn),units='N',desc='net thrust')
        self.add_input('Wt',val=np.ones(nn), units='kg', desc='weight')
        self.add_output('Obj',val=np.ones(nn), units='lbm/s', desc='fuel usage')
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='Obj',wrt=['Qp','Fn','Wt'], rows=arange, cols=arange)
    def compute(self,inputs,outputs):
        s_Q = self.options['s_Q']
        s_F = self.options['s_F']
        s_W = self.options['s_W']
        if self.options['DEFAULT']:
            outputs['Obj']=0.03*inputs['Wt']*2.2*s_W+0.3*inputs['Qp']*0.00134*s_Q-0.5*inputs['Fn']*0.225*s_F
        else :
            outputs['Obj']=inputs['Wt']*s_W + inputs['Qp']*s_Q - inputs['Fn']*s_F
        # outputs['Obj']=0.03*inputs['Wt']+0.3*inputs['Qp']-0.5*inputs['Fn']
    def compute_partials(self, inputs, J):
        s_Q = self.options['s_Q']
        s_F = self.options['s_F']
        s_W = self.options['s_W']
        if self.options['DEFAULT']:
            J['Obj','Qp'] = 0.3*0.00134*s_Q
            J['Obj','Fn'] = -0.5*0.225*s_F
            J['Obj','Wt'] = 0.03*2.2*s_W
        else :
            J['Obj','Qp'] = s_Q
            J['Obj','Fn'] = -s_F
            J['Obj','Wt'] = s_W



if __name__ == "__main__":

    import time
    from openmdao.api import Problem, Group, IndepVarComp, ExplicitComponent

    p2 = Problem()
    Vars =  p2.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    Vars.add_output('Wt', 200, units='kg')
    Vars.add_output('Fn', 2000, units='N')
    Vars.add_output('Qp', 150000, units='W')
    p2.model.add_subsystem('Obj',general_obj_fun(num_nodes=1),promotes_inputs=['*'])
    p2.setup()
    p2.run_model()
    p2.check_partials(compact_print=True)

    print('Obj = '+str(p2['Obj.Obj'][0]))
