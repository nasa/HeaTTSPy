from openmdao.api import ExplicitComponent
import numpy as np

class HE_out_Tout(ExplicitComponent):
    """ Calculate output temperatures"""

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int,
                             desc='Number of nodes to be evaluated')
        self.options.declare('dim', default=2, values=(1,2),
                             desc='number of temperatures to caculate')

    def setup(self):
        nn = self.options['num_nodes']
        dim = self.options['dim']
        if dim ==1:
            self.add_input('C', val=np.ones(nn), units='W/degK', desc='cold side capacity rate')
            self.add_input('T_in', val=np.ones(nn), units='degK', desc='cold side input temperature')
            self.add_input('q', val=np.ones(nn), units='W', desc='heat transfer rate')

            self.add_output('T_out', val=np.ones(nn), units='degK', desc='cold side output temperature')

            arange = np.arange(self.options['num_nodes'])
            self.declare_partials(of='T_out', wrt=['C','q','T_in'], rows=arange, cols=arange)

        elif dim==2:
            self.add_input('C_c', val=np.ones(nn), units='W/degK', desc='cold side capacity rate')
            self.add_input('C_h', val=np.ones(nn), units='W/degK', desc='hot side capacity rate')
            self.add_input('T_c_in', val=np.ones(nn), units='degK', desc='cold side input temperature')
            self.add_input('T_h_in', val=np.ones(nn), units='degK', desc='hot side input temperature')
            self.add_input('q', val=np.ones(nn), units='W', desc='heat transfer rate')

            self.add_output('T_c_out', val=np.ones(nn), units='degK', desc='cold side output temperature',lower = 250)
            self.add_output('T_h_out', val=np.ones(nn), units='degK', desc='hot side output temperature',lower = 250)

            arange = np.arange(self.options['num_nodes'])
            self.declare_partials(of='T_c_out', wrt=['C_c','q','T_c_in'], rows=arange, cols=arange)
            self.declare_partials(of='T_h_out', wrt=['C_h','q','T_h_in'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        dim = self.options['dim']
        q  = inputs['q']
        if dim==1:
            C  = inputs['C']
            T_in = inputs['T_in']
            outputs['T_out']= T_in + q/C
        elif dim==2:
            C_c  = inputs['C_c']
            C_h  = inputs['C_h']
            T_c_in = inputs['T_c_in']
            T_h_in = inputs['T_h_in']

            # Kays and London Eqn: 2-6
            Th_out= T_h_in - q/C_h
            Tc_out= T_c_in + q/C_c
            # outputs['T_h_out'] = np.where(Th_out<T_c_in,T_c_in,Th_out)
            # outputs['T_c_out'] = np.where(Tc_out>T_h_in,T_h_in,Tc_out)

            outputs['T_h_out'] = Th_out= T_h_in - q/C_h
            outputs['T_c_out'] = Tc_out= T_c_in + q/C_c


    def compute_partials(self, inputs, J):
        dim = self.options['dim']
        q  = inputs['q']
        if dim==1:
            J['T_out','q']    = 1/inputs['C']
            J['T_out','T_in'] = 1
            J['T_out','C'] = -inputs['q']/inputs['C']**2
        elif dim==2:
            C_c  = inputs['C_c']
            C_h  = inputs['C_h']

            T_c_in = inputs['T_c_in']
            T_h_in = inputs['T_h_in']

            J['T_h_out','q']    = -1/C_h
            J['T_h_out','T_h_in'] = 1
            J['T_h_out','C_h'] = q/C_h**2

            J['T_c_out','q']    = 1/C_c
            J['T_c_out','T_c_in'] = 1
            J['T_c_out','C_c'] = -q/C_c**2


if __name__ == "__main__":

    import time
    from openmdao.api import Problem, IndepVarComp

    prob = Problem()
    Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    # Flow properties
    Vars.add_output('C_c', val=40000, units='Btu/h/degR', desc='maximum capacity rate')
    Vars.add_output('C_h', val=48800, units='Btu/h/degR', desc='minimum capacity rate')
    Vars.add_output('q', val=2574,units='W',desc='heat transfer')
    Vars.add_output('T_c_in', val=60, units='degF', desc='cold side input temperature')
    Vars.add_output('T_h_in', val=260, units='degF', desc='hot side input temperature')

    Blk = prob.model.add_subsystem('prop_calc', HE_out_Tout(),
        promotes_inputs=['C_c','C_h','q','T_c_in','T_h_in'])
    prob.model.add_subsystem('prop_calc1', HE_out_Tout(dim=1),
        promotes_inputs=[('C','C_c'),'q',('T_in','T_c_in')])

    # Blk.set_check_partial_options(wrt='C_c', step_calc='rel')
    prob.setup(force_alloc_complex=True)
    prob.run_model()
    prob.check_partials(compact_print=True, method='cs')
    #
    print('T_c_out = '+str(prob['prop_calc.T_c_out'][0]))
    print('T_h_out = '+str(prob['prop_calc.T_h_out'][0]))
    print('T_c_out = '+str(prob['prop_calc1.T_out'][0]))
