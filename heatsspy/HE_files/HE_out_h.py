from openmdao.api import ExplicitComponent
import numpy as np

class HE_out_hout(ExplicitComponent):
    """ Calculate output enthaply"""

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int,
                             desc='Number of nodes to be evaluated')
        self.options.declare('dim', default=2, values=(1,2),
                             desc='number of temperatures to caculate')

    def setup(self):
        nn = self.options['num_nodes']
        dim = self.options['dim']
        if dim ==1:
            self.add_input('h', val=np.ones(nn), units='J/kg', desc='initial enthalpy')
            self.add_input('mdot', val=np.ones(nn), units='kg/s', desc='flow rate')
            self.add_input('q', val=np.ones(nn), units='W', desc='heat transfer rate')

            self.add_output('h_out', val=np.ones(nn), units='J/kg', desc='output enthalpy')

            arange = np.arange(self.options['num_nodes'])
            self.declare_partials(of=['h_out'], wrt=['h','q','mdot'], rows=arange, cols=arange)

        elif dim==2:
            self.add_input('h1', val=np.ones(nn), units='J/kg', desc='initial cold surface enthalpy')
            self.add_input('h2', val=np.ones(nn), units='J/kg', desc='initial hot surface enthalpy')
            self.add_input('mdot1', val=np.ones(nn), units='kg/s', desc='cold surface flow rate')
            self.add_input('mdot2', val=np.ones(nn), units='kg/s', desc='hot surface flow rate')
            self.add_input('q', val=np.ones(nn), units='W', desc='heat transfer rate')

            self.add_output('h1_out', val=np.ones(nn), units='J/kg', desc='cold side output enthalpy',lower = 250)
            self.add_output('h2_out', val=np.ones(nn), units='J/kg', desc='hot side output enthalpy',lower = 250)

            arange = np.arange(self.options['num_nodes'])
            self.declare_partials(of='h1_out', wrt=['h1','mdot1','q'], rows=arange, cols=arange)
            self.declare_partials(of='h2_out', wrt=['h2','mdot2','q'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        dim = self.options['dim']
        q  = inputs['q']
        if dim==1:
            h  = inputs['h']
            mdot = inputs['mdot']
            outputs['h_out']= h + q/mdot

        elif dim==2:
            h1  = inputs['h1']
            h2  = inputs['h2']
            mdot1 = inputs['mdot1']
            mdot2 = inputs['mdot2']

            outputs['h1_out'] = h1_out = h1 + q/mdot1
            outputs['h2_out'] = h2_out = h2 - q/mdot2

    def compute_partials(self, inputs, J):
        dim = self.options['dim']
        q  = inputs['q']
        if dim==1:
            mdot = inputs['mdot']
            J['h_out','h']    = 1
            J['h_out','q']    = 1/mdot
            J['h_out','mdot'] = - q/mdot**2


        elif dim==2:
            h1  = inputs['h1']
            h2  = inputs['h2']
            mdot1 = inputs['mdot1']
            mdot2 = inputs['mdot2']

            J['h1_out','h1'] = 1
            J['h1_out','mdot1'] = - q/mdot1**2
            J['h1_out','q']    = 1/mdot1

            J['h2_out','h2'] = 1
            J['h2_out','mdot2'] =  q/mdot2**2
            J['h2_out','q']    = - 1/mdot2


if __name__ == "__main__":

    import time
    from openmdao.api import Problem, IndepVarComp

    prob = Problem()
    Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    # Flow properties
    Vars.add_output('mdot', val=[3,3], units='kg/s', desc='flow rate')
    Vars.add_output('mdot1', val=[1,1], units='kg/s', desc='cold surface flow rate')
    Vars.add_output('mdot2', val=[2,2], units='kg/s', desc='hot surface flow rate')
    Vars.add_output('q', val=[2.1,2.1],units='W',desc='heat transfer')
    Vars.add_output('h', val=[3,3], units='J/kg', desc='flow enthalpy')
    Vars.add_output('h1', val=[1,1], units='J/kg', desc='cold side enthalpy')
    Vars.add_output('h2', val=[2,2], units='J/kg', desc='hot side enthalpy')

    Blk = prob.model.add_subsystem('prop_calc', HE_out_hout(num_nodes=2),
        promotes_inputs=['*'])
    prob.model.add_subsystem('prop_calc1', HE_out_hout(num_nodes=2,dim=1),
        promotes_inputs=[('*')])

    # Blk.set_check_partial_options(wrt='C_c', step_calc='rel')
    prob.setup(force_alloc_complex=True)
    prob.run_model()
    prob.check_partials(compact_print=True, method='cs')
    #
    print('h1_out = '+str(prob['prop_calc.h1_out']))
    print('h2_out = '+str(prob['prop_calc.h2_out']))
    print('h_out = '+str(prob['prop_calc1.h_out']))
