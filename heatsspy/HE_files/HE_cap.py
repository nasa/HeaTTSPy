from openmdao.api import ExplicitComponent
import numpy as np


class HE_cap(ExplicitComponent):
    """ Calculate capcity rates"""

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int,
                             desc='Number of nodes to be evaluated')
        self.options.declare('dim',default=2, desc='flow dimension',values=(1, 2))

    def setup(self):
        nn = self.options['num_nodes']
        dim = self.options['dim']
        arange = np.arange(self.options['num_nodes'])
        if dim == 1:
            self.add_input('mdot', val=np.ones(nn), units="kg/s", desc='mass flow input')
            self.add_input('Cp', val=np.ones(nn), units='J/kg/degK', desc='specific heat with constant pressure')

            self.add_output('C', val=np.ones(nn), units='W/degK', desc='capacity rate')
            self.add_output('CR', val=np.ones(nn), desc='capacity rate ratio')

            self.declare_partials(of='C', wrt=['mdot','Cp'], rows=arange, cols=arange)
            self.declare_partials(of='CR', wrt=['mdot','Cp'], val=0)

        else:
            self.add_input('mdot1', val=np.ones(nn), units="kg/s", desc='cold side mass flow input')
            self.add_input('Cp1', val=np.ones(nn), units='J/kg/degK', desc='cold side specific heat with constant pressure')

            self.add_input('mdot2', val=np.ones(nn), units="kg/s", desc='hot side mass flow input')
            self.add_input('Cp2', val=np.ones(nn), units='J/kg/degK', desc='hot side specific heat with constant pressure')

            self.add_output('C1', val=np.ones(nn), units='W/degK', desc='cold side capacity rate',lower = 1e-5)
            self.add_output('C2', val=np.ones(nn), units='W/degK', desc='hot side capacity rate',lower = 1e-5)
            self.add_output('C_max', val=np.ones(nn), units='W/degK', desc='max capacity rate',lower = 1e-5)
            self.add_output('C_min', val=np.ones(nn), units='W/degK', desc='min capacity rate',lower = 1e-5)
            self.add_output('CR', val=np.ones(nn), desc='capacity rate ratio',lower = 1e-6)

            self.declare_partials(of='C1', wrt=['mdot1','Cp1'], rows=arange, cols=arange)
            self.declare_partials(of='C2', wrt=['mdot2','Cp2'], rows=arange, cols=arange)
            self.declare_partials(of=['C_max','C_min','CR'], wrt=['mdot2','Cp2','mdot1','Cp1'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        dim = self.options['dim']
        if dim ==1:
            mdot  = inputs['mdot']
            Cp  = inputs['Cp']

            outputs['C'] = mdot*Cp # incropera eqn 11.10
            outputs['CR'] = 0
        else:
            mdot2  = inputs['mdot2']
            Cp2  = inputs['Cp2']
            mdot1  = inputs['mdot1']
            Cp1  = inputs['Cp1']
            C_max = np.zeros(nn) #initialize
            C_min = np.zeros(nn) #initialize
            outputs['C2'] = C2 = mdot2*Cp2 # incropera eqn 11.10
            outputs['C1'] = C1 = mdot1*Cp1 # incropera eqn 11.11
            # if C2>C1:
            #     print('C1 low')
            # else:
            #     print('C2 low')
            hgc = np.where(C2 > C1) # C1 is minimum
            outputs['C_max'][hgc] = C_max[hgc] = C2[hgc]
            outputs['C_min'][hgc] = C_min[hgc] = C1[hgc]
            cgh = np.where(C2 <= C1) # C2 is minimum
            outputs['C_max'][cgh] = C_max[cgh] = C1[cgh]
            outputs['C_min'][cgh] = C_min[cgh] = C2[cgh]

            outputs['CR'] = C_min/C_max

    def compute_partials(self, inputs, J):
        nn = self.options['num_nodes']
        dim = self.options['dim']
        if dim == 1:
            mdot = inputs['mdot']
            Cp = inputs['Cp']

            J['C','mdot'] = Cp
            J['C','Cp'] = mdot

        else:
            mdot2  = inputs['mdot2']
            Cp2  = inputs['Cp2']
            mdot1  = inputs['mdot1']
            Cp1  = inputs['Cp1']

            J['C2','mdot2'] = Cp2
            J['C2','Cp2'] = mdot2

            J['C1','mdot1'] = Cp1
            J['C1','Cp1'] = mdot1

            hgc = np.where(mdot2*Cp2 > mdot1*Cp1) # C1 is minimum
            J['C_min','mdot2'][hgc] =  0
            J['C_min','mdot1'][hgc] =  Cp1[hgc]
            J['C_min','Cp2'][hgc] =  0
            J['C_min','Cp1'][hgc] =  mdot1[hgc]

            J['C_max','mdot2'][hgc] =  Cp2[hgc]
            J['C_max','mdot1'][hgc] =  0
            J['C_max','Cp2'][hgc] =  mdot2[hgc]
            J['C_max','Cp1'][hgc] =  0

            J['CR','mdot2'][hgc]  = - mdot1[hgc]*Cp1[hgc]/mdot2[hgc]**2/Cp2[hgc]
            J['CR','mdot1'][hgc] = Cp1[hgc]/mdot2[hgc]/Cp2[hgc]
            J['CR','Cp2'][hgc]  = - mdot1[hgc]*Cp1[hgc]/mdot2[hgc]/Cp2[hgc]**2
            J['CR','Cp1'][hgc] =  mdot1[hgc]/mdot2[hgc]/Cp2[hgc]

            cgh = np.where(mdot2*Cp2 <= mdot1*Cp1) # C2 is minimum
            J['C_min','mdot2'][cgh] = Cp2[cgh]
            J['C_min','mdot1'][cgh] =  0.0
            J['C_min','Cp2'][cgh] = mdot2[cgh]
            J['C_min','Cp1'][cgh] =  0.0

            J['C_max','mdot2'][cgh] =  0
            J['C_max','mdot1'][cgh] =  Cp1[cgh]
            J['C_max','Cp2'][cgh] =  0
            J['C_max','Cp1'][cgh] =  mdot1[cgh]

            J['CR','mdot2'][cgh]  = Cp2[cgh]/mdot1[cgh]/Cp1[cgh]
            J['CR','mdot1'][cgh] = - mdot2[cgh]*Cp2[cgh]/mdot1[cgh]**2/Cp1[cgh]
            J['CR','Cp2'][cgh]  = mdot2[cgh]/mdot1[cgh]/Cp1[cgh]
            J['CR','Cp1'][cgh] = - mdot2[cgh]*Cp2[cgh]/mdot1[cgh]/Cp1[cgh]**2


if __name__ == "__main__":

    import time
    from openmdao.api import Problem, IndepVarComp

    nn = 2
    prob = Problem()
    Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    # Flow properties
    Vars.add_output('mdot2', val=np.array([25.2,37.8]), units="kg/s", desc='hot side mass flow input')
    # Vars.add_output('mdot2', val=np.array([200000,300000]), units="lbm/h", desc='hot side mass flow input')
    Vars.add_output('Cp2', val=np.array([1022,4187]), units='J/kg/degK', desc='hot side specific heat with constant pressure')
    # Vars.add_output('Cp2', val=np.array([0.244,1.0]), units='Btu/lbm/degF', desc='hot side specific heat with constant pressure')

    Vars.add_output('mdot1', val=np.array([50.4,25.2]), units="kg/s", desc='cold side mass flow input')
    # Vars.add_output('mdot1', val=np.array([400000,200000]), units="lbm/h", desc='cold side mass flow input')
    Vars.add_output('Cp1', val=np.array([4187,1022]), units='J/kg/degK', desc='cold side specific heat with constant pressure')
    # Vars.add_output('Cp1', val=np.array([1.0,0.244]), units='Btu/lbm/degF', desc='cold side specific heat with constant pressure')

    Blk = prob.model.add_subsystem('prop_calc', HE_cap(num_nodes=nn),
        promotes_inputs=['*'])

    Blk1 = prob.model.add_subsystem('prop_calc1', HE_cap(num_nodes=nn, dim=1),
        promotes_inputs=[('mdot','mdot2'), ('Cp','Cp2')])

    # Blk.set_check_partial_options(wrt='mu', step_calc='rel')
    prob.setup(force_alloc_complex=True)
    prob.run_model()
    prob.check_partials(compact_print=True, method='fd')
    #
    print('C2 = ',prob.get_val('prop_calc.C2', units='kW/degK'))
    print('C1 = ',prob.get_val('prop_calc.C1', units='kW/degK'))
    print('C_max = '+str(prob['prop_calc.C_max']))
    print('C_min = ',prob.get_val('prop_calc.C_min', units='kW/degK'))
    print('CR = '+str(prob['prop_calc.CR']))
    print('C = '+str(prob['prop_calc1.C']))
