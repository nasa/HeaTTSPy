
import numpy as np
from openmdao.api import Group, IndepVarComp, ExplicitComponent
from heatsspy.api import FlowIn, FlowStart
from heatsspy.api import SetTotal, PassThrough

class line_eq(ExplicitComponent):
    def initialize(self):
        self.options.declare('units', default='kg/s')
        self.options.declare('num_nodes', default=1, types=int,
                                 desc='Number of nodes to be evaluated')

    def setup(self):
        nn = self.options['num_nodes']
        units = self.options['units']
        self.add_input('X', val=np.ones(nn), units=units, desc='input X')
        self.add_input('M', val=np.ones(nn), desc='Multiplier')
        self.add_input('b', val=0.0*np.ones(nn), desc='y intercept')

        self.add_output('Y', val=np.ones(nn), units=units, desc='output Y')
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='Y', wrt=['X','M','b'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        outputs['Y'] = inputs['M']*inputs['X'] + inputs['b']

    def compute_partials(self, inputs, J):
        J['Y','X'] = inputs['M']
        J['Y','M'] = inputs['X']
        J['Y','b'] = 1.0


class weighted_average(ExplicitComponent):
    def initialize(self):
        self.options.declare('Wunits', default='kg/s')
        self.options.declare('hunits', default='J/kg')
        self.options.declare('num_nodes', default=1, types=int,
                             desc='Number of nodes to be evaluated')
    def setup(self):
        nn = self.options['num_nodes']
        Wunits = self.options['Wunits']
        hunits = self.options['hunits']
        self.add_input('W1',val=np.ones(nn), units=Wunits)
        self.add_input('W2',val=np.ones(nn), units=Wunits)
        self.add_input('h1',val=np.ones(nn), units=hunits)
        self.add_input('h2',val=np.ones(nn), units=hunits)

        self.add_output('W_sum',val=np.ones(nn), units=Wunits)
        self.add_output('h_wavg',val=np.ones(nn), units=hunits)
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='W_sum', wrt=['W1','W2'], val = 1, rows=arange, cols=arange)
        self.declare_partials(of='h_wavg', wrt=['W1','W2','h1','h2'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        W1 = inputs['W1']
        W2 = inputs['W2']
        h1 = inputs['h1']
        h2 = inputs['h2']
        outputs['W_sum'] = W1+W2
        outputs['h_wavg'] = (W1*h1+W2*h2)/(W1+W2)

    def compute_partials(self,inputs,J):
        W1 = inputs['W1']
        W2 = inputs['W2']
        h1 = inputs['h1']
        h2 = inputs['h2']

        J['h_wavg', 'W1'] = W2*(h1-h2)/(W1+W2)**2
        J['h_wavg', 'W2'] = W1*(h2-h1)/(W1+W2)**2
        J['h_wavg', 'h1'] = W1/(W1+W2)
        J['h_wavg', 'h2'] = W2/(W1+W2)


class FlowSplit(Group):
    def initialize(self):
        self.options.declare('thermo', default='file')
        self.options.declare('fluid', default='oil')
        self.options.declare('unit_type', default='SI')
        self.options.declare('s_W',default=1.0)
        self.options.declare('s_b',default=0.0)
        self.options.declare('num_nodes', default=1, types=int,
                             desc='Number of nodes to be evaluated')
    def setup(self):
    	nn = self.options['num_nodes']
    	thermo = self.options['thermo']
    	fluid = self.options['fluid']
    	unit_type = self.options['unit_type']
    	s_W = self.options['s_W']
    	s_b = self.options['s_b']

    	flow_in = FlowIn(fl_name='Fl_I',unit_type=unit_type, num_nodes=nn)
    	self.add_subsystem('flow_in', flow_in, promotes=['Fl_I:tot:*', 'Fl_I:stat:*'])

    	Vars =  self.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    	Vars.add_output('M', val=s_W*np.ones(nn), units=None, desc='flow split')
    	Vars.add_output('b', val=s_b*np.ones(nn), units=None, desc='flow adder')

    	self.add_subsystem('Wcalc',line_eq(units='kg/s',num_nodes=nn),
    		promotes_inputs=['M',('X','Fl_I:stat:W'),'b'],
    		promotes_outputs=[('Y','W_out')])

    	self.add_subsystem('FlowStart', FlowStart(thermo=thermo , fluid=fluid, unit_type=unit_type,num_nodes=nn),
    		promotes_inputs=[('W','W_out'),('T','Fl_I:tot:T'),('P','Fl_I:tot:P')],
    		promotes_outputs=['Fl_O*'])


class FlowCombine(Group):
    def initialize(self):
        self.options.declare('thermo', default='file')
        self.options.declare('fluid', default='oil')
        self.options.declare('unit_type', default='SI')
        self.options.declare('num_nodes', default=1, types=int, desc='Number of nodes to be evaluated')
    def setup(self):
        nn = self.options['num_nodes']
        thermo = self.options['thermo']
        fluid = self.options['fluid']
        unit_type = self.options['unit_type']

        flow_in1 = FlowIn(fl_name='Fl_I1',unit_type=unit_type, num_nodes=nn)
        self.add_subsystem('flow_in1', flow_in1, promotes=['Fl_I1:tot:*', 'Fl_I1:stat:*'])

        flow_in2 = FlowIn(fl_name='Fl_I2',unit_type=unit_type, num_nodes=nn)
        self.add_subsystem('flow_in2', flow_in2, promotes=['Fl_I2:tot:*', 'Fl_I2:stat:*'])
        # combine Flows,  weighted average temperatures
        self.add_subsystem('Wcalc',weighted_average(Wunits='kg/s',hunits='J/kg', num_nodes=nn),
            promotes_inputs=[('W1','Fl_I1:stat:W'),('W2','Fl_I2:stat:W'),('h1','Fl_I1:tot:h'),('h2','Fl_I2:tot:h')],
            promotes_outputs=[('W_sum','W_out'),('h_wavg','h_out')])
        # recalc flow properties, assume pressure = P1
        self.add_subsystem('set_total',SetTotal(mode='h',thermo=thermo, fluid = fluid, num_nodes=nn,fl_name='Fl_O:tot'),
            promotes_inputs=[('h','h_out'),('P','Fl_I1:tot:P')],
            promotes_outputs=['Fl_O:tot:*'])

        self.add_subsystem('W_passthru', PassThrough('W_out', 'Fl_O:stat:W', val=np.ones(nn), units= "kg/s"), promotes=['*'])


if __name__ == "__main__":
    import time
    from openmdao.api import Problem, Group, IndepVarComp, ExplicitComponent
    from heatsspy.api import FlowStart
    from heatsspy.api import connect_flow
    from heatsspy.include.props_water import water_props

    prob = Problem()
    Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    # Flow properties
    Vars.add_output('W1_in', 0.25, units='kg/s')
    Vars.add_output('T1_in', 300.0, units='degK')
    Vars.add_output('P1_in', 1e5, units='Pa')

    Vars.add_output('W2_in', 0.75, units='kg/s')
    Vars.add_output('T2_in', 310.0, units='degK')
    Vars.add_output('P2_in', 1.1e5, units='Pa')

    fluid = water_props()
    tval = 'file'
    prob.model.add_subsystem('FS1', FlowStart(thermo=tval, fluid=fluid, unit_type='SI'),
        promotes_inputs=[('W','W1_in'),('T','T1_in'),('P','P1_in')])
    prob.model.add_subsystem('Fsplt',FlowSplit(s_W=0.5,thermo=tval, fluid=fluid, unit_type='SI'))
    connect_flow(prob.model, 'FS1.Fl_O', 'Fsplt.Fl_I')

    prob.model.add_subsystem('FS2', FlowStart(thermo=tval, fluid=fluid, unit_type='SI'),
        promotes_inputs=[('W','W2_in'),('T','T2_in'),('P','P2_in')])
    prob.model.add_subsystem('Fa',FlowCombine(thermo=tval, fluid=fluid, unit_type='SI'))
    connect_flow(prob.model, 'FS1.Fl_O', 'Fa.Fl_I1')
    connect_flow(prob.model, 'FS2.Fl_O', 'Fa.Fl_I2')

    prob.setup()
    prob.run_model()
    prob.check_partials(compact_print=True)

    print('W1 = '+str(prob['FS1.Fl_O:stat:W']))
    print('W2 = '+str(prob['FS2.Fl_O:stat:W']))
    print('Wsplt = '+str(prob['Fsplt.Fl_O:stat:W']))
    print('Wsum = '+str(prob['Fa.Fl_O:stat:W']))
    print('Tavg = '+str(prob['Fa.Fl_O:tot:T']))
    print('Pout = '+str(prob['Fa.Fl_O:tot:P']))
