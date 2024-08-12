from __future__ import print_function, division, absolute_import
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.api import Group
from heatsspy.api import FlowIn, FlowStart, connect_flow, PassThrough

class TestGroup(Group):
    def initialize(self):
        self.options.declare('fluid', desc='fluid type')
        self.options.declare('num_nodes', default=1, types=int)
        self.options.declare('thermo', desc='thermo package', default='file' )
    def setup(self):
        nn = self.options['num_nodes']
        fluid = self.options['fluid']
        thermo = self.options['thermo']

        flow_in = FlowIn(fl_name='Fl_I',unit_type='SI', num_nodes=nn)
        self.add_subsystem('flow_in', flow_in, promotes=['Fl_I:tot:*', 'Fl_I:stat:*'])

        self.add_subsystem('W_passthru', PassThrough('Fl_I:stat:W', 'Fl_O:stat:W', val=np.ones(nn), units= "kg/s"),
            promotes=['*'])
        self.add_subsystem('P_passthru', PassThrough('Fl_I:tot:P', 'Fl_O:tot:P', val=np.ones(nn), units= "Pa"),
            promotes=['*'])


class TestPropLookup(unittest.TestCase):
    def test_ts_run(self):
        import time
        from openmdao.api import Problem, IndepVarComp
        from heatsspy.include.props_water import water_props
        fluid = water_props()

        prob = Problem()
        nn = 2
        Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes=['*'])
        # Flow properties
        Vars.add_output('W', 10*np.ones(nn), units='kg/s')
        Vars.add_output('P', 1e5*np.ones(nn), units='Pa')
        Vars.add_output('T', 350*np.ones(nn), units='degK')

        prob.model.add_subsystem('FS', FlowStart(thermo='file' , fluid=fluid, num_nodes=nn),
                    promotes_inputs=['*'])
        prob.model.add_subsystem('flow_pass', TestGroup(fluid=fluid, num_nodes=nn, thermo='file'),promotes_outputs=['*'])
        connect_flow(prob.model, 'FS.Fl_O', 'flow_pass.Fl_I')

        # Blk.set_check_partial_options(wrt='*', step_calc='rel')
        st=time.time()
        prob.setup(check=False)

        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=1)
        prob.run_model()

        # check values
        print('nn ='+str(np.size(prob['Fl_O:tot:P'])))
        assert_near_equal(np.size(prob['Fl_O:tot:P']), 2.0, 1e-4)
        print('P ='+str(prob['Fl_O:tot:P'][0]))
        assert_near_equal(prob['Fl_O:tot:P'][0], 1e5, 1e-4)
        print('W ='+str(prob['Fl_O:stat:W'][0]))
        assert_near_equal(prob['Fl_O:stat:W'][0], 10.0, 1e-4)


if __name__ == "__main__":

    unittest.main()
