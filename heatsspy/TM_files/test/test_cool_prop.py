from __future__ import print_function, division, absolute_import
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal
from heatsspy.api import cool_prop

class TestPropLookup(unittest.TestCase):
    def test_ts_run(self):
        import time
        from openmdao.api import Problem, IndepVarComp

        prob = Problem()
        nn = 2
        Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
        # Flow properties
        Vars.add_output('Pset', 2e5*np.ones(nn), units='Pa')  
        Vars.add_output('Tset', 300*np.ones(nn), units='degK')  
        Vars.add_output('Sset', 723*np.ones(nn), units='J/kg/degK')  
        Vars.add_output('Hset', 1.3e5*np.ones(nn), units='J/kg')  

        Blk = prob.model.add_subsystem('setTotal_T',cool_prop(mode='T', fluid = 'water', num_nodes=nn),promotes_inputs=['Tset','Pset'])
        Blk2 = prob.model.add_subsystem('setTotal_S',cool_prop(mode='S', fluid = 'water', num_nodes=nn),promotes_inputs=['Pset','Sset'])
        Blk3 = prob.model.add_subsystem('setTotal_H',cool_prop(mode='H', fluid = 'water', num_nodes=nn),promotes_inputs=['Pset','Hset'])

        # Blk.set_check_partial_options(wrt='*', step_calc='rel')
        st=time.time()
        prob.setup(check=False)

        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=1)
        prob.run_model()

        # check values
        print('nn ='+str(np.size(prob['setTotal_T.P'])))
        assert_near_equal(np.size(prob['setTotal_T.P']), 2.0, 1e-4)

        print('h_T ='+str(prob['setTotal_T.H'][0]))
        assert_near_equal(prob['setTotal_T.H'][0], 112745.74907582274, 1e-4)
        print('s_T ='+str(prob['setTotal_T.S'][0]))
        assert_near_equal(prob['setTotal_T.S'][0], 393.034854, 1e-4)
        print('mu_T ='+str(prob['setTotal_T.mu'][0]))
        assert_near_equal(prob['setTotal_T.mu'][0], 0.0008537335667733899, 1e-7)
        print('rho_T ='+str(prob['setTotal_T.rho'][0]))
        assert_near_equal(prob['setTotal_T.rho'][0], 996.6012320, 1e-4)

        print('T_H ='+str(prob['setTotal_H.T'][0]))
        assert_near_equal(prob['setTotal_H.T'][0], 304.1279966132795, 1e-4)
        print('s_H ='+str(prob['setTotal_H.S'][0]))
        assert_near_equal(prob['setTotal_H.S'][0], 450.1569346932293, 1e-4)
        print('mu_H ='+str(prob['setTotal_H.mu'][0]))
        assert_near_equal(prob['setTotal_H.mu'][0], 0.0007808961787681463, 1e-7)
        print('rho_H ='+str(prob['setTotal_H.rho'][0]))
        assert_near_equal(prob['setTotal_H.rho'][0], 995.3937943735558, 1e-4)

        print('T_S ='+str(prob['setTotal_S.T'][0]))
        assert_near_equal(prob['setTotal_S.T'][0], 324.64334470468884, 1e-4)
        print('H_S ='+str(prob['setTotal_S.H'][0]))
        assert_near_equal(prob['setTotal_S.H'][0], 215747.757670402, 1e-4)
        print('mu_S ='+str(prob['setTotal_S.mu'][0]))
        assert_near_equal(prob['setTotal_S.mu'][0], 0.0005331130332852482, 1e-7)
        print('rho_S ='+str(prob['setTotal_S.rho'][0]))
        assert_near_equal(prob['setTotal_S.rho'][0], 987.3953939789972, 1e-4)



        # outputs['Cp'] = CP.PropsSI(f'd(Hmass)/d(T)|P','P',Pset,f'{mode}',set_value,fluid)
        # outputs['Cv'] = CP.PropsSI(f'd(Umass)/d(T)|Dmass','P',Pset,f'{mode}',set_value,fluid)
        # outputs['mu'] = CP.PropsSI('viscosity','P',Pset,f'{mode}',set_value,fluid)
        # outputs['k'] = CP.PropsSI('conductivity','P',Pset,f'{mode}',set_value,fluid)
        # outputs['rho'] = CP.PropsSI('Dmass','P',Pset,f'{mode}',set_value,fluid)



if __name__ == "__main__":

    unittest.main()
