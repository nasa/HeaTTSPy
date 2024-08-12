""" FlowIN component which serves as an input flowstation for cycle components.
"""
import numpy as np

from openmdao.api import ExplicitComponent


class FlowIn(ExplicitComponent):
    """
    Provides a central place to connect flow information to in a component
    but doesn't actually do anything on its own
    """

    def initialize(self):
        self.options.declare('fl_name', default='flow',
                              desc='thermodynamic data set')
        self.options.declare('num_prods', default=0,
                              desc='concentrations of products in mixture')
        self.options.declare('unit_type', default='ENG', desc='output unit type', values=('SI', 'ENG', 'IMP'))
        self.options.declare('num_nodes', default=1, types=int,
                             desc='Number of nodes to be evaluated')

    def setup(self):
        nn = self.options['num_nodes']
        fl_name = self.options['fl_name']
        unit_type = self.options['unit_type']
        if unit_type == 'SI':
            h_units   = 'J/kg'
            T_units   = 'degK'
            P_units   = 'Pa'
            rho_units = 'kg/m**3'
            Cp_units  = 'J/kg/degK'
            Cv_units  = 'J/kg/degK'
            S_units   = 'J/kg/degK'
            mu_units  = 'Pa*s'
            k_units   = 'W/m/degK'
            W_units   = 'kg/s'
            V_units   = 'm/s'
            A_units   = 'm**2'
        else:
            h_units   ='Btu/lbm'
            T_units   ='degR'
            P_units   ='lbf/inch**2'
            rho_units ='lbm/ft**3'
            Cp_units  ='Btu/(lbm*degR)'
            Cv_units  ='Btu/(lbm*degR)'
            S_units   ='Btu/(lbm*degR)'
            mu_units  ='lbf*s/inch**2'
            k_units   ='Btu/(s*ft*degR)'
            W_units   ='lbm/s'
            V_units   ='ft/s'
            A_units   ='inch**2'

        self.add_output('fooBar')

        self.add_input('%s:tot:h'%fl_name, val=np.ones(nn), desc='total enthalpy', units=h_units)
        self.add_input('%s:tot:T'%fl_name, val=518.*np.ones(nn), desc='total temperature', units=T_units)
        self.add_input('%s:tot:P'%fl_name, val=np.ones(nn), desc='total pressure', units=P_units)
        self.add_input('%s:tot:rho'%fl_name, val=np.ones(nn), desc='total density', units=rho_units)
        self.add_input('%s:tot:gamma'%fl_name, val=np.ones(nn), desc='total ratio of specific heats')
        self.add_input('%s:tot:Cp'%fl_name, val=np.ones(nn), desc='total Specific heat at constant pressure', units=Cp_units)
        self.add_input('%s:tot:Cv'%fl_name, val=np.ones(nn), desc='total Specific heat at constant volume', units=Cv_units)
        self.add_input('%s:tot:S'%fl_name, val=np.ones(nn), desc='total entropy', units=S_units)
        self.add_input('%s:tot:mu'%fl_name, val=np.ones(nn), desc='viscosity', units=mu_units)
        self.add_input('%s:tot:k'%fl_name, val=np.ones(nn), desc='thermal conductivity', units=k_units)

        self.add_input('%s:stat:W'%fl_name, val= np.ones(nn), desc='weight flow', units=W_units)

    def compute(self, inputs, outputs):
        pass
