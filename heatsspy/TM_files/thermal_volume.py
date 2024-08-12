from openmdao.api import ExplicitComponent
import numpy as np

class thermal_volume(ExplicitComponent):
    ''' Define reservoir temperature'''
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('calc_weight_based', default=False)

    def setup(self):
        nn = self.options['num_nodes']
        calc_weight_based = self.options['calc_weight_based']
        # Inputs
        self.add_input('Cp',val=np.ones(nn),units='J/(kg * K)',desc='specific heat of fluid')
        self.add_input('h_in',val=np.ones(nn),units='J/kg',desc='enthalpy into reservoir')
        self.add_input('h_out',val=np.ones(nn),units='J/kg',desc='heat out of reservoir')
        self.add_input('W',val=np.ones(nn),units='kg/s',desc='mass flow of fluid')
        if calc_weight_based:
            self.add_input('fluid_weight',val=np.ones(nn),units='kg',desc='reservoir fluid weight')
        else:
            self.add_input('rho',val=np.ones(nn),units='kg/m**3',desc='density')
            self.add_input('vol',val=np.ones(nn),units='m**3',desc='reservoir volume')

        self.add_output('Tdot',val=np.ones(nn),units='K/s',desc='dT')
        arange = np.arange(self.options['num_nodes'])
        if calc_weight_based:
            self.declare_partials(of='Tdot', wrt=['Cp','h_in','h_out','W', 'fluid_weight'], rows=arange, cols=arange)
        else:
            self.declare_partials(of='Tdot', wrt=['Cp','h_in','h_out','rho','W', 'vol'], rows=arange, cols=arange)


    def compute(self, inputs, outputs):
        if self.options['calc_weight_based']:
            outputs['Tdot'] = inputs['W']*(inputs['h_in']-inputs['h_out'])/(inputs['fluid_weight']*inputs['Cp'])
        else:
            outputs['Tdot'] = inputs['W']*(inputs['h_in']-inputs['h_out'])/(inputs['rho']*inputs['vol']*inputs['Cp'])
        # print(self.pathname)
        # print(inputs['h_in'])

    def compute_partials(self, inputs, J):
        if self.options['calc_weight_based']:
            fluid_weight = inputs['fluid_weight']
            J['Tdot','fluid_weight'] = -inputs['W']*(inputs['h_in']-inputs['h_out'])/(inputs['fluid_weight']**2*inputs['Cp'])
        else:
            fluid_weight = inputs['rho']*inputs['vol']
            J['Tdot','rho'] = -inputs['W']*(inputs['h_in']-inputs['h_out'])/(inputs['rho']**2*inputs['vol']*inputs['Cp'])
            J['Tdot','vol'] = -inputs['W']*(inputs['h_in']-inputs['h_out'])/(inputs['rho']*inputs['vol']**2*inputs['Cp'])

        J['Tdot','Cp'] = - inputs['W']*(inputs['h_in']-inputs['h_out'])/(fluid_weight*inputs['Cp']**2)
        J['Tdot','h_in'] = inputs['W']/(fluid_weight*inputs['Cp'])
        J['Tdot','h_out'] = - inputs['W']/(fluid_weight*inputs['Cp'])
        J['Tdot','W'] = (inputs['h_in']-inputs['h_out'])/(fluid_weight*inputs['Cp'])


class thermal_volume_weight(ExplicitComponent):
    ''' Estimate reservoir weight'''
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
        self.options.declare('Al_rho', default=2700, desc='density of aluminum')# units='kg/m**3'
        self.options.declare('t', default=0.003, desc='reservoir thickness')# units='m'
        self.options.declare('include_tank', default=True, desc='consider tank in calculation')
    def setup(self):
        nn = self.options['num_nodes']
        # Inputs
        self.add_input('rho',val=np.ones(nn),units='kg/m**3',desc='density')
        self.add_input('vol',val=np.ones(nn),units='m**3',desc='reservoir volume')

        self.add_output('Wt_res',val=np.ones(nn),units='kg',desc='reservoir weight')

        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='Wt_res', wrt=['rho', 'vol'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        rho_Al = self.options['Al_rho']
        t = self.options['t']
        include_tank = self.options['include_tank']
        if include_tank:
          outputs['Wt_res'] = inputs['rho']*inputs['vol']+6*rho_Al*t*inputs['vol']**(2/3)
        else:
          outputs['Wt_res'] = inputs['rho']*inputs['vol']
        # print(self.pathname)

    def compute_partials(self, inputs, J):
        rho_Al = self.options['Al_rho']
        t = self.options['t']
        include_tank=self.options['include_tank']
        vol = inputs['vol']
        rho = inputs['rho']

        J['Wt_res','rho']= vol
        vlz = np.where(vol > 0)
        vgez = np.where(vol <= 0)

        if include_tank:
            J['Wt_res','vol'][vlz] = rho[vlz]+4*rho_Al*t*vol[vlz]**(-1/3)
            J['Wt_res','vol'][vgez] = 9999
        else:
          J['Wt_res','vol']= rho


if __name__ == "__main__":

    import time
    from openmdao.api import Problem, Group, IndepVarComp, ExplicitComponent

    prob = Problem()
    Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    # Flow properties
    Vars.add_output('Cp',val=0.004186,units='J/(kg * K)',desc='specific heat')
    Vars.add_output('h_in',val=2,units='J/kg',desc='enthalpy into reservoir')
    Vars.add_output('h_out',val=1,units='J/kg',desc='enthalpy out of reservoir')
    Vars.add_output('rho',val=997,units='kg/m**3',desc='density')
    Vars.add_output('vol',val=0.01,units='m**3',desc='reservoir volume')
    Vars.add_output('W',val=1.2,units='kg/s',desc='incoming flow')

    Blk = prob.model.add_subsystem('res_comp',thermal_volume(num_nodes=1),promotes=['*'])
    Blk2 = prob.model.add_subsystem('Wt_res_comp',thermal_volume_weight(num_nodes=1,include_tank=True),promotes=['*'])
    # Blk.set_check_partial_options(wrt=['vol','Cp'], step_calc='rel')
    prob.setup(force_alloc_complex=True)

    prob.run_model()
    prob.check_partials(compact_print=True, method='cs')
    print('dT ='+str(prob['Tdot'][0]))
    print('Wt ='+str(prob['Wt_res'][0]))
