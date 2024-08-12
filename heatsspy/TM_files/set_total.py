from openmdao.api import ExplicitComponent, Group, IndepVarComp
import numpy as np

import inspect
from six import iteritems, PY3

if PY3:
    _full_out_args = inspect.getfullargspec(ExplicitComponent.add_output)
    _allowed_out_args = set(_full_out_args.args[3:] + _full_out_args.kwonlyargs)
else:
    _full_out_args = inspect.getargspec(ExplicitComponent.add_output)
    _allowed_out_args = set(_full_out_args.args[3:])


class UnitConv(ExplicitComponent):
    def initialize(self):
        self.options.declare('fl_name', default="flow", desc='flowstation name of the output flow variables')
        self.options.declare('unit_type', default='SI', desc='output unit type', values=('SI', 'ENG', 'IMP'))
        self.options.declare('num_nodes', default=1, types=int)

    def setup(self):
        nn = self.options['num_nodes']
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

        self.add_input('h', val=np.ones(nn), desc='total enthalpy', units=h_units)
        self.add_input('T', val=300*np.ones(nn), desc='total temperature', units=T_units)
        self.add_input('P', val=1e5*np.ones(nn), desc='total pressure', units=P_units)
        self.add_input('rho', val=np.ones(nn), desc='total density', units=rho_units)
        self.add_input('gamma', val=np.ones(nn), desc='total ratio of specific heats')
        self.add_input('Cp', val=np.ones(nn), desc='total Specific heat at constant pressure', units=Cp_units)
        self.add_input('Cv', val=np.ones(nn), desc='total Specific heat at constant volume', units=Cv_units)
        self.add_input('S', val=np.ones(nn), desc='total entropy', units=S_units)
        self.add_input('mu', val=np.ones(nn), desc='viscosity', units=mu_units)
        self.add_input('k', val=np.ones(nn), desc='thermal conductivity', units=k_units)

        rel2meta = self._var_rel2meta

        fl_name = self.options['fl_name']

        for in_name in self._var_rel_names['input']:

            meta = rel2meta[in_name]
            val = meta['val'].copy()
            new_meta = {k:v for k, v in iteritems(meta) if k in _allowed_out_args}

            out_name = '{0}:{1}'.format(fl_name, in_name)

            self.add_output(out_name, val=val, **new_meta)

        rel2meta = self._var_rel2meta

        for in_name, out_name in zip(self._var_rel_names['input'], self._var_rel_names['output']):

            shape = rel2meta[in_name]['shape']
            size = np.prod(shape)
            row_col = np.arange(size, dtype=int)

            self.declare_partials(of=out_name, wrt=in_name,
                                  val=np.ones(size), rows=row_col, cols=row_col)
    def compute(self, inputs, outputs):
        outputs._data[:] = inputs._data


class gamma_calc(ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('Cp', val=np.ones(nn), units='J/kg/degK', desc='specific heat with constant pressure')
        self.add_input('Cv', val=np.ones(nn), units='J/kg/degK', desc='specific heat with constant volume')

        self.add_output('gamma', val=np.ones(nn), desc='ratio of specific heat values',lower= 1e-5)
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='gamma',wrt=['Cp','Cv'], rows=arange, cols=arange)

    def compute(self,inputs,outputs):
        Cp = inputs['Cp']
        Cv = inputs['Cv']
        outputs['gamma'] = Cp/Cv

    def compute_partials(self,inputs,J):
        Cp = inputs['Cp']
        Cv = inputs['Cv']

        J['gamma','Cp'] = 1.0/Cv
        J['gamma','Cv'] = - Cp/Cv**2


class SetTotal(Group):
    def initialize(self):
        self.options.declare('fl_name', default="flow", desc='flowstation name of the output flow variables')
        self.options.declare('fluid',default='oil',desc='fluid properties class or cool prop fluid name')
        self.options.declare('mode', desc='the input variable that defines the total properties', default='T', values=('T', 'h'))
        self.options.declare('num_nodes', default=1, types=int)
        self.options.declare('thermo', default='file', desc='thermo package', values=('cool_prop', 'file'))
        self.options.declare('unit_type', default='SI', desc='output unit type', values=('SI', 'ENG', 'IMP'))

    def setup(self):
        fl_name = self.options['fl_name']
        fluid = self.options['fluid']
        mode = self.options['mode']
        nn = self.options['num_nodes']
        thermo = self.options['thermo']
        unit_type = self.options['unit_type']

        if thermo == 'cool_prop':
            from heatsspy.TM_files.cool_prop import cool_prop
            cool_vars = ['rho','Cp','Cv','mu','k']
            if mode == 'T':
                cool_vars = cool_vars+['S',('H','h')]
                self.add_subsystem('setTotal',cool_prop(mode='T', fluid=fluid,num_nodes=nn),
                    promotes_inputs=[('Pset','P'),('Tset','T')],
                    promotes_outputs=cool_vars)
            # elif mode == 'S':
            #     cool_vars = cool_vars+['T',('H','h')]
            #     self.add_subsystem('setTotal',cool_prop(mode='S', fluid=fluid,num_nodes=nn),
            #         promotes_inputs=[('Pset','P'),('Sset','S')],
            #         promotes_outputs=cool_vars)
            elif mode == 'h':
                cool_vars = cool_vars+['T','S']
                self.add_subsystem('setTotal',cool_prop(mode='H', fluid=fluid,num_nodes=nn),
                    promotes_inputs=[('Pset','P'),('Hset','h')],
                    promotes_outputs=cool_vars)
        elif thermo == 'file':
            from heatsspy.TM_files.prop_lookup import prop_lookup
            not_calculated_variables =  self.add_subsystem('not_calculated_variables',IndepVarComp() ,promotes_outputs=['*'])
            not_calculated_variables.add_output('Cv', 1.0*np.ones(nn), units='J/kg/degK')
            not_calculated_variables.add_output('S', 1.0*np.ones(nn), units='J/kg/degK')
            cool_vars = ['rho','Cp','mu','k']
            if mode == 'T':
                cool_vars = cool_vars+['h']
                self.add_subsystem('prop_lookup_T',prop_lookup(mode='T', fluid_props=fluid,num_nodes=nn),
                    promotes_inputs=[('Tset','T'),('Pset','P')],
                    promotes_outputs=cool_vars)
            elif mode == 'h':
                cool_vars = cool_vars+[('Tbal','T')]
                self.add_subsystem('prop_lookup_h',prop_lookup(mode='h', fluid_props=fluid,num_nodes=nn),
                    promotes_inputs=[('hset','h'),('Pset','P')],
                    promotes_outputs=cool_vars)

        self.add_subsystem('gamma_calc',gamma_calc(num_nodes=nn),
            promotes_inputs=['Cp','Cv'],
            promotes_outputs=['gamma'])
        all_vars = ['P','T','S','h','rho','gamma','Cp','Cv','mu','k']
        all_outputs = [f'{fl_name}:{var}' for var in all_vars]
        self.add_subsystem('SIorEng',UnitConv(fl_name=fl_name, unit_type=unit_type,num_nodes=nn),
            promotes_inputs=all_vars, promotes_outputs=all_outputs)
        # from heatsspy.TM_files.group_print import GroupPrint
        # print(self.pathname)
        # self.add_subsystem('Print1',GroupPrint(num_nodes=nn),
        #     promotes_inputs=[('print', 'mu')])
        # self.add_subsystem('Print2',GroupPrint(num_nodes=nn),
        #     promotes_inputs=[('print', f'{fl_name}:mu')])


if __name__ == "__main__":

    import time
    from openmdao.api import Problem, Group, IndepVarComp, ExplicitComponent

    from heatsspy.include.props_water import water_props
    from heatsspy.include.props_oil import oil_props
    from heatsspy.include.props_jetA import jetA_props
    from heatsspy.include.props_air import air_props

    prob = Problem()
    Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    nn=2
    # Flow properties
    Vars.add_output('P', 1.2e5, units='Pa')
    Vars.add_output('T', 301, units='degK')
    Vars.add_output('Twtr', 310, units='degK')
    Vars.add_output('S', 723, units='J/kg/degK')
    Vars.add_output('h', 1.3e5, units='J/kg')
    Vars.add_output('P1', 1.2e5*np.ones(nn), units='Pa')
    Vars.add_output('T1', 301*np.ones(nn), units='degK')
    Vars.add_output('Twtr1', 310*np.ones(nn), units='degK')
    Vars.add_output('S1', 723*np.ones(nn), units='J/kg/degK')
    Vars.add_output('h1', 1.3e5*np.ones(nn), units='J/kg')
    fluid = water_props()
    prob.model.add_subsystem('setTotal1',SetTotal(mode='T', fluid = fluid,unit_type='SI'),promotes_inputs=['P','T'])
    # prob.model.add_subsystem('setTotal2',SetTotal(mode='S', fluid = fluid,unit_type='SI'),promotes_inputs=['P','S'])
    prob.model.add_subsystem('setTotal3',SetTotal(mode='h', fluid = fluid,unit_type='SI'),promotes_inputs=['P','h'])

    fluid = oil_props()
    prob.model.add_subsystem('setTotal4',SetTotal(mode='T', fluid = fluid, thermo='file',unit_type='SI',num_nodes=nn),
        promotes_inputs=[('P','P1'),('T','T1')])
    fluid = jetA_props()
    prob.model.add_subsystem('setTotal5',SetTotal(mode='h', fluid = fluid,thermo='file',unit_type='SI'),promotes_inputs=['P','h'])
    fluid = air_props()
    prob.model.add_subsystem('setTotal6',SetTotal(mode='T', fluid = fluid, thermo='file',unit_type='SI'),promotes_inputs=['P','T'])
    fluid = water_props()
    prob.model.add_subsystem('setTotal7',SetTotal(mode='T', fluid = fluid, thermo='file',unit_type='SI'),promotes_inputs=['P',('T','Twtr')])


    prob.setup(force_alloc_complex=True)
    prob.run_model()
    prob.check_partials(compact_print=True,method='fd')
    print('CP mode')
    print('mode T')
    print('P = '+str(prob['P'][0]))
    print('P = '+str(prob['setTotal1.flow:P'][0]))
    print('T = '+str(prob['setTotal1.flow:T'][0]))
    print('S = '+str(prob['setTotal1.flow:S'][0]))
    print('H = '+str(prob['setTotal1.flow:h'][0]))
    print('rho = '+str(prob['setTotal1.flow:rho'][0]))
    print('Cp = '+str(prob['setTotal1.flow:Cp'][0]))
    print('Cv = '+str(prob['setTotal1.flow:Cv'][0]))
    print('mu = '+str(prob['setTotal1.flow:mu'][0]))
    print(' ')
    # print('mode S')
    # print('P = '+str(prob['setTotal2.flow:P'][0]))
    # print('T = '+str(prob['setTotal2.flow:T'][0]))
    # print('S = '+str(prob['setTotal2.flow:S'][0]))
    # print('H = '+str(prob['setTotal2.flow:h'][0]))
    # print('rho = '+str(prob['setTotal2.flow:rho'][0]))
    # print('Cp = '+str(prob['setTotal2.flow:Cp'][0]))
    # print('Cv = '+str(prob['setTotal2.flow:Cv'][0]))
    # print('mu = '+str(prob['setTotal2.flow:mu'][0]))
    # print(' ')
    print('mode H')
    print('P = '+str(prob['setTotal3.flow:P'][0]))
    print('T = '+str(prob['setTotal3.flow:T'][0]))
    print('S = '+str(prob['setTotal3.flow:S'][0]))
    print('H = '+str(prob['setTotal3.flow:h'][0]))
    print('rho = '+str(prob['setTotal3.flow:rho'][0]))
    print('Cp = '+str(prob['setTotal3.flow:Cp'][0]))
    print('Cv = '+str(prob['setTotal3.flow:Cv'][0]))
    print('mu = '+str(prob['setTotal3.flow:mu'][0]))
    print(' ')
    print('file mode')
    print('mode T')
    print('P = '+str(prob['setTotal6.flow:P'][0]))
    print('T = '+str(prob['setTotal6.flow:T'][0]))
    print('S = '+str(prob['setTotal6.flow:S'][0]))
    print('H = '+str(prob['setTotal6.flow:h'][0]))
    print('rho = '+str(prob['setTotal6.flow:rho'][0]))
    print('Cp = '+str(prob['setTotal6.flow:Cp'][0]))
    print('Cv = '+str(prob['setTotal6.flow:Cv'][0]))
    print('mu = '+str(prob['setTotal6.flow:mu'][0]))
    print(' ')
