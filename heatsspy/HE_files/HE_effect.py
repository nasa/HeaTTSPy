from openmdao.api import ExplicitComponent,Group
import numpy as np

class HE_effectQ(ExplicitComponent):
    """ Calculate output effectiveness based on Q"""

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int,desc='Number of nodes to be evaluated')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('q', val=np.ones(nn), units='W', desc='heat transfer rate')
        self.add_input('q_max', val=np.ones(nn), units='W', desc='heat transfer rate')

        self.add_output('effect', val=np.ones(nn), desc='cooler effectiveness',lower = 1e-4)

        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='effect', wrt=['q', 'q_max'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        q  = inputs['q']
        q_max = inputs['q_max']

        outputs['effect'] = q/q_max

    def compute_partials(self, inputs, J):
        q  = inputs['q']
        q_max  = inputs['q_max']

        J['effect','q']  = 1./q_max
        J['effect','q_max']  = - q/q_max**2


class HE_effectiveness(ExplicitComponent):
    """ Calculate intercoolter effectiveness"""

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int, desc='Number of nodes to be evaluated')
        self.options.declare('HE_type', default='Xflow', desc='Number of nodes to be evaluated', values=('Xflow', 'sink', 'CALC'))
        self.options.declare('hex_def',default='hex_props',desc='heat exchanger definition')

    def setup(self):
        nn = self.options['num_nodes']
        HE_type = self.options['HE_type']
        if HE_type=='sink':
            pass
        else:
            self.add_input('CR', val=np.ones(nn), desc='capactiy rate ratio, C_min/C_max')

        self.add_input('NTU', val=np.ones(nn), desc='heat transfer units')

        self.add_output('effect', val=np.ones(nn), desc='cooler effectiveness',lower = 1e-3)

        arange = np.arange(self.options['num_nodes'])
        if HE_type=='sink':
            self.declare_partials(of='effect', wrt=['NTU'], rows=arange, cols=arange)
        else:
            self.declare_partials(of='effect', wrt=['NTU','CR'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        HE_type = self.options['HE_type']
        hex_def = self.options['hex_def']

        NTU  = inputs['NTU']
        if HE_type == 'sink':
            outputs['effect'] = 1 - np.exp(-NTU) # Moffat, "Modeling Air-Cooled Heat Sinks as Heat Exchangers"
        elif HE_type == 'Xflow':
            CR  = inputs['CR']
            # print(self.pathname)
            NTUg = np.where(NTU>=0.0001)
            outputs['effect'][NTUg] = 1 - np.exp((1/CR[NTUg])*NTU[NTUg]**0.22*(np.exp(-CR[NTUg]*NTU[NTUg]**0.78)-1)) # Incropera, Table 11.3, eqn. 11.32
            NTUlt = np.where(NTU<0.0001)
            outputs['effect'][NTUlt] = 1 - np.exp((1/CR[NTUlt])*(217.16*NTU[NTUlt]+0.1101)*(np.exp(-CR[NTUlt]*(5.44*NTU[NTUlt]+2.2e-4))-1))
            
        elif HE_type == 'CALC':
            CR  = inputs['CR']
            hex_def.get_eff(NTU,CR)
            outputs['effect'] = hex_def.effect

    def compute_partials(self, inputs, J):
        nn = self.options['num_nodes']
        HE_type = self.options['HE_type']
        hex_def = self.options['hex_def']
        N  = inputs['NTU']
        if HE_type == 'sink':
            J['effect','NTU'] = np.exp(-N)
        elif HE_type == 'Xflow':
            E = np.ones(nn)
            Em1 = np.ones(nn)
            N22 = np.ones(nn)
            N78 = np.ones(nn)

            CR  = inputs['CR']
            NTUg = np.where(N>=0.0001)
            if len(NTUg[0]) > 0:
                E[NTUg] = np.exp(-CR[NTUg]*N[NTUg]**0.78)
                Em1[NTUg] = np.exp(-CR[NTUg]*N[NTUg]**0.78)-1
                N22[NTUg] = N[NTUg]**0.22
                J['effect','CR'][NTUg]  =  np.exp(N22[NTUg]*Em1[NTUg]/CR[NTUg])*(N22[NTUg]*Em1[NTUg]/CR[NTUg]**2 + N[NTUg]*E[NTUg]/CR[NTUg])
                J['effect','NTU'][NTUg] = -np.exp(N22[NTUg]*Em1[NTUg]/CR[NTUg])*(0.22*Em1[NTUg]/CR[NTUg]/N[NTUg]**0.78 - 0.78*E[NTUg]*N[NTUg]**2.77566e-17)
            NTUlt = np.where(N<0.0001)
            if len(NTUlt[0]) > 0:
                N22[NTUlt] = 217.16*N[NTUlt]+0.1101
                N78[NTUlt] = 5.44*N[NTUlt]+2.2e-4
                print(len(NTUlt))
                print('CR is',np.size(CR))
                E[NTUlt] = np.exp(-CR[NTUlt]*N78)
                Em1[NTUlt] = np.exp(-CR[NTUlt]*N78)-1
                J['effect','CR'][NTUlt] = - np.exp(N22[NTUlt]*N78[NTUlt]*Em1[NTUlt]/CR[NTUlt]) * (-N78[NTUlt]*N22[NTUlt]*E[NTUlt]/CR[NTUlt] - N22[NTUlt]*Em1[NTUlt]/CR[NTUlt]**2)
                J['effect','NTU'][NTUlt] = - np.exp(N22[NTUlt]*Em1[NTUlt]/CR[NTUlt]) * (217.16*Em1[NTUlt]/CR[NTUlt] - 5.44*(N22[NTUlt]*E[NTUlt]))

        elif HE_type == 'CALC':
            CR  = inputs['CR']
            hex_def.get_eff_partials(N,CR)
            J['effect','CR'] = hex_def.deffect_dCR
            J['effect','NTU'] = hex_def.deffect_dNTU

class HE_NTU(ExplicitComponent):
    """ Calculate coefficient of heat transfer"""

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int, desc='Number of nodes to be evaluated')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('C_min', val=np.ones(nn), units='W/degK', desc='minimum capacity rate')
        self.add_input('AU', val=np.ones(nn), desc='Area * heat transfer coefficient', units='W/degK')

        self.add_output('NTU', val=np.ones(nn), desc='heat transfer units',lower = 1e-5)

        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='NTU', wrt=['C_min','AU'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        C_min  = inputs['C_min']
        AU = inputs['AU']

        outputs['NTU'] = NTU = AU/C_min # Kays and London, eqn 2-7
    def compute_partials(self, inputs, J):
        C_min  = inputs['C_min']
        AU  = inputs['AU']

        J['NTU','AU'] = 1/C_min
        J['NTU','C_min'] = - AU/C_min**2


class HE_AU(ExplicitComponent):
    """ Calculate coefficient of heat transfer"""

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int, desc='Number of nodes to be evaluated')
        self.options.declare('hex_def',default='hex_props',desc='heat exchanger definition')
        self.options.declare('side_number',default='' ,desc= 'side number for lookup reference')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('U', val=np.ones(nn), units='W/m**2/degK', desc='coefficient of heat transfer of a selected side(same as for alpha)')
        self.add_input('vol', val=np.ones(nn), units='m**3', desc='heat exchanger total volume')

        self.add_output('AU', val=np.ones(nn), desc='Area * heat transfer coefficient', units='W/degK', lower = 1e-5)

        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='AU', wrt=['U','vol'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        hex_def = self.options['hex_def']
        alpha = getattr(self.options['hex_def'], 'alpha'+str(self.options['side_number']))
        U  = inputs['U']
        vol  = inputs['vol']
        outputs['AU'] = alpha*vol*U

    def compute_partials(self, inputs, J):
        hex_def = self.options['hex_def']
        alpha = getattr(self.options['hex_def'], 'alpha'+str(self.options['side_number']))
        U  = inputs['U']
        vol  = inputs['vol']

        J['AU','U'] = alpha*vol
        J['AU','vol'] = alpha*U


class HE_AU_effect(Group):
    def initialize(self):
        self.options.declare('calc_AU_en', default=True, desc='calculate AU')
        self.options.declare('HE_type', default='Xflow', desc='type of heat exchanger', values=('Xflow', 'sink', 'CALC'))
        self.options.declare('hex_def',default='hex_props',desc='heat exchanger definition')
        self.options.declare('side_number',default='' ,desc= 'side number for lookup reference')
        self.options.declare('num_nodes', default=1, types=int, desc='Number of nodes to be evaluated')

    def setup(self):
        HE_type = self.options['HE_type']
        nn = self.options['num_nodes']
        hex_def = self.options['hex_def']
        side_number = self.options['side_number']

        if self.options['calc_AU_en']:
	        self.add_subsystem('AU_calc',HE_AU(num_nodes=nn, hex_def=hex_def, side_number=side_number),
	            promotes_inputs=['U','vol'],
	            promotes_outputs=['AU'])

        self.add_subsystem('NTU_calc',HE_NTU(num_nodes=nn),
            promotes_inputs=['C_min','AU'],
            promotes_outputs=['NTU'])
        if HE_type=='sink':
            prom_in = ['NTU']
        else:
            prom_in = ['NTU','CR']
        self.add_subsystem('eff_calc',HE_effectiveness(hex_def=hex_def, HE_type=HE_type, num_nodes=nn),
            promotes_inputs=prom_in,
            promotes_outputs=['effect'])


if __name__ == "__main__":

    import time
    from openmdao.api import Problem, IndepVarComp
    from heatsspy.include.HexParams_Regenerator import hex_params_regenerator
    hex_def = hex_params_regenerator()

    prob = Problem()
    Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    nn = 1
    # Flow properties
    Vars.add_output('C_min', val=25500*np.ones(nn), units='W/degK', desc='minimum capacity rate')
    Vars.add_output('U1', val=70.9*np.ones(nn), units='W/m**2/degK', desc='coefficient of heat transfer of a selected side(same as for alpha)')
    Vars.add_output('U2', val=49.47*np.ones(nn), units='W/m**2/degK', desc='coefficient of heat transfer of a selected side(same as for alpha)')
    Vars.add_output('AU',val=108547.9*np.ones(nn), units='W/degK')
    Vars.add_output('vol', val=3.8275*np.ones(nn), units='m**3', desc='heat exchanger total volume')
    Vars.add_output('NTU', val=4.25*np.ones(nn), desc='NTU')
    Vars.add_output('CR', val=0.955*np.ones(nn), desc='capacity rate ratio')
    Vars.add_output('T_c_in', val=300*np.ones(nn),units='degK', desc='cold side temp')
    Vars.add_output('T_h_in', val=320*np.ones(nn),units='degK', desc='hot side temp')
    Vars.add_output('q', val=1000*np.ones(nn),units='W', desc='capacity rate ratio')
    Vars.add_output('q_max', val=2000*np.ones(nn),units='W', desc='max capacity rate ratio')

    Blk = prob.model.add_subsystem('NTU_calc', HE_NTU(num_nodes=nn),
        promotes_inputs=['*'])
    Blk2 = prob.model.add_subsystem('eff_calc', HE_effectiveness(num_nodes=nn, HE_type='sink', hex_def=hex_def),
        promotes_inputs=['*'])
    Blk3 = prob.model.add_subsystem('HE_eff', HE_AU_effect(num_nodes=nn, HE_type='CALC', hex_def=hex_def, side_number=1),
        promotes_inputs=['C_min','CR',('U','U1'),'vol'])
    Blk4 = prob.model.add_subsystem('HE_eff_sink', HE_AU_effect(num_nodes=nn, HE_type='Xflow', hex_def=hex_def, side_number=2),
        promotes_inputs=['C_min','CR',('U','U2'),'vol'])
    Blk5 = prob.model.add_subsystem('Q_calc', HE_effectQ(num_nodes=nn),
        promotes_inputs=['q','q_max'])

    # Blk.set_check_partial_options(wrt=['*'], step_calc='rel')
    prob.setup(force_alloc_complex=True)
    prob.run_model()
    prob.check_partials(compact_print=True, method='cs')
    #
    print('NTU = '+str(prob['NTU_calc.NTU'][0]))
    print('eff = '+str(prob['eff_calc.effect'][0]))

    print('NTU2 = '+str(prob['HE_eff.NTU'][0]))
    print('eff2 = '+str(prob['HE_eff.effect'][0]))

    print('NTU4 = '+str(prob['HE_eff_sink.NTU'][0]))
    print('eff4 = '+str(prob['HE_eff_sink.effect'][0]))

    print('effect = '+str(prob['Q_calc.effect'][0]))
