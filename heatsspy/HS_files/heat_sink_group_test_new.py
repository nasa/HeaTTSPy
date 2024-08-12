import openmdao.api as om
import numpy as np

from heatsspy.HS_files.h_comp import Pr_calc, h_calc
from heatsspy.HS_files.heat_sink_R_group import HeatSinkResistanceGroup
from heatsspy.HS_files.fin_geom_comp import FinArrayGeomComp
from heatsspy.HS_files.dP_comp import dP_calc
from heatsspy.HS_files.f_comp import f_calc
from heatsspy.HS_files.fapp_comp import fapp_calc
from heatsspy.HS_files.Re_comp import Re_calc
from heatsspy.HS_files.Dh_comp import Dh_calc
from heatsspy.HS_files.Ke_comp import Ke_calc
from heatsspy.HS_files.Kc_comp import Kc_calc
from heatsspy.HS_files.Vch_comp import Vch_calc
from heatsspy.HS_files.xp_comp import xp_calc
from heatsspy.HS_files.weight_comp import weight_calc
from heatsspy.TM_files.flow_in import FlowIn

class HeatSinkGroup(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('h_calc_method', values=('teerstra', 'nellis', 'input_h'), default = 'nellis') # default='teerstra')

    def setup(self):
        nn = self.options['num_nodes']
        h_calc_method = self.options['h_calc_method']

        self.set_input_defaults('Ht', val=0.04*np.ones(nn), units='m')
        self.set_input_defaults('Wth', val=0.2*np.ones(nn), units='m')
        self.set_input_defaults('Lng', val=0.1*np.ones(nn), units='m')
        self.set_input_defaults('t_fin', val=0.001*np.ones(nn), units='m')
        self.set_input_defaults('t_base', val=0.005*np.ones(nn), units='m')
        self.set_input_defaults('N_fins', val=32*np.ones(nn), units=None)
        self.set_input_defaults('k_sink', val=218*np.ones(nn), units='W/(m*K)')
        self.set_input_defaults('R_th_cont_per_area', val=1E-3*np.ones(nn), units='m**2*K/W')

        flow_in = FlowIn(fl_name='Fl_I',unit_type='SI', num_nodes=nn)
        self.add_subsystem('flow_in', flow_in, promotes=['Fl_I:tot:*', 'Fl_I:stat:*'])

        self.add_subsystem('fin_geom', FinArrayGeomComp(num_nodes=nn), promotes_inputs=['N_fins', ('L', 'Lng'), ('W', 'Wth'), 't_fin', 't_base', 'Ht'],
                                                                        promotes_outputs=['A_c', 'D_h', 'Pm', 'Sp', 'Vol'])

        self.add_subsystem('Dh', Dh_calc(num_nodes=nn), promotes=['*'])
        self.add_subsystem('Dh_far', Dh_calc(num_nodes=nn), promotes_inputs = [('Sp','Wth'),'*'],
                                                            promotes_outputs = [('Dh','Dh_far')])


        self.add_subsystem('V_ch', Vch_calc(num_nodes=nn), promotes_inputs=[('W_flow', 'Fl_I:stat:W'), ('rho_air', 'Fl_I:tot:rho'), 'Sp', 'N_fins', 'Ht'],
                                                            promotes_outputs=['Vch'])

        if h_calc_method == 'input_h':

            self.set_input_defaults('h', val=1.0*np.ones(nn), units='W/(m*K)')

        else:

            if h_calc_method == 'teerstra':

                self.add_subsystem('Re_therm', Re_calc(num_nodes=nn), promotes_inputs=['Vch', ('mu', 'Fl_I:tot:mu'), ('rho_air', 'Fl_I:tot:rho')],
                                                                    promotes_outputs=[('Re', 'Re_therm')])
                self.add_subsystem('Pr', Pr_calc(num_nodes=nn), promotes_inputs=[('mu_air', 'Fl_I:tot:mu'),
                                                                ('cp_air', 'Fl_I:tot:Cp'),('k_air', 'Fl_I:tot:k')],
                                                                promotes_outputs=['Pr'])

                self.connect('Sp', 'Re_therm.char_length')

                inputs=[('Re', 'Re_therm'), 'Pr', 'k_sink', 'Sp', ('L','Lng'), 'Ht', ('k_air', 'Fl_I:tot:k'), 't_fin']

            elif h_calc_method == 'nellis':

                #inputs=['*']
                inputs = [('k_air', 'Fl_I:tot:k'),'*']

            self.add_subsystem('h_calc', h_calc(num_nodes=nn, h_calc_method=h_calc_method), promotes_inputs=inputs,
                                                            promotes_outputs=['*'])

        self.add_subsystem('thermal_resistance', HeatSinkResistanceGroup(num_nodes=nn),
            promotes_inputs=['h', 'k_sink', 't_base', 'R_th_contact','R_th_cont_per_area', 'Pm', 'A_c', 'Ht', ('L', 'Lng'), ('W', 'Wth'), 'N_fins'],
            promotes_outputs=['*'])

        self.add_subsystem('Re', Re_calc(num_nodes=nn), promotes_inputs=['Vch', 'char_length', ('mu', 'Fl_I:tot:mu'), ('rho_air', 'Fl_I:tot:rho')],
                                                        promotes_outputs=['Re'])

        self.add_subsystem('Re_Dh', Re_calc(num_nodes=nn), promotes_inputs=['Vch', ('mu', 'Fl_I:tot:mu'), ('rho_air', 'Fl_I:tot:rho')],
                                                            promotes_outputs=[('Re', 'Re_Dh')])

        self.connect('Dh', 'Re_Dh.char_length')


        self.add_subsystem('xp', xp_calc(num_nodes=nn), promotes=['*'])

        #self.add_subsystem('f_calc', f_calc(num_nodes=nn), promotes=['*'])
        #self.add_subsystem('fapp', fapp_calc(num_nodes=nn), promotes=['*'])

        #self.add_subsystem('f_calc', f_calc(num_nodes=nn), promotes_inputs=[('Re_Dh','Re_Dh'),'*'],
        #                                                promotes_outputs=['*'])
        #self.add_subsystem('fapp', fapp_calc(num_nodes=nn), promotes_inputs=[('Re_Dh','Re_Dh'),('L_channel','Lng'),'*'],
        #                                                promotes_outputs=['*'])

        self.add_subsystem('f_calc', f_calc(num_nodes=nn), promotes_inputs=['*'],
                                                        promotes_outputs=['*'])
        self.add_subsystem('fapp', fapp_calc(num_nodes=nn), promotes_inputs=[('L_channel','Lng'),'*'],
                                                        promotes_outputs=['*'])

        self.add_subsystem('Kc', Kc_calc(num_nodes=nn), promotes=['*'])
        self.add_subsystem('Ke', Ke_calc(num_nodes=nn), promotes=['*'])
        #self.add_subsystem('dP', dP_calc(num_nodes=nn), promotes_inputs=['fapp', 'Ht', 'Kc', 'Ke', 'Lng', 'N_fins', ('P','Fl_I:tot:P'), ('rho_air', 'Fl_I:tot:rho'), 'Sp', 'Vch', 'Wth'],
        #                                                promotes_outputs=['dPqP', 'dP'])

        self.add_subsystem('dP', dP_calc(num_nodes=nn), promotes_inputs=['fapp', 'Kc', 'Ke', 'Lng', ('P','Fl_I:tot:P'), ('rho_air', 'Fl_I:tot:rho'),'Vch', 'Dh'],
                                                        promotes_outputs=['dPqP', 'dP'])
        self.add_subsystem('weight', weight_calc(num_nodes=nn, rho_sink=2700), promotes=['*'])

        # Don't use the hydraulic diameter, use the channel spacing**********
        #self.connect('Dh', 'char_length')
        self.connect('Sp', 'char_length')

if __name__ == "__main__":
    nn = 1

    import matplotlib.pyplot as plt

    #fig, ax = plt.subplots()

    from heatsspy.api import FlowStart
    from heatsspy.api import connect_flow
    from heatsspy.include.props_air import air_props

    tval = 'file'
    air_props = air_props()

    for h_calc_method in ['nellis', 'teerstra']:
        prob = om.Problem()
        prob.model.add_subsystem('FS', FlowStart(thermo=tval, fluid=air_props, unit_type='SI', num_nodes=nn),
            promotes_inputs=['W', 'T', 'P'])
        prob.model.add_subsystem('heat_sink', HeatSinkGroup(num_nodes=nn, h_calc_method=h_calc_method))

        connect_flow(prob.model, 'FS.Fl_O', 'heat_sink.Fl_I')

        prob.setup()

        prob.set_val('W', val=0.0009, units='kg/s')
        prob.set_val('T', val=63.0, units='degC')
        #prob.set_val('T', val=(350-273.15), units='degC')
        prob.set_val('P', val=101.325, units='kPa')
        prob.set_val('heat_sink.Ht', val=20.0, units='mm' )
        prob.set_val('heat_sink.Wth', val=8., units='mm')
        prob.set_val('heat_sink.Lng', val=25, units='mm')
        prob.set_val('heat_sink.t_fin', val=0.5, units='mm')
        prob.set_val('heat_sink.t_base', val=.003, units='m')
        prob.set_val('heat_sink.N_fins', val=6, units=None)
        prob.set_val('heat_sink.k_sink', val=218*np.ones(nn), units='W/(m*K)')
        prob.set_val('heat_sink.R_th_cont_per_area', val=0E-3*np.ones(nn), units='m**2*K/W')

        prob.run_model()

        #ax.set_xlabel('spacing, m')
        #ax.set_ylabel('R_th, K/W')
        #ax.plot(prob.get_val('heat_sink.Sp'), prob.get_val('heat_sink.R_th_tot'), label = h_calc_method)


        #print('{} Nu_bar W/m-K: '.format(h_calc_method),prob.get_val('heat_sink.h_calc.Nu') )
        #print('{} Dh h_comp W/m-K: '.format(h_calc_method),prob.get_val('heat_sink.h_calc.Dh') )
        #print('{} Dh heat_sink W/m-K: '.format(h_calc_method),prob.get_val('heat_sink.Dh') )
        #print('{} h_bar W/m-K: '.format(h_calc_method),prob.get_val('heat_sink.thermal_resistance.h') )
        #print('{} R_contact K/W: '.format(h_calc_method),prob.get_val('heat_sink.thermal_resistance.base_resistance.R_th_contact') )
        #print('{} R_base_conduction K/W: '.format(h_calc_method),prob.get_val('heat_sink.thermal_resistance.base_resistance.R_th_base_cond') )
        #print('{} R_base_convection K/W: '.format(h_calc_method),prob.get_val('heat_sink.thermal_resistance.base_resistance.R_th_base_conv') )
        #print('{} R_fins K/W: '.format(h_calc_method),prob.get_val('heat_sink.thermal_resistance.fin_resistance.R_th_fins') )
        #print('{} R_th C/W: '.format(h_calc_method),prob.get_val('heat_sink.R_th_tot') )
        #print('\n')
    #ax1=ax.twinx()
    #ax1.set_ylabel('dP, Pa', color='r')
    #ax1.plot(prob.get_val('heat_sink.Sp'), prob.get_val('heat_sink.dPqP'), color='r')
    #print('\n')
    #print('Channel Velocity, m/s: ', prob.get_val('heat_sink.Vch'))
    #print('Channel Reynolds #: ', prob.get_val('heat_sink.Re'),'\n')

    print('L =', prob.get_val('heat_sink.h_calc.L'))
    #print('Pr =',prob.get_val('heat_sink.h_calc.Pr'))
    print('Re_teerstra =',prob.get_val('heat_sink.h_calc.Re'))
    print('Re_star =', prob.get_val('heat_sink.h_calc.Re')*prob.get_val('heat_sink.h_calc.Sp')/prob.get_val('heat_sink.h_calc.L'))
    print('Re_DH =',prob.get_val('heat_sink.Re_Dh'))
    print('dp_total =', prob.get_val('heat_sink.dP'))
    #print('k_flow =',prob.get_val('heat_sink.h_calc.k_air'))
    print('Sp =',prob.get_val('heat_sink.h_calc.Sp'))

    Re_in = prob.get_val('heat_sink.Re')
    b = prob.get_val('heat_sink.Sp')
    l = prob.get_val('heat_sink.Lng')

    #print('Teerstra Re_Star =', Re_in*b/l)
    #print('dP Pa: ', prob.get_val('heat_sink.dP', units='Pa'))
    #print('weight, kg: ', prob.get_val('heat_sink.Wt', units='kg'))
    #print('spacing, mm: ', prob.get_val('heat_sink.Sp', units='mm'))

    #ax.legend()

    # plt.plot(prob.get_val('heat_sink.Sp'), prob.get_val('heat_sink.R_th_global'), label='R_th')
    # plt.plot(prob.get_val('heat_sink.Sp'), prob.get_val('heat_sink.dPqP'), label='dPqP')

    # plt.show()
