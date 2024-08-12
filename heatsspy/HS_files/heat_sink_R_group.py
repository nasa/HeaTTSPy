import openmdao.api as om 
import numpy as np

from heatsspy.HS_files.fin_R_comp import FinResistanceComp
from heatsspy.HS_files.base_R_comp import BaseResistanceComp
from heatsspy.HS_files.total_R_comp import TotalResistanceComp

class HeatSinkResistanceGroup(om.Group):

    def initialize(self): 
        self.options.declare('num_nodes')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem('fin_resistance', FinResistanceComp(num_nodes=nn),
                            promotes_inputs=['N_fins', 'h', 'k_sink', 'Ht', 'A_c', 'Pm'],
                            promotes_outputs=['R_th_fins'])
        self.add_subsystem('base_resistance', BaseResistanceComp(num_nodes=nn),
                            promotes_inputs=['t_base', 'L', 'W', 'k_sink', 'h', 'N_fins', 'A_c', 'R_th_cont_per_area'])
        self.add_subsystem('total_resistance', TotalResistanceComp(num_nodes=nn),
                            promotes_inputs=['R_th_contact'],
                            promotes_outputs=['R_th_tot'])

        self.connect('R_th_fins', 'total_resistance.R_th_fins')
        self.connect('base_resistance.R_th_base_conv', 'total_resistance.R_th_base_conv')
        self.connect('base_resistance.R_th_base_cond', 'total_resistance.R_th_base_cond')
        self.connect('base_resistance.R_th_contact', 'R_th_contact')
