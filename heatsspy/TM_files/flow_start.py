from openmdao.api import Group, ExecComp
import numpy as np

from heatsspy.api import PassThrough, SetTotal

class FlowStart(Group):
    def initialize(self):
        self.options.declare('fluid', default='water', desc='fluid properties class or cool prop fluid name')
        self.options.declare('unit_type', default='SI', desc='output unit type')
        self.options.declare('thermo', default='file', desc='output unit type')
        self.options.declare('num_nodes', default=1, types=int)

    def setup(self):
        nn = self.options['num_nodes']
        thermo = self.options['thermo']
        if thermo == 'cool_prop' or thermo == 'file':
            unit_type = self.options['unit_type']
            fluid = self.options['fluid']

            if unit_type == 'SI':
                Wunits = 'kg/s'
            else:
                Wunits = 'lbm/s'

            fl_name = 'Fl_O:tot'
            self.add_subsystem('W_passthru', PassThrough('W', 'Fl_O:stat:W', val=np.ones(nn), units=Wunits),
                               promotes=['*'])
            #Calculate flow data based on temperature and pressure values
            self.add_subsystem('set_TP',SetTotal(fl_name=fl_name, fluid=fluid,mode='T', unit_type=unit_type, thermo=thermo, num_nodes=nn),
                promotes_inputs=['P','T'],
                promotes_outputs =[ f'{fl_name}:P',
                                    f'{fl_name}:T',
                                    f'{fl_name}:S',
                                    f'{fl_name}:h',
                                    f'{fl_name}:rho',
                                    f'{fl_name}:gamma',
                                    f'{fl_name}:Cp',
                                    f'{fl_name}:Cv',
                                    f'{fl_name}:mu',
                                    f'{fl_name}:k'])
            self.set_input_defaults('T', val=65*np.ones(nn), units='degC')
            self.set_input_defaults('P', val=1e5, units='Pa')
        else:
            print(self.pathname)
            print('Bad setting')
            print(thermo)
            # raise AnalysisError('bad setting')


if __name__ == "__main__": # pragma: no cover
    from collections import OrderedDict

    from openmdao.api import Problem, IndepVarComp
    from heatsspy.include.props_oil import oil_props

    # np.seterr(all='raise')

    p = Problem()
    p.model.add_subsystem('temp', IndepVarComp('T', 350., units="degK"), promotes=["*"])
    p.model.add_subsystem('pressure', IndepVarComp('P', 1.4e5, units="Pa"), promotes=["*"])
    p.model.add_subsystem('W', IndepVarComp('W', 100.0, units='kg/s'), promotes=['*'])
    p.model.add_subsystem('temp2', IndepVarComp('T2', [350.,370], units="degK"), promotes=["*"])
    p.model.add_subsystem('pressure2', IndepVarComp('P2', [1.4e5,2e5], units="Pa"), promotes=["*"])
    p.model.add_subsystem('W2', IndepVarComp('W2', [100.0,110], units='kg/s'), promotes=['*'])

    # p.model.add_subsystem('FS1',FlowStart(), promotes_inputs=['W','P','T'])
    p.model.add_subsystem('FS2', FlowStart(thermo='cool_prop' , fluid='water', unit_type='SI'),
            promotes_inputs=['W','P','T'])
    fluid = oil_props()
    p.model.add_subsystem('FS3', FlowStart(thermo='file' , fluid=fluid, unit_type='SI', num_nodes=2),
            promotes_inputs=[('W','W2'),('P','P2'),('T','T2')])

    p.setup()


    # order = find_order(p.root)
    # import json
    # print(json.dumps(order, indent=4))
    # exit()

    # p['exit_static.mach_calc.Ps_guess'] = .97
    import time
    st = time.time()
    p.run_model()
    # p.check_partials(compact_print=True)

    print("time", time.time() - st)

    # print("Temp", p['T'], p['FS1.Fl_O:tot:T'])
    # print("Pressure", p['P'], p['FS1.Fl_O:tot:P'])
    # print("h", p['FS1.totals.h'], p['FS1.Fl_O:tot:h'])
    # print("S", p['FS1.totals.S'])
    # print("actual Ps", p['FS1.exit_static.Ps'],p['FS1.Fl_O:stat:P'])
    # print("Mach", p['FS1.Fl_O:stat:MN'])
    # print("n tot", p['FS1.Fl_O:tot:n'])
    # print("n stat", p['FS1.Fl_O:stat:n'])
    print("Temp2", p['FS2.Fl_O:tot:T'])
    print("Pressure2", p['FS2.Fl_O:tot:P'])
    print("h2", p['FS2.Fl_O:tot:h'])
    print("rho2", p['FS2.Fl_O:tot:rho'])
    print("Cp2", p['FS2.Fl_O:tot:Cp'])
    print("mu2", p['FS2.Fl_O:tot:mu'])
    print("k2", p['FS2.Fl_O:tot:k'])

    print("Temp3", p['FS3.Fl_O:tot:T'])
    print("Pressure3", p['FS3.Fl_O:tot:P'])
    print("h3", p['FS3.Fl_O:tot:h'])
    print("rho3", p['FS3.Fl_O:tot:rho'])
    print("Cp3", p['FS3.Fl_O:tot:Cp'])
    print("mu3", p['FS3.Fl_O:tot:mu'])
    print("k3", p['FS3.Fl_O:tot:k'])
