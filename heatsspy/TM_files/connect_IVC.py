# This component will connect all Indep Var Comp outputs to like inputs of another subsystem.

def connect_IVC(group, IVC, IVC_target):
    print(IVC)
    # print(IVC.name)
    for i in range(len(IVC._static_var_rel_names['output'])):
        # print(IVC._static_var_rel_names['output'][i][0])
        group.connect(str(IVC.name+'.'+IVC._static_var_rel_names['output'][i][0]), str(IVC_target+'.'+IVC._static_var_rel_names['output'][i][0]))
        # print(str(IVC_target+'.'+IVC._static_var_rel_names['output'][i][0]))

    # print(IVC_target)
    # print(group)
    # group.connect

if __name__ == "__main__":
    import time
    from openmdao.api import Problem, Group, IndepVarComp, AddSubtractComp

    prob = Problem()
    Vars =  prob.model.add_subsystem('Vars',IndepVarComp())
    Vars.add_output('A', 5, units='kN')
    Vars.add_output('B', 5, units='kN')

    adder = AddSubtractComp()
    adder.add_equation('ApB', input_names=['A', 'B'],
                        scaling_factors=[1, 1], units='kN')

    prob.model.add_subsystem(name='sum_here', subsys=adder)

    # prob.model.connect('Vars.A','SUMhere.A')
    connect_IVC(prob.model, Vars, 'sum_here')

    prob.setup()
    prob.run_model()

    print(' Output value 5+5 = ', prob.get_val('sum_here.ApB')[0])
