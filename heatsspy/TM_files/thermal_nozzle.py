import numpy as np
from openmdao.api import Group,ExplicitComponent,IndepVarComp,BalanceComp, ExecComp
from openmdao.api import DirectSolver,BoundsEnforceLS,NewtonSolver

class calc_MN(ExplicitComponent):
    ''' calculate Mach Number'''
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int,
                             desc='Number of nodes to be evaluated')
        self.options.declare('gamma',default=1.4)
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('Pt', val=np.ones(nn), units='Pa', desc='total pressure of flow')
        self.add_input('Ps', val=np.ones(nn), units='Pa', desc='static pressure')
        self.add_output('MN', val=np.ones(nn), units=None, desc='Mach Number')
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials('MN', ['Pt','Ps'], rows=arange, cols=arange)
    def compute(self,inputs,outputs):
        nn = self.options['num_nodes']
        gam=self.options['gamma']
        Pt=inputs['Pt']
        Ps=inputs['Ps']
        PR=Ps/Pt
        CPR=(2/(gam+1))**(gam/(gam-1)) # critical pressure ratio

        prg1 = np.where(PR>=1e-5)
        outputs['MN'][prg1] = np.sqrt((PR[prg1]**((1-gam)/gam)-1)*2/(gam-1))
        prlow = np.where(PR<1e-5)
        outputs['MN'][prlow] = - 26.268*PR[prlow]+6.9578
        # if np.any(outputs['MN'] > 1):
        #     print(self.pathname, 'MN is ', outputs['MN'])
        #     print(self.pathname, 'Pt is ', inputs['Pt'])
        #     print(self.pathname, 'Ps is ', inputs['Ps'])



    def compute_partials(self, inputs, J):
        gam=self.options['gamma']
        Pt=inputs['Pt']
        Ps=inputs['Ps']
        CPR=(2/(gam+1))**(gam/(gam-1)) # critical pressure ratio
        PR = Ps/Pt
        g1 = (1-gam)/gam
        g2 = 2/(gam-1)

        prg1 = np.where(PR>=1e-5)
        J['MN','Pt'][prg1] =   -(1-gam)*Ps[prg1]*(Ps[prg1]/Pt[prg1])**(((1-gam)/gam)-1)/(np.sqrt(2)*(gam-1)*gam*Pt[prg1]**2*np.sqrt(((Ps[prg1]/Pt[prg1])**((1-gam)/gam)-1)/(gam-1)))
        J['MN','Ps'][prg1] = (1-gam)*(Ps[prg1]/Pt[prg1])**(((1-gam)/gam)-1)/(np.sqrt(2)*(gam-1)*gam*Pt[prg1]*np.sqrt(((Ps[prg1]/Pt[prg1])**((1-gam)/gam)-1)/(gam-1)))
        prlow = np.where(PR<1e-5)
        J['MN','Pt'][prlow] = 26.268*Ps[prlow]/Pt[prlow]**2
        J['MN','Ps'][prlow] = - 26.268/Pt[prlow]


class calc_Ts(ExplicitComponent):
    ''' calculate Static Temperature'''
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int,
                             desc='Number of nodes to be evaluated')
        self.options.declare('gamma',default=1.4)
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('MN', val=np.ones(nn), units=None, desc='Mach Number')
        self.add_input('Tt', val=np.ones(nn), units='degK', desc='total temperature')
        self.add_output('Ts', val=np.ones(nn), units='degK', desc='static temperature')
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials('Ts', ['MN','Tt'], rows=arange, cols=arange)
    def compute(self,inputs,outputs):
        gam=self.options['gamma']
        MN=inputs['MN']
        Tt=inputs['Tt']
        outputs['Ts']=Tt*(1+0.5*(gam-1)*MN**2)**-1
    def compute_partials(self, inputs, J):
        gam=self.options['gamma']
        MN=inputs['MN']
        Tt=inputs['Tt']
        J['Ts','MN'] = - ((gam-1)*MN*Tt)/(0.5*(gam-1)*MN**2+1)**2
        J['Ts','Tt'] = (1+0.5*(gam-1)*MN**2)**-1


class calc_a(ExplicitComponent):
    ''' calculate speed of sound'''
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int,
                             desc='Number of nodes to be evaluated')
        self.options.declare('gamma',default=1.4)
        self.options.declare('R',default=286.9) # J/kg K
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('Ts', val=np.ones(nn), units='degK', desc='static temperature')
        self.add_output('a', val=np.ones(nn), units='m/s', desc='speed of sound')
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials('a', ['Ts'], rows=arange, cols=arange)
    def compute(self,inputs,outputs):
        gam=self.options['gamma']
        R=self.options['R']
        Ts=inputs['Ts']
        outputs['a']=np.sqrt(gam*R*Ts)
    def compute_partials(self, inputs, J):
        gam=self.options['gamma']
        R=self.options['R']
        Ts=inputs['Ts']
        J['a','Ts'] = gam*R/2/np.sqrt(gam*R*Ts)


class calc_rhos(ExplicitComponent):
    ''' calculate speed of sound'''
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int,
                             desc='Number of nodes to be evaluated')
        self.options.declare('R',default=286.9) # J/kg K
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('Ps', val=np.ones(nn), units='Pa', desc='static pressure')
        self.add_input('Ts', val=np.ones(nn), units='degK', desc='static temperature')
        self.add_output('rhos', val=np.ones(nn), units='kg/m**3', desc='static density')
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials('rhos', ['Ps','Ts'], rows=arange, cols=arange)
    def compute(self,inputs,outputs):
        outputs['rhos'] = inputs['Ps']/self.options['R']/inputs['Ts']
    def compute_partials(self, inputs, J):
        R=self.options['R']
        Ps=inputs['Ps']
        Ts=inputs['Ts']
        J['rhos','Ps'] = 1/R/Ts
        J['rhos','Ts'] =-Ps/R/Ts**2


class calc_v(ExplicitComponent):
    ''' calculate velocity'''
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int,
                             desc='Number of nodes to be evaluated')
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('MN', val=np.ones(nn), units=None, desc='Mach number')
        self.add_input('a', val=np.ones(nn), units='m/s', desc='speed of sound')
        self.add_output('v', val=np.ones(nn), units='m/s', desc='velocity')
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials('v', ['a','MN'], rows=arange, cols=arange)
    def compute(self,inputs,outputs):
        MN=inputs['MN']
        a=inputs['a']
        outputs['v']=MN*a
    def compute_partials(self, inputs, J):
        MN=inputs['MN']
        a=inputs['a']
        J['v','MN']=a
        J['v','a']=MN


class calc_A(ExplicitComponent):
    ''' calculate speed of sound'''
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int,
                             desc='Number of nodes to be evaluated')
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('rhos', val=np.ones(nn), units='kg/m**3', desc='static density')
        self.add_input('v', val=np.ones(nn), units='m/s', desc='velocity')
        self.add_input('W', val=np.ones(nn), units='kg/s', desc='mass flow')
        self.add_output('A', val=np.ones(nn), units='m**2', desc='area')
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials('A', ['W','v','rhos'], rows=arange, cols=arange)
    def compute(self,inputs,outputs):
        rhos=inputs['rhos']
        v=inputs['v']
        W=inputs['W']

        outputs['A']=W/rhos/v

    def compute_partials(self, inputs, J):
        nn = self.options['num_nodes']
        rhos=inputs['rhos']
        v=inputs['v']
        W=inputs['W']

        J['A','rhos']=-W/v/rhos**2
        J['A','v']=-W/rhos/v**2
        J['A','W']=1/rhos/v


class calc_W(ExplicitComponent):
    ''' calculate mass flow'''
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int,
                             desc='Number of nodes to be evaluated')
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('A', val=np.ones(nn), units='m**2', desc='area')
        self.add_input('rhos', val=np.ones(nn), units='kg/m**3', desc='static density')
        self.add_input('v', val=np.ones(nn), units='m/s', desc='velocity')
        self.add_output('W', val=np.ones(nn), units='kg/s', desc='mass flow')
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials('W', ['v','rhos','A'], rows=arange, cols=arange)
    def compute(self,inputs,outputs):
        outputs['W']=inputs['A']*inputs['v']*inputs['rhos']
    def compute_partials(self, inputs, J):
        J['W','rhos']=inputs['A']*inputs['v']
        J['W','v']=inputs['A']*inputs['rhos']
        J['W','A']=inputs['v']*inputs['rhos']


class calc_Fg(ExplicitComponent):
    ''' calculate gross thrust'''
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int,
                             desc='Number of nodes to be evaluated')
        self.options.declare('Cfg', default=0.95,
                             desc='coefficient of thrust')
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('W', val=np.ones(nn), units='kg/s', desc='mass flow')
        self.add_input('v', val=np.ones(nn), units='m/s', desc='velocity')
        self.add_output('Fg',val=np.ones(nn),units='N',desc='gross thrust')
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials('Fg', ['v','W'], rows=arange, cols=arange)
    def compute(self,inputs,outputs):
        Cfg = self.options['Cfg']
        outputs['Fg']=inputs['W']*inputs['v'] *Cfg
    def compute_partials(self, inputs, J):
        Cfg = self.options['Cfg']
        J['Fg','v']=inputs['W']*Cfg
        J['Fg','W']=inputs['v']*Cfg


class calc_PR(ExplicitComponent):
    ''' calculate nozzle pressure ratio'''
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int,
                             desc='Number of nodes to be evaluated')
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('Pt', val=np.ones(nn), units='Pa', desc='total pressure')
        self.add_input('Ps', val=np.ones(nn), units='Pa', desc='static pressure')
        self.add_output('PR',val=np.ones(nn),units=None,desc='pressure ratio')
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials('PR', ['Pt','Ps'], rows=arange, cols=arange)
    def compute(self,inputs,outputs):
        outputs['PR']=inputs['Pt']/inputs['Ps']
    def compute_partials(self, inputs, J):
        J['PR','Pt']=1.0/inputs['Ps']
        J['PR','Ps']=-inputs['Pt']/inputs['Ps']**2.0


class fpr_calc(ExplicitComponent):
    ''' calculate ideal fan pressure ratio'''
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int,
                             desc='Number of nodes to be evaluated')
        self.options.declare('NPR_des',default= 1.01)
    def setup(self):
        nn=self.options['num_nodes']
        self.add_input('Pamb', val=np.ones(nn), units='Pa', desc='ambient pressure')
        self.add_input('P_in', val=np.ones(nn), units='Pa', desc='input pressure')
        self.add_output('FPR', val=np.ones(nn), units=None, desc='fan pressure ratio')
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials('FPR', ['P_in','Pamb'], rows=arange, cols=arange)
    def compute(self,inputs,outputs):
        FPR= self.options['NPR_des']*inputs['Pamb']/inputs['P_in']

        outputs['FPR']=np.where(FPR <=1,1.0,FPR)
        # print(self.pathname)
        # print('Pamb =',inputs['Pamb'])
        # print('P_in =',inputs['P_in'])
        # print('dP_calc =', inputs['P_in']-inputs['Pamb'])
        # print(self.pathname, 'P_in is ', inputs['P_in'])
        # print(self.pathname, 'FPR is ', FPR)
    def compute_partials(self, inputs, J):
        FPR=self.options['NPR_des']*inputs['Pamb']/inputs['P_in']
        J['FPR','Pamb']=np.where(FPR<=1,0,self.options['NPR_des']/inputs['P_in'])
        J['FPR','P_in']=np.where(FPR<=1,0,-self.options['NPR_des']*inputs['Pamb']/inputs['P_in']**2)


class pr_compressor(ExplicitComponent):
    ''' pressure ratio to temperature ratio'''
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int,
                             desc='Number of nodes to be evaluated')
        self.options.declare('eff',default=0.95)
        self.options.declare('gamma',default=1.4)
        self.options.declare('Cp',default=1005) # J/kg/K
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('P_in', val=np.ones(nn), units='Pa', desc='input pressure')
        self.add_input('PR', val=np.ones(nn), units=None, desc='pressure ratio')
        self.add_input('T_in', val=np.ones(nn), units='degK', desc='input temperature')
        self.add_input('W', val=np.ones(nn), units='kg/s', desc='mass flow')

        self.add_output('P_out', val=np.ones(nn), units='Pa', desc='output pressure')
        self.add_output('Pwr',val=np.ones(nn),units='W',desc='power')
        self.add_output('T_out', val=np.ones(nn), units='degK', desc='output temperature')
        self.add_output('TR',val=np.ones(nn),units=None,desc='temperature ratio')
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials('P_out', ['PR','P_in'], rows=arange, cols=arange)
        self.declare_partials('T_out', ['T_in','PR'], rows=arange, cols=arange)
        self.declare_partials('TR', ['PR'], rows=arange, cols=arange)
        self.declare_partials('Pwr', ['W','T_in','PR'], rows=arange, cols=arange)

    def compute(self,inputs,outputs):
        eff=self.options['eff']
        gam=self.options['gamma']
        Cp=self.options['Cp']
        PR = inputs['PR']
        P_in = inputs['P_in']
        T_in = inputs['T_in']
        W = inputs['W']
        # print(self.pathname)
        # print(inputs['PR'])
        # print(inputs['T_in'])

        pr1 = np.where(PR<=1) # if PR <=1
        outputs['P_out'][pr1]=P_in[pr1]
        outputs['T_out'][pr1]=T_in[pr1]
        outputs['TR'][pr1]=1
        outputs['Pwr'][pr1]=0
        prg1 = np.where(PR>1) # else
        outputs['P_out'][prg1]=P_out=PR[prg1]*P_in[prg1]
        outputs['T_out'][prg1]=T_out=(T_in[prg1]*PR[prg1]**((gam-1)/gam) - T_in[prg1])/eff + T_in[prg1]
        outputs['TR'][prg1]=T_out/T_in[prg1]
        outputs['Pwr'][prg1]=W[prg1]*Cp*(T_out-T_in[prg1])

        # print(self.pathname, 'PR is ', PR)
        # print(self.pathname, 'Pt is ', outputs['P_out'])
        # print(self.pathname, 'Ps is ', inputs['Ps'])

    def compute_partials(self, inputs, J):
        eff=self.options['eff']
        gam=self.options['gamma']
        Cp=self.options['Cp']
        PR = inputs['PR']
        P_in = inputs['P_in']
        T_in = inputs['T_in']
        W = inputs['W']
        PR=np.where(PR<=1,1,PR)
        J['P_out','P_in']=PR
        J['P_out','PR']=P_in
        J['T_out','T_in']=(PR**((gam-1)/gam) - 1)/eff + 1
        J['T_out','PR']=(gam-1)*T_in*PR**(-1/gam)/eff/gam
        J['TR','PR']=(gam-1)*PR**(-1/gam)/eff/gam
        J['Pwr','W']=Cp*((T_in*PR**((gam-1)/gam) - T_in)/eff)
        J['Pwr','T_in']=W*Cp*(PR**((gam-1)/gam) - 1)/eff
        J['Pwr','PR']=W*Cp*(gam-1)*T_in*PR**(-1/gam)/eff/gam


class fan_weight_calc(ExplicitComponent):
    """ Estimate TMS fan weight"""
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int,
                             desc='Number of nodes to be evaluated')
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('W', val=np.ones(nn), units='lbm/s', desc='mass flow')
        self.add_output('weight', val=np.ones(nn), units='lbm', desc='fan weight')

        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='weight', wrt=['W'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        W  = inputs['W']
        outputs['weight'] = 0.4386 * W + 0.1104

    def compute_partials(self, inputs, J):
        W  = inputs['W']
        J['weight','W'] = 0.4386


class nozzle(Group):
    ''' Nozzle'''
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int,
                             desc='Number of nodes to be evaluated')
    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('calc_PR',calc_PR(num_nodes=nn),promotes_inputs=['*'],promotes_outputs=['*'])
        self.add_subsystem('calc_MN',calc_MN(num_nodes=nn),promotes_inputs=['*'],promotes_outputs=['*'])
        self.add_subsystem('calc_Ts',calc_Ts(num_nodes=nn),promotes_inputs=['*'],promotes_outputs=['*'])
        self.add_subsystem('calc_a',calc_a(num_nodes=nn),promotes_inputs=['*'],promotes_outputs=['*'])
        self.add_subsystem('calc_rhos',calc_rhos(num_nodes=nn),promotes_inputs=['*'],promotes_outputs=['*'])
        self.add_subsystem('calc_v',calc_v(num_nodes=nn),promotes_inputs=['*'],promotes_outputs=['*'])
        self.add_subsystem('calc_A',calc_A(num_nodes=nn),promotes_inputs=['*'],promotes_outputs=['*'])
        self.add_subsystem('calc_Fg',calc_Fg(num_nodes=nn),promotes_inputs=['*'],promotes_outputs=['*'])


class puller_fan(Group):
    ''' puller_fan network'''
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int,
                             desc='Number of nodes to be evaluated')
        self.options.declare('set_fpr',default=True)
    def setup(self):
        set_fpr=self.options['set_fpr']
        nn = self.options['num_nodes']
        if set_fpr:
            self.add_subsystem('fpr_calc',fpr_calc(num_nodes=nn),promotes=['*'])
        self.add_subsystem('pr_compressor',pr_compressor(num_nodes=nn),
            promotes_inputs=[('PR','FPR'),'P_in','T_in','W'],
            promotes_outputs=[('P_out','P5'),('T_out','T5'),('Pwr','Qfan')])
        self.add_subsystem('fan_weight',fan_weight_calc(num_nodes=nn),
            promotes_inputs=['W'],
            promotes_outputs=[('weight', 'fan_weight')])
        self.add_subsystem('nozzle',nozzle(num_nodes=nn),
            promotes_inputs=[('Pt','P5'),('Ps','Pamb'),('Tt','T5'),'W'],
            promotes_outputs=[('A','Ath'),'Fg','MN',('PR','NPR')])




if __name__ == "__main__":

    import time
    from openmdao.api import Problem, Group, IndepVarComp, ExplicitComponent
    from openmdao.api import BalanceComp, DirectSolver, BoundsEnforceLS, NewtonSolver

    prob = Problem()

    model = prob.model
    newton = model.nonlinear_solver = NewtonSolver()
    newton.options['atol'] = 1e-6
    newton.options['rtol'] = 1e-10
    newton.options['iprint'] = -1
    newton.options['maxiter'] = 10
    newton.options['solve_subsystems'] = True
    newton.options['max_sub_solves'] = 1001
    newton.linesearch = BoundsEnforceLS()
    # newton.linesearch.options['maxiter'] = 1
    newton.linesearch.options['bound_enforcement'] = 'scalar'
    newton.linesearch.options['iprint'] = 2

    model.linear_solver = DirectSolver(assemble_jac=True)

    Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    # Flow properties
    Vars.add_output('Pt2',val=[18.458,19],units='psi')
    Vars.add_output('Ps2',val=[12.227,20],units='psi')
    Vars.add_output('Tt2',val=[605.47,600],units='degR')
    # Vars.add_output('W',val=1.105,units='lbm/s') # 0.5012 kg/s
    Vars.add_output('W2',val=[16.076, 16],units='lbm/s') # 0.5012 kg/s
    Vars.add_output('A2',val=[0.001862626764,0.002],units='m**2')
    # SLS design
    Vars.add_output('P_in2',val=[95077.3246,1.0e5],units='Pa')
    Vars.add_output('P_amb2',val=[101146.09, 1.1e5],units='Pa')
    # Vars.add_output('PR',val=1.1,units=None)
    Vars.add_output('T_in2',val=[316.483,320],units='degK')
    # Alt OD
    Vars.add_output('P_inA2',val=[0.112518e4,0.11e4],units='Pa')
    Vars.add_output('P_ambA2',val=[0.1197e4,0.12e4],units='Pa')
    # Vars.add_output('PR',val=1.1,units=None)
    Vars.add_output('T_inA2',val=[226.51,230.0],units='degK')

    Vars.add_output('Pt',val=18.458,units='psi')
    Vars.add_output('Ps',val=12.227,units='psi')
    Vars.add_output('Tt',val=605.47,units='degR')
    # Vars.add_output('W',val=1.105,units='lbm/s') # 0.5012 kg/s
    Vars.add_output('W',val=16.076,units='lbm/s') # 0.5012 kg/s
    Vars.add_output('A',val=0.001862626764,units='m**2')
    # SLS design
    Vars.add_output('P_in',val=1.0e5,units='Pa')
    Vars.add_output('P_amb',val=101146.09,units='Pa')
    # Vars.add_output('PR',val=1.1,units=None)
    Vars.add_output('T_in',val=316.483,units='degK')
    # Alt OD
    Vars.add_output('P_inA',val=0.112518e4,units='Pa')
    Vars.add_output('P_ambA',val=0.1197e4,units='Pa')
    # Vars.add_output('PR',val=1.1,units=None)
    Vars.add_output('T_inA',val=226.51,units='degK')

    # # SLS design
    # Vars.add_output('P_in',val=12.227,units='psi')
    # Vars.add_output('PR',val=15,units=None)
    # Vars.add_output('T_in',val=537.84,units='degR')
    # # Alt OD
    # Vars.add_output('P_inA',val=0.1197e4,units='Pa')
    # # Vars.add_output('PR',val=1.1,units=None)
    # Vars.add_output('T_inA',val=226.51,units='degK')

    BlkU=prob.model.add_subsystem('PF',puller_fan(num_nodes=1),
        promotes_inputs=[('P_in','Pt'),('T_in','Tt'),'W',('Pamb','Ps')])

    BlkU2=prob.model.add_subsystem('PF2',puller_fan(num_nodes=1),
        promotes_inputs=[('P_in','Ps'),('T_in','Tt'),'W',('Pamb','Pt')])

    # Blk1=prob.model.add_subsystem('nozzle',nozzle(),
    #     promotes_inputs=['Pt','Ps','Tt','W'])

    # Blk2= prob.model.add_subsystem('pr_comp',pr_compressor(),
    #     promotes_inputs=['PR','P_in','T_in','W'],
    #     promotes_outputs=[('P_out','P5'),('T_out','T5')])
    # Blk3=prob.model.add_subsystem('NOZ',nozzle(),
    #     promotes_inputs=[('Pt','P5'),('Ps','P_amb'),('Tt','T5'),'W'],
    #     promotes_outputs=[('A','Ath')])

    # Blk4= prob.model.add_subsystem('pr_compA',pr_compressor(),
    #     promotes_inputs=['PR',('P_in','P_inA'),('T_in','T_inA'),('W','WA')],
    #     promotes_outputs=[('P_out','P5A'),('T_out','T5A')])
    # Blk5=prob.model.add_subsystem('NOZA',nozzle(),
    #     promotes_inputs=[('Pt','P5A'),('Ps','P_ambA'),('Tt','T5A'),('W','WA')],
    #     promotes_outputs=[('A','Ath_calc')])

    # balanceWA = prob.model.add_subsystem('balanceWA',BalanceComp(eq_units='m**2'),
    #         promotes_inputs=[('lhs:WA','Ath_calc'),('rhs:WA','Ath')],
    #         promotes_outputs=['WA'])
    # balanceWA.add_balance('WA',val= 0.004,eq_units='m**2', units='kg/s', lower= 0.001, upper = 10)
    # balancePR = prob.model.add_subsystem('balancePR',BalanceComp(eq_units=None),
    #         promotes_inputs=[('lhs:PR','NOZ.PR')],
    #         promotes_outputs=['PR'])
    # balancePR.add_balance('PR',val= 1.3,eq_units=None, units=None,rhs_val=1.05, lower= 1.01, upper = 10)

    # Blk.set_check_partial_options(wrt=['Ht','Wth'], step_calc='rel')
    prob.setup()

    prob.setup(force_alloc_complex=True)
    prob.run_model()
    prob.check_partials(compact_print=True,method='cs')
    print('FPR ='+str(prob['PF.FPR']))
    print('NPR ='+str(prob['PF.NPR']))

    print('FPR ='+str(prob['PF2.FPR']))
    print('NPR ='+str(prob['PF2.NPR']))
    # print('--------------------------')
    # print('Nozzle Solver for Area')
    # print('--------------------------')
    # print('W2 ='+str(prob['W'][0]))
    # print('MN ='+str(prob['nozzle.MN'][0]))
    # print('Ts ='+str(prob['nozzle.Ts'][0]))
    # print('v ='+str(prob['nozzle.v'][0]))
    # print('rhos ='+str(prob['nozzle.rhos'][0]))
    # print('Ath ='+str(prob['nozzle.A'][0]))
    # print('Fg ='+str(prob['nozzle.Fg'][0]))
    # print('NPR ='+str(prob['Pt'][0]/prob['Ps'][0]))
    # print('--------------------------')
    # print('pr_comp+NOZ')
    # print('--------------------------')
    # print('T_out ='+str(prob['pr_comp.T_out'][0]))
    # print('TR ='+str(prob['pr_comp.TR'][0]))
    # print('P_out ='+str(prob['pr_comp.P_out'][0]))
    # print('Pwr ='+str(prob['pr_comp.Pwr'][0]))
    # print('MN ='+str(prob['NOZ.MN'][0]))
    # print('Ath ='+str(prob['Ath'][0]))
    # print('Fg ='+str(prob['NOZ.Fg'][0]))
    # print('NPR ='+str(prob['NOZ.PR'][0]))
    # print('PR ='+str(prob['PR'][0]))
    # print('--------------------------')
    # print('Off nom at alt')
    # print('--------------------------')
    # print('WA ='+str(prob['WA'][0]))
    # print('Pwr ='+str(prob['pr_compA.Pwr'][0]))
    # print('Ath ='+str(prob['Ath'][0]))
    # print('Ath_calc ='+str(prob['Ath_calc'][0]))
    # print('Fg ='+str(prob['NOZA.Fg'][0]))
    # print('NPR ='+str(prob['NOZA.PR'][0]))

    # p2 = Problem()
    # Vars =  p2.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    # Vars.add_output('Pt', 1e5, units='Pa')
    # Vars.add_output('Ps', 87356.575, units='Pa')
    # p2.model.add_subsystem('MN',calc_MN(num_nodes=1),promotes_inputs=['*'])
    # p2.setup()
    # p2.run_model()
    # p2.check_partials(compact_print=True)

    # p3 = Problem()
    # Vars =  p3.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    # Vars.add_output('MN', 0.2, units=None)
    # Vars.add_output('Tt', 320, units='degK')
    # p3.model.add_subsystem('Ts',calc_Ts(num_nodes=1),promotes_inputs=['*'])
    # p3.setup()
    # p3.run_model()
    # p3.check_partials(compact_print=True)

    # p4 = Problem()
    # Vars =  p4.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    # Vars.add_output('Ts', 320, units='degK')
    # p4.model.add_subsystem('a',calc_a(num_nodes=1),promotes_inputs=['*'])
    # p4.setup()
    # p4.run_model()
    # p4.check_partials(compact_print=True)

    # p5 = Problem()
    # Vars =  p5.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    # Vars.add_output('Ps', 1e5, units='Pa')
    # Vars.add_output('Ts', 320, units='degK')
    # p5.model.add_subsystem('rhos',calc_rhos(num_nodes=1),promotes_inputs=['*'])
    # p5.setup()
    # p5.run_model()
    # p5.check_partials(compact_print=True)
