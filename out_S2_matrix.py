from __future__ import print_function
from dolfin import *
import numpy as np
import math
import getopt, sys


# SS added
parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['optimize'] = True
parameters["ghost_mode"] = "shared_facet"
set_log_active(False)

# SS added
iMin=2; iMax = 7
jMin=0; jMax = 2
# Define parameters
alpha = 3.0
beta = 3.0
k=1

def usage():
  print("-h   or --help")
  print("-a a or --alpha a       to specify \alpha")
  print("-b b or --beta  b       to specify \beta")
  print("-k       to specify k")
  print("-i i or --iMin  i       to specify iMin")
  print("-j j or --jMin  j       to specify jMin")
  print("-I i or --iMax  i       to specify iMax")
  print("-J j or --jMax  j       to specify jMax")
  print(" ")
  os.system('date +%Y_%m_%d_%H-%M-%S')
  print (time.strftime("%d/%m/%Y at %H:%M:%S"))

# parse the command line
try:
  opts, args = getopt.getopt(sys.argv[1:], "ha:b:k:i:I:j:J:",
                   [
                    "help",           # obvious
                    "alpha=",         # alpha
                    "beta=",          # beta
                    "k=",          # degree of polynomials
                    "iMin=",          # iMin
                    "iMax=",          # iMax
                    "jMin=",          # jMin
                    "jMax=",          # jMax
                    ])

except getopt.GetoptError as err:
  # print help information and exit:
  print(err) # will print something like "option -a not recognized"
  usage()
  sys.exit(2)

for o, a in opts:
  if o in ("-h", "--help"):
    usage()
    sys.exit()
  elif o in ("-a", "--alpha"):
    alpha = float(a)
    print('setting: alpha = %f;' % alpha),
  elif o in ("-b", "--beta"):
    beta = float(a)
    print('setting:  beta = %f;' % beta),    
  elif o in ("-k"):
    k = int(a)
    print('setting:  k = %d;' % k),
  elif o in ("-i", "--iMin"):
    iMin = int(a)
    print('setting:  iMin = %f;' % iMin),
  elif o in ("-I", "--iMax"):
    iMax = int(a)
    print('setting:  iMax = %f;' % iMax),
  elif o in ("-j", "--jMin"):
    jMin = int(a)
    print('setting:  jMin = %f;' % jMin),
  elif o in ("-J", "--jMax"):
    jMax = int(a)
    print('setting:  jMax = %f;' % jMax),
  else:
    assert False, "unhandled option"



#Save data for error
L2_error=np.zeros((iMax-iMin+1,jMax-jMin+1), dtype=np.float64)
Lu_error=np.zeros((iMax-iMin+1,jMax-jMin+1), dtype=np.float64)
H1_error=np.zeros((iMax-iMin+1,jMax-jMin+1), dtype=np.float64)
Hu_error=np.zeros((iMax-iMin+1,jMax-jMin+1), dtype=np.float64)

# problem data
T = 1.0     # total simulation time

varphi1=0.1
varphi2=0.4
varphi0=1-varphi1-varphi2

tau1=0.5
tau2=1.5
#-----------------------------------------------------------------------------------------------------------------

ux = Expression(("x[0]*x[1]*exp(1.0 - tn)","cos(tn)*sin(x[0]*x[1])"), tn=0, degree=5)
wx = Expression(("-x[0]*x[1]*exp(1.0-tn)","-sin(x[0]*x[1])*sin(tn)"), tn=0, degree=5)

zetax1 = Expression(("varphi1*x[0]*x[1]*(tau1*exp(-tn + 1)/(tau1 - 1) - tau1*exp(-tn/tau1 + 1)/(tau1 - 1))","-(tau1*tau1*exp(-tn/tau1)/(tau1*tau1 + 1) - (tau1*tau1*cos(tn) - tau1*sin(tn))/(tau1*tau1 + 1))*varphi1*sin(x[0]*x[1])"),tau1=tau1, varphi1=varphi1,  tn=0, degree=5)
#[ p1=1-exp(-tn/tau1) q1=(tn-tau1+tau1*exp(-tn/tau1)) ]

zetax2 = Expression(("varphi2*x[0]*x[1]*(tau2*exp(-tn + 1)/(tau2 - 1) - tau2*exp(-tn/tau2 + 1)/(tau2 - 1))","-(tau2*tau2*exp(-tn/tau2)/(tau2*tau2 + 1) - (tau2*tau2*cos(tn) - tau2*sin(tn))/(tau2*tau2 + 1))*varphi2*sin(x[0]*x[1])"),tau2=tau2, varphi2=varphi2,  tn=0, degree=5)
#===================================================================================================================
tol=1E-15
for i in range(iMin,iMax):
    for j in range(jMin,jMax):
        
        Nxy=pow(2,i)
        print('  i = %d (to %d), Nxy = %d; j = %d (to %d)' % (i, iMax-1, Nxy, j, jMax-1))
        Nt=(2**(j))    # mesh density and number of time steps
        dt = T/Nt      # time step
        mesh = UnitSquareMesh(Nxy, Nxy)
        V = VectorFunctionSpace(mesh, 'DG', k)
            # Define boundary conditions

    #bottom edge
        def bottom(x, on_boundary):
            return near(x[1],0.0,tol) and on_boundary
    # left edge
        def left(x, on_boundary):
            return near(x[0],0.0,tol) and on_boundary


    # define the boundary partition
        boundary_parts =MeshFunction("size_t", mesh,mesh.topology().dim()-1, 0)

    # Mark subdomain 0 for \Gamma_0 etc
    #Gamma0 has Neumann BC(top)
        class Gamma0(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[1],1.0,tol)
        Gamma_0 = Gamma0()
        Gamma_0.mark(boundary_parts, 0)

    #Gamma1 has Neumann BC(right)
        class Gamma1(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[0],1.0,tol)
        Gamma_1 = Gamma1()
        Gamma_1.mark(boundary_parts, 1)

        class GammaDirichlet(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[0]*x[1],0.0,tol)
        Gamma_Dirichlet = GammaDirichlet()
        Gamma_Dirichlet.mark(boundary_parts, 2)

        def DiricletBoundary(x, on_boundary):
            return near(x[0]*x[1],0.0,tol) and on_boundary

        bc = DirichletBC(V,Constant((0.0,0.0)),DiricletBoundary,method="geometric")
        ds = ds(subdomain_data = boundary_parts)
        
        dx=Measure('dx')
    # Define normal vector and mesh size
        n = FacetNormal(mesh)
        h = FacetArea(mesh)
        h_avg = (h('+') + h('-'))/2
        def epsilon(v):
            Dv=grad(v)
            return 0.5*(Dv+Dv.T)
        
    # Initial condition
        ux.tn=0.0; wx.tn=0.0;zetax1.tn=0.0; zetax2.tn=0.0;

        U = VectorFunctionSpace(mesh, 'Lagrange', 5)
        U0=project(ux,U)
        u=TrialFunction(V)
        v=TestFunction(V)
        Ux=interpolate(ux,U)
        A0=inner(epsilon(u),epsilon(v))*dx- inner(avg(epsilon(u)), outer(v('+'),n('+'))+outer(v('-'),n('-')))*dS \
        -inner(avg(epsilon(v)), outer(u('+'),n('+'))+outer(u('-'),n('-')))*dS \
        + alpha/(h_avg**beta)*inner(jump(u), jump(v))*dS \
        - inner(epsilon(u), outer(v,n))*ds(2) \
        - inner(outer(u,n), epsilon(v))*ds(2) \
        + alpha/(h**beta)*dot(u,v)*ds(2)
    
        L0=inner(epsilon(Ux),epsilon(v))*dx- inner(avg(epsilon(Ux)), outer(v('+'),n('+'))+outer(v('-'),n('-')))*dS \
        -inner(avg(epsilon(v)), outer(Ux('+'),n('+'))+outer(Ux('-'),n('-')))*dS \
        + alpha/(h_avg**beta)*inner(jump(Ux), jump(v))*dS \
        - inner(epsilon(Ux), outer(v,n))*ds(2) \
        + inner(outer(Ux,n), epsilon(v))*ds(2) \
        + alpha/(h**beta)*dot(Ux,v)*ds(2)
    
        u0=Function(V)
        solver = LUSolver('mumps')
        solve(A0==L0,u0,bc)
        

        A1=dot(u,v)*dx
        L1=dot(wx,v)*dx
        w0=Function(V)
        solve(A1==L1,w0,bc)

        zeta01 = interpolate(zetax1, V)
        zeta02 = interpolate(zetax2, V)
        
        #================================================================================================#
        fav = Expression(("0.5*(0.5*varphi1*x[0]*x[1]*(tau1*exp(-tn/tau1)/(tau1*tau1 + 1) - (tau1*tau1*sin(tn) + tau1*cos(tn))/(tau1*tau1 + 1))*sin(x[0]*x[1])/tau1 + 0.5*varphi2*x[0]*x[1]*(tau2*exp(-tn/tau2)/(tau2*tau2 + 1) - (tau2*tau2*sin(tn) + tau2*cos(tn))/(tau2*tau2 + 1))*sin(x[0]*x[1])/tau2 + 0.5*x[0]*x[1]*cos(tn)*sin(x[0]*x[1]) + x[0]*x[1]*exp(-tn + 1) - 0.5*varphi1*(tau1*exp(-tn/tau1)/(tau1*tau1 + 1) - (tau1*tau1*sin(tn) + tau1*cos(tn))/(tau1*tau1 + 1))*cos(x[0]*x[1])/tau1 - 0.5*varphi2*(tau2*exp(-tn/tau2)/(tau2*tau2 + 1) - (tau2*tau2*sin(tn) + tau2*cos(tn))/(tau2*tau2 + 1))*cos(x[0]*x[1])/tau2 - 0.5*cos(x[0]*x[1])*cos(tn)            +0.5*varphi1*x[0]*x[1]*(tau1*exp(-(tn+dt)/tau1)/(tau1*tau1 + 1) - (tau1*tau1*sin((tn+dt)) + tau1*cos((tn+dt)))/(tau1*tau1 + 1))*sin(x[0]*x[1])/tau1 + 0.5*varphi2*x[0]*x[1]*(tau2*exp(-(tn+dt)/tau2)/(tau2*tau2 + 1) - (tau2*tau2*sin((tn+dt)) + tau2*cos((tn+dt)))/(tau2*tau2 + 1))*sin(x[0]*x[1])/tau2 + 0.5*x[0]*x[1]*cos((tn+dt))*sin(x[0]*x[1]) + x[0]*x[1]*exp(-(tn+dt) + 1) - 0.5*varphi1*(tau1*exp(-(tn+dt)/tau1)/(tau1*tau1 + 1) - (tau1*tau1*sin((tn+dt)) + tau1*cos((tn+dt)))/(tau1*tau1 + 1))*cos(x[0]*x[1])/tau1 - 0.5*varphi2*(tau2*exp(-(tn+dt)/tau2)/(tau2*tau2 + 1) - (tau2*tau2*sin((tn+dt)) + tau2*cos((tn+dt)))/(tau2*tau2 + 1))*cos(x[0]*x[1])/tau2 - 0.5*cos(x[0]*x[1])*cos((tn+dt)))","0.5*(varphi1*x[0]*x[0]*(tau1*exp(-tn/tau1)/(tau1*tau1 + 1) - (tau1*tau1*sin(tn) + tau1*cos(tn))/(tau1*tau1 + 1))*sin(x[0]*x[1])/tau1 + 0.5*varphi1*x[1]*x[1]*(tau1*exp(-tn/tau1)/(tau1*tau1 + 1) - (tau1*tau1*sin(tn) + tau1*cos(tn))/(tau1*tau1 + 1))*sin(x[0]*x[1])/tau1 + varphi2*x[0]*x[0]*(tau2*exp(-tn/tau2)/(tau2*tau2 + 1) - (tau2*tau2*sin(tn) + tau2*cos(tn))/(tau2*tau2 + 1))*sin(x[0]*x[1])/tau2 + 0.5*varphi2*x[1]*x[1]*(tau2*exp(-tn/tau2)/(tau2*tau2 + 1) - (tau2*tau2*sin(tn) + tau2*cos(tn))/(tau2*tau2 + 1))*sin(x[0]*x[1])/tau2 + x[0]*x[0]*cos(tn)*sin(x[0]*x[1]) + 0.5*x[1]*x[1]*cos(tn)*sin(x[0]*x[1]) - cos(tn)*sin(x[0]*x[1]) - 0.5*varphi1*(tau1*exp(-tn + 1)/(tau1 - 1) - tau1*exp(-tn/tau1 + 1)/(tau1 - 1))/tau1 - 0.5*varphi2*(tau2*exp(-tn + 1)/(tau2 - 1) - tau2*exp(-tn/tau2 + 1)/(tau2 - 1))/tau2 - 0.5*exp(-tn + 1)            +varphi1*x[0]*x[0]*(tau1*exp(-(tn+dt)/tau1)/(tau1*tau1 + 1) - (tau1*tau1*sin((tn+dt)) + tau1*cos((tn+dt)))/(tau1*tau1 + 1))*sin(x[0]*x[1])/tau1 + 0.5*varphi1*x[1]*x[1]*(tau1*exp(-(tn+dt)/tau1)/(tau1*tau1 + 1) - (tau1*tau1*sin((tn+dt)) + tau1*cos((tn+dt)))/(tau1*tau1 + 1))*sin(x[0]*x[1])/tau1 + varphi2*x[0]*x[0]*(tau2*exp(-(tn+dt)/tau2)/(tau2*tau2 + 1) - (tau2*tau2*sin((tn+dt)) + tau2*cos((tn+dt)))/(tau2*tau2 + 1))*sin(x[0]*x[1])/tau2 + 0.5*varphi2*x[1]*x[1]*(tau2*exp(-(tn+dt)/tau2)/(tau2*tau2 + 1) - (tau2*tau2*sin((tn+dt)) + tau2*cos((tn+dt)))/(tau2*tau2 + 1))*sin(x[0]*x[1])/tau2 + x[0]*x[0]*cos((tn+dt))*sin(x[0]*x[1]) + 0.5*x[1]*x[1]*cos((tn+dt))*sin(x[0]*x[1]) - cos((tn+dt))*sin(x[0]*x[1]) - 0.5*varphi1*(tau1*exp(-(tn+dt) + 1)/(tau1 - 1) - tau1*exp(-(tn+dt)/tau1 + 1)/(tau1 - 1))/tau1 - 0.5*varphi2*(tau2*exp(-(tn+dt) + 1)/(tau2 - 1) - tau2*exp(-(tn+dt)/tau2 + 1)/(tau2 - 1))/tau2 - 0.5*exp(-(tn+dt) + 1))"),tau1=tau1,tau2=tau2,varphi1=varphi1,varphi2=varphi2, dt=dt, tn=0, degree=5)
        
        g0av =Expression(("0.5*(0.5*varphi1*x[1]*(tau1*exp(-tn/tau1)/(tau1*tau1 + 1) - (tau1*tau1*sin(tn) + tau1*cos(tn))/(tau1*tau1 + 1))*cos(x[0]*x[1])/tau1 + 0.5*varphi2*x[1]*(tau2*exp(-tn/tau2)/(tau2*tau2 + 1) - (tau2*tau2*sin(tn) + tau2*cos(tn))/(tau2*tau2 + 1))*cos(x[0]*x[1])/tau2 + 0.5*x[1]*cos(x[0]*x[1])*cos(tn) + 0.5*varphi1*x[0]*(tau1*exp(-tn + 1)/(tau1 - 1) - tau1*exp(-tn/tau1 + 1)/(tau1 - 1))/tau1 + 0.5*varphi2*x[0]*(tau2*exp(-tn + 1)/(tau2 - 1) - tau2*exp(-tn/tau2 + 1)/(tau2 - 1))/tau2 + 0.5*x[0]*exp(-tn + 1)            +0.5*varphi1*x[1]*(tau1*exp(-(tn+dt)/tau1)/(tau1*tau1 + 1) - (tau1*tau1*sin((tn+dt)) + tau1*cos((tn+dt)))/(tau1*tau1 + 1))*cos(x[0]*x[1])/tau1 + 0.5*varphi2*x[1]*(tau2*exp(-(tn+dt)/tau2)/(tau2*tau2 + 1) - (tau2*tau2*sin((tn+dt)) + tau2*cos((tn+dt)))/(tau2*tau2 + 1))*cos(x[0]*x[1])/tau2 + 0.5*x[1]*cos(x[0]*x[1])*cos((tn+dt)) + 0.5*varphi1*x[0]*(tau1*exp(-(tn+dt) + 1)/(tau1 - 1) - tau1*exp(-(tn+dt)/tau1 + 1)/(tau1 - 1))/tau1 + 0.5*varphi2*x[0]*(tau2*exp(-(tn+dt) + 1)/(tau2 - 1) - tau2*exp(-(tn+dt)/tau2 + 1)/(tau2 - 1))/tau2 + 0.5*x[0]*exp(-(tn+dt) + 1))","0.5*(varphi1*x[0]*(tau1*exp(-tn/tau1)/(tau1*tau1 + 1) - (tau1*tau1*sin(tn) + tau1*cos(tn))/(tau1*tau1 + 1))*cos(x[0]*x[1])/tau1 + varphi2*x[0]*(tau2*exp(-tn/tau2)/(tau2*tau2 + 1) - (tau2*tau2*sin(tn) + tau2*cos(tn))/(tau2*tau2 + 1))*cos(x[0]*x[1])/tau2 + x[0]*cos(x[0]*x[1])*cos(tn)            +varphi1*x[0]*(tau1*exp(-(tn+dt)/tau1)/(tau1*tau1 + 1) - (tau1*tau1*sin((tn+dt)) + tau1*cos((tn+dt)))/(tau1*tau1 + 1))*cos(x[0]*x[1])/tau1 + varphi2*x[0]*(tau2*exp(-(tn+dt)/tau2)/(tau2*tau2 + 1) - (tau2*tau2*sin((tn+dt)) + tau2*cos((tn+dt)))/(tau2*tau2 + 1))*cos(x[0]*x[1])/tau2 + x[0]*cos(x[0]*x[1])*cos((tn+dt)))"),tau1=tau1,tau2= tau2, varphi1=varphi1,varphi2=varphi2, dt=dt, tn=0, degree=5)

        g1av =Expression(("0.5*(varphi1*x[1]*(tau1*exp(-tn + 1)/(tau1 - 1) - tau1*exp(-tn/tau1 + 1)/(tau1 - 1))/tau1 + varphi2*x[1]*(tau2*exp(-tn + 1)/(tau2 - 1) - tau2*exp(-tn/tau2 + 1)/(tau2 - 1))/tau2 + x[1]*exp(-tn + 1)            +varphi1*x[1]*(tau1*exp(-(tn+dt) + 1)/(tau1 - 1) - tau1*exp(-(tn+dt)/tau1 + 1)/(tau1 - 1))/tau1 + varphi2*x[1]*(tau2*exp(-(tn+dt) + 1)/(tau2 - 1) - tau2*exp(-(tn+dt)/tau2 + 1)/(tau2 - 1))/tau2 + x[1]*exp(-(tn+dt) + 1))","0.5*(0.5*varphi1*x[1]*(tau1*exp(-tn/tau1)/(tau1*tau1 + 1) - (tau1*tau1*sin(tn) + tau1*cos(tn))/(tau1*tau1 + 1))*cos(x[0]*x[1])/tau1 + 0.5*varphi2*x[1]*(tau2*exp(-tn/tau2)/(tau2*tau2 + 1) - (tau2*tau2*sin(tn) + tau2*cos(tn))/(tau2*tau2 + 1))*cos(x[0]*x[1])/tau2 + 0.5*x[1]*cos(x[0]*x[1])*cos(tn) + 0.5*varphi1*x[0]*(tau1*exp(-tn + 1)/(tau1 - 1) - tau1*exp(-tn/tau1 + 1)/(tau1 - 1))/tau1 + 0.5*varphi2*x[0]*(tau2*exp(-tn + 1)/(tau2 - 1) - tau2*exp(-tn/tau2 + 1)/(tau2 - 1))/tau2 + 0.5*x[0]*exp(-tn + 1)            +0.5*varphi1*x[1]*(tau1*exp(-(tn+dt)/tau1)/(tau1*tau1 + 1) - (tau1*tau1*sin((tn+dt)) + tau1*cos((tn+dt)))/(tau1*tau1 + 1))*cos(x[0]*x[1])/tau1 + 0.5*varphi2*x[1]*(tau2*exp(-(tn+dt)/tau2)/(tau2*tau2 + 1) - (tau2*tau2*sin((tn+dt)) + tau2*cos((tn+dt)))/(tau2*tau2 + 1))*cos(x[0]*x[1])/tau2 + 0.5*x[1]*cos(x[0]*x[1])*cos((tn+dt)) + 0.5*varphi1*x[0]*(tau1*exp(-(tn+dt) + 1)/(tau1 - 1) - tau1*exp(-(tn+dt)/tau1 + 1)/(tau1 - 1))/tau1 + 0.5*varphi2*x[0]*(tau2*exp(-(tn+dt) + 1)/(tau2 - 1) - tau2*exp(-(tn+dt)/tau2 + 1)/(tau2 - 1))/tau2 + 0.5*x[0]*exp(-(tn+dt) + 1))"),tau1=tau1,tau2= tau2, varphi1=varphi1,varphi2=varphi2, dt=dt, tn=0, degree=5)
        Rav = Expression('(varphi1*(exp(-tn/tau1)+exp(-(tn+dt)/tau1))+varphi2*(exp(-tn/tau2)+exp(-(tn+dt)/tau2)))*0.5',tau1=tau1,tau2= tau2, varphi1=varphi1,varphi2=varphi2,dt=dt, tn=0, degree=5)
        #================================================================================================#
        u, v = TrialFunction(V), TestFunction(V)
        
# the unknown at a new time level
        uh = Function(V)   
        wh = Function(V)
        zetah1 = Function(V)
        zetah2 = Function(V)

# bilinear form for the solver
        mass = inner(u,v)*dx
        stiffness_sym = inner(epsilon(u),epsilon(v))*dx- inner(avg(epsilon(u)), outer(v('+'),n('+'))+outer(v('-'),n('-')))*dS \
        - inner(avg(epsilon(v)), outer(u('+'),n('+'))+outer(u('-'),n('-')))*dS \
        + alpha/(h_avg**beta)*dot(jump(u), jump(v))*dS \
        - inner(epsilon(u), outer(v,n))*ds(2) \
        - inner(outer(u,n), epsilon(v))*ds(2) \
        + alpha/(h**beta)*dot(u,v)*ds(2)
        jump_penalty =  alpha/(h_avg**beta)*dot(jump(u), jump(v))*dS  + alpha/(h**beta)*dot(u,v)*ds(2)  
        
        
# linear form for the right hand side and internal variables
        L=dot(fav,v)*dx+dot(g0av,v)*ds(0)+dot(g1av,v)*ds(1)\
                    -(Rav*inner(epsilon(U0), epsilon(v))*dx\
                - Rav*inner(avg(epsilon(U0)), outer(v('+'),n('+'))+outer(v('-'),n('-')))*dS \
        -Rav*inner(outer(U0('+'),n('+'))+outer(U0('-'),n('-')), avg(epsilon(v)))*dS \
        + Rav*alpha/(h_avg**beta)*inner(jump(U0), jump(v))*dS \
        - Rav*inner(epsilon(U0), outer(v,n))*ds(2) \
        - Rav*inner(outer(U0,n), epsilon(v))*ds(2) \
        + Rav*alpha/(h**beta)*dot(U0,v)*ds(2)) 

# assemble the system matrix once and for all
        M = assemble(mass)
        A = assemble(stiffness_sym)
        J = assemble(jump_penalty)  
        CurlB=(tau1*varphi1/(2*tau1+dt)+tau2*varphi2/(2*tau2+dt))*A
        B=(2.0/dt/dt*M)+CurlB+(varphi0/2*A)+1.0/dt*J

# assemble only once, before the time stepping
        b = None 
        b2=None
        for n in range(0,Nt):
         #   progbar += 1;
            
            # update data and solve for tn+k
            tn = n*dt; ux.tn = tn; wx.tn = tn; fav.tn = tn; g0av.tn = tn; g1av.tn = tn;Rav.tn=tn;
            b = assemble(L, tensor=b)
            b2=2.0/dt*M*w0.vector().get_local()+(2.0/dt/dt*M-varphi0/2.0*A+CurlB+1.0/dt*J)*u0.vector().get_local()-A*(2.0*tau1/(2*tau1+dt)*zeta01.vector().get_local()+2.0*tau2/(2*tau2+dt)*zeta02.vector().get_local())     
            b.add_local(b2)
            
            solver = LUSolver('mumps')
            bc.apply(B,b)
            
            solve(B, uh.vector(), b)   
            wh.assign(2.0/dt*(uh-u0)-w0)
            zetah1.assign(2.0*tau1*varphi1/(2*tau1+dt)*(uh-u0)+(2*tau1-dt)/(2*tau1+dt)*zeta01)
            zetah2.assign(2.0*tau2*varphi2/(2*tau2+dt)*(uh-u0)+(2*tau2-dt)/(2*tau2+dt)*zeta02)
            
            #update old terms
            w0.assign(wh);u0.assign(uh);zeta01.assign(zetah1);zeta02.assign(zetah2)
            
        # compute error at last time step
        ux.tn = T; wx.tn = T; 
        err1 = errornorm(wx,w0,'L2')        
        err2 = errornorm(ux,u0,'L2')
        err3=sqrt(errornorm(wx,w0,'H10')**2+err1**2)
        err4=sqrt(errornorm(ux,u0,'H10')**2+err2**2)
        
        L2_error[0,j-jMin+1]=Nt; L2_error[i-iMin+1,0]=Nxy; L2_error[i-iMin+1,j-jMin+1]=err1;
        Lu_error[0,j-jMin+1]=Nt; Lu_error[i-iMin+1,0]=Nxy; Lu_error[i-iMin+1,j-jMin+1]=err2;
        H1_error[0,j-jMin+1]=Nt; H1_error[i-iMin+1,0]=Nxy; H1_error[i-iMin+1,j-jMin+1]=err3;
        Hu_error[0,j-jMin+1]=Nt; Hu_error[i-iMin+1,0]=Nxy; Hu_error[i-iMin+1,j-jMin+1]=err4;
        
        
# SS altered the following loop limits
print ('L2_error for w')
print('\\begin{tabular}{|c|',end="")
for j in range(jMin,jMax): print('c',end="")
print('|}\\hline')
for j in range(jMin,jMax):
  if j==jMin: print('    ', end="")
  print(' & %10d ' % L2_error[0,j-jMin+1], end="")
print('  \\\\\\hline')
for i in range(iMin,iMax):
  print('%4d ' % L2_error[i-iMin+1,0], end="")
  for j in range(jMin,jMax):
      print('& %11.4le ' % L2_error[i-iMin+1,j-jMin+1], end="")
  print(' \\\\')
print('\\hline\\end{tabular}')

print ('L2_error for u')
print('\\begin{tabular}{|c|',end="")
for j in range(jMin,jMax): print('c',end="")
print('|}\\hline')
for j in range(jMin,jMax):
  if j==jMin: print('    ', end="")
  print(' & %10d ' % Lu_error[0,j-jMin+1], end="")
print('  \\\\\\hline')
for i in range(iMin,iMax):
  print('%4d ' % Lu_error[i-iMin+1,0], end="")
  for j in range(jMin,jMax):
      print('& %11.4le ' % Lu_error[i-iMin+1,j-jMin+1], end="")
  print(' \\\\')
print('\\hline\\end{tabular}')

print ('H1_error for w')
print('\\begin{tabular}{|c|',end="")
for j in range(jMin,jMax): print('c',end="")
print('|}\\hline')
for j in range(jMin,jMax):
  if j==jMin: print('    ', end="")
  print(' & %10d ' % H1_error[0,j-jMin+1], end="")
print('  \\\\\\hline')
for i in range(iMin,iMax):
  print('%4d ' % H1_error[i-iMin+1,0], end="")
  for j in range(jMin,jMax):
      print('& %11.4le ' % H1_error[i-iMin+1,j-jMin+1], end="")
  print(' \\\\')
print('\\hline\\end{tabular}')

print ('H1_error for u')
print('\\begin{tabular}{|c|',end="")
for j in range(jMin,jMax): print('c',end="")
print('|}\\hline')
for j in range(jMin,jMax):
  if j==jMin: print('    ', end="")
  print(' & %10d ' % Hu_error[0,j-jMin+1], end="")
print('  \\\\\\hline')
for i in range(iMin,iMax):
  print('%4d ' % Hu_error[i-iMin+1,0], end="")
  for j in range(jMin,jMax):
      print('& %11.4le ' % Hu_error[i-iMin+1,j-jMin+1], end="")
  print(' \\\\')
print('\\hline\\end{tabular}')

# Compute the rates of convergence

l2Diag=np.diag(L2_error)[1:]
l2uDiag=np.diag(Lu_error)[1:]
h1Diag=np.diag(H1_error)[1:]
h1uDiag=np.diag(Hu_error)[1:]

m= len(l2Diag)

if m>2:    
    v1=np.array(l2Diag)
    t1=np.log(v1[0:-1]/v1[1:])
    d1=np.mean(t1/np.log(2))

    v2=np.array(l2uDiag)
    t2=np.log(v2[0:-1]/v2[1:])
    d2=np.mean(t2/np.log(2))

    v3=np.array(h1Diag)
    t3=np.log(v3[0:-1]/v3[1:])
    d3=np.mean(t3/np.log(2))
    
    v4=np.array(h1uDiag)
    t4=np.log(v4[0:-1]/v4[1:])
    d4=np.mean(t4/np.log(2))

    print('L2 of w orders')
    print(t1/np.log(2))
    print('L2 of u orders')
    print(t2/np.log(2))
    print('H1 of w orders')
    print(t3/np.log(2))
    print('H1 of u orders')
    print(t4/np.log(2))
    print('Numeical convergent order when h=dt: L2 error of w = %5.4f,  L2 error of u= %5.4f, H1 error of w = %5.4f, H1 error of u = %5.4f' %(d1,d2,d3,d4))  

    if k==1:        
        np.savetxt("L2_error_linear_velo_S2.txt",L2_error,fmt="%2.3e")
        np.savetxt("L2_error_linear_disp_S2.txt",Lu_error,fmt="%2.3e")
        np.savetxt("H1_error_linear_velo_S2.txt",H1_error,fmt="%2.3e")
        np.savetxt("H1_error_linear_disp_S2.txt",Hu_error,fmt="%2.3e")
        
    elif k==2:
        np.savetxt("L2_error_quad_velo_S2.txt",L2_error,fmt="%2.3e")
        np.savetxt("L2_error_quad_disp_S2.txt",Lu_error,fmt="%2.3e")
        np.savetxt("H1_error_quad_velo_S2.txt",H1_error,fmt="%2.3e")
        np.savetxt("H1_error_quad_disp_S2.txt",Hu_error,fmt="%2.3e")

if jMax-jMin==1:
    
    v1=L2_error[1:,1]
    t1=np.log(v1[0:-1]/v1[1:])
    d1=np.mean(t1/np.log(2))
    
    v2=Lu_error[1:,1]
    t2=np.log(v2[0:-1]/v2[1:])
    d2=np.mean(t2/np.log(2))

    v3=H1_error[1:,1]
    t3=np.log(v3[0:-1]/v3[1:])
    d3=np.mean(t3/np.log(2))
    
    v4=Hu_error[1:,1]
    t4=np.log(v4[0:-1]/v4[1:])
    d4=np.mean(t4/np.log(2))

    
    print('L2 of w orders')
    print(t1/np.log(2))
    print('L2 of u orders')
    print(t2/np.log(2))
    print('H1 of w orders')
    print(t3/np.log(2))
    print('H1 of u orders')
    print(t4/np.log(2))
    print('Numeical convergent order for fixed dt: L2 error of w = %5.4f,  L2 error of u= %5.4f, H1 error of w = %5.4f, H1 error of u = %5.4f' %(d1,d2,d3,d4))

    
if iMax-iMin==1:
    
    v1=L2_error[1,1:]
    t1=np.log(v1[0:-1]/v1[1:])
    d1=np.mean(t1/np.log(2))
    
    v2=Lu_error[1,1:]
    t2=np.log(v2[0:-1]/v2[1:])
    d2=np.mean(t2/np.log(2))

    v3=H1_error[1,1:]
    t3=np.log(v3[0:-1]/v3[1:])
    d3=np.mean(t3/np.log(2))
    
    v4=Hu_error[1,1:]
    t4=np.log(v4[0:-1]/v4[1:])
    d4=np.mean(t4/np.log(2))

    
    print('L2 of w orders')
    print(t1/np.log(2))
    print('L2 of u orders')
    print(t2/np.log(2))
    print('H1 of w orders')
    print(t3/np.log(2))
    print('H1 of u orders')
    print(t4/np.log(2))
    print('Numeical convergent order for fixed h: L2 error of w = %5.4f,  L2 error of u= %5.4f, H1 error of w = %5.4f, H1 error of u = %5.4f' %(d1,d2,d3,d4))          
