import numpy as np
import sys
from scipy.sparse import coo_array, diags, bmat, eye, kron
from scipy.optimize import linprog
import matplotlib.pyplot as plt

def observationOp(n,K):
    # image is (2n+1) x (2n+1), representing periodic domain [-1,1]^2, Fourier coefficients go from (-K,-K) to (K,K), observationOp produces the real and imaginary part separately
    freq = np.arange(-K,K+1)*np.pi
    [freq2,freq1] = np.meshgrid(freq,freq)
    pos = np.arange(-n,n+1)/n
    [pos2,pos1] = np.meshgrid(pos,pos)
    freq1 = freq1[:,:,np.newaxis,np.newaxis]
    freq2 = freq2[:,:,np.newaxis,np.newaxis]

    Ob = np.reshape( np.exp(-1j*(freq1*pos1+freq2*pos2)), (-1,(2*n+1)**2), order = 'F' )
    Ob = (np.concatenate([Ob.real,Ob.imag], axis = 0)) 
    Ob = coo_array(Ob)
    Ob.eliminate_zeros()
    return Ob

def RadonOp(n,m,theta):
    #image is (2n+1) x (2n+1), representing domain [-1,1]^2, Radon transform is 2m+1, representing domain [-sqrt2,sqrt2], at each angle theta (Rd is list of sparse matrices)
    x = np.arange(-n,n+1)/n
    [Y,X] = np.meshgrid(x,x)
    Rd = []
    for j in theta:
        proj = X*np.cos(j)+Y*np.sin(j)
        projIdx = np.round(m/np.sqrt(2)*proj)+m
        Rd.append(coo_array((np.ones(X.size, dtype= int),(projIdx.flatten(order = 'F'), np.arange(0,X.size))), shape= np.array([2*m+1,X.size])))
    return Rd

def moveOp(n,m,t):
    #space-velocity projection is (2n+1) x (2n+1), representing domain [-sqrt2,sqrt2]^2, result is 2m+1, representing domain [-sqrt2,sqrt2], at all times in t\in[-1,1] (Mv is list of sparse matrices)
    x = np.sqrt(2)/n*np.arange(-n,n+1)
    [V,X] = np.meshgrid(x,x)
    Mv = []

    for s in t:
        proj = X + s * V
        projIdx = np.round(m/np.sqrt(2)*proj)+m
        origIdx = np.arange(0,X.size)
        origIdx = origIdx[np.logical_not(np.logical_or((projIdx>2*m),(projIdx<0))).flatten(order = 'F')]
        projIdx = projIdx.transpose()[np.logical_not(np.logical_or((projIdx.transpose()>2*m),(projIdx.transpose()<0)))] #transpose needed because matlab uses order F to find the elements satisfying the constraint
        Mv.append(coo_array((np.ones_like(origIdx, dtype=int),(projIdx, origIdx)), shape= np.array([2*m+1,X.size])))

    return Mv

def divergence(m):
    #image is 2m+1, representing domain [-sqrt2,sqrt2]
    div = diags([np.ones(2*m+1), -np.ones(2*m)], [0,-1], shape = (2*m+1, 2*m+1)) #note: the discrete derivative is not scaled with the grid width since it is the divergence of Dirac masses! A value x at a pixel of the argument means mass x, not mass (grid width times x)!
    return div
    #note: the first line just returns the first entry of the argument; this is as if the argument has another entry at index -1 with value 0, i.e. we compute divergence with zero boundary conditions


def inexactLinearTracking():

    #set up concrete reconstruction problem example
    #define particle number N, image size 2n+1, projection size 2m+1, Fourier cutoff frequency K, etc.
    N = 3  # number of particles
    n = 16 # 2n+1 grid points in x- and y-direction representing [-1,1]^2
    m = 16 # 2m+1 grid points along 1d domains representing [-sqrt(2),sqrt(2)]
    K = 2  # cutoff frequency
    T = 11 # number of reconstruction times in [-1,1]
    L = 5  # number of angles between [0,pi)
    rho = .2

    #define measurement times, reconstruction times and projection angles
    tRecon = np.linspace(-1,1,T)
    tObInd = np.arange(1,T+1)
    theta = np.linspace(0,np.pi,L+1)[:-1]

    #define parabolic particle trajectories; rows correspond to particles, each row contains initial location, initial velocity, constant acceleration, particle mass
    eps = .2
    X_V_A_m = np.array([[-.2,-.2,.75,-.75,0,eps,.6], #(x_1,x_2),(v_1,v_2),(a_1,a_2),m
                [-.4,-.2,-.3,-.7,-eps,0,.8],
                [.2,0,.3,.9,0,eps,.7],
                [.2,.2,-.5,.5,-eps,-eps,.5],
                [.6,.2,.2,-.5,eps,0,.9]])

    X_V_A_m = X_V_A_m[:N,:,np.newaxis]

    t= tRecon
    # compute Fourier measurements f (2 for real and imaginary part  x number of time steps x (2K+1) x (2K+1))
    freq = np.arange(-K,K+1)*np.pi
    [freq2,freq1] = np.meshgrid(freq,freq)

    particleLocations = X_V_A_m[:,0:2]+ t* X_V_A_m[:,2:4] + t**2 /2* X_V_A_m[:,4:6] 
    particleLocations = particleLocations[:,:,:,np.newaxis,np.newaxis]
    f = np.sum(np.exp(-1j*(freq1*particleLocations[:,0,:] + freq2*particleLocations[:,1,:])) * np.squeeze(X_V_A_m[:,6])[:,np.newaxis, np.newaxis, np.newaxis], axis = 0)
    f= np.array([f.real,f.imag])
    totalMass = f[0,0,K,K]

    # set up linear program (order of variables u_t,\gamma_\theta,v_{t,\theta},V_{t,\theta})
    # create basic operators
    div = divergence(m)
    Mv = moveOp(n,m,tRecon)
    Rd = RadonOp(n,m,theta)
    Ob = observationOp(n,K)
    lengthFT = np.shape(Ob)[0]
    lengthUT = np.shape(Rd[0])[1]
    lengthGammaTheta = np.shape(Mv[0])[1]
    lengthUGamma = T*lengthUT+L*lengthGammaTheta
    lengthV = L*T*(2*m+1)

    #constraint div v_{t,\theta} = Mv_t \gamma_\theta - Rd_\theta u_t

    for k in range(T):
        for j in range(L):
            if k == 0 and j == 0:
                consistencyConstraint = bmat([[coo_array((2*m+1,(k)*lengthUT)),Rd[j],coo_array((2*m+1,(T-k-1)*lengthUT)),
                                            coo_array((2*m+1,(j)*lengthGammaTheta)),-Mv[k],coo_array((2*m+1,(L-j-1)*lengthGammaTheta)),
                                            coo_array((2*m+1,(2*m+1)*(L*(k)+j))),-div,coo_array((2*m+1,(2*m+1)*(L*(T-k-1)+L-j-1))),
                                            coo_array((2*m+1,lengthV)) ]])
            else:
                add = bmat([[coo_array((2*m+1,(k)*lengthUT)),Rd[j],coo_array((2*m+1,(T-k-1)*lengthUT)),
                                            coo_array((2*m+1,(j)*lengthGammaTheta)),-Mv[k],coo_array((2*m+1,(L-j-1)*lengthGammaTheta)),
                                            coo_array((2*m+1,(2*m+1)*(L*(k)+j))),-div,coo_array((2*m+1,(2*m+1)*(L*(T-k-1)+L-j-1))),
                                            coo_array((2*m+1,lengthV))]])
                consistencyConstraint = bmat([[consistencyConstraint],[add]])

    #constraints V >= v, V >= -v
    absConstraint = bmat([[coo_array((lengthV,lengthUGamma)),-eye(lengthV,lengthV),-eye(lengthV,lengthV)],
                        [coo_array((lengthV,lengthUGamma)), eye(lengthV,lengthV),-eye(lengthV,lengthV)]])

    #constraints h sum V_{t,\theta} <= eps t^2 mass, where mass = (f_0)_0
    numWassersteinConstraints = L*T

    WassersteinConstraint = bmat([[coo_array((numWassersteinConstraints,lengthUGamma+lengthV)),
                            kron(eye(numWassersteinConstraints) , coo_array((np.sqrt(2)/m * np.ones(2*m+1), (np.zeros(2*m+1) , np.arange(0,2*m+1))), shape = (1,2*m+1)))]])
    WassersteinConstraintRHS = kron(eps * totalMass * tRecon**2 ,np.ones(L)).toarray()[0,:]
    lengthNonUGamma = 2 * lengthV

    #constraint |(Ob_t u_t-f_t)_\xi| \leq h/2 \pi |\xi|_1 u_t(\Omega)
    for k in tObInd:
        if k == 1:
            observationConstraint = bmat([[coo_array((lengthFT,(k-1)*lengthUT)), Ob, coo_array((lengthFT,(T-k)*lengthUT)), coo_array((lengthFT,L*lengthGammaTheta+lengthNonUGamma))]])
        else:
            add = bmat([[coo_array((lengthFT,(k-1)*lengthUT)), Ob, coo_array((lengthFT,(T-k)*lengthUT)), coo_array((lengthFT,L*lengthGammaTheta+lengthNonUGamma))]])
            observationConstraint = bmat([[observationConstraint],
                                        [add]])
    observationConstraint = bmat([[observationConstraint],[-observationConstraint]])
    observationConstraintMeasurement = np.transpose(np.array([f,-f]),axes = (3,4,1,2,0)) #x-frequency, y-frequency, real-imag, time, both inequalities
    observationConstraintDeviation = np.transpose(np.tile( totalMass / n / 2 * (abs(freq1)+abs(freq2)), (2, T, 2, 1, 1) ),axes = (3,4,2,1,0))
    observationConstraintMeasurement = observationConstraintMeasurement + rho*observationConstraintDeviation

    #assemble all constraints and define the cost vector
    Aeq = consistencyConstraint
    beq = np.zeros(np.shape(consistencyConstraint)[0])
    A = bmat([[observationConstraint],[absConstraint],[WassersteinConstraint]])
    b = np.concatenate([observationConstraintMeasurement.flatten(order = 'F'),np.zeros(np.shape(absConstraint)[0]),WassersteinConstraintRHS])
    costVec = np.concatenate([np.ones(lengthUGamma),np.zeros(lengthNonUGamma)])
    bound = np.full((np.size(costVec),2), None)
    bound[:lengthUGamma,0] = 0

    #solve linear programs
    #solution with time coupling
    UGammaVV = linprog(costVec, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq, bounds = bound, options = { "disp" : True}, method='highs-ipm') #the interior-point-method from scipy with the HiGHS interface is used
    uRes = np.reshape(UGammaVV.x[:T*lengthUT],(2*n+1,2*n+1,T), order = 'F')
    gammaRes = np.reshape(UGammaVV.x[T*lengthUT:T*lengthUT+L*lengthGammaTheta],(2*n+1,2*n+1,L), order = 'F')

    #solution without time coupling
    UGammaVVStatic = linprog(costVec,A_ub = observationConstraint, b_ub = observationConstraintMeasurement.flatten(order = 'F'), options = { "disp" : True}, method = 'highs-ipm')
    uStatic = np.reshape(UGammaVVStatic.x[:T*lengthUT],(2*n+1,2*n+1,T), order = 'F')

    #ground truth
    DiracIndices = (np.round(particleLocations*n)+n).astype(int)
    uGT = np.zeros((2*n+1,2*n+1,T))
    for j in range(T):
        for k in range(N):
            uGT[DiracIndices[k,0,j,0,0],DiracIndices[k,1,j,0,0],j] = X_V_A_m[k,6,0]


    liftedParticles = np.array([np.sum(X_V_A_m[:,:2]*np.array([np.cos(theta),np.sin(theta)]), axis = 1),
                                np.sum(X_V_A_m[:,2:4]*np.array([np.cos(theta),np.sin(theta)]), axis = 1)])

    liftedParticles = np.transpose(liftedParticles, axes = (1,0,2))
    DiracIndices2 = (np.round(liftedParticles*n/np.sqrt(2))+n).astype(int)
    gammaGT = np.zeros((2*n+1,2*n+1,L))

    for j in range(L):
        for k in range(N):
            gammaGT[DiracIndices2[k,0,j],DiracIndices2[k,1,j],j] = X_V_A_m[k,6,0]
    
    #Plot
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
    })
    fig, axes = plt.subplots(2, 3)

    axes[0,0].imshow(1 - np.sum(uGT, axis=2), cmap='gray')
    axes[0,0].set_axis_off()
    axes[0,0].set_title(r'$\sum_{t}u_t^\dagger$', fontsize=14)

    axes[0,1].imshow(1 - uGT[:, :, 3], cmap='gray')
    axes[0,1].set_axis_off()
    axes[0,1].set_title(r'$u_{-2/5}^\dagger$', fontsize=14)

    axes[0,2].imshow(1 - gammaGT[:, :, 1], cmap='gray')
    axes[0,2].set_axis_off()
    axes[0,2].set_title(r'$\gamma_{\pi/5}^\dagger$', fontsize=14)

    axes[1,0].imshow(1 - uStatic[:, :, 3], cmap='gray')
    axes[1,0].set_axis_off()
    axes[1,0].set_title(r'$\mathrm{argmin}_{u\,\mathrm{s.t.\ Ob} u=f_{-2/5}}\|u\|$', fontsize=14)

    
    axes[1,1].imshow(1 - uRes[:, :, 3], cmap='gray')
    axes[1,1].set_axis_off()
    axes[1,1].set_title(r'$u_{-2/5}$', fontsize=14)

    axes[1,2].imshow(1 - gammaRes[:, :, 1], cmap='gray')
    axes[1,2].set_axis_off()
    axes[1,2].set_title(r'$\gamma_{\pi/5}$', fontsize=14)

    fig.tight_layout()
    plt.show()

inexactLinearTracking()
        