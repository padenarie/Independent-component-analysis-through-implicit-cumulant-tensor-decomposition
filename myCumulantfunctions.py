from json import tool
import numpy as np
from scipy import linalg
import scipy as sc
import myTensorfunctions as mtf
from scipy import signal
import timeit
import matplotlib.pyplot as plt
import svd_update as su
import sys
eps = sys.float_info.epsilon


# Cumulant tensor diags entries
def cum4_diagsWHITE(Z0,U0):
    P = U0.shape[1]; I = Z0.shape[1]
    Z = U0.T @ np.copy(Z0)
    DIAG = np.zeros(P)
    for p in range(P):
        DIAG[p] = 1/I * Z[p,:].reshape([1,I]) @ (Z[p,:].reshape([1,I])**3).T - 3/(I**2)* (Z[p,:] @ Z[p,:].T)**2
    return DIAG

def measdiag4WHITE(T0,Z0,U0):
    T = np.copy(T0)
    Tdiag = cum4_diagsWHITE(Z0,U0)
    return np.sqrt(mtf.my_inner_prod(Tdiag,Tdiag))/np.sqrt(mtf.my_inner_prod(T,T))
    

# own timer function
def my_perftimer(func, *args, num_runs, num_repeat,average):
    ex_time = timeit.Timer(lambda: func(*args)).repeat(repeat=num_repeat,number=num_runs)
    if average == True:
        print(f'It took {np.average(np.asarray(ex_time)/num_runs)}')
    else:
        print(f'It took {np.asarray(ex_time)/num_runs}')

# statistic COVARIANCE - 2ND MOMENT
def cum2(a,b):
    I = len(a)
    cum2 = np.sum(np.multiply(a,b))/I
    return cum2

# statistic KURTOSIS - 4th MOMENT
def cum4(a,b,c,d):
    I = len(a)
    cum4 = np.sum(np.multiply(np.multiply(a,b),np.multiply(c,d)))/I - cum2(a,b)*cum2(c,d) - cum2(a,c)*cum2(b,d) - cum2(a,d)*cum2(b,c)
    return cum4

def cum4tensor(X):
    P = X.shape[0]
    cum4tensor = np.zeros([P,P,P,P])
    Teye = np.multiply.outer(np.eye(P),np.eye(P)) + np.multiply.outer(np.eye(P),np.eye(P)).transpose([0,2,1,3]) +  np.multiply.outer(np.eye(P),np.eye(P)).transpose([0,3,2,1])
    for i in range(P):
        for j in range(P):
            for k in range(P):
                for l in range(P):
                    cum4tensor[i,j,k,l] = cum4(X[i,:],X[j,:],X[k,:],X[l,:])
    return cum4tensor, Teye

def mom4tensor(X):
    P,I = X.shape
    M = 1/I*X @ (sc.linalg.khatri_rao(X,sc.linalg.khatri_rao(X,X))).T 
    return M.reshape([P,P,P,P])

# ALternative version of cumulant tensor
def cum4tensor2(X):
    I = X.shape[1]
    P = X.shape[0]
    M4 = 0
    for l in range(I):
        L = (X[:,l],X[:,l],X[:,l],X[:,l])
        M4 += 1/I*mtf.my_out_prod(L)

    M2a = 0

    for l2 in range(I):
        L2 = (X[:,l2],X[:,l2])
        M2a += mtf.my_out_prod(L2) 


    M2wholea = 1/I**2*np.multiply.outer(M2a, M2a)
    M2wholeb = np.transpose(np.copy(M2wholea),(0,2,1,3))
    M2wholec = np.transpose(np.copy(M2wholea),(0,2,3,1))
    
    return M4 -M2wholea -M2wholeb -M2wholec, M4, M2wholea, M2wholeb, M2wholec

def cum4tensor3(X):
    I = X.shape[1]
    P = X.shape[0]
    M4 = 1/I * X @ (sc.linalg.khatri_rao(X,sc.linalg.khatri_rao(X,X))).T
    M4 = M4.reshape([P,P,P,P],order='F')
    M2a = 1/I * X @ X.T

    M2wholea = np.multiply.outer(M2a, M2a)
    M2wholeb = np.transpose(np.copy(M2wholea),(0,2,1,3))
    M2wholec = np.transpose(np.copy(M2wholea),(0,2,3,1))
    
    return M4 -M2wholea -M2wholeb -M2wholec

# For Fair comparison: use the per slice computation
def cum4tensor4(X):
    I = X.shape[1]
    P = X.shape[0]
    C4t = np.zeros([P,P,P,P])
    for p1 in range(P):
        for p2 in range(P):
            MC = 1/I*X@X.T
            MT = 1/I*X @ (X[p1,:].reshape([1,I])*X[p2,:].reshape([1,I]) * X).T - MC*MC[p1,p2] - np.multiply.outer(MC[:,p1],MC[:,p2]) - np.multiply.outer(MC[:,p2],MC[p1,:])
            C4t[:,:,p1,p2] = MT
    return C4t

# Whiten data SVD more efficient
def whitendataSVD(X,removemean):
    if X.shape[0] > X.shape[1]:
        X = X.T
    I = X.shape[1]   
    if removemean == True:
        X = (X.T-np.mean(X,axis=1)).T
    U,Sigma,Vh = sc.linalg.svd(X,full_matrices=False)
    return Vh*np.sqrt(I),U,Sigma*(np.sqrt(I)**-1)

# Initialize with SVD:
def SVDinit(X,R,removemean):
    X,U00,Sigma00 = whitendataSVD(X,removemean)
    return Sigma00[0:R], U00[:,0:R], X[0:R,:]

# Generate testset
def testset(n,I,noise_factor,start_end=[0,8],random_mixmatrix=True):
    # Generate sample data
    time = np.linspace(start_end[0], start_end[1], I)
    s1 = 0.6*np.sin(2 * time)  # Signal 1 : sinusoidal signal
    s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
    s3 = 1.2*signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal
    s4 = 1*np.sqrt(time)  # Signal 4: square root signal

    if n <= 4:
        S = np.c_[s1,s2,s3,s4]; S = S[:,0:n].T
        if random_mixmatrix == True:
            Amix = np.random.rand(4,4)
            # Amix += 0.1
        else:
            Amix = np.array([[1, 1, 1, 1], [0.5, 2, 1.0, 0.5], [1.5, 1.0, 2.0, 1.1], [1, 2.0, 1.50, 0.2]])  # Mixing matrix
        # S +=  np.random.normal(size=S.shape,scale=noise_factor)  # Add noise
        Xmix = Amix[0:n,0:n] @ S   # Generate observations
        Noise = np.random.normal(size=Xmix.shape,scale=noise_factor)
        Xmix +=  Noise # Add noise
        # Xmix += noise_factor*np.random.normal(size=Xmix.shape)  # Add noise
    else:
        S = np.c_[s1,s2,s3,s4].T
        Amix = np.random.rand(n,4) 
        # Amix += 0.1
        # S +=  np.random.normal(size=S.shape,scale=noise_factor)  # Add noise
        Xmix = Amix @ S   # Generate observations
        Noise = np.random.normal(size=Xmix.shape,scale=noise_factor)
        Xmix +=  Noise # Add noise
        # Xmix += noise_factor*np.random.normal(size=Xmix.shape)  # Add noise
    return Xmix,S,Amix, Noise

# Generate testset
def testset2(n,I,noise_factor,start_end=[0,8],random_mixmatrix=True):
    # Generate sample data
    time = np.linspace(start_end[0], start_end[1], I)
    s1 = 0.6*np.sin(2 * time)  # Signal 1 : sinusoidal signal
    s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
    s3 = 1.2*signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal
    s4 = 1*np.sqrt(time)  # Signal 4: square root signal

    if n <= 4:
        S = np.c_[s1,s2,s3,s4]; S = S[:,0:n].T
        if random_mixmatrix == True:
            Amix = np.random.rand(4,4)
            Amix += 0.1
        else:
            Amix = np.array([[1, 1, 1, 1], [0.5, 2, 1.0, 0.5], [1.5, 1.0, 2.0, 1.1], [1, 2.0, 1.50, 0.2]])  # Mixing matrix
        S += 4*noise_factor * np.random.normal(size=S.shape)  # Add noise
        Xmix = Amix[0:n,0:n] @ S   # Generate observations
        Xmix += np.random.normal(size=Xmix.shape,scale=noise_factor)  # Add noise
        # Xmix += noise_factor*np.random.normal(size=Xmix.shape)  # Add noise
    else:
        S = np.c_[s1,s2,s3,s4].T
        Amix = np.random.randn(n,4)
        # Amix += 0.1
        S += 4*noise_factor * np.random.normal(size=S.shape)  # Add noise
        Xmix = Amix @ S   # Generate observations
        Xmix += np.random.normal(size=Xmix.shape,scale=noise_factor)  # Add noise
        # Xmix += noise_factor*np.random.normal(size=Xmix.shape)  # Add noise
    return Xmix,S,Amix

# Define function recreate tensor from CPD form
def CPDrecon(U,lam):
    R = U.shape[1]
    T = lam[0]*mtf.my_out_prod([U[:,0],U[:,0],U[:,0],U[:,0]])
    for i in range(1,R):
        T = T + lam[i]*mtf.my_out_prod([U[:,i],U[:,i],U[:,i],U[:,i]])
    return T


# Function for error tracking
def Errorfunc(S,S_hat):
    # Init test loop
    P,I = S_hat.shape
    P0 = S.shape[0]
    Error = np.zeros(P)
    Finalerror = np.zeros(P0)
    Finalkeys = np.zeros(P0,dtype=int)
    minus_final = np.zeros(P)

    L = [*range(P)]
    L2 = []
    noise_components = [*range(P)]

    S_norm_zmean = np.zeros(S.shape)
    for p in range(P0):
        S[p,:] += - np.mean(S[p,:])
        S_norm_zmean[p,:] = S[p,:]/np.sqrt(mtf.my_inner_prod(S[p,:],S[p,:]))
    for p in range(P):
        S_hat[p,:] += - np.mean(S_hat[p,:])
        S_hat[p,:] = S_hat[p,:]/np.sqrt(mtf.my_inner_prod(S_hat[p,:],S_hat[p,:]))

    cor = np.abs(np.corrcoef(S_norm_zmean,S_hat,rowvar=True))[0:P0,P0::]
    Lsort = np.argsort(np.max(cor,axis=0))[::-1]

    for j in range(P0):
        minus_sub = np.ones(P)
        for p in Lsort:
            if p in L2:
                Error[p] = 1000
            else:
                Smin = np.copy(S_norm_zmean[P0-1-j,:]) - np.copy(S_hat[p,:])
                Splus = np.copy(S_norm_zmean[P0-1-j,:]) + np.copy(S_hat[p,:])
                if np.argmin([np.sqrt(mtf.my_inner_prod(Smin,Smin)),np.sqrt(mtf.my_inner_prod(Splus,Splus))]) == 1:
                    minus_sub[p] = -1
                else:
                    minus_sub[p] = 1
                Error[p] = np.min([np.sqrt(mtf.my_inner_prod(Smin,Smin)),np.sqrt(mtf.my_inner_prod(Splus,Splus))])
        
        if np.sum(Error >= 1000) <= 3:
            L2.append(np.argmin(Error))
            L.remove(np.argmin(Error))

        minus_final[P0-1-j] = minus_sub[np.argmin(Error)]       # Signs of signals in order of Lsort
        Finalkeys[P0-1-j] = np.argmin(Error)          # indices of how unmixed data matches source components in order
        Finalerror[P0-1-j] = np.min(Error)               # errors in order of the error of the matched observation with the j'th source component
        noise_components.remove(np.argmin(Error))
    
    

    Shat_sorted = np.zeros([P0,I])
    for p,pval in enumerate(Finalkeys):
        Shat_sorted[p] = S_hat[pval]

    cor = np.abs(np.corrcoef(S,Shat_sorted,rowvar=True))[0:P0,P0::]
    Error = np.sqrt(mtf.my_inner_prod(Finalerror,Finalerror))/np.sqrt(mtf.my_inner_prod(S_norm_zmean,S_norm_zmean))
    
    
    # if P <= 4:            # OLD
    #     # Margins for P=4
    #     ERRmarg = [0.7,0.7,0.7,0.7]
    #     cormarg = [0.75,0.8,0.8,0.75]
    # else:
    #     # Margins for P>4


    ERRmarg = [0.6,0.6,0.6,0.6]
    cormarg = [0.8,0.8,0.8,0.8]

    # Compare found sources with error margins
    fail = []; found = [*range(P0)]
    for j in range(P0):
        if (Finalerror[j] > ERRmarg[j]) or (cor[j,j]<cormarg[j]):
            fail.append(j)
            found.remove(j)

    result_dict = {}
    if len(found) == 4:
        result_dict['succes'] = 1
    else:
        result_dict['succes'] = 0
    result_dict['found'] = found
    result_dict['fails'] = fail
    result_dict['noise_comps'] = noise_components
    result_dict['Finalkeys'] = Finalkeys
    result_dict['Error'] = Error
    result_dict['Errors'] = Finalerror
    result_dict['cor'] = cor
    result_dict['signs'] = minus_final
    return result_dict

## ------------3dtestset----------------------------

# create 3d data signal set
def testset3d(I,noise_factorX,noise_factorY,noise_factorZ,mixdimP):
    # Create voxel test data set
    z1 = np.linspace(0, 15, I)
    x1 = 3*np.sin(z1)+8
    y1 = 5*np.cos(z1)+10

    x2 = np.linspace(0, 15, I)
    z2 = np.sin(x2)
    y2 = 2*x2

    x3 = np.linspace(0, 15, I)
    z3 = 0.5*np.sin(x3)+6
    y3 = np.sign(np.sin(3 *x3))+12

    y4 = np.linspace(0, 25, I)
    x4 = signal.sawtooth(1 * np.pi * y4) +6
    z4 = np.sin(x4) +10

    Ax = np.random.randn(mixdimP,4)
    Ay = np.random.randn(mixdimP,4)
    Az = np.random.randn(mixdimP,4)
    
    Sx = np.c_[x1,x2,x3,x4].T
    Sy = np.c_[y1,y2,y3,y4].T
    Sz = np.c_[z1,z2,z3,z4].T

    Sxn = Sx + noise_factorX * np.random.normal(size=Sx.shape) 
    Syn = Sy + noise_factorY * np.random.normal(size=Sx.shape) 
    Szn = Sz + noise_factorZ * np.random.normal(size=Sx.shape) 

    Xx = np.dot(Ax,Sxn); Xy = np.dot(Ay,Syn); Xz = np.dot(Az,Szn)
    return [Xx,Xy,Xz], [Sx,Sy,Sz], [Sxn,Syn,Szn]

def plot3dtestset(Xx,Xy,Xz):
    print('Type above your code to get ienractive plots: %matplotlib widget')
    plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')
    for i in range(Xx.shape[0]):
        ax.plot3D(Xx[i,:], Xy[i,:], Xz[i,:],linewidth=3)
    ax.set_xlabel('X',fontsize=15)
    ax.set_ylabel('Y',fontsize=15)
    ax.set_zlabel('Z',fontsize=15)

def plotvoxeldata(vox):
    print('Type above your code to get ienractive plots: %matplotlib widget')
    ax = plt.figure().add_subplot(projection='3d')
    for n in range(len(vox)):
        ax.voxels(vox[n], edgecolor='k')
    ax.set_xlabel('X',fontsize=15)
    ax.set_ylabel('Y',fontsize=15)
    ax.set_zlabel('Z',fontsize=15)

def voxelizer(Sx,Sy,Sz,Xdim,fscale):
    # Dimensions for Voxel data
    Ydim = Xdim
    Zdim = Xdim

    # create value funcions
    I = Sx.shape[1]

    # Separate data range into voxelranges
    max_xrange = np.max(Sx); max_yrange = np.max(Sy); max_zrange = np.max(Sz)
    min_xrange = np.min(Sx); min_yrange = np.min(Sy); min_zrange = np.min(Sz)

    rangeSx = np.linspace(min_xrange,max_xrange,Xdim+1)
    rangeSy = np.linspace(min_yrange,max_yrange,Ydim+1)
    rangeSz = np.linspace(min_zrange,max_zrange,Zdim+1)

    
    c = {}
    for n in range(4):
        x, y, z = np.indices((Xdim, Ydim, Zdim))

        cx_min = rangeSx[x]; cx_max = rangeSx[x+1]
        cy_min = rangeSy[y]; cy_max = rangeSy[y+1]
        cz_min = rangeSz[z]; cz_max = rangeSz[z+1]

        c[n] = np.full((Xdim, Ydim, Zdim), False).astype(float)
        for t1 in range(Xdim):
            for t2 in range(int(I/Xdim)):
                t = t1*int(I/Xdim)+t2
                c[n] += (((cx_min <= Sx[n,t]) & (Sx[n,t] < cx_max)) & ((cy_min <= Sy[n,t]) & (Sy[n,t] < cy_max)) & ((cz_min <= Sz[n,t]) & (Sz[n,t] < cz_max))).astype(float)*fscale
    return c



##-------------- COM2 algorithm---------------------

# COM2 algrithm

# Test tensor Jacobi method COM2
def offdiag4(T):
    I = T.shape[0]
    C = np.copy(T)
    for i in range(I):
        C[i,i,i,i] = 0
    return np.sqrt(mtf.my_inner_prod(np.abs(C),np.abs(C)))

def diag4(T):
    return np.sqrt(mtf.my_inner_prod(trace4D(T),trace4D(T)))

def trace4D(T):
    I = T.shape[0]
    trace = np.array([T[i,i,i,i] for i in range(I)])
    return trace

def offdiag3(T):
    I = T.shape[0]
    C = np.copy(T)
    for i in range(I):
        C[i,i,i] = 0
    return np.sqrt(mtf.my_inner_prod(np.abs(C),np.abs(C)))

def diag3(T):
    I = T.shape[0]
    trace = np.array([T[i,i,i] for i in range(I)])
    return np.sqrt(mtf.my_inner_prod(trace,trace))

# measure of diagonality for symmetric tensor
def measdiag4(T0):
    T = np.copy(T0)
    Tdiag = np.zeros(T.shape)
    for i in range(T.shape[0]):
        Tdiag[i,i,i,i] = T[i,i,i,i]
    return np.sqrt(mtf.my_inner_prod(Tdiag,Tdiag))/np.sqrt(mtf.my_inner_prod(T,T))

# optimal jacobi rotation
def opt_jacobi_rot4_A(T,p,q):
    u = T[p,p,p,p]**2+T[q,q,q,q]**2
    v = (T[p,p,p,p]+T[q,q,q,q]-6*T[p,p,q,q])*(T[p,q,q,q]-T[p,p,p,q])

    d4 = T[p,p,p,p]*T[p,p,p,q] - T[p,q,q,q]*T[q,q,q,q]
    d3 = u - 4*(T[p,p,p,q]**2+T[p,q,q,q]**2) - 3*T[p,p,q,q]*(T[p,p,p,p]+T[q,q,q,q])
    d2 = 3*v
    d1 = 3*u -2*T[p,p,p,p]*T[q,q,q,q]-32*T[p,p,p,q]*T[p,q,q,q]-36*T[p,p,q,q]**2
    d0 = -4*(v+4*d4)
    roots = np.roots([d4,d3,d2,d1,d0])
    
    a0 = T[p,p,p,p]; a1 = 4*T[p,p,p,q]; a2 = 6*T[p,p,q,q]; a3 = 4*T[p,q,q,q]; a4 = T[q,q,q,q]

    b4 = a0**2 + a4**2; b3 = 2*(a2*a4 - a0*a1); b2 = 4*a0**2 + 4*a4**2 + a1**2 + a3**2 + 2*a0*a2 + 2*a2*a4
    b1 = 2*(-3*a0*a1 + 3*a3*a4 + a1*a4 - a0*a3 + a2*a3 - a1*a2); b0 = 2*(a0**2 + a1**2 + a2**2 + a3**2 + a4**2 +2*a0*a2 + 2*a0*a4 + 2*a1*a3 + 2*a2*a4)

    zeta = np.real(roots[np.iscomplex(roots) == False])
    psi = np.power((np.power(zeta,2) + 4),-2)*(b0 + b1*zeta + b2*np.power(zeta,2) + b3*np.power(zeta,3) + b4*np.power(zeta,4))

    thetas = np.roots([1,-np.real(zeta[np.where(psi == np.amax(psi))[0][0]]),-1])
    loc = np.where(np.logical_and(thetas>-1,thetas<=1))
    theta = thetas[loc]

    c = 1/np.sqrt(1+theta**2); s = -theta/np.sqrt(1+theta**2)

    J1 = np.identity(T.shape[0])
    J1[p,p] = c; J1[p,q] = -s; J1[q,p] = np.conj(s); J1[q,q] = c
    return J1

# For the literature version   with Tpair
def opt_jacobi_rot4_B(T,p,q,Orig_shape):
    p2 = np.copy(p); p = 0; q2 = np.copy(q); q = 1
    u = T[p,p,p,p]**2+T[q,q,q,q]**2
    v = (T[p,p,p,p]+T[q,q,q,q]-6*T[p,p,q,q])*(T[p,q,q,q]-T[p,p,p,q])

    d4 = T[p,p,p,p]*T[p,p,p,q] - T[p,q,q,q]*T[q,q,q,q]
    d3 = u - 4*(T[p,p,p,q]**2+T[p,q,q,q]**2) - 3*T[p,p,q,q]*(T[p,p,p,p]+T[q,q,q,q])
    d2 = 3*v
    d1 = 3*u -2*T[p,p,p,p]*T[q,q,q,q]-32*T[p,p,p,q]*T[p,q,q,q]-36*T[p,p,q,q]**2
    d0 = -4*(v+4*d4)
    roots = np.roots([d4,d3,d2,d1,d0])
    
    a0 = T[p,p,p,p]; a1 = 4*T[p,p,p,q]; a2 = 6*T[p,p,q,q]; a3 = 4*T[p,q,q,q]; a4 = T[q,q,q,q]

    b4 = a0**2 + a4**2; b3 = 2*(a2*a4 - a0*a1); b2 = 4*a0**2 + 4*a4**2 + a1**2 + a3**2 + 2*a0*a2 + 2*a2*a4
    b1 = 2*(-3*a0*a1 + 3*a3*a4 + a1*a4 - a0*a3 + a2*a3 - a1*a2); b0 = 2*(a0**2 + a1**2 + a2**2 + a3**2 + a4**2 +2*a0*a2 + 2*a0*a4 + 2*a1*a3 + 2*a2*a4)

    zeta = np.real(roots[np.iscomplex(roots) == False])
    psi = np.power((np.power(zeta,2) + 4),-2)*(b0 + b1*zeta + b2*np.power(zeta,2) + b3*np.power(zeta,3) + b4*np.power(zeta,4))

    thetas = np.roots([1,-np.real(zeta[np.where(psi == np.amax(psi))[0][0]]),-1])
    loc = np.where(np.logical_and(thetas>-1,thetas<=1))
    theta = thetas[loc]

    c = 1/np.sqrt(1+theta**2); s = -theta/np.sqrt(1+theta**2)

    J1 = np.identity(Orig_shape)
    J1[p2,p2] = c; J1[p2,q2] = -s; J1[q2,p2] = np.conj(s); J1[q2,q2] = c
    return J1

# Own version
def COM2_own(Xmix,tol,U0,n_iter,print_f):
    P = Xmix.shape[0]
    C = cum4tensor3(Xmix)

    T = np.copy(C)
    U = U0
    for i in range(n_iter):
        for p in range(P-1):
            for q in range(p+1,P):
                J1 = opt_jacobi_rot4_A(T,p,q)

                for i in range(len(T.shape)):
                    T = mtf.my_Nmode_prod(T,J1,i)
                U = np.dot(J1,U)
                if print_f == True:
                    print(offdiag4(T))

    # Get source signals
    return np.dot(U,Xmix),offdiag4(T),U

# Own version, (p,q) selection based on current lowest kurtosis value
def COM2_own_tracemin(Xmix,tol,n_iter,print_f):
    P = Xmix.shape[0]
    C = cum4tensor(Xmix)

    T = np.copy(C)
    U = np.identity(P)
    for i in range(n_iter):
        Pstar = np.argmin(np.abs(trace4D(T)))
        for p in range(Pstar-1):
            for q in range(p+1,Pstar):
                J1 = opt_jacobi_rot4_A(T,p,q)

                for i in range(len(T.shape)):
                    T = mtf.my_Nmode_prod(T,J1,i)
                U = np.dot(J1,U)
                if print_f == True:
                    print(offdiag4(T))
    # Get source signals
    return np.dot(U,Xmix),offdiag4(T),U

## ------------Implicit CPD-----------------  S. Sherman and T. Kolda

# Final version, version 3, for simplicity reasons
def y_rfun(X,u_r):
    I = X.shape[1]; 
    z_r = np.dot(X.T,u_r)
    return 1/I*np.dot(X,z_r**3) - 3*1/(I**2)*np.dot(z_r.T,z_r)*np.dot(X,z_r)

def y_rfunWHITE(X,u_r):
    I = X.shape[1]; 
    return 1/I*X @ (X.T @ u_r)**3  - 3*u_r

# Explicit y_r computation
def y_r_expfun(C4t,u_r):
    u_r = u_r.reshape([3,1])
    T = mtf.my_Nmode_prod(C4t,u_r.T,0)
    T = mtf.my_Nmode_prod(T,u_r.T,1)
    T = mtf.my_Nmode_prod(T,u_r.T,2)
    return T.squeeze()


# intermediate computation of w_r
def w_rfun(y_r,u_r):
    return np.dot(y_r.T,u_r)


# Implicit CPD algorithm for optimization
def IMPCPDfunOPT(lamU,X,alpha,R):
    P = X.shape[0]
    lam = np.copy(lamU[0:R]); U = np.copy(lamU[R::]).reshape([P,R],order='F')

    U = _sym_decorrelation(U)

    # Compute Y
    Y = np.zeros([P,R])
    for r in range(R):
        Y[:,r] = y_rfun(X,U[:,r]).squeeze()

    # Compute w
    w = np.zeros(R)
    for r in range(R):
        w[r] = w_rfun(Y[:,r],U[:,r]).squeeze()
   
    # Additional steps for storage reduction and speed increase
    B = np.dot(U.T,U)
    C = B**3
    v = np.dot(np.multiply(B,C),lam)

    # Function value update
    f = alpha + np.dot(lam.T,v) -2*np.dot(w.T,lam)

    # Compute gradients
    glam = -2*(w-v)
    GU = -2*4*np.dot((Y-np.dot(U,np.dot(np.diag(lam),C))),np.diag(lam))

    # GU = _sym_decorrelation(GU)

    glamU = np.r_[glam,GU.reshape(P*R,order='F')]
    return [f, glamU]


# Implicit CPD algorithm for fixed point iteration, S. Sherman and T. Kolda
def IMPCPDfunFixedPoint(X,lam,U,alpha,R,IMPCPD_or_fastICA,WHITE=True):
    P,I = X.shape

    # Compute Y
    if WHITE == False:
        Y = np.zeros([P,R])
        for r in range(R):
            Y[:,r] = y_rfun(X,U[:,r]).squeeze()
    elif WHITE == True:
        I = X.shape[1]
        Y = 1/I*X @ (X.T @ U)**3 - 3*U 

    # Compute w
    w = np.sum(Y*U,axis=0)
   
    # Additional steps for storage reduction and speed increase

    if WHITE == False:
        B = np.dot(U.T,U)
        C = B**3
    elif WHITE == True:
        B = np.eye(U.shape[1])
        C = B

    v = B*C @ lam

    # Function value update
    f = alpha + np.dot(lam.T,v) -2*np.dot(w.T,lam)
 

    # Compute gradients
    if IMPCPD_or_fastICA == 'IMPCPD':
        if WHITE == False:
            glam = (w-v)     # gradient implicit CPD
            GU = (Y- U @ np.diag(lam) @ C) @ np.diag(lam)

        elif WHITE == True:
            glam = (w-v) 
            GU = (Y- U @ np.diag(lam) @ C) @ np.diag(lam)

        return f, glam, GU
    elif IMPCPD_or_fastICA == 'fastICA':
        glam = (w)    # gradient fastICA
        GU = (Y) @ np.diag(lam)       # When changing the gradient to that of fastICA

        if WHITE == False:
            glam = (w)    # gradient fastICA
            GU = (Y) @ np.diag(lam)       # When changing the gradient to that of fastICA
        elif WHITE == True:
            glam = (w) 
            GU = (1/I*X @ (X.T @ U)**3 - 3*U)
        return f, glam, GU

# Clear NaN and inf instances
def clearNaNinf(X):
    X[np.argwhere(np.isnan(X))]=0
    X[np.argwhere(np.isinf(X))]=0
    return X


# symmetric orthogonalization from fastICA package
def _sym_decorrelation(W):
    """Symmetric decorrelation
    i.e. W <- (W * W.T) ^{-1/2} * W
    """
    s, u = linalg.eigh(np.dot(W, W.T))
    # s, u = linalg.eig(np.dot(W, W.T))
    # s = np.real(s); u = np.real(u)
    # u (resp. s) contains the eigenvectors (resp. square roots of
    # the eigenvalues) of W * W.T
    return np.linalg.multi_dot([u * (1.0 / np.sqrt(s)), u.T, W])


# full implicit CPD fixed-point symmetric solution 
def symIMPCPDff(X,lamff,Uff,alphaff,Rff,n_iter,IMPCPD_or_fastICA,symorth,f_stopcrit,WHITE=True):
    f = np.zeros(n_iter)
    
    ## Separate gradients
    for i in range(n_iter):
        # gradient from implicit CPD
        f[i],glamff,GUff = IMPCPDfunFixedPoint(X,lamff,Uff,alphaff,Rff,IMPCPD_or_fastICA,WHITE)

        if i >= 1 and f_stopcrit == True and f[i]>f[i-1]:
            break

        Uff = GUff/np.sqrt(mtf.my_inner_prod(GUff,GUff))   
        if WHITE == False:
            lamff = glamff/np.sqrt(mtf.my_inner_prod(glamff,glamff))
        elif WHITE == True:
            lamff = glamff/np.sqrt(mtf.my_inner_prod(glamff,glamff))
        
        if symorth == True:
            # Symmetric orthogonalization of Uff
            Uff = _sym_decorrelation(Uff)
    return np.real(f[0:i+1]),np.real(lamff),np.real(Uff)

# full implicit CPD fixed-point symmetric solution for whitened data
def symIMPCPDffWHITEFAST(X,lamff0,Uff0,Rff,n_iter,tol):
    P,I = X.shape
    lamff = np.copy(lamff0)
    Uff = np.copy(Uff0)
    err_store = np.zeros(n_iter)
    ## Separate gradients
    for i in range(n_iter):
        # Compute gradients
        M4 = 1/I*X@(X.T @ Uff)**3
        glamff =  np.diag(Uff.T @ M4) - (3*np.ones(P)+lamff)
        
        GUff = (M4 - Uff @ (3*np.eye(P) + np.diag(lamff)) ) @ np.diag(lamff)
        # lamff = glamff/np.sqrt(mtf.my_inner_prod(glamff,glamff))
        lamff = np.ones(P)
        Uff2 = np.diag(glamff)@GUff/np.sqrt(mtf.my_inner_prod(np.diag(glamff)@GUff,np.diag(glamff)@GUff))   
    
        Uffnew = _sym_decorrelation(np.copy(Uff2))

        if np.isnan(Uffnew).any():
            break

        # fastICA error convergence
        err = max(abs(abs(np.diag(np.dot(Uffnew, Uff.T))) - 1))
        err_store[i] = err
        # err_list[i] = err

        if err < tol:
            break
        Uff = Uffnew

    return np.real(lamff),np.real(Uff),i+1,err_store






## QR-T method
# QRT on full cumulant tensor
def QRT_fulltensor(T0,R,n_repeat):
    T = np.copy(T0)
    P = T.shape[0]
    Q = np.eye(P)
    R2 = np.copy(R)

    if R == P:
        R2 += -1
    offdiag = np.zeros([R2,n_repeat])
    diag = np.zeros([R2,n_repeat])
    for p in range(R2):
        for i in range(n_repeat):
            Q2 = sc.linalg.qr(T[p:P,p:P,p,p])[0].T
            # Q2 = sc.linalg.qr(T[p:P,p:P,p,p][:,0].reshape([P-p,1]))[0].T

            Qeye = np.eye(P)
            Qeye[p:P,p:P] = Q2
            Q = Qeye @ Q 

            T = mtf.my_Nmode_prod(T,Qeye,0)
            T = mtf.my_Nmode_prod(T,Qeye,1)
            T = mtf.my_Nmode_prod(T,Qeye,2)
            T = mtf.my_Nmode_prod(T,Qeye,3)

            T[abs(T) < 10**(-12)] = 0
            
            offdiag[p,i] = offdiag4(T)
            diag[p,i] = diag4(T)
    return Q,T,offdiag,diag


# Compute unique values of cumulant tensor and selector polynomial

# Function for computing only the unique values of the cumulant tensor
def c4mat_uniq(Z):
    P,I = Z.shape
    MC4unique_kr = sc.linalg.khatri_rao(Z[0,:].reshape([1,I]),sc.linalg.khatri_rao(Z[0,:].reshape([1,I]),Z[0:P,:]))
    for i in range(1,P-1):
        MC4unique_kr = np.r_[MC4unique_kr,sc.linalg.khatri_rao(Z[0:i+1,:],sc.linalg.khatri_rao(Z[i].reshape([1,I]),Z[i:P,:]))]
    MC4unique_kr = np.r_[MC4unique_kr,sc.linalg.khatri_rao(Z[0:P,:],sc.linalg.khatri_rao(Z[P-1].reshape([1,I]),Z[P-1].reshape([1,I])))]
    MC4unique = 1/I*Z @ MC4unique_kr.T
    return MC4unique

# Function for producing the relevant indices for the corresponding unique values
def idxc4mat_uniq(P):
    Puniq = 0
    for i in range(1,P+1):
        Puniq += i*(P+1-i)
    idx = []
    for i in range(P):
        for j in range(i,P):
            idx.append([i*P**2 + j*P + j,i*P**2 + (j+1)*P])
    return idx, Puniq

# Function for producing the uniqe Meye values   -> still have to take in account a multiplicity (?)
def Meye_uniqf(P,D,Meye_or_D):
    idx_uniq,_ = idxc4mat_uniq(P)
    if Meye_or_D == True:
        Meye = mtf.my_modeN_mat(np.multiply.outer(np.eye(P,P),np.eye(P,P)).squeeze() + np.multiply.outer(np.eye(P,P),np.eye(P,P)).squeeze().transpose([0,2,1,3])
                                                                                  + np.multiply.outer(np.eye(P,P),np.eye(P,P)).squeeze().transpose([0,3,2,1]),0)
        Meye_uniq = Meye[:,idx_uniq[0][0]:idx_uniq[0][1]]
        for _,val in enumerate(idx_uniq[1::]):
            Meye_uniq = np.c_[Meye_uniq,Meye[:,val[0]:val[1]]]
        return Meye_uniq
    else:
        D_uniq = D[:,idx_uniq[0][0]:idx_uniq[0][1]]
        for _,val in enumerate(idx_uniq[1::]):
            D_uniq = np.c_[D_uniq,D[:,val[0]:val[1]]]
        return D_uniq



## ---------------------------------------
# QR methods

# Create alternating QR tensor function with iplicit use of the cumulant tensor, straight from data
def QRT_Final(Z0,R,Q0,n_repeat=10,tol=10**(-6),V_error='fastICA',Method='parallel'):
    Z = Q0 @ np.copy(Z0)
    P = Z.shape[0]
    I = Z.shape[1]
    Q = Q0
    err_list = np.zeros(n_repeat)
    Qold = np.eye(P)

    # Deflation algorithm
    if Method == 'deflationary':
        err_list = np.zeros([R,n_repeat])
        Meye = np.eye(P)*3
        if R == P:
            R += -1

        for p in range(R):
            for i in range(n_repeat): 
                MC4_sub = 1/I*Z[p::,:] @ (Z[p,:]*Z[p,:]*Z[p,:]).T
                MT = MC4_sub - Meye[p::,p]

                Q2 = sc.linalg.qr(MT.reshape([P-p,1]))[0].T
                Q2old = np.abs(np.copy(Q))
                Qeye = np.eye(P); Qeye[p::,p::] = Q2

                Q = Qeye @ Q 
                err = np.abs(np.abs((Q[p::,p] * Q2old[p::,p]).sum()) - 1)
                err_list[p,i] = err
                
                Z = Qeye @ Z

    # Parallel algorithm
    else:
        Meye = np.eye(P)*3
        if R == P:
            R += -1

        for i in range(n_repeat): 
            MC4_sub = 1/I*Z @ (Z[0:R,:]*Z[0:R,:]*Z[0:R,:]).T
            MT = MC4_sub - Meye[:,0:R]
            Q2 = sc.linalg.qr(MT)[0].T
            Q = Q2 @ Q 

            # Compute error for tolerance
            if V_error == 'abs_identity':
                # V1, absolute error compared to identity matrix
                E = np.eye(P) - np.abs(Q2)
                err = np.sqrt(np.dot(E.ravel(),E.ravel()))
                err_list[i] = err
        
            elif V_error == 'abs_difference':
                # V2, absolute error of difference between old and new mixing matrix estimate
                E = np.abs(Q) - Qold
                err = np.sqrt(np.dot(E.ravel(),E.ravel()))
                Qold = np.abs(np.copy(Q))
                err_list[i] = err
    
            elif V_error == 'fastICA':   
                # fastICA error convergence translated into QRT
                err = max(abs(abs(np.diag(np.dot(Q, Qold.T))) - 1))
                Qold = np.abs(np.copy(Q))
                err_list[i] = err

            if err < tol:
                break
            
            Z = Q2 @ Z
    return Z, Q, err_list

# Create alternating QR tensor function with iplicit use of the cumulant tensor, straight from data
def QRT4iterIMPLICIT(Z,R,n_repeat,tol):
    P = Z.shape[0]
    Q = np.eye(R)
    I = Z.shape[1]
    Qold = np.eye(P)

    for p in range(P-1):
        for _ in range(n_repeat):
            MC4_sub = 1/I*Z[p:P,:] @ sc.linalg.khatri_rao(Z[p,:].reshape([1,I])**2,Z[p:P,:]).T

            # # Version 1
            Meye = np.eye(R-p)
            Meye[0,0] = 3
            MT = MC4_sub - Meye

            ## Version with Q multiplied with the Meye part, not needed
            # Qkron = sc.linalg.kron(Qeye,sc.linalg.kron(Qeye,Qeye)).T
            # Meye = (Qeye @ Meye @ Qkron)
            # MT = MC4_sub - Meye[p:P,p*P**2+p*P+p:p*P**2+p*P+P]

            Q2 = sc.linalg.qr(MT)[0].T
            # Q2 = sc.linalg.eig(MT)[1].T
            Qeye = np.eye(R)
            Qeye[p:R,p:R] = Q2 @ Qeye[p:R,p:R]

            Q = Qeye @ Q 
            Z = Qeye @ Z

            # fastICA error convergence translated into QRT
            err = max(abs(abs(np.diag(np.dot(Q, Qold.T))) - 1))
            Qold = np.copy(Q)

            if err < tol:
                break
    return Z,Q

def QRT_FinalFASTWHITE(Z0,R,Q0,n_repeat=10,tol=10**(-6)):
    Z = Q0 @ np.copy(Z0)
    P = Z.shape[0]
    I = Z.shape[1]
    Q = Q0
    err_list = np.zeros(n_repeat)
    Qold = np.eye(P)

    # Parallel algorithm
    Meye = np.eye(P)*3
    if R == P:
        R += -1

    for i in range(n_repeat): 
        MT = 1/I*Z @ (Z[0:R,:]*Z[0:R,:]*Z[0:R,:]).T - Meye[:,0:R]
        Q2 = sc.linalg.qr(MT)[0].T
        Q = Q2 @ Q 

        # fastICA error convergence translated into QRT
        err = max(abs(abs(np.diag(np.dot(Q, Qold.T))) - 1))
        Qold = np.copy(Q)
        err_list[i] = err

        Z = Q2 @ Z

        if err < tol:
            break
        
    return Z, Q, err_list, i+1

def QRT_FinalFAST(Z0,R,Q0,n_repeat=10,tol=10**(-6)):
    Z = Q0 @ np.copy(Z0)
    P = Z.shape[0]
    I = Z.shape[1]
    Q = Q0
    err_list = np.zeros(n_repeat)
    Qold = np.eye(P)

    # Parallel algorithm
    if R == P:
        R += -1

    for i in range(n_repeat): 
        MC = 1/I*Z @ Z[0:R].T
        MT = 1/I*Z @ (Z[0:R,:]*Z[0:R,:]*Z[0:R,:]).T - 3*MC * MC 
        Q2 = sc.linalg.qr(MT)[0].T
        Q = Q2 @ Q 

        # fastICA error convergence translated into QRT
        err = max(abs(abs(np.diag(np.dot(Q, Qold.T))) - 1))
        Qold = np.copy(Q)
        err_list[i] = err

        Z = Q2 @ Z

        if err < tol:
            break
        
    return Z, Q, err_list, i+1


## Kims QRST algorithm
def QRST4(T0,tol,kmax):
    N = len(T0.shape)
    P = T0.shape[0]
    EYE = np.eye(P)

    for i in range(P):
        # print('i:',i)
        T = np.copy(T0)
        Tprime = np.copy(T)
        for n in range(0,N-1):
            Tprime = mtf.my_Nmode_prod(Tprime,EYE[i,:].reshape([1,P],order='F'),n)
            if n == 1:
                s = -np.real(np.min(sc.linalg.eig(Tprime.squeeze())[0]))

        E = Tprime.squeeze() - EYE[i,:]
        eps= np.sqrt(mtf.my_inner_prod(E,E))/np.sqrt(mtf.my_inner_prod(T[:,:,i,i],T[:,:,i,i]))

        Qstore = np.eye(P)
        Q = {}
        k = 0
        while (eps > tol):
            if k >= kmax:
                break
            Q[k] = sc.linalg.qr(T[:,:,i,i].squeeze()+s*np.eye(P))[0]
            
            for n in range(0,N):
                T = mtf.my_Nmode_prod(T,Q[k],n)
            Qstore = Qstore @ Q[k]
            
            Tprime = np.copy(T)
            for n in range(0,N-1):
                Tprime = mtf.my_Nmode_prod(Tprime,EYE[i,:].reshape([1,P],order='F'),n)
                if n == 1:
                    s = -np.min(sc.linalg.eig(Tprime.squeeze())[0])
            E = Tprime.squeeze() - EYE[i,:]
            eps= np.sqrt(mtf.my_inner_prod(E,E))/np.sqrt(mtf.my_inner_prod(T[:,:,i,i],T[:,:,i,i]))
            k += 1
        

    lam = np.asarray([T[i,i,i,i] for i in range(P)])
    return lam, Qstore, T



# CPD-GEVD from Matlab
def CPD_gevd(T0,R):
    T = np.copy(T0)
    Tshape = list(T.shape)
    N = len(Tshape)

    # Directr trilinear decomposition
    ## original HOSVD for CPD-GEVD
    G,U,_,mlsv = mtf.my_Tucker1(T,0)

    # tolerance for truncation (not needed)
    eps = sys.float_info.epsilon
    tol = np.max((Tshape,np.divide(np.prod(Tshape),Tshape)),axis=0)*eps
    size_core = [np.sum(mlsv[n] > tol[n]*np.max(mlsv[n])) for n in range(N)]
    size_core.sort(reverse=True)

    idxs = sorted(range(len(size_core)), key=lambda k: size_core[k], reverse=True)

    T = T.transpose(idxs)
    Tshape = list(T.shape)
    G = G.transpose(idxs)
    Unew = {}; iperm = np.zeros(len(idxs))
    for idx, val in enumerate(idxs):
        Unew[idx] = U[val]
        iperm[val] = idx
    
    # Test to see if the method can be applied. Take only the first 2 slices
    size_core[0:2] = np.min((size_core[0:2],np.tile(R,(2))),axis=0)
    size_core[2] = np.min((size_core[2],2))
    for n in range(3,N):
        size_core[n] = 1

    idxs_dict = {}
    for i,val in enumerate(size_core):
        idxs_dict[i] = [*range(val)]

    G = G[idxs_dict[0][0]:idxs_dict[0][-1]+1,idxs_dict[1][0]:idxs_dict[1][-1]+1,idxs_dict[2][0]:idxs_dict[2][-1]+1][(..., *[0]*(N-3))].squeeze()
    Unew[0] = Unew[0][:,0:size_core[0]]
    

    # Compute first factor matrix using generalized eigenvectors with sc.linalg.eig
    Deig,Veig = sc.linalg.eig(G[:,:,0].squeeze(), G[:,:,1].squeeze())

    # Change the complex conjugate vectors into the only complex part of the conjugate vector by making the
    idx_imag = [i for i, x in enumerate(np.isreal(Deig)) if not x]
    for i in range(int(len(idx_imag)/2)):
        Veig[:,idx_imag[i]] = np.real(Veig[:,idx_imag[i]])
        Veig[:,idx_imag[i+1]] = np.imag(Veig[:,idx_imag[i+1]])
    
        # Compute remaining factor matrices
    Uout = {}
    for n in range(N):
        Uout[n] = np.zeros([Tshape[n],R])
    
    # Old version
    T1 = T.reshape([Tshape[0],np.prod(Tshape[1:])],order='F')
    X = T1.T @ np.conj(Unew[0]) @ Veig
    for r in range(R):
        _,u,_,_ = mtf.my_Tucker1(np.real(X[:,r]).reshape(Tshape[1:],order='F'),0)
        for n in range(1,N):
            Uout[n][:,r] = u[n-1][:,0]    
    KR = sc.linalg.khatri_rao(Uout[2],Uout[1])
    for i in range(3,N):
        KR = sc.linalg.khatri_rao(Uout[i],KR)

    Uout[0] = np.real(np.dot(T1,np.linalg.pinv(KR.T)))                

    # Normalize factors of rank-one components
    nrm = np.zeros([len(Uout),R])
    for i,(key,Uval) in enumerate(Uout.items()):
        nrm[key,:] = np.sqrt(np.sum(np.abs(Uval)**2,axis=0))
        Uout[key] = Uval/nrm[key,:]

    nrm2 = np.prod(nrm,axis=0) #**(1/N)
    # for i,(key,Uval) in enumerate(Uout.items()):
    #     Uout[key] = Uval*nrm2

    # # Inverse permute the factor matrices to original mode sequence
    for idx, val in enumerate(iperm):
        Uout[idx] = Uout[val]
    return Uout,nrm2

# Try-out iterative HOSVD 2    -      mehod 2 is fastest
def HOSVD_iterFINAL(Zorig,Core_only=False,method=2,reorth_step=0):
    Z = np.copy(Zorig)
    P,I = Z.shape

    if Core_only == False:
        # alternative using svd update and using unique slices with right multiplicities of the cumulant tensor
        for p1 in range(P):
            for p2 in range(p1,P):
                MT = 1/I*Z @ (Z[p1,:].reshape([1,I])*Z[p2,:].reshape([1,I]) * Z).T - np.eye(P)*np.eye(P)[p1,p2]
                MT[p1,p2] += -1; MT[p2,p1] += -1

                if (p1 == 0) & (p2 == 0):
                    Utest,Stest,Vhtest = sc.linalg.svd(MT,full_matrices=False,)
                    svdobj = su.SVD_updater(Utest,Stest,Vhtest,update_V=False,reorth_step=method)
                else:
                    if p1 == p2:
                        [svdobj.add_column(MT[:,m],method=method) for m in range(0,P)]

                    else:
                        [svdobj.add_column(MT[:,int(m - np.floor(m/(P))*P)],method=method) for m in range(0,2*P)]

    elif Core_only == 'HALF':
        # alternative using svd update and using unique slices only once of the cumulant tensor
        for p1 in range(P):
            for p2 in range(p1,P):
                MT = 1/I*Z @ (Z[p1,:].reshape([1,I])*Z[p2,:].reshape([1,I]) * Z).T - np.eye(P)*np.eye(P)[p1,p2]
                MT[p1,p2] += -1; MT[p2,p1] += -1

                if (p1 == 0) & (p2 == 0):
                    Utest,Stest,Vhtest = sc.linalg.svd(MT,full_matrices=False,)
                    svdobj = su.SVD_updater(Utest,Stest,Vhtest,update_V=False,reorth_step=method)
                else:
                    [svdobj.add_column(MT[:,m],method=method) for m in range(0,P)]

    elif Core_only == True:
        # alternative using svd update and using core slices of the cumulant tensor 
        Meye = np.eye(P)*3
        for p1 in range(P):
            MT = 1/I*Z @ (Z[p1,:].reshape([1,I])**2 * Z).T - Meye

            if p1 == 0:
                Utest,Stest,Vhtest = sc.linalg.svd(MT,full_matrices=False,)
                svdobj = su.SVD_updater(Utest,Stest,Vhtest,update_V=False,reorth_step=method)
            else:
                [svdobj.add_column(MT[:,m],method=method) for m in range(0,P)]

    return svdobj.get_current_svd()[0:2]


def HOSVD_iterFINAL_CPDGEVD(Zorig,R,Core_only=False,method=2,reorth_step=0):
    Z = np.copy(Zorig)
    P,I = Z.shape
    MTstore = {}
    
    if Core_only == False:
    # alternative using svd update and using selective slices of the cumulant tensor
        for p1 in range(P):
            for p2 in range(p1,P):
                MT = 1/I*Z @ (Z[p1,:].reshape([1,I])*Z[p2,:].reshape([1,I]) * Z).T - np.eye(P)*np.eye(P)[p1,p2]
                MT[p1,p2] += -1; MT[p2,p1] += -1

                if (p1 == 0) & (p2 == 0):
                    Utest,Stest,Vhtest = sc.linalg.svd(MT,full_matrices=False,)
                    svdobj = su.SVD_updater(Utest,Stest,Vhtest,update_V=False,reorth_step=method)
                else:
                    if p1 == p2:
                        [svdobj.add_column(MT[:,m],method=method) for m in range(0,P)]

                    else:
                        [svdobj.add_column(MT[:,int(m - np.floor(m/(P))*P)],method=method) for m in range(0,2*P)]

                if (p1 <= R) and (p2 <= R) and (p1 == p2):
                    MTstore[p1] = MT

    elif Core_only == 'HALF':
        # alternative using svd update and using unique slices only once of the cumulant tensor
        for p1 in range(P):
            for p2 in range(p1,P):
                MT = 1/I*Z @ (Z[p1,:].reshape([1,I])*Z[p2,:].reshape([1,I]) * Z).T - np.eye(P)*np.eye(P)[p1,p2]
                MT[p1,p2] += -1; MT[p2,p1] += -1

                if (p1 == 0) & (p2 == 0):
                    Utest,Stest,Vhtest = sc.linalg.svd(MT,full_matrices=False,)
                    svdobj = su.SVD_updater(Utest,Stest,Vhtest,update_V=False,reorth_step=method)
                else:
                    [svdobj.add_column(MT[:,m],method=method) for m in range(0,P)]

                if (p1 <= R) and (p2 <= R) and (p1 == p2):
                    MTstore[p1] = MT

    elif Core_only == True:
        # alternative using svd update and using core slices of the cumulant tensor 
        Meye = np.eye(P)*3
        for p1 in range(P):
            MT = 1/I*Z @ (Z[p1,:].reshape([1,I])**2 * Z).T - Meye

            if p1 == 0:
                Utest,Stest,Vhtest = sc.linalg.svd(MT,full_matrices=False,)
                svdobj = su.SVD_updater(Utest,Stest,Vhtest,update_V=False,reorth_step=method)
            else:
                [svdobj.add_column(MT[:,m],method=method) for m in range(0,P)]
            MTstore[p1] = MT

    return svdobj.get_current_svd()[0],MTstore


def CPD_gevdIMPLICIT_FINAL(Z,R,version,SVDCore_only=False,method=2):
    N = 4
    P = Z.shape[0]
    I = Z.shape[1]
    
    U,MTstore = HOSVD_iterFINAL_CPDGEVD(Z,R,Core_only=SVDCore_only,method=method)
  
    G1 = U.T @ MTstore[0]
    G2 = U.T @ MTstore[1]

    Deig,Veig = sc.linalg.eig(G1, G2)

    # Change the complex conjugate vectors into the only complex part of the conjugate vector by making the
    idx_imag = [i for i, x in enumerate(np.isreal(Deig)) if not x]
    for i in range(int(len(idx_imag)/2)):
        Veig[:,idx_imag[i]] = np.real(Veig[:,idx_imag[i]])
        Veig[:,idx_imag[i+1]] = np.imag(Veig[:,idx_imag[i+1]])
    
    # # Compute remaining factor matrices
    Uout = {}
    for n in range(N):
        Uout[n] = np.zeros([P,R])

    if version == 1:
        MTT = np.zeros([P,P]); KR = np.zeros([P,R])
        for r in range(R):
            # u = sc.linalg.svd(MTstore[r] @ U.T @ np.real(Veig))[0]
            # u = sc.linalg.svd((Veig.T @ U @MT[r]))[2].T
            u = sc.linalg.svd(MTstore[r] @ U @ sc.linalg.inv(Veig.T))[0]
            Uout[1][:,r] = u[:,0] 
            KR += Uout[1][r,:]**2 * Uout[1]
            MTT += MTstore[r]
 
        # MTT = U @ (G1 + G2)
        Uout[0] = np.real(MTT @ KR)
        Uout[2] = Uout[1]; Uout[3] = Uout[1]

    else:
        #---OLD---
        X = np.zeros([P,1]); KR = np.zeros([1,R])
        for r in range(R):
            X = np.c_[X,MTstore[r]]
            # u = sc.linalg.svd(MTstore[r] @ U.T @ Veig)[0]
            # u = sc.linalg.svd((Veig.T @ U @MT[r]))[2].T
            u = sc.linalg.svd(MTstore[r] @ U @ sc.linalg.inv(Veig.T))[0]
            Uout[1][:,r] = np.real(u[:,0]) 
        for r in range(R):
            KR = np.r_[KR,Uout[1][r,:]**2 * Uout[1]]

        X = X[:,1::]; KR = KR[1::,:]
        Uout[2] = np.real(Uout[1]); Uout[3] = np.real(Uout[1])
        Uout[0] = np.real(np.dot(X,np.linalg.pinv(KR.T)))
        #---OLD---


    # Normalize factors of rank-one components
    nrm = np.zeros([len(Uout),R])
    for i,(key,Uval) in enumerate(Uout.items()):
        nrm[key,:] = np.sqrt(np.sum(np.abs(Uval)**2,axis=0))
        Uout[key] = Uval/nrm[key,:]

    nrm2 = np.prod(nrm,axis=0) #**(1/N)
    # for i,(key,Uval) in enumerate(Uout.items()):
    #     Uout[key] = Uval*nrm2
    return Uout,nrm2

def CPD_gevdIMPLICIT_FINAL_ORTHO(Z,R,version):
    N = 4
    P = Z.shape[0]
    I = Z.shape[1]
    
    U,MTstore = HOSVD_iterFINAL_CPDGEVD(Z,R)
  
    G1 = U.T @ MTstore[0]
    G2 = U.T @ MTstore[1]

    Deig,Veig = sc.linalg.eig(G1, G2)

    # Change the complex conjugate vectors into the only complex part of the conjugate vector by making the
    idx_imag = [i for i, x in enumerate(np.isreal(Deig)) if not x]
    for i in range(int(len(idx_imag)/2)):
        Veig[:,idx_imag[i]] = np.real(Veig[:,idx_imag[i]])
        Veig[:,idx_imag[i+1]] = np.imag(Veig[:,idx_imag[i+1]])
    
    # # Compute remaining factor matrices
    Uout = {}
    for n in range(N):
        Uout[n] = np.zeros([P,R])

    if version == 1:
        MTT = np.zeros([P,P]); KR = np.zeros([P,R])
        for r in range(R):
            # u = sc.linalg.svd(MTstore[r] @ U.T @ np.real(Veig))[0]
            # u = sc.linalg.svd((Veig.T @ U @MT[r]))[2].T
            u = sc.linalg.svd(MTstore[r] @ U @ sc.linalg.inv(Veig.T))[0]
            Uout[1][:,r] = u[:,0] 
            KR += Uout[1][r,:]**2 * Uout[1]
            MTT += MTstore[r]
 
        # MTT = U @ (G1 + G2)
        Uout[0] = np.real(MTT @ KR)
        Uout[2] = Uout[1]; Uout[3] = Uout[1]

    else:
        #---OLD---
        X = np.zeros([P,1]); KR = np.zeros([1,R])
        for r in range(R):
            X = np.c_[X,MTstore[r]]
            # u = sc.linalg.svd(MTstore[r] @ U.T @ Veig)[0]
            # u = sc.linalg.svd((Veig.T @ U @MT[r]))[2].T
            u = sc.linalg.svd(MTstore[r] @ U @ sc.linalg.inv(Veig.T))[0]
            Uout[1][:,r] = np.real(u[:,0]) 
        for r in range(R):
            KR = np.r_[KR,Uout[1][r,:]**2 * Uout[1]]

        X = X[:,1::]; KR = KR[1::,:]
        Uout[2] = np.real(Uout[1]); Uout[3] = np.real(Uout[1])
        Uout[0] = np.real(np.dot(X,np.linalg.pinv(KR.T)))
        Uout[0] = _sym_decorrelation(Uout[0])
        #---OLD---


    # Normalize factors of rank-one components
    nrm = np.zeros([len(Uout),R])
    for i,(key,Uval) in enumerate(Uout.items()):
        nrm[key,:] = np.sqrt(np.sum(np.abs(Uval)**2,axis=0))
        Uout[key] = Uval/nrm[key,:]

    nrm2 = np.prod(nrm,axis=0) #**(1/N)
    return Uout,nrm2


def CPDerror(T,lam,DU,symmetric):
    CP = {}
    CPf = np.zeros(T.shape)

    if symmetric == True:
        R = DU.shape[1]
        for r in range(R):
            L = [DU[:,r] for n in range(len(T.shape))]
            CP[r] = mtf.my_out_prod(L)
            CPf += lam[r]*CP[r]
    else:
        R = DU[0].shape[1]
        for r in range(R):
            L = [DU[n][:,r] for n in range(len(T.shape))]
            CP[r] = mtf.my_out_prod(L)
            CPf += lam[r]*CP[r]
    E = T - CPf
    return np.sqrt(mtf.my_inner_prod(E,E))/np.sqrt(mtf.my_inner_prod(T,T))


