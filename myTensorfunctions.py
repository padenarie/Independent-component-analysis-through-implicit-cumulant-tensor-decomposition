import numpy as np
from scipy import linalg
from itertools import chain
import scipy as sc
import sys
eps = sys.float_info.epsilon

############# Starterkit 1 #############
# Function to compute outer product of an arbitrary number of vectors, insert the vectors for the outer product as the elements of a single list L
def my_out_prod(L):
    T_out = np.copy(L[0])
    for i in range(1,len(L)):
        T_out = np.multiply.outer(T_out,L[i])
    return T_out

# Tensor inner product using ravel
def my_inner_prod(T1,T2):
    return np.dot(T1.ravel(),T2.ravel())

# Mode-n matricization (n starts at 0)
def my_modeN_mat(T,mode):
    L = list(range(len(T.shape))); L.remove(len(T.shape)-(mode+1))
    return T.T.transpose([len(T.shape)-(mode+1),*L]).reshape((T.shape[mode],-1),order='C')

# Mode-n matricization reversal (n starts at 0)
def my_modeN_mat_reverse(Tm,Tshape,mode):
    if mode == -1:
        mode = len(Tshape)
    L = list(range(len(Tshape))); L.remove(mode)
    return Tm.T.reshape((np.asarray(Tshape)[[*L[::-1],mode]]),order='C').T.transpose((np.insert(list(range(1,len(Tshape))),mode,0)))

# Dot product of Tensor N-mode with second dimension of Matrix M (rows), mode starts at 0
def my_Nmode_prod(T,M,mode):
    Tshape = np.asarray(T.shape)
    Tshape[mode] = M.shape[0]

    L1 = list(range(len(T.shape))); L1.remove(len(T.shape)-(mode+1))
    L2 = list(range(len(Tshape))); L2.remove(mode)
    return np.dot(T.T.transpose([len(T.shape)-(mode+1),*L1]).reshape((T.shape[mode],-1),order='C').T,M.T).reshape((np.asarray(Tshape)[[*L2[::-1],mode]]),order='C').T.transpose((np.insert(list(range(1,len(Tshape))),mode,0)))

# Return the k'th folding matrix of a tensor T where rk is the rank of Tk (k starts at 0)
def k_unfold_Tensor(T,k):
    prod1 = np.prod(np.asarray(T.shape)[0:k+1],axis=0)
    prod2 = np.prod(np.asarray(T.shape)[k+1:],axis=0)
    return np.reshape(T.T,([prod1,prod2])[::-1],'C').T

# Return the k'th folding matrix of a tensor T with a custom new rank rk = r at k and original rank nk at k (k starts at 1)
def nk_unfold_Tensor(T,nk,r):
    numel = np.prod(np.asarray(T.shape),axis=0)
    return np.reshape(T.T,([int(r*nk),int(numel/(r*nk))])[::-1],'C').T

# Unfold complete Tensor T and return list A containing the fold matrices
def unfold_Tensor(T):
    A = {}
    for k in range(1,len(T.shape)+1):
        A[k-1] = k_unfold_Tensor(T,k)
    return A

# Unfold the Tensor T and return an array containing the ranks of the folding matrices
def unfold_rank(T):
    A = unfold_Tensor(T)
    rank = np.empty(len(A),dtype=int)
    for i in range(len(A)):
        rank[i] = np.linalg.matrix_rank(np.asarray(A[i]))
    return rank

# Delta-rank function
def delta_rank(M,delta):
    S = np.linalg.svd(M,full_matrices=False)[1]
    i = 0
    while np.linalg.norm(S[-1:-2-i:-1]) <= delta:
        i = i + 1
    return len(S)-i

# TT-SVD algorithm
def my_TT_SVD(T,e):
    # Init
    d = len(T.shape)
    delta = e/np.sqrt(d-1)*np.sqrt(my_inner_prod(T,T))

    # Make copy of Tensor
    C = T
    r = {}; r[0] = 1; r[d] = 1
    G = {}
    n = np.asarray(T.shape)

    # For loop to create the tensor cores up to Gd
    for k in range(0,d-1):
        C = np.reshape(C.T,([int(r[k]*n[k]),int(np.prod(np.asarray(C.shape),axis=0)/(r[k]*n[k]))])[::-1],'C').T

        # U,S,Vh = np.linalg.svd(C,full_matrices=False)
        U,S,Vh = sc.linalg.svd(C,full_matrices=False,compute_uv=True)
        i = 0
        while np.sqrt(my_inner_prod(S[-1:-2-i:-1],S[-1:-2-i:-1])) <= delta:
            i = i + 1
        r[k+1] = len(S)-i
        G[k] = np.reshape(U[:,0:r[k+1]].T,([r[k],n[k],r[k+1]])[::-1],'C').T
        C = np.dot(np.diag(S[0:r[k+1]]),Vh[0:r[k+1],:])
    G[d-1] = C[...,np.newaxis]
    return G, r

def rgb2gray(img_rgb):
    return np.dot(img_rgb[...,:3], [1/3, 1/3, 1/3])


def Tensor_recon(L):
    T = my_Tensordot(np.asarray(L[0]),np.asarray(L[1]),[-1,0])

    for i in range(2,len(L)):
        T = my_Tensordot(T,np.asarray(L[i]),[-1,0])
    return T.squeeze()

    
# Function to compute non-truncated TT-ranks of a tensor
def predict_TTranks(dims):
    r = {}; r[0] = 1; r[len(dims)] = 1; i_left = 0; i_right = 0

    for _ in range(len(dims)-1):
        if r[i_left]*dims[i_left] >= r[len(dims)-i_right]*dims[len(dims)-1-i_right]:
            r[len(dims)-1-i_right] = r[len(dims)-i_right]*dims[len(dims)-1-i_right]
            i_right += 1
        elif r[i_left]*dims[i_left] < r[len(dims)-i_right]*dims[len(dims)-1-i_right]:
            r[i_left+1] = r[i_left]*dims[i_left]
            i_left += 1
    r = [r[i] for i in range(len(r))]
    return r

def my_Tucker1(T,e):
    # Init
    d = len(T.shape)
    delta = e/np.sqrt(d)*np.sqrt(my_inner_prod(T,T))

    U = {}
    G = np.copy(T)
    r = {} 
    S_store = {}

    for k in range(d):
        # U[k],S = np.linalg.svd(my_modeN_mat(T,k), full_matrices=False)[0:2]
        U[k],S = sc.linalg.svd(my_modeN_mat(T,k),full_matrices=False,compute_uv=True)[0:2]
        i = 0
        while np.sqrt(my_inner_prod(S[-1:-2-i:-1],S[-1:-2-i:-1])) <= delta:
            i = i + 1
        r[k] = len(S)-i
        U[k] = U[k][:,0:r[k]]
        S_store[k] = S[0:r[k]]
        G = my_Nmode_prod(G,np.asarray(U[k]).T,k)
    return G,U,r,S_store

def my_Tucker1SYM(T,e):
    # Init
    d = len(T.shape)
    delta = e/np.sqrt(d)*np.sqrt(my_inner_prod(T,T))


    U,S = sc.linalg.svd(my_modeN_mat(T,0),full_matrices=False,compute_uv=True)[0:2]
    i = 0
    while np.sqrt(my_inner_prod(S[-1:-2-i:-1],S[-1:-2-i:-1])) <= delta:
        i = i + 1
    r = len(S)-i
    U = U[:,0:r]
    S_store = S[0:r]
    return U,r,S_store

def Tuck1_tensordot(G,U):
    T = np.copy(G)
    for i in range(len(np.asarray(T.shape))):
        T = my_Nmode_prod(T,np.asarray(U[i]),i)
    return T

# Approximation error, give list of cores for TT-SVD as input and give final core for Tucker 1 as input
def prox_error(GA,UA,GB,UB,method):
    if method == 'TTSVD':
        if type(GA)==dict:
            A2 = Tensor_recon(GA)
        else:
            A2 = GA
        if type(GB)==dict:
            B2 = Tensor_recon(GB)
        else:
            B2 = GB
        return np.sqrt(my_inner_prod(A2-B2,A2-B2))/np.sqrt(my_inner_prod(A2,A2))
    elif method == 'Tuck1':
        if type(UA)==dict:
            A = Tuck1_tensordot(GA,UA)
        else:
            A = GA
        if type(UB)==dict:
            B = Tuck1_tensordot(GB,UB)
        else:
            B = GB
        return np.sqrt(my_inner_prod(A-B,A-B))/np.sqrt(my_inner_prod(A,A))

############# Starterkit 2 #############

def my_innerprodTTAA(GA,GB):
    return my_inner_prod(list(GA.values())[-1],list(GB.values())[-1])

# Own version of Tensordot, modes start at 0, possible to input multipled modes, !number of modes must be even!
def my_Tensordot(A,B,modes):
    for i in range(len(modes)):
        if modes[i] < 0 and i < int(len(modes)/2):
            modes[i] = len(A.shape) + modes[i]
        elif modes[i] < 0 and i >= int(len(modes)/2):
            modes[i] = len(B.shape) + modes[i]

    A_order = list(range(len(A.shape))); [A_order.remove(elA) for elA in modes[:int(len(modes)/2)]]
    B_order = list(range(len(B.shape))); [B_order.remove(elB) for elB in modes[int(len(modes)/2):]]

    Anew = A.transpose([*A_order,*modes[:int(len(modes)/2)]])
    Bnew = B.transpose([*modes[int(len(modes)/2):],*B_order])

    if len(Anew.shape) == int(len(modes)/2):
        Anew2 = Anew.reshape([np.prod(Anew.shape[-int(len(modes)/2):])])
        Bnew2 = Bnew.reshape([np.prod(Bnew.shape[:int(len(modes)/2)])])
    else:
        Anew2 = Anew.reshape([np.prod(Anew.shape[:-int(len(modes)/2)]),np.prod(Anew.shape[-int(len(modes)/2):])])
        Bnew2 = Bnew.reshape([np.prod(Bnew.shape[:int(len(modes)/2)]),np.prod(Bnew.shape[int(len(modes)/2):])])

    T = np.dot(Anew2,Bnew2)

    return np.reshape(T,[*Anew.shape[0:-int(len(modes)/2)],*Bnew.shape[int(len(modes)/2):]])

# innerproduct of Tensors using numpy tensordot
def my_innerprodTTAB_numpy(GA,GB):
    InProd = np.tensordot(GA[0],GB[0],([1],[1]))
    InProd = np.tensordot(InProd,GA[1],(1,0))
    for i in range(1,len(GA)-1):
        InProd = np.tensordot(InProd,GB[i],([-3,-2],[0,1]))
        InProd = np.tensordot(InProd,GA[i+1],(2,0))
    return np.tensordot(InProd,GB[len(GB)-1],([-3,-2],[0,1])).squeeze()

# Own tensortrain inner product
def my_innerprodTTAB(GA,GB):
    InProd = my_Tensordot(GA[0],GB[0],[1,1])

    for i in range(1,len(GA)):
        # Add last tensor rank border dimension if missing
        if len(GA[i].shape) == 2:
            GA[i] =  GA[i][..., np.newaxis]
        if len(GB[i].shape) == 2:
            GB[i] =  GB[i][..., np.newaxis]

        if i == 1:
            InProd = my_Tensordot(InProd,GA[i],[1,0])
        else:
            InProd = my_Tensordot(InProd,GA[i],[-2,0])
        InProd = my_Tensordot(InProd,GB[i],[-3,-2,0,1])
    return InProd.squeeze()

# Innerproduct of TT train from literature using Kronecker product and columnwise Kronecker product (Khatri Rao product)
def my_innerprodTTABLit(GA,GB):
    d = len(GA)
    v = np.sum(linalg.khatri_rao(GA[0][0,:,:].T,GB[0][0,:,:].T),axis=1)

    for k in range(1,d-1):
          v = np.dot(v,(np.sum([np.kron(GA[k][:,i],GB[k][:,i]) for i in range(GA[k].shape[1])],axis=0)))
    return np.dot(v,(np.sum([np.kron(GA[d-1][:,i],GB[d-1][:,i]) for i in range(GA[d-1].shape[1])],axis=0)))


# site-k-mixed canonical form, l starts at 1
def site_k_mixed_canon(GA,l):
    if l == len(GA):
        return GA, 0,0
    else:
        T = my_Tensordot(np.asarray(GA[len(GA)-2]),np.asarray(GA[len(GA)-1]),[len(np.asarray(GA[0]).shape)-1,0])
        for i in range(3,len(GA)-l+2):
            T = my_Tensordot(GA[len(GA)-i],np.asarray(T),[-1,0])

        # site-k-canonical mixed form other side
        C = np.copu(T)
        n = [GA[i].shape[1] for i in range(len(GA))]
        r = [GA[i].shape[-1] for i in range(len(GA))]; r[len(GA)-1] = 1; r.insert(0,1); G = {}

        # Decompose current tensor C with SVD, store S, go further with Vh  
        C = np.reshape(C.T,([int(r[l-1]*n[l-1]),int(np.prod(np.asarray(C.shape),axis=0)/(r[l-1]*n[l-1]))])[::-1],'C').T
        U,S_core,Vh = np.linalg.svd(C,full_matrices=False)
        Gl = {i:GA[i] for i in range(l-1)}
        Gl[l-1] = np.reshape(U.T,([r[l-1],n[l-1],S_core.shape[0]])[::-1],'C').T
        C = Vh; r_right = {}; r_right[l] = S_core.shape[0]; r_right[0] = 1; n_right = n[l::][::-1]

        for k in range(1,len(GA)-l):
            C = np.reshape(C.T,[r_right[l]*np.prod(n_right[k::]),n_right[k-1]*r_right[k-1]][::-1],order='C').T
            R,Q = sc.linalg.rq(C)
            r_right[k] = Q.shape[0]
            G[len(GA)-k] = np.reshape(Q.T,([r_right[k],n_right[k-1],r_right[k-1]][::-1]),'C').T
            C = R
        
        if len(GA)-l == 1:
            k = 0
            
        G[l] = np.reshape(C.T,[len(S_core),n_right[-1],r_right[k]][::-1],'C').T
        return Gl,S_core,G



# Function to reconstruct site-k-mixed-canonical form
def site_k_recon(Gl,S_core,Gr):
    l = len(Gl); Tl = Gl[0]; Tr = Gr[len(Gl)+len(Gr)-1]
    for i in range(1,l):
        Tl = my_Tensordot(Tl,Gl[i],[-1,0])

    for i in range(l+1,len(Gl)+len(Gr)):
        Tr = my_Tensordot(Gr[len(Gl)+len(Gr)+l-1-i],Tr,[-1,0])
        
    Trec_left = my_Tensordot(Tl,np.diag(S_core),[-1,0]).squeeze()
    return my_Tensordot(Trec_left,Tr,[-1,0]).squeeze()

# Function to sum up 2 tensors in tensor train format
def addTT(GA,GB):
    GC = {}
    # First core
    GC[0] = np.concatenate((GA[0],GB[0]),axis=-1)

    # Middle cores
    rC = {}; rC[0] = 1; rC[len(GA)] = 1
    for i in range(1,len(GA)-1):
        rC[i] = GA[i].shape[0] + GB[i].shape[0]
        GC[i] = np.concatenate((np.concatenate((GA[i],np.zeros(GA[i].shape)),axis=-1),np.concatenate((np.zeros(GB[i].shape),GB[i]),axis=-1)),axis=0)

    # Last core
    if len(GA)==2:
        i = 0

    GC[i+1] = np.concatenate((GA[i+1],GB[i+1]),axis=0)
    rC[i+1] = GA[i+1].shape[0] + GB[i+1].shape[0]
    return GC, rC

# TT-rounding algorithm
def TT_rounding(GA,e):
    # Init
    d = len(GA); Gqr = {}
    delta = e/np.sqrt(d-1)*np.sqrt(my_inner_prod(GA[len(GA)-1],GA[len(GA)-1]))

    if len(GA[len(GA)-1].shape) == 2:
            GA[len(GA)-1] = GA[len(GA)-1][..., np.newaxis]

    # Right-to-left truncation
    Gqr[d-1] = GA[d-1]

    for k in range(d-1,0,-1):  
        R, Gqr[k]= sc.linalg.rq(np.reshape(Gqr[k].T,[Gqr[k].shape[0],np.prod(Gqr[k].shape[1:])][::-1],'C').T)
        Gqr[k] = np.reshape(Gqr[k].T,[Gqr[k].shape[0],GA[k].shape[1],int(Gqr[k].shape[-1]/GA[k].shape[1])][::-1],'C').T
        Gqr[k-1] = my_Nmode_prod(GA[k-1],R.T,2)
        
    # Compression for orthogonalized representation
    r = {}; r[0] = 1; r[d] = 1

    for k in range(0,d-1):
        U,S,Vh = sc.linalg.svd(np.reshape(Gqr[k].T,[Gqr[k].shape[0]*GA[k].shape[1],Gqr[k].shape[-1]][::-1],'C').T,full_matrices=False,compute_uv=True)

        i = 0
        while np.sqrt(my_inner_prod(S[-1:-2-i:-1],S[-1:-2-i:-1])) <= delta:
            i = i + 1
        r[k+1] = len(S)-i

        Gqr[k] = U[:,0:r[k+1]]; S = S[0:r[k+1]]; Vh = Vh[0:r[k+1],:]
        Gqr[k+1] = my_Nmode_prod(Gqr[k+1],np.dot(np.diag(S),Vh),0)
        Gqr[k] = np.reshape(Gqr[k].T,[r[k],GA[k].shape[1],r[k+1]][::-1],'C').T
    return Gqr,r


# TTm-SVD algorithm Must give an equal amount of dimensions for dims argument
def my_TTm_SVD(M,dims,e):
    if np.prod(dims) != np.prod(M.shape):
        print('ERROR Chosen tensor dimensions do not match up with number of elements in matrix M:',np.prod(dims),'!=',np.prod(M.shape))
    T = np.reshape(M.T,[*dims][::-1],'C').T
    T = np.transpose(T,list(chain.from_iterable((i,int(len(T.shape)/2+i)) for i in range(int(len(T.shape)/2)))))
    # TT-SVD algorithm
    d = int(len(T.shape)/2); delta = e/np.sqrt(d-1)*np.sqrt(my_inner_prod(M,M))
    # Make copy of Tensor
    C = T; r = {}; r[0] = 1; r[d] = 1; G = {}; ni = np.asarray(T.shape)[::2]; nj = np.asarray(T.shape)[1::2]
    # For loop to create the tensor cores up to Gd
    for k in range(1,d):
        C = np.reshape(C.T,([int(r[k-1]*ni[k-1]*nj[k-1]),int(np.prod(np.asarray(C.shape),axis=0)/(r[k-1]*ni[k-1]*nj[k-1]))])[::-1],'C').T
        U,S,Vh = np.linalg.svd(C,full_matrices=False)
        i = 0
        while np.sqrt(my_inner_prod(S[-1:-2-i:-1],S[-1:-2-i:-1])) <= delta:
            i = i + 1
        r[k] = len(S)-i
        G[k-1] = np.reshape(U[:,0:r[k]].T,([r[k-1],ni[k-1],nj[k-1],r[k]])[::-1],'C').T
        C = np.dot(np.diag(S[0:r[k]]),Vh[0:r[k],:])
    G[d-1] = np.reshape(C.T,([r[d-1],ni[d-1],nj[d-1],r[d]])[::-1],'C').T
    return G, r

def TTm_recon(L,Mshape):
    T = my_Tensordot(np.asarray(L[0]),np.asarray(L[1]),[-1,0])

    for i in range(2,len(L)):
        T = my_Tensordot(T,np.asarray(L[i]),[-1,0]).squeeze()

    M = np.reshape(T.T.squeeze().transpose([*[i for i in range(0,len(T.squeeze().shape),2)],*[i for i in range(1,len(T.squeeze().shape),2)]]),Mshape[::-1],'C').T
    return M

#TTm-rounding
def TTm_rounding(GM,e):
    GM3d = {}; GM4d = {}
    for i in range(len(GM)):
        GM3d[i] = np.reshape(GM[i].T,[GM[i].shape[0],GM[i].shape[1]*GM[i].shape[2],GM[i].shape[3]][::-1],order='C').T

    GM3d_rounded,rM_rounded = TT_rounding(GM3d,e)

    for i in range(len(GM3d)):
        GM4d[i] = np.reshape(GM3d_rounded[i].T,[GM3d_rounded[i].shape[0],GM[i].shape[1],GM[i].shape[2],GM3d_rounded[i].shape[2]][::-1],order='C').T
    return GM4d,rM_rounded

# Transpose a tensor which is in tensor train format, input is the dictionary containing the cores
def my_transpose_TTm(TT):
    TTnew = {}
    for i in range(len(TT)):
        TTnew[i] = np.transpose(TT[i],[0,2,1,3])
    return TTnew

# Dot product of matrix in TT format and vector in TT format
def matrixvec(GM,GV):
    GMVprod = {}; rMVprod= {}; rMVprod[0] = 1

    for i in range(len(GM)):
        GMVprod[i] = np.transpose(my_Tensordot(GM[i],GV[i],[2,1]),[0,3,1,2,4])
        GMVprod[i] = np.reshape(GMVprod[i].T,[np.prod(GMVprod[i].shape[:2]),GMVprod[i].shape[2],np.prod(GMVprod[i].shape[3:])][::-1],'C').T
        rMVprod[i+1] = GM[i].shape[-1]*GV[i].shape[-1]
    return GMVprod, rMVprod


# CPD-GEVD from Matlab, barebones without fancy errors
def CPD_gevd(T,R,Random,method):
    Tshape = list(T.shape)
    N = len(Tshape)

    # Directr trilinear decomposition
    if R > 1:
        if Random == True:
            G,U,_ = my_Tucker1(T,0)
            G = G[0:R,0:R,:]
            G = my_Nmode_prod(G,np.random.randn(G.shape[2],2),2)
            size_core = [R,R,2]
        else:
            # Prepermute the (core) tensor so the largest two modes are the first
            # two, which hopefully maximizes the probability that the first two
            # factor matrices have full column rank.
            G,U,_,mlsv = my_Tucker1(T,0)
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
        
        # Check whether the 2 slices are of full column rank
        if np.any(R > np.asarray(size_core[0:2])) or size_core[2] != 2:
            print('Slices are not full column rank!')
            return
        
        if method == 'qz':
            # Compute first factor matrix using generalized eigenvectors with qz algorithm
            _,_,_,Veig = sc.linalg.qz(G[:,:,0].squeeze(), G[:,:,1].squeeze())

        else:
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
       
        T1 = T.reshape([Tshape[0],np.prod(Tshape[1:])],order='F')
        X = np.dot(np.dot(T1.T,np.conj(Unew[0])),Veig)
        

        for r in range(R):
            _,u,_,_ = my_Tucker1(np.real(X[:,r]).reshape(Tshape[1:],order='F'),0)
            for n in range(1,N):
                Uout[n][:,r] = u[n-1][:,0]       

        KR = sc.linalg.khatri_rao(Uout[2],Uout[1])
        for i in range(3,N):
            KR = sc.linalg.khatri_rao(Uout[i],KR)
            
        # Uout[0] = np.real(np.dot(Unew[0],np.linalg.inv(Veig)))
        Uout[0] = np.real(np.dot(T1,np.linalg.pinv(KR.T)))

        # Normalize factors of rank-one components
        nrm = np.zeros([len(Uout),R])
        for i,(key,Uval) in enumerate(Uout.items()):
            nrm[i,:] = np.sqrt(np.sum(np.abs(Uval)**2,axis=0))
            Uout[key] = Uval/nrm[key,:]

        nrm2 = np.prod(nrm,axis=0)**(1/N)
        for i,(key,Uval) in enumerate(Uout.items()):
            Uout[key] = Uval*nrm2
    else:
        # Compute approximate single rank-one component through HOSVD
        G,Uout,_,_ = my_Tucker1(T,0)
        G = G[np.zeros([1,N])]
        for i,(key,Uval) in enumerate(Uout.items()):
            U[key] = Uval[0]
        Uout[0] = Uout[0]*G

    # Inverse permute the factor matrices to original mode sequence
    for idx, val in enumerate(iperm):
        Uout[idx] = Uout[val]
    return Uout

# Compute CPD error
def CPDerror(T,lam,DU,symmetric):
    CP = {}
    CPf = np.zeros(T.shape)

    if symmetric == True:
        R = DU.shape[1]
        for r in range(R):
            L = [DU[:,r] for n in range(len(T.shape))]
            CP[r] = my_out_prod(L)
            CPf += lam[r]*CP[r]
    else:
        R = DU[0].shape[1]
        for r in range(R):
            L = [DU[n][:,r] for n in range(len(T.shape))]
            CP[r] = my_out_prod(L)
            CPf += lam[r]*CP[r]
    E = T - CPf
    return my_inner_prod(E,E)/my_inner_prod(T,T)

# ---------------------------------------------------------OLD---------------------------------------------------------------------


# # Mode-n matricization (n starts at 1)
# def my_modeN_mat(T,mode):
#     L = list(range(len(T.shape))); L.remove(len(T.shape)-mode)
#     return T.T.transpose([len(T.shape)-(mode),*L]).reshape((T.shape[mode-1],-1),order='C')

# def my_modeN_mat_reverse(Tm,Tshape,mode):
#     mode = mode-1
#     L = list(range(len(Tshape))); L.remove(mode)
#     return Tm.reshape((np.asarray(Tshape)[[mode,*L]]),order='F').transpose((np.insert(list(range(1,len(Tshape))),mode,0)))

# # Dot product of Tensor N-mode with first dimension of Matrix M (rows), mode starts at 1
# def my_Nmode_prod(T,M,mode):
#     Tshape = np.asarray(T.shape)
#     Tshape[mode-1] = M.shape[1]

#     L1 = list(range(len(T.shape))); L1.remove(len(T.shape)-mode)
#     L2 = list(range(len(Tshape))); L2.remove(mode-1)
#     return np.dot(T.T.transpose([len(T.shape)-(mode),*L1]).reshape((T.shape[mode-1],-1),order='C').T,M).reshape((np.asarray(Tshape)[[*L2[::-1],mode-1]]),order='C').T.transpose((np.insert(list(range(1,len(Tshape))),mode-1,0)))


# # Return the k'th folding matrix of a tensor T where rk is the rank of Tk (k starts at 1)   OLD
# def k_unfold_Tensor(T,k):
#     prod1 = np.prod(np.asarray(T.shape)[0:k],axis=0)
#     prod2 = np.prod(np.asarray(T.shape)[k:],axis=0)
#     return np.reshape(T.T,([prod1,prod2])[::-1],'C').T

# # Return the k'th folding matrix of a tensor T with a custom new rank rk = r at k and original rank nk at k (k starts at 1)     OLD 1.2
# def rnk_unfold_Tensor(T,nk,r):
#     numel = np.prod(np.asarray(T.shape),axis=0)
#     return np.reshape(T.T,([int(r*nk),int(numel/(r*nk))])[::-1],'C').T

# # TT-SVD algorithm
# def my_TT_SVD(T,e):
#     # Init
#     d = len(T.shape)
#     delta = e/np.sqrt(d-1)*np.sqrt(my_inner_prod(T,T))

#     # Make copy of Tensor
#     C = T
#     r = {}; r[0] = 1; r[d] = 1
#     G = {}
#     n = np.asarray(T.shape)

#     # For loop to create the tensor cores up to Gd
#     for k in range(0,d-1):
#         C = rnk_unfold_Tensor(C,n[k],r[k])
#         U,S,Vh = np.linalg.svd(C,full_matrices=False)
#         i = 0
#         while np.sqrt(my_inner_prod(S[-1:-2-i:-1],S[-1:-2-i:-1])) <= delta:
#             i = i + 1
#         r[k+1] = len(S)-i
#         G[k] = np.reshape(U[:,0:r[k+1]].T,([r[k],n[k],r[k+1]])[::-1],'C').T
#         C = np.dot(np.diag(S[0:r[k+1]]),Vh[0:r[k+1],:])
#     G[d-1] = C[..., np.newaxis]
#     return G, r

# def Tensor_recon(L):
#     T = my_Tensordot(np.asarray(L[0]),np.asarray(L[1]),[-1,0])

#     for i in range(2,len(L)):
#         T = my_Tensordot(T,np.asarray(L[i]),[-1,0])
#     return T.squeeze()

# # TT-SVD algorithm   OLD
# def my_TT_SVD(T,e):
#     # Init
#     d = len(T.shape)
#     delta = e/np.sqrt(d-1)*np.sqrt(my_inner_prod(T,T))

#     # Make copy of Tensor
#     C = T
#     r = {}; r[0] = 1; r[d] = 1
#     G = {}
#     n = np.asarray(T.shape)

#     # For loop to create the tensor cores up to Gd
#     for k in range(1,d):
#         C = nk_unfold_Tensor(C,n[k-1],r[k-1])
#         U,S,Vh = np.linalg.svd(C,full_matrices=False)
#         i = 0
#         while np.sqrt(my_inner_prod(S[-1:-2-i:-1],S[-1:-2-i:-1])) <= delta:
#             i = i + 1
#         r[k] = len(S)-i
#         G[k-1] = np.reshape(U[:,0:r[k]].T,([r[k-1],n[k-1],r[k]])[::-1],'C').T
#         C = np.dot(np.diag(S[0:r[k]]),Vh[0:r[k],:])
#     G[d-1] = np.reshape(C.T,([r[d-1],n[d-1],r[d]])[::-1],'C').T
#     return G, r

# def Tensor_recon(L):    OLD
#     T = np.tensordot(np.asarray(L[0]),np.asarray(L[1]),axes=[[len(np.asarray(L[0]).shape)-1],[0]]).squeeze()
#     for i in range(len(np.asarray(L[1]).shape)-1,len(L)):
#         T = np.tensordot(np.asarray(T),np.asarray(L[i]),[[-1],[0]])
#     return T.squeeze()

# def TT_rounding(GA,e):      -- OLD
#     # Init
#     d = len(GA); Gqr = {}
#     delta = e/np.sqrt(d-1)*np.sqrt(my_inner_prod(GA[len(GA)-1],GA[len(GA)-1]))

#     if len(GA[len(GA)-1].shape) == 2:
#             GA[len(GA)-1] = GA[len(GA)-1][..., np.newaxis]

#     # Right-to-left truncation
#     Gqr[d-1] = GA[d-1]

#     for k in range(d-1,0,-1):  
#         Gqr[k], R = np.linalg.qr(np.reshape(Gqr[k],[Gqr[k].shape[0],np.prod(Gqr[k].shape[1:])],'F').T)
#         Gqr[k] = np.reshape(Gqr[k].T,[Gqr[k].T.shape[0],GA[k].shape[1],int(Gqr[k].T.shape[-1]/GA[k].shape[1])],'F')
#         Gqr[k-1] = my_Nmode_prod(GA[k-1],R,2)
        
#     # Compression for orthogonalized representation
#     r = {}; r[0] = 1; r[d] = 1

#     for k in range(0,d-1):
#         U,S,Vh = sc.linalg.svd(np.reshape(Gqr[k],[Gqr[k].shape[0]*GA[k].shape[1],Gqr[k].shape[-1]],'F'),full_matrices=False,compute_uv=True)

#         i = 0
#         while np.sqrt(my_inner_prod(S[-1:-2-i:-1],S[-1:-2-i:-1])) <= delta:
#             i = i + 1
#         r[k+1] = len(S)-i

#         Gqr[k] = U[:,0:r[k+1]]; S = S[0:r[k+1]]; Vh = Vh[0:r[k+1],:]
#         Gqr[k+1] = my_Nmode_prod(Gqr[k+1],np.dot(np.diag(S),Vh),0)
#         Gqr[k] = np.reshape(Gqr[k],[r[k],GA[k].shape[1],r[k+1]],'F')
#     return Gqr,r


# # Own version of Tensordot, modes start at 1   --version1
# def my_Tensordot(A,B,modes):
#     Ashape = list(A.shape); del Ashape[modes[0]-1]
#     Bshape = list(B.shape); del Bshape[modes[1]-1]
#     L = [*Ashape, *Bshape]
#     return my_modeN_mat_reverse((my_Nmode_prod(A,my_modeN_mat(B,modes[1]),modes[0])),L,-1) 

# # Own version of Tensordot, modes start at 0     --version2
# def my_Tensordot(A,B,modes):
#     A_order = list(range(len(A.shape))); A_order.remove(modes[0])
#     B_order = list(range(len(B.shape))); B_order.remove(modes[1])

#     Anew = A.transpose([*A_order,modes[0]])
#     Bnew = B.transpose([modes[1],*B_order])

#     Anew2 = Anew.reshape([np.prod(Anew.shape[0:-1]),Anew.shape[-1]])
#     Bnew2 = Bnew.reshape([Bnew.shape[0],np.prod(Bnew.shape[1::])])

#     T = np.dot(Anew2,Bnew2)
#     return np.reshape(T,[*Anew.shape[0:-1],*Bnew.shape[1::]])


## innerproduct of TT train from literature using Kronecker product and columnwise Kronecker product (Khatri Rao product)    OLD
# def my_innerprodTTABLit(GA,GB):     
#     # v = 0
#     d = len(GA)
#     # for i in range(GA[0][0].shape[0]):
#     #     v +=  np.kron(GA[0][0][i],GB[0][0][i])
    
#     v = np.sum(linalg.khatri_rao(GA[0][0,:,:].T,GB[0][0,:,:].T),axis=1)

#     for k in range(1,d-1):
#         # vnew =  np.dot(v,np.kron(GA[k][:,0],GB[k][:,0]))

#         # for i2 in range(1,GA[k].shape[1]):
#         #     vnew +=  np.dot(v,np.kron(GA[k][:,i2],GB[k][:,i2]))
        
#         # v = vnew
        
#         v = np.dot(v,(np.sum([np.kron(GA[k][:,i],GB[k][:,i]) for i in range(GA[k].shape[1])],axis=0)))
#     return np.dot(v,(np.sum([np.kron(GA[d-1][:,i],GB[d-1][:,i]) for i in range(GA[d-1].shape[1])],axis=0)))

# # Own tensortrain inner product  -- version 2 OLD
# def my_innerprodTTAB(GA,GB):
#     InProd = my_Tensordot(GA[0],GB[0],[2,2])
#     InProd = my_Tensordot(InProd,GA[1],[2,1])

#     for i in range(1,len(GA)-1):
#         if len(GB[i].shape) == 2:
#             GB[i] =  GB[i][..., np.newaxis]

#         res_perm = np.transpose(InProd, [0,1,4,2,3])
#         res = np.reshape(res_perm,[res_perm.shape[0]*res_perm.shape[1]*res_perm.shape[2],res_perm.shape[3]*res_perm.shape[4]])
#         resb = np.reshape(GB[i],[GB[i].shape[0]*GB[i].shape[1],GB[i].shape[-1]])
#         Tdot = np.dot(res,resb)
#         InProd = np.reshape(Tdot,[1,1,InProd.shape[-1],InProd.shape[-1]])

#         InProd = my_Tensordot(InProd,GA[i+1],(3,1))
#     return np.tensordot(InProd,GB[len(GB)-1],([-3,-2],[0,1])).squeeze()       

# # Site-k-mixed-canonical form, l starts at 1   OLD
# def site_k_mixed_canon(GA,l,e):
#     if l == len(GA):
#         return GA, 0,0
#     else:
#         T = np.tensordot(np.asarray(GA[len(GA)-2]),np.asarray(GA[len(GA)-1]),axes=[[len(np.asarray(GA[0]).shape)-1],[0]])
#         for i in range(3,len(GA)-l+2):
#             T = np.tensordot(GA[len(GA)-i],np.asarray(T),[[-1],[0]])
            
    
#         # # Code to recreate the used cores (for checking if previous piece of code works)
#         # C = T
#         # d = len(GA)
#         # delta = e/np.sqrt(d-1)*np.sqrt(mtf.my_inner_prod(GA[len(GA)-1],GA[len(GA)-1]))

#         # n = [GA[i].shape[1] for i in range(len(GA))]
        
#         # r = {}; r[len(GA)-1-l] = GA[len(GA)-l-2].shape[-1]
        
#         # G = {}
#         # for k in range(len(GA)-l,d):
            
#         #     C = mtf.nk_unfold_Tensor(C,n[k-1],r[k-1])
            
#         #     U,S,Vh = np.linalg.svd(C,full_matrices=False)
            
#         #     i = 0
#         #     while np.sqrt(mtf.my_inner_prod(S[-1:-2-i:-1],S[-1:-2-i:-1])) <= delta:
#         #         i = i + 1
#         #     r[k] = len(S)-i
#         #     G[k-1] = np.reshape(U[:,0:r[k]].T,([r[k-1],n[k-1],r[k]])[::-1],'C').T
#         #     C = np.dot(np.diag(S[0:r[k]]),Vh[0:r[k],:])
#         # G[d-1] = C
#         # return G, r



#         # site-k-canonical mixed form other side
#         C = T
#         d = len(GA)
#         delta = e/np.sqrt(d-1)*np.sqrt(mtf.my_inner_prod(GA[len(GA)-1],GA[len(GA)-1]))

#         n = [GA[i].shape[1] for i in range(len(GA))]
#         r = [GA[i].shape[-1] for i in range(len(GA))]; r[len(GA)-1] = 1; r.insert(0,1)
 
#         G = {}

#         # Decompose current tensor C with SVD, store S, go further with Vh      
#         C = mtf.nk_unfold_Tensor(C,n[l-1],r[l-1])
          
#         U,S_core,Vh = np.linalg.svd(C,full_matrices=False)
        
#         i = 0
#         while np.sqrt(mtf.my_inner_prod(S_core[-1:-2-i:-1],S_core[-1:-2-i:-1])) <= delta:
#             i = i + 1
        
#         r[l] = len(S_core)-i
     
#         C = Vh
#         r_right = {}; r_right[len(GA)-l] = r[l]; r_right[0] = 1
     
#         n_right = n[l::][::-1]


#         for k in range(1,len(GA)-l):

#             C = np.reshape(C,[r_right[len(GA)-l]*np.prod(n_right[k::]),n_right[k-1]*r_right[k-1]],order='F')

#             U,S,Vh = np.linalg.svd(C,full_matrices=False)

#             i = 0
#             while np.sqrt(mtf.my_inner_prod(S[-1:-2-i:-1],S[-1:-2-i:-1])) <= delta:
#                 i = i + 1
#             r_right[k] = len(S)-i

#             G[len(GA)-k] = np.reshape(Vh[0:r_right[k],:],([r_right[k],n_right[k-1],r_right[k-1]]),'F')
            
#             C = np.dot(U[:,0:r_right[k]],np.diag(S[0:r_right[k]]))
        
        
#         if len(GA)-l == 1:
#             k = 0
            
#         G[l] = np.reshape(C,[len(S_core),n_right[-1],r_right[k]],'F')
        
#         Gl = {i:GA[i] for i in range(l)}
#         return Gl,S_core,G


# # Function to reconstruct site-k-mixed-canonical form     OLD
# def site_k_recon(Gl,S,Gr):
#     l = len(Gl)
#     Tl = Gl[0]
#     for i in range(1,l):
#         Tl = np.tensordot(Tl,Gl[i],[-1,0])
    
#     Tr = Gr[len(GA)-1]
#     for i in range(l+1,len(GA)):
#         Tr = np.tensordot(Gr[len(GA)+l-1-i],Tr,[-1,0])
        
#     S = np.diag(S_core)
#     Trec_left = np.tensordot(Tl,S,[-1,0]).squeeze()

#     return np.tensordot(Trec_left,Tr,[-1,0]).squeeze()


# # TTm-SVD algorithm Must give an equal amount of dimensions for dims argument    - OLD
# def my_TTm_SVD(M,dims,e):
#     if np.prod(dims) != np.prod(M.shape):
#         print('ERROR Chosen tensor dimensions do not match up with number of elements in matrix M:',np.prod(dims),'!=',np.prod(M.shape))
#     T = np.reshape(M,[*dims],'F')
#     T = np.transpose(T,list(chain.from_iterable((i,int(len(T.shape)/2+i)) for i in range(int(len(T.shape)/2)))))

#     # TT-SVD algorithm
#     # Init
#     d = int(len(T.shape)/2)
#     delta = e/np.sqrt(d-1)*np.sqrt(my_inner_prod(M,M))

#     # Make copy of Tensor
#     C = T; r = {}; r[0] = 1; r[d] = 1; G = {}
#     ni = np.asarray(T.shape)[::2]; nj = np.asarray(T.shape)[1::2]

#     # For loop to create the tensor cores up to Gd
#     for k in range(1,d):
#         C = nk_unfold_Tensor(C,ni[k-1]*nj[k-1],r[k-1])

#         U,S,Vh = np.linalg.svd(C,full_matrices=False)
#         i = 0
#         while np.sqrt(my_inner_prod(S[-1:-2-i:-1],S[-1:-2-i:-1])) <= delta:
#             i = i + 1
#         r[k] = len(S)-i
#         G[k-1] = np.reshape(U[:,0:r[k]].T,([r[k-1],ni[k-1],nj[k-1],r[k]])[::-1],'C').T
#         C = np.dot(np.diag(S[0:r[k]]),Vh[0:r[k],:])
#     G[d-1] = np.reshape(C.T,([r[d-1],ni[d-1],nj[d-1],r[d]])[::-1],'C').T
#     return G, r


# # site-k-mixed canonical form, l starts at 1    - OLD version 3
# def site_k_mixed_canon(GA,l,e):
#     if l == len(GA):
#         return GA, 0,0
#     else:
#         T = my_Tensordot(np.asarray(GA[len(GA)-2]),np.asarray(GA[len(GA)-1]),[len(np.asarray(GA[0]).shape)-1,0])
#         for i in range(3,len(GA)-l+2):
#             T = my_Tensordot(GA[len(GA)-i],np.asarray(T),[-1,0])

#         # site-k-canonical mixed form other side
#         C = T; d = len(GA)
#         delta = e/np.sqrt(d-1)*np.sqrt(my_inner_prod(GA[len(GA)-1],GA[len(GA)-1]))

#         n = [GA[i].shape[1] for i in range(len(GA))]
#         r = [GA[i].shape[-1] for i in range(len(GA))]; r[len(GA)-1] = 1; r.insert(0,1); G = {}

#         # Decompose current tensor C with SVD, store S, go further with Vh  
#         C = np.reshape(C.T,([int(r[l-1]*n[l-1]),int(np.prod(np.asarray(C.shape),axis=0)/(r[l-1]*n[l-1]))])[::-1],'C').T
#         U,S_core,Vh = np.linalg.svd(C,full_matrices=False)
        
#         i = 0
#         while np.sqrt(my_inner_prod(S_core[-1:-2-i:-1],S_core[-1:-2-i:-1])) <= delta:
#             i = i + 1
        
#         r[l] = len(S_core)-i
#         C = Vh; r_right = {}; r_right[len(GA)-l] = r[l]; r_right[0] = 1; n_right = n[l::][::-1]

#         for k in range(1,len(GA)-l):
#             C = np.reshape(C,[r_right[len(GA)-l]*np.prod(n_right[k::]),n_right[k-1]*r_right[k-1]],order='F')
#             U,S,Vh = np.linalg.svd(C,full_matrices=False)

#             i = 0
#             while np.sqrt(my_inner_prod(S[-1:-2-i:-1],S[-1:-2-i:-1])) <= delta:
#                 i = i + 1
#             r_right[k] = len(S)-i

#             G[len(GA)-k] = np.reshape(Vh[0:r_right[k],:].T,([r_right[k],n_right[k-1],r_right[k-1]][::-1]),'C').T
#             C = np.dot(U[:,0:r_right[k]],np.diag(S[0:r_right[k]]))
        
#         if len(GA)-l == 1:
#             k = 0
            
#         G[l] = np.reshape(C.T,[len(S_core),n_right[-1],r_right[k]][::-1],'C').T
#         Gl = {i:GA[i] for i in range(l)}
#         return Gl,S_core,G