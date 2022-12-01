# Independent-component-analysis-through-implicit-cumulant-tensor-decomposition
Python files used for the master thesis. The **myCumulantfunctions.py** and **myTensorfunctions.py** are python files containing functions devised and used throughout this thesis. The sources and definitions of the multi-linear functions in **myTensorfunctions.py** can be found in the thesis PDF. The functions in **myCumulantfunctions.py** have been given a description about the function and the input arguments.

##**USERGUIDE**:
The following steps must be performed in order for all files to work.
- Store a copy of the FastICA implementation from sklearn and replace its contents with that of **FASTICA_edited.py**.
- Install the SVD-update package. This can be done directly from https://github.com/AlexGrig/svd_update. However, when doing so the second method in function 'find_roots' (file svd_update.py, line 452-497) must be replaced with the following code:

```
 if method == 2: # Imported lapack function
        d = sigmas[::-1]
        z = m_vec[::-1]
        roots = np.empty_like(d)
        for i in range(len(sigmas)):
            delta, roots[i], work, info = sp.linalg.lapack.dlasd4(i, d, z)
            if (info > 0) or np.isnan(roots[i]):
                raise ValueError("LAPACK root finding dlasd4 failed at {}-th sigma".format(i))
        return roots[::-1]
```

Alternaively, the SVD-package in the **svd_update** folder in this repository can be used.

- When using the data-sets for plotting. The files must be in the same directory as **plot_finalcomps.ipynb** or an additional statement must be added to the top of the code where the files can be found.

## Contents
The repository contains the following:
**master thesis in PDF**:
This is the master thesis for which the provided code has been used to perform the numerical experiments.

**myCumulantfunctions.py**:
This file contains the ICA and BSS functions that were used during the making of this thesis. The following functions are the important ones:
- cum4tensor3            , create the cumulant tensor of data array X.
- testset                , create an instance of the artificial testset used in the testset.
- Errorfunc              , function used to evaluate the estimated results of an algorithm. Additionally sorts and determines the correct sign of each solution if classiifed as correct.
- IMPCPDfunOPT           , the computation of the gradients of the CP-gradient method. This can be used with the scipy.optimize solver package
- symIMPCPDffFAST            , the CP-fixed-point algorithm 
- symIMPCPDffWHITEFAST   , the CP-fixed-point algorithm for whitened data, is faster than the previous function
- QRT_fulltensor         , The QRT algorithm for any 4-way symmetric tensor. Can be used for ICA.
- QRT                    , the final implicit QRT algorithm for whitened data. Allows for a deflationary or parallel computation scheme togeter with various convergence criterions.
- QRST4                  , Python implementation of the QR-for-Symmetric-Tensors algorithm by Batselier, K., & Wong, N. (2014) for computing a symmetric tensor's eigenpairs.
- CPD_gevd               , the CP-GEVD algorithm by Sanchez, E., & Kowalski, B. R. (1990) for computing tha CP of a tensor in deterministic fashion. Code is ported over from the Matlab implementation.
- HOSVD_iterFINAL        ,  the implicit HOSVD algorithm which uses the SVD-update package from https://github.com/AlexGrig/svd_update.
- CPD_gevdIMPLICIT_FINAL , the implicit CP-GEVD algorithm.


**myTensorfunctions.py**:
This file contains the multi-linear operation used throughout the thesis such a mode-N product, mode-N matricization etc. Many of the fuctions in myCumulantfunctions.py use functions from this file.

**DATAsets**:
Folder containing data-sets containing the performance measures of the non-iterative and iterative algorithms with varying initialization schemes. Can be used togetehr with the plot_finalcomps.ipynb file.

**svd_update**:
The svd-update package from https://github.com/AlexGrig/svd_update. 
NOTE: the package has been altered to work with Python 3.10. An issue in the: scipy.linalg.lapack.dlasd4 function causes the svd-update implementation from 2013 by Alexander Grigorievskiy to not work properly. The fix is presented on the github page of Alexander Grigorievskiy too.

**FASTICA_edited.py**:
This is an edited version of the FastICA implementation from sklearn. This version of the implementation stores the convergence error at each iteration for parallel fastICA and gives it as an output. The FastICA calls in the plot_performance.ipynb file only works with this version.

**plot_performance.ipynb**:
The file in which the numerical experiments are performed. The file produces the performance measures which can be stored into .mat files. The file allows for the plotting of independent results. For the plotting of all results combined and pre-processed the plot_finalcomps.ipynb file must be used.

**plot_finalcomps.ipynb**:
The file in which the final plots of the thesis are produced. The file combines the .mat files of experiments with varying initializations schemes such that these results can be ploted next to each other for comparison.

