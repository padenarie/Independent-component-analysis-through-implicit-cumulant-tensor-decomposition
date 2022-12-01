# Independent-component-analysis-through-implicit-cumulant-tensor-decomposition
Python files used for the master thesis. The **myCumulantfunctions.py** and **myTensorfunctions.py** are python files containing functions devised and used throughout this thesis. The sources and definitions of the multi-linear functions in **myTensorfunctions.py** can be found in the thesis PDF. The functions in **myCumulantfunctions.py** have been given a description about the function and the input arguments.

The repository contains the following:
**master thesis in PDF**
This is the master thesis for which the provided code has been used to perform the numerical experiments.

**myCumulantfunctions.py**
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


**myTensorfunctions.py**
This file contains the multi-linear operation used throughout the thesis such a mode-N product, mode-N matricization etc. Many of the fuctions in myCumulantfunctions.py use functions from this file.

**Data-sets**

