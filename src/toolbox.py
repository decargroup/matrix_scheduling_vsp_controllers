# Purpose: Usefull functions
# %------------------------------------------ Packages -------------------------------------------% #
import numpy as np
# %--------------------------------------- Print Function ----------------------------------------% #
def myprint(var_name, var_val):
    print(f'\nEigenvalues of {var_name} are:\n', var_val)
    
# %---------------------------------------- MATH TOOLBOX ------------------------------------------% #
def is_spd(A:np.array,
        verbose:bool = False,
        var_name:str = "") -> bool:
    """
        Check if Matrix is SPD by:
            1) Check symmetry
            2) Check all eig > 0
        
        [OPTIONAL]:verbose with printing var name
    """
    # Check if A is numpy array of 1 element
    if np.isscalar(A) or A.shape == ():
        if verbose and A > 0:
            print(f'\n{var_name} = {A} > 0')
        if verbose and A <= 0:
            print(f'\n{var_name} = {A} <= 0')
        return A > 0
    
    # Check if A is symmetrical
    if not np.allclose(A, A.T):
        return False
    
    # Get Eigenvalues of A
    eigVals = np.linalg.eigvals(A)
    
    if verbose:
        myprint(var_name, eigVals)
        
    # Check if eigenvalues are within tol SPD
    return np.all(eigVals > 0)
        
def is_snd(A:np.array,
        verbose:bool = False,
        var_name:str = "") -> bool:
    """
        Check if Matrix is SPD by:
            1) Check symmetry
            2) Check all eig < 0
        
        [OPTIONAL]:verbose with printing var name
    """
    # Check if A is symmetrical
    if not np.allclose(A, A.T):
        return False
    
    # Get Eigenvalues of A
    eigVals = np.linalg.eigvals(A)
    
    if verbose:
        myprint(var_name, eigVals)
    
    # Check if eigenvalues are within tol SND
    return np.all(eigVals < 0)

def is_spsd(A:np.array,
            verbose:bool = False,
            var_name:str = "") -> bool:
    """
        Check if Matrix is SPSD by:
            1) Check symmetry
            2) Check all eig >= 0
        
        [OPTIONAL]:verbose with printing var name
    """
    # Check if A is symmetrical
    if not np.allclose(A, A.T):
        return False
    
    # Get Eigenvalues of A
    eigVals = np.linalg.eigvals(A)
    
    if verbose:
        myprint(var_name, eigVals)
    
    # Check if eigenvalues are within tol SPSD
    return np.all(eigVals >= 0)

def is_snsd(A:np.array,
            verbose:bool = False,
            var_name:str = "") -> bool:
    """
        Check if Matrix is SPSD by:
            1) Check symmetry
            2) Check all eig <= 0
        
        [OPTIONAL]:verbose with printing var name
    """
    # Check if A is symmetrical
    if not np.allclose(A, A.T):
        return False
    
    # Get Eigenvalues of A
    eigVals = np.linalg.eigvals(A)
    
    if verbose:
        myprint(var_name, eigVals)
    
    # Check if eigenvalues are within tol SNSD
    return np.all(eigVals <= 0)