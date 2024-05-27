# Purpose: VSP Controller Synthesized as per Section V.C
# %------------------------------------------ Packages -------------------------------------------% #
import numpy as np
import scipy
import control

from src.toolbox import is_spd, is_snd, is_snsd
# %--------------------------------------- Abstract Class -----------------------------------------% #
class controller(object):
    def __init__(self, x_ctrl0, Kp):
        # Controller Initial condition
        self.x_ctrl0 = x_ctrl0
        
        # Prewrap P control
        self.Kp = Kp
        
    # Purpose: Overide print(class)
    def __repr__(self) -> str:
        # Printing Msg
        str = ""
        
        # Print Class Name
        str += f'Using {type(self).__name__}\n'
        
        # Format Numpy to print matrix in 1 line
        reformat = lambda m: np.array2string(m, 
                                             formatter={'all':lambda x: f"{x}"},
                                             separator=',').replace(',\n', ';')
        # Print all properties
        props_dic = vars(self)
        for prop, value in props_dic.items():
            if type(value) == np.ndarray:
                value = reformat(value)
            str += f'{prop} = {value}\n'
        return str
        
    # Purpose: Compute xc_dot
    def f(self, x, e) -> np.array:
        return self.A @ x + self.B @ e
    
    # Purpose: Compute u
    def g(self, x, e) -> np.array:
        return self.C @ x + self.D @ e
    
    # Purpose: Compute u_prewrap
    def g_prewrap(self, e) -> np.array:
        return self.Kp @ e

# %-------------------------------------- VSP Control Classes --------------------------------------% #
class VSP_lqr(controller):
    def __init__(self, sys, Q, R, Kp, x_ctrl0, FEED_THROUGH=1e-4) -> None:
        # Controller Initial condition
        super().__init__(x_ctrl0, Kp)
        
        O2 = np.zeros((2,2)) 
        Ap = sys.A + np.block([[O2                        , O2],
                               [-sys.M_inv(sys.r_lin) @ Kp, O2]])
        
        # Compute matrix gain using LQR
        K, _, _ = control.lqr(Ap, sys.B, Q, R)

        # Construct VSP Controller
        self.A = Ap - sys.B @ K
        self.C = K
        self.D = FEED_THROUGH*np.eye(2)

        # Find P using KYP Lemma
        Q = np.eye(len(self.A))
        P = control.lyap(self.A.T, Q)
        self.B = scipy.linalg.solve(P, self.C.T, 
                                    assume_a="pos")
