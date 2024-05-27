# Purpose: Setting up the two-link robot in Figure 2
# %------------------------------------------ Packages -------------------------------------------% #
import pathlib
import numpy as np

from typing import Tuple, List

from src import paper_plot
# %------------------------------------------ Classes --------------------------------------------% #
class TwoLinkManipulatorRobot():
    def __init__(self,
                 m1:float, L1:float, 
                 m2:float, L2:float, 
                 x_sys0:np.ndarray, 
                 trajectory, r_lin:np.ndarray = None,
                 sys_type:str = "Nonlinear") -> None:
        # TwoLinkRobot: Physical properties
        self.m1, self.m2 = m1, m2
        self.L1, self.L2 = L1, L2
        
        # TwoLinkRobot: Initial condition
        self.x_sys0 = x_sys0
        
        # TwoLinkRobot: Desired trajectory
        self.trajectory = trajectory
        
        # TwoLinkRobot: Linearization point
        self.r_lin = r_lin
        if r_lin is None:
            self.r_lin = trajectory.r_end
        
        # Control u = [tau_1, tau_2]
        self.bhat = np.eye(2)
        
        # Measrement y = [theta_1_dot, theta_2_dot]
        self.C = np.array([[0, 0, 1, 0],
                           [0, 0, 0, 1]])
        
        # Compute and Store intertias for each arm
        self._J1 = (m1 * L1**2) / 3
        self._J2 = (m2 * L2**2) / 3
        
        # Compute Linearized Model about r_lin
        self.A = np.array([[0, 0, 1, 0],
                           [0, 0, 0, 1],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        
        self.B = np.block([[np.zeros((2,2))],
                           [self.M_inv(self.r_lin)]]) @ self.bhat
        
        self.C_prewrap = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0]])
        
        self.D = np.zeros((2, 2))
        
        # Choose system type
        self.sys_type = sys_type
        match sys_type:
            case "Nonlinear":
                self.f = lambda xs, u: self._f_nonlinear(xs, u)
            case "Linearized":
                    self._f = lambda xs, u: self._f_linearized(xs, u)
        
        # Define prewrap measurement  
        self.g_prewrap = lambda xs: self.C_prewrap @ xs
        
        # Define Measurement
        self.g = lambda xs: self.C @ xs 
        
    # %---------------------------------------- Methods --------------------------------------% #
    def _M(self, x) -> np.ndarray:
        # Extract states.
        th2 = x[1, 0]  # th2

        mass = np.array([[self._J1 + self.m2 * (self.L1**2) + self._J2 + np.cos(th2) * self.L1 * self.L2 * self.m2, self._J2 + np.cos(th2) * self.L1 * self.L2 * self.m2 / 2],
                         [self._J2 + np.cos(th2) * self.L1 * self.L2 * self.m2 / 2                                ,  self._J2]])
        mass = (mass + mass.T) / 2  # force the mass matrix to be to be symmetric
        return mass

    def _fnon(self, x) -> np.ndarray:
        # Extract states
        _, th2, dot_th1, dot_th2 = x.ravel()
        dot_q = x[2:].reshape((-1, 1))  # angle rates
        a = np.array([[0], 
                      [- 1/2 * dot_th1 * (dot_th1 + dot_th2) * np.sin(th2) * self.L1 * self.L2 * self.m2]])
        dot_M = -np.sin(th2) * self.L1 * self.L2 * self.m2 * dot_th2 * np.array([[1, 0.5], [0.5, 0]])
        nonlinear_forces = dot_M @ dot_q - a
        return nonlinear_forces
    
    def M_inv(self, x) -> np.ndarray:
        return np.linalg.inv(self._M(x))
    
    # Purpose: Non-Linear model
    def _f_nonlinear(self, xs, u) -> np.ndarray:
        # Extract states.
        q_dot = xs[2:]

        RHS = u - self._fnon(xs)
        q_ddot = np.linalg.solve(self._M(xs), RHS)
        return np.vstack((q_dot, q_ddot))  # xs_dot
    
    # Purpose: Linearized model
    def _f_linearized(self, xs, u) -> np.ndarray:
        return self.A @ xs + self.B @ u
    
    # Purpose: Compute control effort for plotting
    def compute_control_effort(self, N, x_ctrl, q, q_dot, ctrl, t) -> np.ndarray:
        # Compute error and control (for plotting purposes)
        u_ctrl   = np.zeros((2, N))
        error_th = np.zeros((2, N))
        error_th_dot = np.zeros((2, N))
        
        for i, ti in enumerate(t):
            error_th[:, [i]]     = self.trajectory.r_des(ti) - q[:, [i]]
            error_th_dot[:, [i]] = self.trajectory.r_des_dot(ti) - q_dot[:, [i]]
            u_ctrl[:, [i]] =  ctrl.g_prewrap(error_th[:, [i]]).reshape(-1, 1) \
                              + self.bhat @ ctrl.g(x_ctrl[:, [i]], error_th_dot[:, [i]]).reshape(-1, 1)
        return error_th, error_th_dot, u_ctrl
    
    # Purpose: Extract States and Compute Plot data
    def extract_states(self, sol, ctrl) -> Tuple[np.ndarray,
                                                 List[np.ndarray],
                                                 List[np.ndarray],
                                                 np.ndarray,
                                                 np.ndarray,
                                                 np.ndarray]:
        # Extract states
        t     = sol.t
        sol_x = sol.y
        
        x_sys  = sol_x[:4, :]
        x_ctrl = sol_x[4:, :]
        
        q = x_sys[:2, :]
        q_dot = x_sys[2:, :]
        
        # Convert angles to degrees
        sol_x_degree = x_sys * 180 / np.pi
        th1, th2, dot_th1, dot_th2 = sol_x_degree
        
        th = [th1, th2]
        dot_th = [dot_th1, dot_th2]
        
        N = th1.size
        
        # Compute Error for plotting
        error_th, error_th_dot, u_ctrl = self.compute_control_effort(N, x_ctrl, q, q_dot, ctrl, t)
        
        return t, th, dot_th, u_ctrl, error_th, error_th_dot
    
    # Purpose: Plot x, u, and energy
    def plot_results(self, sol, ctrl, save_fig=False) -> None:
        # Extract States and Compute Plot data
        t, th, dot_th, u_ctrl, error_th, error_th_dot = self.extract_states(sol, ctrl)
        
        # Plotting Path
        path = None
        if save_fig:
            path = pathlib.Path('figs')
            path.mkdir(exist_ok=True)
            
        # Title
        title = f"{self.sys_type} model using {type(ctrl).__name__}"
        
        # Plot theta and theta_dot
        labels = lambda indx : [rf'$\theta_{indx+1}(t)$ (deg)',
                                rf'$\dot{{\theta}}_{indx+1}(t)$ (deg/s)']

        for i in range(2):
            paper_plot.plt_subplot_vs_t(t, th[i], dot_th[i], 
                                        labels=labels(i), title=title,
                                        trajectory=self.trajectory, 
                                        entry=i,
                                        path=path)
            
        # Plot error and error_dot
        labels = lambda indx : [rf'error in $\theta_{indx+1}(t)$ (deg)',
                                rf'error in $\dot{{\theta}}_{indx+1}(t)$ (deg/s)']

        err     = lambda i : [abs(e) * 180 / np.pi for e in error_th[i]]
        err_dot = lambda i : [abs(e) * 180 / np.pi for e in error_th_dot[i]]
        
        for i in range(2):
            paper_plot.plt_subplot_vs_t(t, err(i), err_dot(i), 
                                        labels=labels(i), title=title,
                                        path=path)

        # Plot control
        labels = [r'$u_1(t)$ (N/m)', r'$u_2(t)$ (N/m)']
        paper_plot.plt_subplot_vs_t(t, u_ctrl[0, :], u_ctrl[1, :], 
                                    labels=labels, title=title, path=path)
        