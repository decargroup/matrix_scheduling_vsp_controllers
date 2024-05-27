# Purpose: Simulate the Control of the Two-Link Robot in Figure 2 using a gain-scheduling
# %------------------------------------------ Packages -------------------------------------------% #
import pathlib
import numpy as np
import matplotlib.pyplot as plt

from scipy import integrate

from src import initialize
from src import paper_plot

# %----------------------------------------- Initializing ------------------------------------------% #
def init_linearization(model_uncertainty="False"):
    # [PLANT PARAMS]: Pick Linearization points
    r_lin_1 = np.array([[-np.pi/2],    # theta_1
                        [5*np.pi/6]])  # theta_2
    
    r_lin_2 = np.array([[np.pi/3],   # theta_1
                        [np.pi/3]])  # theta_2
    
    r_lin_3 = np.array([[5*np.pi/6], # theta_1
                        [-np.pi/2]]) # theta_2

    # Plant Linearization points
    r_lin_points = [r_lin_1, r_lin_2, r_lin_3]
    
    # Construct Linearized Systems
    lin_systems = [initialize.init_sys(sys_type="Linearized", r_lin=r,
                                       model_uncertainty=model_uncertainty) for r in r_lin_points]
    return lin_systems
    
def init_scheduling_signals(scheduling_type="matrix", DOF=2):
    # [Scheduling Signals]: Set up scheduling signals   
    s1 = lambda t: 1 - (t/3)**4 if t<=3 else 0
    s2 = lambda t: 1 - ((t-3)/2.8)**4 if 0.2<=t<=5.8 else 0
    def s3(t):
        if t<5:
            return 0
        elif t<=7:
            return 1 - ((t-7.5)/2.5)**4
        else:
            return 1 
    
    # [Scheduling Signals]: Set up scheduling signals
    match scheduling_type:
        case "matrix":
            # Set alphas
            alphas = [2, 1, 2]
            
            # Set S_u
            Su_1 = lambda t: np.array([[2*s1(t) + 4*s2(t), 0],
                                        [0                , s1(t)]])
            Su_2 = lambda t: np.array([[s2(t), 0],
                                        [s2(t), s2(t)]])
            Su_3 = lambda t: np.array([[s2(t) + 2*s3(t), 0],
                                        [0            , s3(t)]])

            # Set S_y = S_u.T
            Sy_1 = lambda t: alphas[0] * Su_1(t).T
            Sy_2 = lambda t: alphas[1] * Su_2(t).T
            Sy_3 = lambda t: alphas[2] * Su_3(t).T
                
                
        case "scalar":
            # Set all alphas to 1
            alphas = [1, 1, 1]
            
            # Set S_u = s * I
            Su_1 = lambda t: s1(t) * np.eye(DOF)
            Su_2 = lambda t: s2(t) * np.eye(DOF)
            Su_3 = lambda t: s3(t) * np.eye(DOF)
            
            # Set S_y = s * I
            Sy_1 = lambda t: alphas[0] * s1(t) * np.eye(DOF)
            Sy_2 = lambda t: alphas[1] * s2(t) * np.eye(DOF)
            Sy_3 = lambda t: alphas[2] * s3(t) * np.eye(DOF)
        
    signals = [(Su_1, Sy_1), (Su_2, Sy_2), (Su_3, Sy_3)]
    return signals

def init_controllers(sys_non_linear, sys_linearized, ctrl_typ="VSP_lqr"):
    # Construct Controllers
    ctrls = [initialize.init_controller(sys=sys, ctrl_typ=ctrl_typ) for sys in sys_linearized]
                          
    # Set up closed-loop IC
    x_ctrls0 = np.tile(ctrls[0].x_ctrl0, (len(ctrls), 1))
    x_cl0 = np.concatenate((sys_non_linear.x_sys0, x_ctrls0))
    return ctrls, x_cl0
    
# %----------------------------------------- System Dynamics ----------------------------------------------% #
def gain_scheduled_closed_loop(sys, ctrls, signals, t, x) -> np.array:
        """Closed-loop system."""
        # Split state
        x_sys  = x[:4] # [theta_1, theta_2, theta_dot_1, theta_dot_2]
        x_ctrl = x[4:] # [x_c1, x_c2, ..., xc_n]
        
        # Dim of problem
        dim = len(x_sys)

        # Measurment
        y  = sys.g(x_sys)
        yp = sys.g_prewrap(x_sys)
        
        # Compute errors
        error     = sys.trajectory.r_des(t) - yp
        error_dot = sys.trajectory.r_des_dot(t) - y
        
        # Compute control
        u_ctrl = np.zeros((ctrls[0].C.shape[0], 1))
        
        # Advance controller state.
        x_dot_ctrl = np.zeros_like(x_ctrl)
        
        # Iterate through controllers
        for i, (ctrl, signal) in enumerate(zip(ctrls, signals)):
            # Extract controller state
            x_ctrl_i = x_ctrl[i*dim:(i+1)*dim]
            
            # Compute signal
            Su_i, Sy_i = signal
            Su_i, Sy_i = Su_i(t), Sy_i(t)
            
            # Compute error for each controller
            error_dot_i = Su_i @ error_dot
            
            # Advance controller state.
            x_dot_ctrl[i*dim:(i+1)*dim] = ctrl.f(x_ctrl_i, error_dot_i)
            
            # Compute control
            u_ctrl = u_ctrl + Sy_i @ ctrl.g(x_ctrl_i, error_dot_i)
        
        # Add Proportional control prewrap
        u_ctrl = ctrl.g_prewrap(error) + sys.bhat @ u_ctrl
        
        # Advance system state
        x_dot_sys = sys.f(x_sys, u_ctrl)

        # Concatenate state derivatives
        return np.concatenate((x_dot_sys, x_dot_ctrl)) # x_dot

# %-------------------------------------------- Helper Functions ----------------------------------------------% #
def compute_control_effort(sys, t, N, x_ctrl, q, q_dot, ctrls, signals) -> np.ndarray:
    # Compute error and control (for plotting purposes)
    u_ctrl   = np.zeros((2, N))
    error_th = np.zeros((2, N))
    error_th_dot = np.zeros((2, N))

    dim = len(q) + len(q_dot)
    
    # Control Dim
    u_dim = np.zeros((ctrls[0].C.shape[0], 1))
    
    for i, ti in enumerate(t):
        error_th[:, [i]]     = sys.trajectory.r_des(ti) - q[:, [i]]
        error_th_dot[:, [i]] = sys.trajectory.r_des_dot(ti) - q_dot[:, [i]]
        
        u_ctrl_i = np.zeros_like(u_dim)
        
        for j, (ctrl, signal) in enumerate(zip(ctrls, signals)):
            # Extract controller state
            x_ctrl_i = x_ctrl[j*dim:(j+1)*dim, [i]]
            
            # Compute signal
            Su_i, Sy_i = signal
            Su_i, Sy_i = Su_i(ti), Sy_i(ti)
            
            # Compute error for each controller
            error_dot_i = Su_i @ error_th_dot[:, [i]]
            
            # Compute control
            u_ctrl_i = u_ctrl_i + Sy_i @ ctrl.g(x_ctrl_i, error_dot_i)
       
        # Add Proportional control prewrap
        u_ctrl[:, [i]] = ctrl.g_prewrap(error_th[:, [i]]) + sys.bhat @ u_ctrl_i
    return error_th, error_th_dot, u_ctrl

def extract_states(sys, sol, ctrls, signals):
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
    
    th     = [th1, th2]
    dot_th = [dot_th1, dot_th2]
    
    N = th1.size
    
    # Compute Error for plotting
    error_th, error_th_dot, u_ctrl = compute_control_effort(sys, t, N, x_ctrl, q, q_dot, ctrls, signals)
    
    return t, th, dot_th, u_ctrl, error_th, error_th_dot, sys.trajectory 
    
def plot_results(sys_non_linear, sol, ctrls, signals, save_fig=False):
    # Plotting Path
    path = None
    if save_fig:
        path = pathlib.Path('figs')
        path.mkdir(exist_ok=True)
        
    # Extract states
    t, th, dot_th, u_ctrl, error_th, error_th_dot, _ = extract_states(sys_non_linear, sol, ctrls, signals)
    
    # Extract signals
    S1 = signals[0]
    s1t = np.array([S1(ti)[0, 0] for ti in t])
    s4t = np.array([S1(ti)[1, 1] for ti in t])
    
    # Title
    title = f"{sys_non_linear.sys_type} model using gain scheduling VSP_lqr"
    
    # Plot theta and theta_dot
    labels = lambda indx : [rf'$\theta_{indx+1}(t)$ (deg)',
                            rf'$\dot{{\theta}}_{indx+1}(t)$ (deg/s)']

    for i in range(2):
        paper_plot.plt_subplot_vs_t(t, th[i], dot_th[i], 
                            labels=labels(i), title=title,
                            trajectory=sys_non_linear.trajectory, 
                            entry=i)
        
    # Plot error and error_dot
    labels = lambda indx : [rf'error in $\theta_{indx+1}(t)$ (deg)',
                            rf'error in $\dot{{\theta}}_{indx+1}(t)$ (deg/s)']
    
    err     = lambda i : [e * 180 / np.pi for e in error_th[i]]
    err_dot = lambda i : [e * 180 / np.pi for e in error_th_dot[i]]
    
    for i in range(2):
        paper_plot.plt_subplot_vs_t(t, err(i), err_dot(i), labels=labels(i), title=title)
        
    # Plot control
    labels = [r'$u_1(t)$ (N/m)', r'$u_2(t)$ (N/m)']
    paper_plot.plt_subplot_vs_t(t, u_ctrl[0, :], u_ctrl[1, :], 
                        labels=labels, title=title, path=path)
        
    # Plot signals
    plt.figure(figsize=(15,8))
    plt.plot(t, s1t, '--', label=r'$s_{11}^{(1)}$')
    plt.plot(t, 1-s1t, '--', label=r'$s_{11}^{(2)}$')
    plt.plot(t, s4t, '--', label=r'$s_{22}^{(1)}$')
    plt.plot(t, 1-s4t, '--', label=r'$s_{22}^{(2)}$')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.xlabel(r'$t$')
    plt.ylabel(r'$s(t)$')
    
# %----------------------------------------- simulate ------------------------------------------% #
def simulate(scheduling_type="matrix",
             controller_type="VSP_lqr",
             model_uncertainty=False,
             T_END=25,
             plot=True,
             save_fig=False):
    # [SOLVER PARAMS]: Set IVP Solver parameters
    IVP_PARAM = initialize.init_ivp(T_END=T_END)

    # Construct Nonlinear System
    sys_non_linear = initialize.init_sys(sys_type="Nonlinear")
    
    # Construct Linearized Systems
    lin_systems = init_linearization(model_uncertainty=model_uncertainty)
    
    # Construct Controllers
    ctrls, x_cl0 = init_controllers(sys_non_linear=sys_non_linear, 
                                    sys_linearized=lin_systems, 
                                    ctrl_typ=controller_type)
    
    # Construct scheduling signals
    signals = init_scheduling_signals(scheduling_type=scheduling_type, DOF=2)
    
    # Construct gain scheduled system closed_loop dynamics
    sys_closed_loop = lambda t, x : gain_scheduled_closed_loop(sys=sys_non_linear, 
                                                               ctrls=ctrls, 
                                                               signals=signals, 
                                                               t=t, x=x)
    
    # Find time-domain response by integrating the ODE
    sol = integrate.solve_ivp(sys_closed_loop,
                              (IVP_PARAM.t_start, IVP_PARAM.t_end),
                              x_cl0.ravel(),
                              t_eval=IVP_PARAM.t_eval,
                              rtol=IVP_PARAM.rtol,
                              atol=IVP_PARAM.atol,
                              method=IVP_PARAM.method,
                              vectorized=True)
    
    # Plot results
    if plot:
        plot_results(sys_non_linear, sol, ctrls, signals, save_fig=save_fig)
    
    # Return solution
    return extract_states(sys_non_linear, sol, ctrls, signals)