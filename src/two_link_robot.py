# Purpose: Simulate the Control of the Two-Link Robot in Figure 2 using a single controller
# %------------------------------------------ Packages -------------------------------------------% #
import numpy as np

from scipy import integrate

from src import initialize
# %------------------------------------------ Functions ------------------------------------------% #
# Purpose: Closed Loop dynamics of the system
def closed_loop(sys, ctrl, t, x) -> np.array:
    """Closed-loop system."""
    # Split state
    x_sys  = x[:4] # [theta_1, theta_2, theta_dot_1, theta_dot_2]
    x_ctrl = x[4:] # [x_c]

    # Measurment
    y  = sys.g(x_sys)
    yp = sys.g_prewrap(x_sys)
    
    # Compute errors
    error = sys.trajectory.r_des(t) - yp
    error_dot = sys.trajectory.r_des_dot(t) - y
    
    # Compute control
    u_ctrl = ctrl.g_prewrap(error) + sys.bhat @ ctrl.g(x_ctrl, error_dot)
    
    # Advance controller state.
    x_dot_ctrl = ctrl.f(x_ctrl, error_dot)
    
    # Advance system state
    x_dot_sys = sys.f(x_sys, u_ctrl)

    # Concatenate state derivatives
    return np.concatenate((x_dot_sys, x_dot_ctrl)) # x_dot

def simulate(controller_type="VSP_lqr",
             sys_type="Nonlinear", 
             model_uncertainty=False,
             T_END=25,
             plot=True,
             save_fig=False):
    # Set IVP Solver parameters
    IVP_PARAM = initialize.init_ivp(T_END=T_END)

    # Construct system and controller
    sys, ctrl, x_cl0 = initialize.problem(controller_type=controller_type, 
                                          sys_type=sys_type, 
                                          model_uncertainty=model_uncertainty)
    
    # Construct system closed_loop
    sys_closed_loop = lambda t, x : closed_loop(sys, ctrl, t, x)
    
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
        sys.plot_results(sol, ctrl, save_fig)
        
    # Return solution
    return sys.extract_states(sol, ctrl)