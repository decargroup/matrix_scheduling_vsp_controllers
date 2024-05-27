# Purpose: Initialize the system and controller for the problem
# %------------------------------------------ Packages -------------------------------------------% #
import numpy as np

from typing import NamedTuple, Tuple

from src import controller
from src import system
from src import tracking
# %----------------------------------- IVP Solver Parameters ------------------------------------% #
# Purpose: Immutable Struct for solve_ivp params
class num_integ_settings(NamedTuple):
    t_start : float
    t_end   : float
    t_eval  : np.array
    rtol    : float
    atol    : float
    method  : str
    
def init_ivp(DT=1e-3, T_START=0, T_END=30, 
             RTOL=1e-5, ATOL=1e-5,
             METHOD='RK45') -> NamedTuple:
    # Solve_ivp: Parameters
    T_EVAL = np.arange(T_START, T_END, DT)
    IVP_PARAM = num_integ_settings(t_start=T_START,
                                   t_end=T_END,
                                   t_eval=T_EVAL,
                                   rtol=RTOL,
                                   atol=ATOL,
                                   method=METHOD)
    return IVP_PARAM

# %--------------------------------------- Tracking ---------------------------------------------% #
def init_trajectory() -> tracking.Trajectory:
    tk = [[ 0.0, np.array([[ -np.pi/2], [5*np.pi/6]])],
            [ 0.5, np.array([[ -np.pi/2], [5*np.pi/6]])], 
            [ 1.0, np.array([[ -np.pi/3], [  np.pi/2]])], 
            [ 2.0, np.array([[ -np.pi/3], [  np.pi/2]])], 
            [ 3.0, np.array([[  np.pi/4], [  np.pi/3]])], 
            [ 5.0, np.array([[  np.pi/3], [  np.pi/4]])], 
            [ 6.0, np.array([[  np.pi/2], [ -np.pi/3]])], 
            [ 6.5, np.array([[  np.pi/2], [ -np.pi/3]])], 
            [ 7.5, np.array([[5*np.pi/6], [ -np.pi/2]])], 
            [ 8.5, np.array([[5*np.pi/6], [ -np.pi/2]])]]
    return tracking.Trajectory(tk=tk)

# %------------------------------------ System Parameters --------------------------------------% #
# Purpose: Return only necessary parameters for robot system
def init_sys(sys_type="Nonlinear", 
             r_lin=None,
             model_uncertainty=False) -> system.TwoLinkManipulatorRobot:
    # TwoLinkRobot: Physical properties
    M_1, M_2 = 0.4, 0.9   # kg, mass
    L_1, L_2 = 1.1, 0.85  # m
    
    # TwoLinkRobot: Add model uncertainty
    if model_uncertainty:
        print("Adding Model Uncertainty!")
        M_1 = 1.10 * M_1
        M_2 = 1.10 * M_2
        L_1 = 0.98 * L_1
        L_2 = 0.98 * L_2
    
    # TwoLinkRobot: initialize trajectory to track
    trajectory = init_trajectory()
    
    # TwoLinkRobot: Initial condition
    x_sys0 = trajectory.x0
    
    # TwoLinkRobot: linearization point
    if r_lin is None:
        r_lin = trajectory.r_end
    
    # Initiate TwoLinkRobot instance to create an TwoLinkRobot object.
    return system.TwoLinkManipulatorRobot(m1=M_1, L1=L_1, 
                                          m2=M_2, L2=L_2, 
                                          x_sys0=x_sys0, 
                                          trajectory=trajectory, r_lin=r_lin,
                                          sys_type=sys_type)

# %------------------------------------ Controller Picker --------------------------------------% #0.98
def init_controller(sys, ctrl_typ="VSP_lqr") -> controller.controller:
    match ctrl_typ:
        case "VSP_lqr": ctrl = VSP_lqr(sys)
        case _: raise NotImplementedError
    return ctrl

# %-------------------------------------- VSP Controllers --------------------------------------% #
def VSP_lqr(sys) -> controller.controller:
    # [Controller]: VSP_lqr Proportional control prewrap and gain
    Kp = np.diag([35, 35])
    
    # [Controller]: VSP_lqr State and input weight matrices
    u1_max = 15
    u2_max = 15
    
    theta_1_max = 1/3
    theta_2_max = 1/4
    theta_1_dot_max = 180/180 * np.pi
    theta_2_dot_max = 180/180 * np.pi
    
    Q = np.diag([1/theta_1_max, 1/theta_2_max, 1/theta_1_dot_max, 1/theta_2_dot_max])**2
    R = np.diag([1/u1_max, 1/u2_max])**2
    
    # VSP_lqr: Initial condition
    x_ctrl0 = np.array([[0, 0, 0, 0]]).T
    
    return controller.VSP_lqr(sys=sys, 
                              Q=Q, R=R, 
                              Kp=Kp, 
                              x_ctrl0=x_ctrl0, 
                              FEED_THROUGH=1e-4)

# %------------------------------------- Problem Parameters ------------------------------------% #
# Purpose: Return only necessary parameters for the problem
def problem(controller_type="VSP_lqr",
            sys_type="Nonlinear", 
            model_uncertainty=False) -> Tuple[system.TwoLinkManipulatorRobot, 
                                            controller.controller, 
                                            np.array]:
    # Initialize True system with potential disturbance but no model uncertainty 
    sys = init_sys(sys_type=sys_type)
    
    # Initialize Measured system subject to model uncertainty
    sys_measured = init_sys(sys_type=sys_type, model_uncertainty=model_uncertainty)
    
    # Initialize controller
    ctrl = init_controller(sys=sys_measured, ctrl_typ=controller_type)
    
    # Set up closed-loop IC.
    x_cl0 = np.concatenate((sys.x_sys0, ctrl.x_ctrl0))
    
    return sys, ctrl, x_cl0