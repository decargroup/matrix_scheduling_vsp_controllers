# Purpose: Generate Plots in Figure 3 and Figures 5-7
# %------------------------------------------ Packages -------------------------------------------% #
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from src import two_link_robot
from src import two_link_robot_scheduling
# %------------------------------------------ Plots ----------------------------------------------% #
def simulate(sys_type= "Nonlinear",
             controller_type="VSP_lqr",
             model_uncertainty=False,
             T_END=12.5,
             save_fig=False):
    # Print padding
    padding = 50
    
    # Check if the directory exists
    if save_fig:
        Path('./Figures').mkdir(parents=True, exist_ok=True)
    
    # Results: t, th, dot_th, u_ctrl, error_th, error_th_dot
    print(f'{"Simulating No Scheduling " + controller_type:-^{padding}}')
    single_lqr_results = two_link_robot.simulate(controller_type=controller_type,
                                                 sys_type=sys_type,
                                                 model_uncertainty=model_uncertainty,
                                                 T_END=T_END,
                                                 plot=False)
    
    # Results: t, th, dot_th, u_ctrl, error_th, error_th_dot, trajectory
    print(f'{"Simulating Scalar Scheduling " + controller_type:-^{padding}}')
    scalar_scheduled_lqr_results = two_link_robot_scheduling.simulate(controller_type=controller_type,
                                                                      scheduling_type="scalar",
                                                                      model_uncertainty=model_uncertainty,
                                                                      T_END=T_END,
                                                                      plot=False)
    
    print(f'{"Simulating Matrix Scheduling " + controller_type:-^{padding}}')
    matrix_scheduled_lqr_results = two_link_robot_scheduling.simulate(controller_type=controller_type,
                                                                      scheduling_type="matrix",
                                                                      model_uncertainty=model_uncertainty,
                                                                      T_END=T_END,
                                                                      plot=False)
    
    # Generate Tracking Trajectory
    print(f'{"Generating Trajectory":-^{padding}}')
    trajectory = scalar_scheduled_lqr_results[6]
    rs, r_dots = trajectory.generate_trajectory(single_lqr_results[0], type="deg")
    
    # Plot theta and theta_dot
    t = single_lqr_results[0]
    results_th           = [rs, single_lqr_results[1], scalar_scheduled_lqr_results[1], matrix_scheduled_lqr_results[1]]
    results_dot_th       = [r_dots, single_lqr_results[2], scalar_scheduled_lqr_results[2], matrix_scheduled_lqr_results[2]]
    results_u_ctrl       = [single_lqr_results[3], scalar_scheduled_lqr_results[3], matrix_scheduled_lqr_results[3]]
    results_error_th     = [single_lqr_results[4], scalar_scheduled_lqr_results[4], matrix_scheduled_lqr_results[4]]
    results_error_th_dot = [single_lqr_results[5], scalar_scheduled_lqr_results[5], matrix_scheduled_lqr_results[5]]

    # %----------------------------------------------- Trajectory ----------------------------------------------% #
    colors = ['k', '#377eb8', '#ff7f00','#e41a1c']
    line_styles = ['-', ':', '-.', '--']
    labels = ["Desired", "Unscheduled", "Scalar GS", "Matrix GS"]
    lws    = [6, 5, 5, 4]
    fig, axs = plt.subplots(2, 2, sharex=True, figsize=(12, 8))
    for i in range(len(axs)):
        for j in range(len(results_th)):
            axs[0, i].plot(t, results_th[j][i], c=colors[j], ls=line_styles[j], lw=lws[j], label=labels[j])
            axs[1, i].plot(t, results_dot_th[j][i], c=colors[j], ls=line_styles[j], lw=lws[j],  label=labels[j])
        axs[0, i].set_xlim([0, T_END])
        axs[1, i].set_xlim([0, T_END])
        axs[0, i].set_ylabel(rf'$\theta_{i+1}(t)$ [deg]')
        axs[1, i].set_ylabel(rf'$\dot{{\theta}}_{i+1}(t)$ [deg/s]')
        axs[1, i].set_xlabel(r'Time $[s]$')
    fig.align_labels()
    axs[1][1].legend(loc='lower right', bbox_to_anchor=(1.0, 0.0), borderaxespad=0.05) 
    
    plt.subplots_adjust(wspace=0.4)
    plt.subplots_adjust(hspace=0.1)
    if save_fig:
        file_name = "trajctory"
        fig.savefig('./Figures/' + file_name + ".pdf")
    
    # %----------------------------------------------- Error ----------------------------------------------% #
    colors = ['#377eb8', '#ff7f00','#e41a1c']
    lws    = [5, 5, 4]
    line_styles = [':', '-.', '-']
    labels = ["No GS", "Scalar GS", "Matrix GS"]
    col_label = ["theta_1", "theta_2"]
    rmse_errors = []
    rmse_errors_dot = []
    fig, axs = plt.subplots(2, 2, sharex=True, figsize=(12, 9))
    for i in range(len(axs)):
        for j in range(len(results_error_th)):
            err = [e * 180 / np.pi for e in results_error_th[j][i]]
            err_dot = [e * 180 / np.pi for e in results_error_th_dot[j][i]]
            axs[0, i].plot(t, err, c=colors[j], ls=line_styles[j], lw=lws[j], label=labels[j])
            axs[1, i].plot(t, err_dot, c=colors[j], ls=line_styles[j], lw=lws[j], label=labels[j])
            rmse_errors+=[f"RMSE Error of {col_label[i]} for {labels[j]}: {np.sqrt(np.mean(np.square(err))):.4f}"]
            rmse_errors_dot+=[f"RMSE Error of {'dot ' + col_label[i]} for {labels[j]}: {np.sqrt(np.mean(np.square(err_dot))):.4f}"]
        axs[0, i].set_xlim([0, T_END])
        axs[1, i].set_xlim([0, T_END]) 
        axs[0, i].set_ylabel(rf'$e_{i+1}(t)$ [deg]')
        axs[1, i].set_ylabel(rf'$\dot{{e}}_{i+1}(t)$ [deg/s]')
        axs[1, i].set_xlabel(r'Time $[s]$')
    fig.align_labels()
    axs[1][1].legend(loc='lower right', bbox_to_anchor=(1.0, 0.0), borderaxespad=0.05)
    plt.subplots_adjust(wspace=0.35)
    plt.subplots_adjust(hspace=0.1)
    if save_fig:
        file_name = "error"
        fig.savefig('./Figures/' + file_name + ".pdf")
    
    # %---------------------------------------------- Torque ----------------------------------------------% #
    col_label = [r"$\tau_1$", r"$\tau_1$"]
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 6.5))
    for i in range(len(axs)):
        for j in range(len(results_error_th)):
            axs[i].plot(t, results_u_ctrl[j][i], c=colors[j], ls=line_styles[j], lw=lws[j], label=labels[j])
        axs[i].set_xlim([0, T_END])
        axs[i].set_ylabel(rf'$\tau_{i+1}(t)$ [N$\,$m]')
    axs[1].set_xlabel(r'Time $[s]$')
    fig.align_labels()
    axs[1].legend(loc='lower right', bbox_to_anchor=(1.0, 0.0), borderaxespad=0.05)
    plt.subplots_adjust(hspace=0.1)
    if save_fig:
        file_name = "control_effort"
        fig.savefig('./Figures/' + file_name + ".pdf")
        
    # %----------------------------------------------- Signals ----------------------------------------------% #
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
    xnew = np.arange(trajectory.tk[0][0], 8.5, 0.01)
    lw=4
    fig = plt.figure(figsize=(12, 4))
    plt.plot(xnew, [s1(t) for t in xnew], c='#377eb8', ls='-', lw=lw, label=r"$s_1$")
    plt.plot(xnew, [s2(t) for t in xnew], c='#ff7f00', ls='--', lw=lw, label=r"$s_2$")
    plt.plot(xnew, [s3(t) for t in xnew], c='#e41a1c', ls='-.',lw=lw,  label=r"$s_3$")
    plt.legend(loc='center right')
    plt.xlabel(r'Time $[s]$')
    plt.ylabel(r'Scheduling Signals')
    plt.xlim(0, xnew[-1])
    if save_fig:
        fig.savefig('./Figures/scheduling_signal.pdf')
        
    # %----------------------------------------------- Print Errors ----------------------------------------------% #
    print(f'{"Printing Table Values":-^{padding}}')
    # Print RMSE Errors
    for rmse in rmse_errors:
        print(rmse)
    print("")
    # Print RMSE Error Rates
    for rmse in rmse_errors_dot:
        print(rmse)
    print("")