# Purpose: Setup plotting settings and Plot Subplot
# %------------------------------------------ Packages -------------------------------------------% #
import matplotlib.pyplot as plt
# %------------------------------------------ Functions ------------------------------------------% #
# Purpose: Set global figure preferences
def set_fig_preferences():
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=25)
    plt.rc('lines', lw=6, c='k')
    plt.rc('axes', grid=True)
    plt.rc('grid', ls='--', lw=0.5, c='0.5')
    plt.rcParams['axes.grid.which'] = 'both'
    plt.rcParams["figure.figsize"] = (12, 10)
    plt.rcParams['legend.framealpha'] = 1.0
    plt.rcParams.update({
        'savefig.bbox':'tight',
        'savefig.pad_inches': 0.07,
        'legend.loc': 'best',
        'legend.fontsize': 20})
    plt.rcParams['savefig.dpi'] = 400
# Purpose: Plot Subplot
def plt_subplot_vs_t(t, y1, y2, labels, title="", 
                     legends=None, trajectory=None, entry=0, path=None) -> None:
    fig, axs = plt.subplots(2, 1)
    
    # Format axes
    for indx, a in enumerate(axs):
        a.set_xlabel(r'$t(s)$')
        axs[indx].set_ylabel(labels[indx])
    
    # Plot data
    axs[0].plot(t, y1)
    axs[1].plot(t, y2)
    
    # Add title
    axs[0].set_title(title)
    
    # Add legend
    if legends is not None:
        for ax, l in zip(axs, legends):
            ax.legend([l],loc='upper right')
    
    # Add desired line
    if trajectory is not None:
        rs, r_dots = trajectory.generate_trajectory(t, "deg")
        axs[0].plot(t, rs[entry], color='red', linestyle='dotted', label=r'$r_{desired}(t)$')
        axs[1].plot(t, r_dots[entry], color='red', linestyle='dotted', label=r'$\dot{r}_{desired}(t)$')
        axs[0].legend(loc='upper right')
        axs[1].legend(loc='upper right')
    
    if path is not None:
        fig.savefig(path.joinpath(".png"))