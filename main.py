# Title   : Passivity-Based Gain-Scheduled Control with Scheduling Matrices
# Authors : Sepehr Moalemi and James Richard Forbes
# Code    : Minimum Code to Reproduce the Application Example in Section V
# %------------------------------------------ Packages -------------------------------------------% #
import argparse
import matplotlib.pyplot as plt

from src import paper_plot
from src import two_link_robot_compare 
# %-------------------------------------------- Main ---------------------------------------------% #
def main():
    # Set Figure Save Preferences
    paper_plot.set_fig_preferences()
    parser = argparse.ArgumentParser()
    parser.add_argument('--savefig', action='store_true', default=False)
    save_fig = parser.parse_args().savefig
    
    # Run Simulation
    two_link_robot_compare.simulate(sys_type="Nonlinear",
                                    controller_type="VSP_lqr",
                                    model_uncertainty=True,
                                    T_END=12,
                                    save_fig=save_fig)
# %--------------------------------------------- Run ---------------------------------------------% #
if __name__ == '__main__':
    print(f'{"Start":-^{50}}')
    main()
    plt.show()
    print(f'{"End":-^{50}}')