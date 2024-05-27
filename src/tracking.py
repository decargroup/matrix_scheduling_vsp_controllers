# Purpose: Trajectory Generation using (22)
# %------------------------------------------ Packages -------------------------------------------% #
import numpy as np

from src import paper_plot
# %------------------------------------------ Class ------------------------------------------% #
# Purpose: Trajectory class to generate desired trajectory
class Trajectory():
    def __init__(self, tk, DOF=2) -> None:
        """
            if tk is a collection of time and position pairs:
                tk = [(t_0, theta_0), (t_1, theta_1), ...]     
                then calculate smooth r_des and r_des_dot
            if tk is just a final state:
                tk = [theta_0, theta_1]
                then just set a constant r_des 
                and r_des_dot = 0
        """
        self.DOF = DOF
        self.tk = tk
        
        if isinstance(tk, list):
            self.x0 = np.vstack((self.r_des_smooth(0), 
                                 self.r_des_dot_smooth(0)))
            self.r_end     = tk[-1][1]  
            self.r_des     = self.r_des_smooth
            self.r_des_dot = self.r_des_dot_smooth
        else:
            self.x0 = np.vstack((tk, np.zeros_like(tk)))
            self.r_end     = tk
            self.r_des     = lambda _: tk
            self.r_des_dot = lambda _: np.zeros_like(tk)
        
    # %---------------------------------------- Methods --------------------------------------% #
    # Purpose: Generate [theta_1, theta_2] at time t
    def r_des_smooth(self, t):
        # If t is greater than the last time in tk, return r_end
        if t >= self.tk[-1][0]:
            return self.r_end
        
        # Otherwise, find the segment that t is in and interpolate
        for i in range(len(self.tk)-1):
            start, theta_start = self.tk[i]
            end, theta_end     = self.tk[i + 1]
            if start <= t <= end:
                alpha = (t - start) / (start - end)
                return alpha**3 * (10 + 15*alpha + 6*alpha**2) * (theta_start - theta_end) + theta_start

    # Purpose: Generate [theta_1_dot, theta_2_dot] at time t
    def r_des_dot_smooth(self, t):
        # If t is greater than the last time in tk, return 0 since r_des is constant
        if t >= self.tk[-1][0]:
            return np.zeros_like(self.r_end)
        
        # Otherwise, find the segment that t is in and interpolate
        for i in range(len(self.tk)-1):
            start, theta_start = self.tk[i]
            end, theta_end     = self.tk[i + 1]
            if start <= t <= end:
                return 30 * (t - start)**2 * (t - end)**2 * (theta_start - theta_end) / (start - end)**5
        
    # Purpose: Generate [theta_1, theta_2] and [theta_1_dot, theta_2_dot] for time grid
    def generate_trajectory(self, t_grid, type="rad"):
        match type:
            case "deg":
                factor = 180 / np.pi
            case "rad":
                factor = 1
            case _:
                raise ValueError(f"Invalid type: {type}. Must be 'deg' or 'rad'.") 
        rs = []
        r_dots = []
        for j in range(self.DOF):
            rs.append([self.r_des(ti)[j, 0] * factor for ti in t_grid])
            r_dots.append([self.r_des_dot(ti)[j, 0] * factor for ti in t_grid])
        return rs, r_dots
    
    # Purpose: Plot trajectory vs time
    def plt_trajectory_vs_t(self, dt=1e-3, type="rad"):
        t = np.arange(self.tk[0][0], self.tk[-1][0], dt)
        rs, r_dots = self.generate_trajectory(t, type)
        for i in range(self.DOF):
            paper_plot.plt_subplot_vs_t(t, 
                                        rs[i], 
                                        r_dots[i],
                                        title="Trajectory",
                                        labels=[rf'$\theta_{i+1}$', rf'$\dot{{\theta}}_{i+1}$'],
                                        legends=[r'$r_{desired}(t)$', r'$\dot{r}_{desired}(t)$'])