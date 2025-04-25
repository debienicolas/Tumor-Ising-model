import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import laplace
import matplotlib.cm as cm

class FrontInvasionSimulation:
    """
    Simulation of mean-field dynamics of front invasion in branching morphogenesis
    based on the Fisher-KPP equation system.
    """
    
    def __init__(self, length=100, dx=1.0, dt=0.1, D=1.0, rb=1.0, re=0.5, a0=1.0, 
                 save_frames=False, num_frames=100):
        """
        Initialize the simulation parameters.
        
        Parameters:
        -----------
        length : int
            Size of the spatial domain
        dx : float
            Spatial discretization step
        dt : float
            Time discretization step
        D : float
            Diffusion constant for active particles
        rb : float
            Branching rate for active particles
        re : float
            Production rate of inactive particles when active particles move
        a0 : float
            Saturation density for active particles
        save_frames : bool
            Flag to save animation frames
        num_frames : int
            Number of frames to save if save_frames is True
        """
        self.length = length
        self.dx = dx
        self.dt = dt
        self.D = D
        self.rb = rb
        self.re = re
        self.a0 = a0
        self.save_frames = save_frames
        self.num_frames = num_frames
        
        # Initialize spatial grid
        self.x = np.arange(0, length, dx)
        self.nx = len(self.x)
        
        # Initialize concentration fields
        self.a = np.zeros(self.nx)  # Active particles
        self.i = np.zeros(self.nx)  # Inactive particles
        
        # Set initial condition: single active walker at one side
        self.a[0:5] = self.a0  # Start with active particles at the left edge
        
        # Theoretical wave velocity
        self.theoretical_velocity = 2 * np.sqrt(D * rb)
        
        # For animation
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.line_a = None
        self.line_i = None
        
        # Track front position
        self.front_positions = []
        self.times = []
        
    def laplacian(self, field):
        """
        Compute the Laplacian of a field using a second-order finite difference scheme.
        """
        # Using scipy's laplace filter with proper scaling for dx
        return laplace(field) / (self.dx**2)
    
    def update(self):
        """
        Update the concentration fields according to the coupled PDEs.
        """
        # Calculate Laplacian of active particles
        lap_a = self.laplacian(self.a)
        
        # Update active particles (Fisher-KPP equation)
        da_dt = self.D * lap_a + self.rb * self.a * (1 - self.a / self.a0)
        self.a += da_dt * self.dt
        
        # Update inactive particles (slave to active particles)
        di_dt = self.re * self.a + self.rb * (self.a0 - self.a)**2
        self.i += di_dt * self.dt
        
    def track_front(self, threshold=0.5):
        """
        Track the position of the propagating front.
        """
        # Define the front as the position where a = threshold * a0
        front_indices = np.where(self.a >= threshold * self.a0)[0]
        if len(front_indices) > 0:
            return front_indices[-1] * self.dx
        return 0
    
    def run(self, total_time, track_interval=1.0):
        """
        Run the simulation for a specified time.
        
        Parameters:
        -----------
        total_time : float
            Total simulation time
        track_interval : float
            Time interval for tracking front position
        """
        steps = int(total_time / self.dt)
        track_steps = int(track_interval / self.dt)
        
        for step in range(steps):
            self.update()
            
            # Track front position at specified intervals
            if step % track_steps == 0:
                t = step * self.dt
                front_pos = self.track_front()
                self.front_positions.append(front_pos)
                self.times.append(t)
        
    def animate(self, total_time, interval=50):
        """
        Create an animation of the simulation.
        
        Parameters:
        -----------
        total_time : float
            Total simulation time
        interval : int
            Interval between animation frames in milliseconds
        """
        steps = int(total_time / self.dt)
        
        # Create figure and axes
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Initialize lines
        self.line_a, = self.ax1.plot(self.x, self.a, 'b-', label='Active (a)')
        self.line_i, = self.ax2.plot(self.x, self.i, 'r-', label='Inactive (i)')
        
        # Set axis labels and titles
        self.ax1.set_title('Mean-Field Dynamics of Front Invasion')
        self.ax1.set_ylabel('Active Particles (a)')
        self.ax1.set_ylim(0, self.a0 * 1.1)
        self.ax1.legend()
        
        self.ax2.set_xlabel('Position (x)')
        self.ax2.set_ylabel('Inactive Particles (i)')
        self.ax2.legend()
        
        # Create frames for animation if saving
        if self.save_frames:
            frame_step = max(1, steps // self.num_frames)
            for step in range(0, steps, frame_step):
                # Update simulation
                for _ in range(frame_step):
                    self.update()
                
                # Save frame
                plt.savefig(f'frame_{step//frame_step:04d}.png', dpi=100)
                
                # Update plot data
                self.line_a.set_ydata(self.a)
                self.line_i.set_ydata(self.i)
                self.ax2.set_ylim(0, max(1, np.max(self.i) * 1.1))
                
                # Update title with time information
                time = step * self.dt
                theoretical_pos = self.theoretical_velocity * time
                self.ax1.set_title(f'Time: {time:.1f}, Theoretical Front Position: {theoretical_pos:.1f}')
                
                plt.tight_layout()
                plt.draw()
        else:
            # Create animation function
            def animate(frame):
                # Update simulation multiple times per frame for smoother animation
                for _ in range(5):
                    self.update()
                
                # Update plot data
                self.line_a.set_ydata(self.a)
                self.line_i.set_ydata(self.i)
                self.ax2.set_ylim(0, max(1, np.max(self.i) * 1.1))
                
                # Update title with time information
                time = frame * 5 * self.dt
                theoretical_pos = self.theoretical_velocity * time
                self.ax1.set_title(f'Time: {time:.1f}, Theoretical Front Position: {theoretical_pos:.1f}')
                
                return self.line_a, self.line_i
            
            # Create animation
            ani = FuncAnimation(self.fig, animate, frames=steps//5, interval=interval, blit=True)
            plt.tight_layout()
            plt.show()
            
            return ani
    
    def plot_front_velocity(self):
        """
        Plot the front position over time and calculate the velocity.
        """
        if not self.front_positions:
            print("No front position data. Run the simulation first.")
            return
        
        # Calculate numerical velocity by linear regression
        times_array = np.array(self.times)
        positions_array = np.array(self.front_positions)
        
        # Use linear regression to find the velocity (slope)
        coeffs = np.polyfit(times_array, positions_array, 1)
        numerical_velocity = coeffs[0]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        plt.plot(self.times, self.front_positions, 'bo-', label='Front Position')
        plt.plot(times_array, np.polyval(coeffs, times_array), 'r--', 
                 label=f'Linear Fit (v = {numerical_velocity:.3f})')
        
        # Add theoretical velocity
        plt.axline((0, 0), slope=self.theoretical_velocity, color='g', linestyle='-.',
                  label=f'Theoretical (v = {self.theoretical_velocity:.3f})')
        
        plt.xlabel('Time')
        plt.ylabel('Front Position')
        plt.title('Front Propagation Analysis')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        print(f"Theoretical velocity: {self.theoretical_velocity:.3f}")
        print(f"Numerical velocity: {numerical_velocity:.3f}")
        
    def plot_final_state(self):
        """
        Plot the final state of active and inactive particles.
        """
        plt.figure(figsize=(10, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(self.x, self.a, 'b-')
        plt.ylabel('Active Particles (a)')
        plt.title('Final State')
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(self.x, self.i, 'r-')
        plt.xlabel('Position (x)')
        plt.ylabel('Inactive Particles (i)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def create_spacetime_plot(self, total_time, dt_record=1.0):
        """
        Create a space-time plot of the active particle concentration.
        
        Parameters:
        -----------
        total_time : float
            Total simulation time
        dt_record : float
            Time interval for recording the state
        """
        steps = int(total_time / self.dt)
        record_steps = int(dt_record / self.dt)
        num_records = steps // record_steps + 1
        
        # Initialize active particle history array
        a_history = np.zeros((num_records, self.nx))
        t_history = np.zeros(num_records)
        
        # Reset initial condition
        self.a = np.zeros(self.nx)
        self.i = np.zeros(self.nx)
        self.a[0:5] = self.a0
        
        # Record initial state
        a_history[0] = self.a.copy()
        t_history[0] = 0
        
        # Run simulation and record states
        record_idx = 1
        for step in range(steps):
            self.update()
            
            if (step + 1) % record_steps == 0:
                a_history[record_idx] = self.a.copy()
                t_history[record_idx] = (step + 1) * self.dt
                record_idx += 1
        
        # Create space-time plot
        plt.figure(figsize=(10, 8))
        plt.imshow(a_history, aspect='auto', origin='lower', 
                  extent=[0, self.length, 0, total_time],
                  cmap=cm.viridis, interpolation='nearest')
        plt.colorbar(label='Active Particle Concentration (a)')
        plt.xlabel('Position (x)')
        plt.ylabel('Time (t)')
        plt.title('Space-Time Plot of Active Particle Front Invasion')
        
        # Add theoretical front velocity line
        x_vals = np.array([0, self.theoretical_velocity * total_time])
        t_vals = np.array([0, total_time])
        plt.plot(x_vals, t_vals, 'r--', linewidth=2, 
                label=f'Theoretical Velocity (v = {self.theoretical_velocity:.3f})')
        plt.legend()
        
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Create simulation instance
    sim = FrontInvasionSimulation(
        length=100,     # Domain size
        dx=0.5,         # Spatial step
        dt=0.01,        # Time step
        D=1.0,          # Diffusion constant
        rb=1.0,         # Branching rate
        re=0.1,         # Production rate of inactive particles
        a0=1.0          # Saturation density
    )
    
    # Run simulation
    total_time = 20.0
    sim.run(total_time)
    
    # Plot final state
    sim.plot_final_state()
    
    # Analyze front velocity
    sim.plot_front_velocity()
    
    # Create space-time plot
    sim.create_spacetime_plot(total_time, dt_record=0.2)
    
    # Create animation (comment out if not needed)
    # ani = sim.animate(total_time, interval=50)
    
    # Uncomment to save the animation
    # from matplotlib.animation import FFMpegWriter
    # writer = FFMpegWriter(fps=15)
    # ani.save('front_invasion.mp4', writer=writer)