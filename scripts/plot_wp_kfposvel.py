#!/usr/bin/python
"""
Created by Teddy Herrera, 11 Sept 2025
Plots for Kalman Filter Performance Analysis

Usage:
    python plot_wp_kfpos.py <Waypoints> <Data Log>

Requirements from HW:
- State estimates vs. true states (separate subplots for x/vx/ax and y/vy/ay)
- Trace of covariance P
- Estimation errors (separate subplots for x/vx/ax and y/vy/ay)
- Kalman gains
- XY trajectory (true vs. estimated) with measurements and uncertainty ellipses
"""

import sys
import numpy as np
import matplotlib
import scipy.stats as stats

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import filterpy.stats as fstats


def load_waypoints(filename):
    """Load waypoints from file"""
    waypoints = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    x, y = map(float, line.split(','))
                    waypoints.append([x, y])
            if len(waypoints) >= 2:
                if waypoints[0] != waypoints[-1]:
                    waypoints.append(waypoints[0])
                print("Loaded %d waypoints" % len(waypoints))
                return waypoints
    except Exception as e:
        print('Error loading waypoints: %s' % str(e))
        return None


def plot_error_distributions(x_err, y_err, vx_err, vy_err, ax_err, ay_err, x_res, y_res, vx_res, vy_res):
    """
    Plot error distributions with histograms and fitted PDFs for position, velocity, and acceleration

    Parameters:
    x_err, y_err: position estimation errors (true - estimated)
    vx_err, vy_err: velocity estimation errors (true - estimated)
    ax_err, ay_err: acceleration estimation errors (true - estimated)
    x_res, y_res: position residuals (measurement - estimated)
    vx_res, vy_res: velocity residuals (measurement - estimated)
    """
    fig = plt.figure(figsize=(16, 12), facecolor='white')
    
    # Fit normal distributions to all error types
    mean_x_err, std_x_err = stats.norm.fit(x_err)
    mean_y_err, std_y_err = stats.norm.fit(y_err)
    mean_vx_err, std_vx_err = stats.norm.fit(vx_err)
    mean_vy_err, std_vy_err = stats.norm.fit(vy_err)
    mean_ax_err, std_ax_err = stats.norm.fit(ax_err)
    mean_ay_err, std_ay_err = stats.norm.fit(ay_err)
    mean_x_res, std_x_res = stats.norm.fit(x_res)
    mean_y_res, std_y_res = stats.norm.fit(y_res)
    mean_vx_res, std_vx_res = stats.norm.fit(vx_res)
    mean_vy_res, std_vy_res = stats.norm.fit(vy_res)
    
    # Create fitting ranges
    x_err_range = np.linspace(np.min(x_err), np.max(x_err), 1000)
    y_err_range = np.linspace(np.min(y_err), np.max(y_err), 1000)
    vx_err_range = np.linspace(np.min(vx_err), np.max(vx_err), 1000)
    vy_err_range = np.linspace(np.min(vy_err), np.max(vy_err), 1000)
    ax_err_range = np.linspace(np.min(ax_err), np.max(ax_err), 1000)
    ay_err_range = np.linspace(np.min(ay_err), np.max(ay_err), 1000)
    x_res_range = np.linspace(np.min(x_res), np.max(x_res), 1000)
    y_res_range = np.linspace(np.min(y_res), np.max(y_res), 1000)
    vx_res_range = np.linspace(np.min(vx_res), np.max(vx_res), 1000)
    vy_res_range = np.linspace(np.min(vy_res), np.max(vy_res), 1000)
    
    # Generate fitted PDFs
    fitted_pdf_x_err = stats.norm.pdf(x_err_range, mean_x_err, std_x_err)
    fitted_pdf_y_err = stats.norm.pdf(y_err_range, mean_y_err, std_y_err)
    fitted_pdf_vx_err = stats.norm.pdf(vx_err_range, mean_vx_err, std_vx_err)
    fitted_pdf_vy_err = stats.norm.pdf(vy_err_range, mean_vy_err, std_vy_err)
    fitted_pdf_ax_err = stats.norm.pdf(ax_err_range, mean_ax_err, std_ax_err)
    fitted_pdf_ay_err = stats.norm.pdf(ay_err_range, mean_ay_err, std_ay_err)
    fitted_pdf_x_res = stats.norm.pdf(x_res_range, mean_x_res, std_x_res)
    fitted_pdf_y_res = stats.norm.pdf(y_res_range, mean_y_res, std_y_res)
    fitted_pdf_vx_res = stats.norm.pdf(vx_res_range, mean_vx_res, std_vx_res)
    fitted_pdf_vy_res = stats.norm.pdf(vy_res_range, mean_vy_res, std_vy_res)
    
    # X Position errors subplot
    plt.subplot(3, 2, 1)
    plt.hist(x_err, bins=30, density=True, alpha=0.7, color='lightblue',
             label=r'$e_{\hat{x}}$ histogram')
    plt.plot(x_err_range, fitted_pdf_x_err, 'r-', linewidth=2,
             label=r'$e_{\hat{x}}=x_{true}-\hat{x}$')
    plt.plot(x_res_range, fitted_pdf_x_res, 'c-.', linewidth=2,
             label=r'$r_x=z_x - \hat{x}$')
    
    plt.ylabel('pdf')
    plt.title(r'X Position Error Distribution:\n  $\mu$ = {:.4f}m,   $\sigma$ = {:.4f}m'.format(mean_x_err, std_x_err))
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.ylim([0, None])
    
    # Y Position errors subplot
    plt.subplot(3, 2, 2)
    plt.hist(y_err, bins=30, density=True, alpha=0.7, color='lightgreen',
             label=r'$e_{\hat{y}}$ histogram')
    plt.plot(y_err_range, fitted_pdf_y_err, 'r-', linewidth=2,
             label=r'$e_{\hat{y}}=y_{true}-\hat{y}$')
    plt.plot(y_res_range, fitted_pdf_y_res, 'c-.', linewidth=2,
             label=r'$r_y=z_y - \hat{y}$')
    
    plt.ylabel('pdf')
    plt.title(r'Y Position Error Distribution:\n  $\mu$ = {:.4f}m,   $\sigma$ = {:.4f}m'.format(mean_y_err, std_y_err))
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.ylim([0, None])
    
    # X velocity errors subplot
    plt.subplot(3, 2, 3)
    plt.hist(vx_err, bins=30, density=True, alpha=0.7, color='lightcoral',
             label=r'$e_{\dot{\hat{x}}}$ histogram')
    plt.plot(vx_err_range, fitted_pdf_vx_err, 'r-', linewidth=2,
             label=r'$e_{\dot{\hat{x}}}=\dot{x}_{true}-\dot{\hat{x}}$')
    plt.plot(vx_res_range, fitted_pdf_vx_res, 'c-.', linewidth=2,
             label=r'$r_{\dot{x}}=z_{\dot{x}} - \dot{\hat{x}}$')
    
    plt.ylabel('pdf')
    plt.title(
        r'X Velocity Error Distribution:\n  $\mu$ = {:.4f}m/s,   $\sigma$ = {:.4f}m/s'.format(mean_vx_err, std_vx_err))
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.ylim([0, None])
    
    # Y velocity errors subplot
    plt.subplot(3, 2, 4)
    plt.hist(vy_err, bins=30, density=True, alpha=0.7, color='plum',
             label=r'$e_{\dot{\hat{y}}}$ histogram')
    plt.plot(vy_err_range, fitted_pdf_vy_err, 'r-', linewidth=2,
             label=r'$e_{\dot{\hat{y}}}=\dot{y}_{true}-\dot{\hat{y}}$')
    plt.plot(vy_res_range, fitted_pdf_vy_res, 'c-.', linewidth=2,
             label=r'$r_{\dot{y}}=z_{\dot{y}} - \dot{\hat{y}}$')
    
    plt.ylabel('pdf')
    plt.title(
        r'Y Velocity Error Distribution:\n  $\mu$ = {:.4f}m/s,   $\sigma$ = {:.4f}m/s'.format(mean_vy_err, std_vy_err))
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.ylim([0, None])
    
    # X acceleration errors subplot
    plt.subplot(3, 2, 5)
    plt.hist(ax_err, bins=30, density=True, alpha=0.7, color='orange',
             label=r'$e_{\ddot{\hat{x}}}$ histogram')
    plt.plot(ax_err_range, fitted_pdf_ax_err, 'r-', linewidth=2,
             label=r'$e_{\ddot{\hat{x}}}=\ddot{x}_{true}-\ddot{\hat{x}}$')
    
    plt.xlabel('Error (m/s^2)')
    plt.ylabel('pdf')
    plt.title(r'X Acceleration Error Distribution:\n  $\mu$ = {:.4f}m/s^2,   $\sigma$ = {:.4f}m/s^2'.format(mean_ax_err,
                                                                                                         std_ax_err))
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.ylim([0, None])
    
    # Y acceleration errors subplot
    plt.subplot(3, 2, 6)
    plt.hist(ay_err, bins=30, density=True, alpha=0.7, color='gold',
             label=r'$e_{\ddot{\hat{y}}}$ histogram')
    plt.plot(ay_err_range, fitted_pdf_ay_err, 'r-', linewidth=2,
             label=r'$e_{\ddot{\hat{y}}}=\ddot{y}_{true}-\ddot{\hat{y}}$')
    
    plt.xlabel('Error (m/s^2)')
    plt.ylabel('pdf')
    plt.title(r'Y Acceleration Error Distribution:\n  $\mu$ = {:.4f}m/s^2,   $\sigma$ = {:.4f}m/s^2'.format(mean_ay_err,
                                                                                                         std_ay_err))
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.ylim([0, None])
    
    plt.tight_layout()
    plt.savefig("kfposvel_error_distributions.png", dpi=300, bbox_inches='tight')
    print("Error distributions plot saved as 'error_distributions.png'")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python plot_wp_kfpos.py waypoint_list.txt log_mydata.txt")
        sys.exit(1)
    
    waypoint_file = sys.argv[1]
    log_file = sys.argv[2]
    
    # Load waypoints
    waypoints = load_waypoints(waypoint_file)
    if waypoints is None:
        print("Failed to load waypoints")
        sys.exit(1)
    
    # Load data
    try:
        with open(log_file, 'r') as f:
            data = np.genfromtxt(f, delimiter=',', names=True, comments=';')
    except Exception as exc:
        print("Error loading data:", exc)
        sys.exit(1)
    
    print("Data columns:", data.dtype.names)
    
    # Extract data
    time = data['SimTime']
    
    # True states
    x_true = data['X_true']
    y_true = data['Y_true']
    heading_true = data['Heading_true']
    vx_true = data['Vx_true']
    vy_true = data['Vy_true']
    
    # Estimated states (now includes acceleration)
    x_est = data['X_est']
    y_est = data['Y_est']
    heading_est = data['Heading_est']
    vx_est = data['Vx_est']
    vy_est = data['Vy_est']
    ax_est = data['Ax_est']  # New acceleration estimates
    ay_est = data['Ay_est']  # New acceleration estimates
    
    # Measurements (now includes velocity measurements)
    x_meas = data['x_meas']
    y_meas = data['y_meas']
    vx_meas = data['vx_meas']  # New velocity measurements
    vy_meas = data['vy_meas']  # New velocity measurements
    
    # Filter diagnostics
    P_trace = data['P_trace']
    K_x_gain = data['K_x_gain']
    K_xdot_gain = data['K_xdot_gain']
    K_y_gain = data['K_y_gain']
    K_ydot_gain = data['K_ydot_gain']
    
    # Position covariance elements for ellipses (updated indices for 6x6 matrix)
    P_xx = data['P_xx']  # Var(x) - now P[0,0]
    P_xy = data['P_xy']  # Cov(x,y) - now P[0,3]
    P_yx = data['P_yx']  # Cov(y,x) - now P[3,0]
    P_yy = data['P_yy']  # Var(y) - now P[3,3]
    
    # Calculate true accelerations from true data
    dt = np.diff(time)
    ax_true = np.zeros_like(vx_true)
    ay_true = np.zeros_like(vy_true)
    
    # Calculate acceleration using differences
    for i in range(1, len(vx_true) - 1):
        if dt[i - 1] > 0:
            ax_true[i] = (vx_true[i + 1] - vx_true[i - 1]) / (2 * dt[i - 1])
            ay_true[i] = (vy_true[i + 1] - vy_true[i - 1]) / (2 * dt[i - 1])
    
    # Forward/backward differences for endpoints
    if len(dt) > 0:
        ax_true[0] = (vx_true[1] - vx_true[0]) / dt[0] if dt[0] > 0 else 0
        ay_true[0] = (vy_true[1] - vy_true[0]) / dt[0] if dt[0] > 0 else 0
        ax_true[-1] = (vx_true[-1] - vx_true[-2]) / dt[-1] if dt[-1] > 0 else 0
        ay_true[-1] = (vy_true[-1] - vy_true[-2]) / dt[-1] if dt[-1] > 0 else 0
    
    # Calculate estimation errors
    error_x = x_true - x_est
    error_y = y_true - y_est
    error_vx = vx_true - vx_est
    error_vy = vy_true - vy_est
    error_ax = ax_true - ax_est
    error_ay = ay_true - ay_est
    
    # Calculate residuals
    residual_x = x_meas - x_est
    residual_y = y_meas - y_est
    residual_vx = vx_meas - vx_est
    residual_vy = vy_meas - vy_est
    
    # Calculate performance metrics for title
    rms_pos_error = np.sqrt(np.mean(error_x ** 2 + error_y ** 2))
    rms_vel_error = np.sqrt(np.mean(error_vx ** 2 + error_vy ** 2))
    rms_acc_error = np.sqrt(np.mean(error_ax ** 2 + error_ay ** 2))
    mission_time = time[-1] - time[0]
    
    # Create the required plots
    plt.ioff()  # Turn off interactive plotting
    fig = plt.figure(figsize=(20, 16))
    
    # 1. XY trajectory with mission metrics in title
    ax1 = plt.subplot(3, 3, (1,3))
    
    # Convert waypoints to meters for plotting
    wp_x = [wp[0] / 1e3 for wp in waypoints]
    wp_y = [wp[1] / 1e3 for wp in waypoints]
    plt.plot(wp_x, wp_y, 'k--', linewidth=2, label='Waypoint Path', alpha=0.7)
    
    # Plot waypoint markers
    for i, wp in enumerate(waypoints[:-1]):
        plt.plot(wp[0] / 1e3, wp[1] / 1e3, 'ko', markersize=6)
        plt.annotate('WP%d' % i, (wp[0] / 1e3, wp[1] / 1e3), xytext=(5, 5),
                     textcoords='offset points', fontsize=8)
    
    # Plot trajectories
    plt.plot(x_true, y_true, 'b-', linewidth=2, label='True Trajectory')
    plt.plot(x_est, y_est, 'r-', linewidth=2, label='Estimated Trajectory')
    
    # Plot measurements (subsampled for clarity)
    subsample = max(1, len(x_meas) // 30)
    plt.scatter(x_meas[::subsample], y_meas[::subsample], c='magenta', s=15,
                alpha=0.6, label='Position Measurements')
    
    # Plot uncertainty ellipses using filterpy.stats
    ellipse_subsample = max(1, len(x_est) // 20)
    for i in range(0, len(x_est), ellipse_subsample):
        if i < len(P_xx):
            # Reconstruct 2x2 position covariance matrix
            P_pos = np.array([[P_xx[i], P_xy[i]],
                              [P_yx[i], P_yy[i]]])
            mean = (x_est[i], y_est[i])
            
            # Check if covariance is valid
            if np.all(np.isfinite(P_pos)) and np.linalg.det(P_pos) > 1e-12:
                try:
                    fstats.plot_covariance(mean, cov=P_pos, std=2,
                                           fc='red', ec='red', alpha=0.2)
                except:
                    continue
    
    # Start/end markers
    plt.plot(x_est[0], y_est[0], 'go', markersize=8, label='Start')
    plt.plot(x_est[-1], y_est[-1], 'ro', markersize=8, label='End')
    
    plt.title('XY Trajectory | Time: {:.1f}s | RMSE - Pos: {:.3f}m, Vel: {:.3f}m/s, Acc: {:.3f}m/s^2'.format(
        mission_time, rms_pos_error, rms_vel_error, rms_acc_error))
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # 2. Positon - States
    ax2 = plt.subplot(3, 3, 4)
    ax2.plot(time, vx_true, 'b-', linewidth=1, label='Vx True', alpha=0.6)
    ax2.plot(time, vx_est, 'g--', linewidth=0.5, label='Vx Est', alpha=0.8)
    ax2.axhline(y=0, color='y', linestyle=':', alpha=0.5)
    ax2.plot(time, vy_true, 'k-', linewidth=1, label='Vy True',  alpha=0.6)
    ax2.plot(time, vy_est, 'm--', linewidth=0.5, label='Vy Est',  alpha=0.8)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)', color='black')
    ax2.set_title('Velocity States')
    ax2.grid(True, alpha=0.3)
    
    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    ax2.legend(lines1, labels1, loc='upper right', fontsize=8)
    
    
    # 3. Acceleration States
    ax3 = plt.subplot(3, 3, 5)
    ax3.plot(time, ax_est, 'c-', linewidth=1, label='Ax Est', alpha=0.8)
    ax3.axhline(y=0, color='y', linestyle=':', alpha=0.5)
    ax3.plot(time, ay_est, 'g-', linewidth=1, label='Ay Est',  alpha=0.8)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Acceleration (m/s^2)', color='black')
    ax3.set_title('Acceleration States')
    ax3.grid(True, alpha=0.3)
    
    # Combined legend
    lines1, labels1 = ax3.get_legend_handles_labels()
    ax3.legend(lines1, labels1, loc='upper right', fontsize=8)
    
    # 4. Trace of covariance P
    ax4 = plt.subplot(3, 3, 6)
    plt.semilogy(time, P_trace, 'k-', linewidth=2)
    plt.title('Trace of Covariance Matrix P')
    plt.xlabel('Time (s)')
    plt.ylabel('Tr(P) [log scale]')
    plt.grid(True, alpha=0.3)
    
    # 5. Errors (Position)
    ax5 = plt.subplot(3, 3, 7)
    ax5.plot(time, error_x, 'b-', linewidth=2, label='X Error',  alpha=0.7)
    ax5.axhline(y=0, color='k', linestyle=':', alpha=0.5)
    ax5.plot(time, error_y, 'r-', linewidth=2, label='Y Error',  alpha=0.7)
    
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Position Error (m)', color='black')
    ax5.set_title('Position Errors')
    ax5.grid(True, alpha=0.3)
    
    # Combined legend
    lines1, labels1 = ax5.get_legend_handles_labels()
    ax5.legend(lines1, labels1, loc='upper right', fontsize=8)

    # 6. Errors (Velocity)
    ax6 = plt.subplot(3, 3, 8)
    ax6.plot(time, error_vx, 'g-', linewidth=1, label='Vx Error',  alpha=0.7)
    ax6.axhline(y=0, color='g', linestyle=':', alpha=0.5)
    ax6.plot(time, error_vy, 'm-', linewidth=1, label='Vy Error',  alpha=0.6)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Velocity Error (m/s)', color='black')
    ax6.set_title('Velocity Errors')
    ax6.grid(True, alpha=0.3)
    
    # Combined legend
    lines1, labels1 = ax6.get_legend_handles_labels()
    ax6.legend(lines1, labels1, loc='upper right', fontsize=8)
    
    # 7. Kalman gains
    ax7 = plt.subplot(3, 3, 9)
    plt.plot(time, K_x_gain, 'b-', linewidth=2, label='K_x')
    plt.plot(time, K_xdot_gain, 'c-',linewidth=2, label='K_xdot')
    plt.plot(time, K_y_gain, 'r-', linewidth=2, label='K_y')
    plt.plot(time, K_ydot_gain,'m-',linewidth=2, label='K_ydot' )
    plt.title('Kalman Gains')
    plt.xlabel('Time (s)')
    plt.ylabel('Gain Value')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("kfposvel_analysis.png", dpi=300, bbox_inches='tight')
    print("Kalman filter analysis plot saved as 'kfposvel_analysis.png'")
    
    # Plot Error Distributions
    plot_error_distributions(error_x, error_y, error_vx, error_vy, error_ax, error_ay,
                             residual_x, residual_y, residual_vx, residual_vy)