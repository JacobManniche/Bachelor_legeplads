import numpy as np
import plotly.graph_objects as go
from Tracer.tracer import Trajectory
from Tracer.windfield import WindField

def animate(ball_speed, launch_angle, spin_rate, spin_axis=0, wind=WindField(profile='log'), dt=0.01):

    t, p, v, w = Trajectory(ball_speed, launch_angle, spin_rate, spin_axis, wind=wind).solve(dt=dt)
    
    # 1. Pre-calculate indices to avoid repeated slicing
    step_size = max(1, len(p) // 100)
    indices = np.arange(0, len(p), step_size)

    # 2. Static Background Elements (The full path)
    # We plot the full trajectory once with low opacity as a "guide"
    full_trajectory = go.Scatter3d(
        x=p[:, 0], y=p[:, 1], z=p[:, 2],
        mode='lines',
        line=dict(color='rgba(255, 0, 0, 0.7)', width=2), # Faded red
        name="Full Path",
        customdata=np.column_stack((v, w)),
        hovertemplate=(
            "<b>Position:\t</b> %{x:.1f}m %{y:.1f}m %{z:.1f}m<br>" +
            "<b>Velocity:\t</b> %{customdata[0]:.1f}, %{customdata[1]:.1f}, %{customdata[2]:.1f} m/s<br>" +
            "<b>Spin:\t</b> %{customdata[4]:.0f} rad/s<br>" +
            "<extra></extra>"    
            )
    )

    # The actual "Moving" Tracer line (initially just the first point)
    moving_ball = go.Scatter3d(
        x=[p[0, 0]], y=[p[0, 1]], z=[p[0, 2]],
        mode='markers+text',
        marker=dict(color='white', size=6, line=dict(color='black', width=1)),
        text=["Ball"],
        name="Ball"
    )

    # 3. Optimized Frames: ONLY update the marker position
    frames = [
        go.Frame(
            data=[go.Scatter3d(x=[p[i, 0]], y=[p[i, 1]], z=[p[i, 2]])],
            name=str(i),
            traces=[1] # This tells Plotly to only update the second trace (moving_ball)
        ) for i in indices
    ]

    # 4. Faster Layout Config
    scene_config = dict(
        xaxis=dict(range=[p[:,0].min()-5, p[:,0].max()+5]),
        yaxis=dict(range=[p[:,1].min()-5, p[:,1].max()+5]),
        zaxis=dict(range=[0, p[:,2].max()+5], backgroundcolor="rgb(34, 139, 34)"),
        aspectmode="data" # 'data' is faster than 'manual' with complex ratios
    )

        ## Define the buttons for play, pause, and reset ##
    updatemenu = [
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": (dt*0.9)*1000*step_size, "redraw": True},  "mode": "immediate", "transition": {"duration": 0}, "fromcurrent": True}], # (0.9 to make up for computation time, 1000 to convert to ms, step_size to adjust for frame skipping)
                    "label": "▶ Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                    "label": "■ Pause",
                    "method": "animate"
                },
                {
                    "args": [["0"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                    "label": "↺ Reset",
                    "method": "animate"
                }
            ],
            "type": "buttons",
            "showactive": True,
            "x": 0.1, "y": 1, "xanchor": "right", "yanchor": "top"
        }
    ]

    ## Add a slider for manual control ##
    sliders = [
        {
            "active": 0,
            "currentvalue": {"prefix": "Time: ", "visible": True, "xanchor": "right"},
            "pad": {"b": 10, "t": 50},
            "steps": [
                {
                    "args": [[str(i)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                    "label": f"{t[i]:.2f} s",
                    "method": "animate"
                } for i in range(0, len(p), step_size)
            ]
        }
    ]

    fig = go.Figure(
        data=[full_trajectory, moving_ball],
        frames=frames,
        layout=go.Layout(
            title="Optimized Golf Trajectory",
            scene=scene_config,
            updatemenus=updatemenu,
            sliders=sliders
        )
    )

    fig.show()


if __name__ == "__main__":
    # This is where we will run our simulation and plot the results
    # We can change the initial conditions and parameters here to see how they affect the trajectory of the ball
    
    animate(76.4, 10.4, -8545, dt=0.01)