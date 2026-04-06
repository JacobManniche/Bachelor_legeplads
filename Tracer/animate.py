import numpy as np
import plotly.graph_objects as go
from tracer import solver, initial_velocity, rpm2radps

def animate(P0, V0, W0, wind, dt=0.01, randomize_wind=False):

    t, p = solver(P0, V0, W0, wind, dt=dt, randomize_wind=randomize_wind)
    
    ## Define the scene configuration with dynamic ranges and aspect ratio ##
    z_max = p[:,2].max()
    x_min, x_max = p[:,0].min(), p[:,0].max()
    y_min, y_max = p[:,1].min(), p[:,1].max()
    padding = 10

    ranges = [
        [x_min-padding, x_max+padding],
        [y_min-padding, y_max+padding],
        [0, z_max+padding]
    ]

    scene_config = dict()
    scene_config["xaxis"] = {"range": ranges[0], 'tickmode': 'linear', 'dtick': padding}
    scene_config["yaxis"] = {"range": ranges[1], 'tickmode': 'linear', 'dtick': padding}
    scene_config["zaxis"] = {"range": ranges[2], "backgroundcolor": "rgb(34, 139, 34)"}
    scene_config["camera"] = {"eye": {"x": 2, "y": 2, "z": 2}}
    scene_config["aspectmode"] = "manual"
    scene_config["aspectratio"] = {"x": ranges[0][1] - ranges[0][0], "y": ranges[1][1] - ranges[1][0], "z": ranges[2][1] - ranges[2][0]}

    ## Render the frames with step size ##
    step_size = max(1, len(p) // 100)  # Adjust step size for smoother animation

    frames = [
        go.Frame(
            data=[
                go.Scatter3d(x=p[:i, 0], y=p[:i, 1], z=p[:i, 2], mode='lines', line={'color': 'red', 'width': 4}),
                go.Scatter3d(x=[p[i, 0]], y=[p[i, 1]], z=[p[i, 2]], mode='markers', marker={'color': 'Lightgray', 'size': 5})
                ],
            name=str(i)
        ) for i in range(0, len(p), step_size)  # Adjust step for smoother animation
    ]

    starting_frame = [
        go.Scatter3d(x=[p[0,0]], y=[p[0,1]], z=[p[0,2]], mode='lines', line={'color': 'red', 'width': 4}, name="Trajectory"),
        go.Scatter3d(x=[p[0,0]], y=[p[0,1]], z=[p[0,2]], mode='markers+text', marker={'color': 'Lightgray', 'size': 5}, text=["Start"], textfont={'color': 'white', 'size': 12}, name="Ball")
    ]

    frames[0] = go.Frame(data=starting_frame, name="0") # Change initial frame for reset button

    ## Define the buttons for play, pause, and reset ##
    updatemenu = [
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": (dt*0.9)*1000*step_size, "redraw": True}, "fromcurrent": True}], # (0.9 to make up for computation time, 1000 to convert to ms, step_size to adjust for frame skipping)
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

    ## Assemble the Figure ##
    fig = go.Figure(
        data=starting_frame,  # Initial empty line for trajectory
        frames=frames,
        layout=go.Layout(title="Golf Ball Trajectory", updatemenus=updatemenu, scene=scene_config, sliders=sliders)
    )

    fig.show()

if __name__ == "__main__":
    # This is where we will run our simulation and plot the results
    # We can change the initial conditions and parameters here to see how they affect the trajectory of the ball
    P0 = [0,0,0]
    V0 = initial_velocity(speed=48, angle=20) 
    W0 = np.array([rpm2radps(8647), 0, 0])
    wind = np.array([8, 3, 6]) # 

    animate(P0, V0, W0, wind, dt=0.01, randomize_wind=True)