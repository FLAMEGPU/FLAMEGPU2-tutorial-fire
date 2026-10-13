import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import traceback

DATA_DIR = "data"
grid = np.array([])
loaded_steps = 0

cmap = ListedColormap([
    "white",     # 0
    "green",     # 1
    "red",       # 2
    "darkred",   # 3
])

norm = BoundaryNorm(
    boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5],
    ncolors=cmap.N
)

def unpack(v):
    # Function is required to convert agent variable states from array to scalars due to bug in RC3
    # I.e. [123] to 123
    return v[0] if isinstance(v, list) else v


def load_data(state_count):
    global grid
    global loaded_steps
    try:
        for i in range(state_count+1):
            file_path = f"{DATA_DIR}/{i}.json"
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            agents = data.get("agents", {}).get("agent", {}).get("default", [])

            if not agents:
                print("No agent data found.")
                return

            # Determine grid size
            max_x = unpack(max(agent.get('x', 0) for agent in agents))
            max_y = unpack(max(agent.get('y', 0) for agent in agents))

            if not grid.any():
                grid = np.zeros((int(max_x) + 1, int(max_y) + 1, int(state_count)+1))

            for agent in agents:
                x, y = unpack(agent.get('x', 0)), unpack(agent.get('y', 0))
                status = unpack(agent.get('status', 0))

                if status != 0:
                    grid[int(x), int(y), i] = status  # Mark active agents with their status
        loaded_steps = state_count
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        print(traceback.format_exc())


def plot_state(state):
    """Reads a JSON file and plots active agents."""

    # if grid is empty force reload of data
    if not grid.any() or loaded_steps < state:
        load_data(state)

    # Plotting
    plt.figure(figsize=(8, 6))

    im = plt.imshow(grid[:,:,state].T, origin='lower', cmap=cmap, norm=norm)

    #plot
    plt.title(f"Agent Sugar Level and Active Agents Step {state}")
    # plt.colorbar(im)
    plt.show()


def animate_states(state_count):
    """Reads a JSON file and plots active agents."""
    global ani

    # if grid is empty force reload of data
    if not grid.any():
        load_data(state_count)

    fig, ax = plt.subplots()

    ims = [] # ims is a list of lists, each row is a list of artists to draw in the
    for i in range(state_count):
        im = ax.imshow(grid[:,:,i].T, animated=True, cmap=cmap, norm=norm, origin='lower')
        if i == 0:
            ax.imshow(grid[:,:,i].T, animated=True, cmap=cmap, norm=norm, origin='lower')  # show an initial one first
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)

    # Plotting
    # plt.colorbar(im, ax=ax)
    plt.show()



def display_states(state_count):
    """Reads a JSON file and plots active agents."""
    global im
    global slider
    im = slider = None

    # if grid is empty force reload of data
    if not grid.any():
        load_data(state_count)

    fig, ax = plt.subplots()
    im = ax.imshow(grid[:,:,0].T, cmap=cmap, norm=norm, origin='lower')
    ax_depth = plt.axes([0.23, 0.02, 0.56, 0.04])
    slider = Slider(ax_depth, 'iteration', 0, grid.shape[2]-1, valinit=0)
    update_slider = lambda v: im.set_data(grid[:, :, int(round(v))].T)
    slider.on_changed(update_slider)

    # Plotting
    # plt.colorbar(im, ax=ax)
    plt.show()