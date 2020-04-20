from AYS_3D_figures import *


def plot_example_figure(learning_progress_arr, cut_shelter=False, num_traj=0,num_hairs=300, ax=None, ticks=True,label=[], colors=[],
                        option=None,plot_traj=True, plot_boundary=True,
                        filename='./images/phase_space_plots/3D_AYS_example_trajectory.pdf', ):
    
    learning_progress= learning_progress_arr[num_traj]
    if ax is None:
        if cut_shelter:       
            fig, ax3d=create_extract_figure(Azimut=-160, plot_boundary=plot_boundary, label=None, colors=None)
        else:
            if option=='DG':
                Azimut=-177
                Elevation=65
            elif option=='ET':
                Azimut=-88
                Elevation=65
            else:
                Azimut=-167
                Elevation=25
            fig, ax3d=create_figure(Azimut=Azimut,Elevation=Elevation, label=label, colors=colors, ticks=ticks, plot_boundary=plot_boundary )
    else:
        ax3d=ax
    
    if option:
        plot_management_dynamics(ax3d, option)
    else: 
        plot_hairy_lines(num_hairs, ax3d)
  
    
    A,Y,S=learning_progress['A'],learning_progress['Y'],learning_progress['S'] 
    
    
    if plot_traj:
        plot_action_trajectory(ax3d, learning_progress, 0, len(S),lw=4)

    fig.savefig(filename)
    
    return ax3d
