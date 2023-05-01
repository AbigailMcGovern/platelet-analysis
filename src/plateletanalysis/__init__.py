#from ...old_modules.io import get_experiment_df, add_info_from_file_name
from .variables.measure import finite_difference_derivatives, stability, \
   contractile_motion #, point_depth
from .variables.transform import adjust_coordinates, spherical_coordinates, cylindrical_coordinates
from .variables.basic import add_basic_variables_to_files
from .variables.calcium import corrected_calcium
from.variables.neighbours import add_neighbour_lists, local_density, average_neighbour_distance
from.analysis.summary_data import rolling_counts_and_growth_plus_peaks, quantile_analysis_data, \
    inside_and_outside_injury_counts_density, \
        inside_and_outside_injury_counts_density_thrombus_size, \
            experiment_time_region_data, region_parallel_coordinate_data
from .analysis.plots import abs_and_pcnt_timeplots, abs_and_pcnt_timeplots, \
    inside_and_outside_injury_barplots, regions_abs_and_pcnt_timeplots, \
        individual_exp_inside_outside_timeplots 

__all__ = [
    # ---------------------
    # Categorical variables
    # ---------------------
    'add_basic_variables_to_files', 
    # ---------------------
    # Coordinate transforms
    # ---------------------
    'adjust_coordinates', # convert from pixel-based coordinates to ROI based (x_s, ys, zs)
    'spherical_coordinates', # rho, theta, phi
    'cylindrical_coordinates', # cyl_radial, cyl_azimuthal (zs is the other part of this coordinate system)
    # ---------------------------------
    # Measurements based on coordinates
    # ---------------------------------
    'finite_difference_derivatives', # dv, dvx, dvy, dvx
    'stability', # stab
    'contractile_motion', # cont
    #'point_depth', depth # this isnt the copy that works... Niklas's code
    # ----------------------------------------
    # Measurements based on neighbor platelets
    # ----------------------------------------
    'average_neighbour_distance', # nba_5, nba_10
    'add_neighbour_lists', # nb_particles_15
    'local_density', # nb_density_15
    # -----------------------------
    # Measurements based on calcium
    # -----------------------------
    'corrected_calcium', # ca_corr
    # ----------------------------
    # Generate summary data frames
    # ----------------------------
    'rolling_counts_and_growth_plus_peaks', 
    'quantile_analysis_data', 
    'inside_and_outside_injury_counts_density', 
    'inside_and_outside_injury_counts_density_thrombus_size', 
    'experiment_time_region_data', 
    'region_parallel_coordinate_data',
    # --------------
    # Generate plots
    # --------------
    'abs_and_pcnt_timeplots',
    'abs_and_pcnt_timeplots', 
    'inside_and_outside_injury_barplots', 
    'regions_abs_and_pcnt_timeplots',
    'individual_exp_inside_outside_timeplots',
    # -----------------
    # Statistical tests
    # -----------------

    # --------
    # Topology
    # --------


]