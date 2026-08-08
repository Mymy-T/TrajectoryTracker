# Sample a camera configuration within a range
#   For each sampled camera config, generate multiple random 3D trajectories (10 maybe?)
#   Project each trajectory into 2D
#   Save each instance into a dataframe with cols for the camera configuration, maybe a custom dtype for storing variable length trajectory data
#
# After saving the df as csv file, load that file, and project each 3d trajectory sequence into a sequence of 2d u, v coords
# To make a dataset, pair each 2d sequence with each 3d sequence