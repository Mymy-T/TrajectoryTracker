# TrajectoryTracker
TrajectoryTracker is an in-progress project that analyzes a video clip of a tennis rally to reconstruct its 3D trajectory and estimate initial conditions after a hit or bounce.

 
## Project Layout
### TrajectoryTrackerSwiftCode
- Contains some code for when the models become good enough to be incorporated into an iOS app
- Currently incomplete.

### TrajectoryTrackerPython
- Contains all of the code for training, evaluating, and running the models
- Split into 2 main independent components: a ball detection/tracking component and a 3D trajectory reconstruction/estimation component. 

In **convert_2d_to_3d**:
| File | Function |
|--|--|
| `bounce_classifier_eval.py` | Obtain the confusion matrix of a particular bounce model |
| `cam_config_explorer.py` | Runs a GUI for visualizing how different camera intrinsics affect the view of a court |
| `checkpointing.py` | Contains classes that manage model checkpointing, the save used for pausing/resuming model training, and loss logging structure |
| `data_models.py` | Contains key variables/properties (e.g. gravitational acceleration, court dimensions, ball properties, etc.); also contains definitions of "setting" class structures |
| `data_pipeline.py` | Wrapper script to generate the synthetic data; can generate both the original physics datasets or the processed datasets (after interpolation to a desired time step & projection into 2D) |
| `data_splitter.py` | Splits the data into train, valid, & test sets; can be set to respect camera configuration boundaries (essentially "blocked" splitting) |
| `dataset.py` | A customized PyTorch dataset class that loads examples and creates batches |
| `dataset_creator.py` | Contains a function that builds the projected 2D trajectories from the 3D trajectories; also interpolates both to a desired time step |
| `dataset_explorer.py` | Runs a GUI that allows for traversing a processed dataset; the trajectories can be viewed and played (as animations) |
| `full_model_eval.py` | Computes MSE of a given model on a test set; also gives different MSE percentiles over the whole test dataset and allows for evaluation under different conditions (e.g. removing noise or using true bounce) |
| `imbalance_reporter.py` | Use to obtain the weights for the BCE losses of the bounce and EOT networks (since the number of negative time steps exceeds the number of positive ones in both cases) |
| `inference.py` | Runs inference on a given example (or the first example of the test set if no example is given); saves the inference results as a .npz file for use by other scripts |
| `inference_visualizer.py` | Shows the results of inference on synthetic data with a GUI |
| `initial_conditions.py` | Estimates the initial conditions (velocity, spin) using reconstructed 3D trajectories |
| `loss.py` | Contains implementations of all of the loss functions that were utilized within this module |
| `manual_keypoint_labeler.py` | Runs a GUI that prompts the user to mark the key points that will be used in estimating the camera parameters for a particular view |
| `model.py` | Contains the "blueprint" of both the 3D reconstruction and bounce models |
| `paths.py` | Contains a class acting as the "master path directory" for this module |
| `physics.py` | Contains the physics simulator, as well as a few other functions to sample initial conditions and simulate trajectories |
| `physics_evaluation.py` | Assess how different physics simulation time steps affect generated trajectories |
| `proj_diameter_comparison.py` | Visualizes how the projected diameter of the ball (in pixels) changes with the ball's position along the court |
| `projector.py` | Contains functions that perform projection from 3D to 2D; also contains functions to sample camera parameters and calibrate cameras as needed |
| `reset_camera_calibration.py` | Deletes saved camera parameters, key points, or both; this leads to re-calibration when camera parameters are needed again |
| `settings.py` | Contains the general configurations of the project (e.g. training hyperparameters, physics trajectory generation settings, trajectory time step, and more) |
| `train_script.py` | Starts the training process; the model, max training time (before pausing), and max number of checkpoints can all be specified |
| `trainer.py` | Contains the train loop; also has train and evaluate functions implemented, which are called by the main train loop |
| `trajectory_generator.py` | Creates a dataset out of physics-generated synthetic data |
| `trajectory_lengths.py` | Shows histograms and percentiles of the trajectory/clip lengths; also computes average trajectory length over the dataset |
| `verify_masking.py` | Acts as a "unit test" to ensure the masking is working when applied to batches with unequal trajectory lengths |
| `visualizer.py` | Contains functions for plotting 3D & 2D trajectories; also contains helper functions utilized for specific tasks (e.g. creating fading trails in the visualizer |



## Technologies

This project is mainly written in Python; some Swift is present since the final goal is an iOS app. Python libraries include PyTorch, OpenCV, Pandas, NumPy, and more.


<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEwODQ5MzQ5NjNdfQ==
-->