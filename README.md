# TrajectoryTracker
TrajectoryTracker is an in-progress project that analyzes a video clip of a tennis rally to reconstruct its 3D trajectory and estimate initial conditions after a hit or bounce.

 
## Project Layout
### TrajectoryTrackerSwiftCode
- Contains some code for when the models become good enough to be incorporated into an iOS app
- Currently incomplete.

### TrajectoryTrackerPython
- Contains all of the code for training, evaluating, and running the models
- Split into 2 main independent components: a ball detection/tracking component and a 3D trajectory reconstruction/estimation component. 
- Currently, work is focused on the reconstruction module; the ball tracking component can be tweaked and improved based on the results of certain tests with the reconstruction component

Python files in **convert_2d_to_3d**:
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
| `verify_masking.py` | Acts as a test to ensure the masking works when applied to batches with unequal trajectory lengths |
| `visualizer.py` | Contains functions for plotting 3D & 2D trajectories; also contains helper functions utilized for specific tasks (e.g. creating fading trails in the visualizer |

Python files in **ball_detection**:
| File | Function |
|--|--|
| `annotation.py` | Saves the annotated data (via the AnyLabeling GUI) in a specific folder |
| `checkpointing.py` | Contains classes that manage model checkpointing and the save used for pausing/resuming model training |
| `data_splitter.py` | Splits the data into train, valid, & test sets; can be set to respect match or clip boundaries |
| `dataset.py` | A customized PyTorch dataset class that loads examples and creates batches |
| `dataset_creator.py` | Processes the clip frames into a format ready for training the model; also creates the heatmaps based on the annotations; the processed dataset is saved in a specific path |
| `eval_script.py` | Prints avg loss and classification metrics on the test set |
| `frame_extractor.py` | Processes the clips into a folder of video frames; saves them to a specific path where they will be annotated by running `annotation.py` |
| `gpu_test.py` | Check the GPUs available (also checks for MPS on Mac) |
| `heatmap_visualizer.py` | Used to view the a heatmap at a specified path |
| `image_from_video.py` | Saves the first frame of a specified video clip to a specific path; this image is used for camera calibration in **convert_2d_to_3d** and in visualizations |
| `imbalance_checker.py` | Used to investigate how the number of positive heatmap pixels compares to the number of negative ones; the results were used in informing loss function choice and weights |
| `inference.py` | Runs a specified model (may be trained or pretrained) on a specified clip and saves the results |
| `model.py` | Contains the "blueprint" of the TrackNet model by specifying its architecture |
| `paths.py` | Contains a class acting as the "master path directory" for this module |
| `settings.py` | Contains global variables specifying device, train clip fps, and inference clip fps; different right now since a 30 fps pretrained model is used (fully trained model is incomplete) |
| `train_script.py` | Starts the training process; the model, max training time (before pausing), and max number of checkpoints can all be specified |
| `train_test.py` | Used as a tester for small test/trial runs; basically a scaled-down version of a full training and evaluation run
| `trainer.py` | Contains the train loop; also has train and evaluate functions implemented, which are called by the main train loop |
| `video_downloader.py` | Used to download video clips from Youtube for later use; clips are saved to a specific path


## Technologies

This project is mainly written in Python; some Swift is present since the final goal is an iOS app. Python libraries include PyTorch, OpenCV, Pandas, NumPy, and more.

## Attributions & References
- Dignall, R. J. (2004). Modelling the impact of tennis balls on court surfaces.
This paper details tennis bounce physics in 2D.

- Eassom, A., Robertson, S., & Reid, M. (2026). Tennis ball trajectory decomposition based on spatiotemporal tracking data. Sports Engineering, 29(1), Article 19. https://doi.org/10.1007/s12283-026-00547-6
This paper was the basis for the initial condition estimation method. The paper utilized the Powell algorithm from SciPy to optimize for the initial conditions that best re-create an observed trajectory.

- Huang, Y.-C., Liao, I.-N., Chen, C.-H., İk, T.-U., & Peng, W.-C. (2019, September 1). TrackNet: A deep learning network for tracking high-speed and tiny objects in sports applications*. IEEE Xplore. https://ieeexplore.ieee.org/document/8909871
This paper introduced the TrackNet architecture, specifically designed and trained to detect small, moving objects like tennis balls. This same architecture was used for the ball detection component of the pipeline.

- Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollar, P. (2020). Focal loss for dense object detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, 42(2), 318–327. https://doi.org/10.1109/tpami.2018.2858826
This paper introduced focal loss, which I am using to help fine-tune a pretrained TrackNet model to detect balls at a court-level view. 

- Ponglertnapakorn, P., & Suwajanakorn, S. (2025). Where is the ball: 3D ball trajectory estimation from 2D monocular tracking (2506.05763V1). arXiv. https://arxiv.org/abs/2506.05763v1
This is the basis of the reconstruction architecture. This version is an **unofficial implementation** of their model (official implementation wasn't publicly available as of 11/1/2026), along with some additions of my own in an attempt to address limitations of the paper.

- yastrebksv. (2022). GitHub - yastrebksv/TrackNet: Unofficial PyTorch implementation of TrackNet. GitHub. https://github.com/yastrebksv/TrackNet
This was the implementation from which the pretrained TrackNet model was obtained. Code in `model.py` for defining the model was reused from this. My file organization was also inspired by this implementation.


<!--stackedit_data:
eyJoaXN0b3J5IjpbMTAzMDkyNzI4XX0=
-->