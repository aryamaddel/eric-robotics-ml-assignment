Object Detection + Distance Estimation for Robotics Navigation
Your goal is to detect navigation-relevant objects and estimate how far they are from the robot (from camera perspective). On top of that, you’ll look into ways of making your model run efficiently on edge devices.

Objective
Detect cones, barriers, stop signs.
Estimate their distance from the robot (camera perspective) and annotate results accordingly.
Explore optimization techniques for running your model on limited hardware.
What to Do

1. Object Detection
   Work with the BDD100K dataset. (Optional, feel free to look into any other relevant datasets.)
   Use transfer learning – pick a suitable pretrained model and fine-tune it.
   Keep your implementation clean and modular.
2. Distance Estimation
   For each detected object, estimate distance to the robot.
   Annotate bounding boxes like this:
   Cone, 1.5m
   Stop Sign, 3.2m
3. Optimization for Edge Devices
   Try out quantization, pruning, or swapping to lightweight backbones.
   Record FPS on CPU and GPU for comparison.

Optional (Extra Credit)
Not mandatory, but good to explore if you’re curious: Epipolar Geometry - derive disparity–depth relation, Homography / Perspective Transform - warp the scene to a bird’s-eye view, Optical Flow - track moving cones across frames.
