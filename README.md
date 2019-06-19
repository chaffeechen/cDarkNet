# cDarkNet
cDarkNet is modified by AlexeyAB's darknet
1. detector test function modified
   usage ./darknet detector test2 <same thing as usual> -iou_thresh X.X
   threshold for objectiveness and classification is seperated
   
2. class_scale, object_scale, coord_scale is added into yolo layer

3. in .cfg, user can define different weights for classification of each class.
   For example: class_weights = 1.0,0.9,1.4 for all three class with different weights
