# cDarkNet
cDarkNet is modified by AlexeyAB's darknet
1. detector test function modified
   usage ./darknet detector test2 <same thing as usual> -iou_thresh X.X
   threshold for objectiveness and classification is seperated
   
2. class_scale, object_scale, coord_scale is added into yolo layer

3. in .cfg, user can define different weights for classification of each class.
   For example: class_weights = 1.0,0.9,1.4 for all three class with different weights
   
4. detector train function modified
   usage ./darknet detector train2 <same thing as usual>
   In train2 mode, there is several things to be check
   
   4.1 [yolo2] should be used in .cfg file in stead of [yolo]. All parameters for yolo2 are same with yolo. Only the layer.truths size is different. yolo2's label is 1 class_id + 4 coordination + 1 trust_type.
   'trust_type' means whether coord label or class label should be trust. If not, it will not take that part into BP.
   
   4.2 For label file like 'xxx.txt', it goes like this:
   id x . y . w . h . trust_type
   0 0.3 0.2 0.4 0.6 1
   1 0.2 0.4 0.3 0.3 0
   Here, 0 stands for that all labels(coord+class) is trusted and can be count into BP.
   1 stands for that only coord label is trusted and class should not be count into BP.
   2 stands for that only class label is trusted and coord should not be count into BP.
   Doing so, you can train your network with mixtured data type such as data A with only detected coordination and data B which is used for classification.
   
   
