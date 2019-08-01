# cDarkNet
cDarkNet is modified by AlexeyAB's darknet

1. detector test function modified

   usage: ./darknet detector test2 [same thing as usual] -iou_thresh X.X

   threshold for objectiveness and classification is seperated
   
2. class_scale, object_scale, coord_scale is added into yolo layer

3. in .cfg, user can define different weights for classification of each class.

   For example: class_weights = 1.0,0.9,1.4 for all three class with different weights
   
4. detector train function modified
   
   usage: ./darknet detector train2 [same thing as usual]

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
   

5. detector map function modified

   usage: ./darknet detector map2 [same thing as usual]

   In map2 mode, more error analysis can be seen.
   Detection result is shown by precision, recall and F1-score, of which objectness is used as a index to judge whether an object is detected.
   The additional results goes like this:

   ##raw classification result##
    for thresh = 0.25, precision = 0.29, recall = 0.16, F1-score = 0.21 
    for thresh = 0.25, TP = 5208, FP = 12673, FN = 26798, average IoU = 20.01 %

   ##detection result##
    for thresh = 0.25, precision = 0.51, recall = 0.12, F1-score = 0.19 ,TP = 3778, FP = 3656 , FN = 28228, average IoU = 33.98 %


6. classifier valid2 function added

    usage: ./darknet classifier valid2 [same thing as usual]

    In valid2 mode, more error analysis can be seen.
    Each class will output its Top1 and Top2 accuracy result.

7. detector train3 function added

    usage: ./darknet train3 [same thing as usual]

    This function will take 2 image as input and combine them into a 4 channel image data as input of the network.
    First image is an original bgr image read from the train.txt, the second image is a single channel heatmap image read by replace XXX.[jpg/tiff/jpeg] into XXX_bg.[jpg/tiff/jpeg].
    The idea behind is using the probability map of other algorithm to help detection. 
    E.g. Use the result of background subtraction as the 4th channel.
    To verify the result, map3 should be used: ./darknet map3 [same thing as usual]

8. writing.c is modified to support gpu calculation and validation is added

    usage:  ./darknet writing traingpu [cfg] [weights] [datacfg]

    usage:  ./darknet writing valid [cfg] [weights] [datacfg]

    In 'traingpu' mode, validation step will not be operated because of some bugs not fixed.
    The bug occurs when the program tries to go into the next training round after validation.
    The status check of gpu will report a failure.
    In 'valid' mode, the evaluation index is defined as the iou of predict binary image and groundtruth binary image.
    In [datacfg] file, only parameters of 'train = xxx' and 'valid = xxx' is supported. If [datacfg] is not defined, 'figures.list' and 'figures_valid.list' will be used as default.

9. map_2stage_v0 and map_2stage_v1 is added to support mAP calculation with 2 stage model.

    usage:  ./darknet detector map_2stage_v0 [datacfg] [detection_cfg] [detection_weights] [classification_cfg] [classification_weights] -thresh -iou_thresh -dont_show

    The mAP will be caculated for models combined by detection and classification. 
    The difference between map_2stage_v0 and map_2stage_v1 is that 'v0' is much faster and classification is predicted after NMS of detection.
    'v1' is accurate and the classification is predicted before NMS, and NMS is done for multi-classification.
    However, 'v0' is also correct because the first stage is responsible for detection and the classification network should only takes care of the candidate bbox produced by the first stage.

10. classifier valid3 function added

    usage: ./darknet classifier valid3 [same thing as usual]

    In valid3 mode, AUC of each class will be calculated.
    All AUCs will be averaged to generate avg_AUC.
    
11. classifier valid4 function added

    usage: ./darknet classifier valid4 [same thing as usual]

    In valid4 mode, AUC of each class will be calculated.
    All AUCs will be averaged to generate avg_AUC.
    valid 3 and valid 4 are equals.
    Difference between them is inner code of calculating AUC, which is written into a function named 'get_roc_auc'  in valid4

12. detector map02 and detector map03 is added to replace map2 and map3

    usage:  ./darknet detector map02 [datacfg] [network_cfg] [network_weights] -thresh -iou_thresh -dont_show

    ./darknet detector map03 [datacfg] [network_cfg] [network_weights] -thresh -iou_thresh -dont_show

    map02 was cloned from the newest map calculation code from AlexeyAB and modified to output more individual results.
    map03 was designed for inputting data with 4 channels, where the 4th channel is another image going like XXX_bg.jpe/tiff/png

13. map0_2stage_v0 and map0_2stage_v1 is added to support mAP calculation with 2 stage model.

    usage:  ./darknet detector map0_2stage_v0 [datacfg] [detection_cfg] [detection_weights] [classification_cfg] [classification_weights] -thresh -iou_thresh -dont_show

    The mAP will be caculated for models combined by detection and classification. 
    The difference between map_2stage_v0 and map_2stage_v1 is that 'v0' is much faster and classification is predicted after NMS of detection.
    'v1' is accurate and the classification is predicted before NMS, and NMS is done for multi-classification.
    However, 'v0' is also correct because the first stage is responsible for detection and the classification network should only takes care of the candidate bbox produced by the first stage.
    Usually, 'v0' is used to test the model in order to meet the fact.

14. bce_layer is added in comparison with softmax layer

    bce_layer can be used instead of softmax layer during the classification task, where the each sample may belong to multi-class. 
    However, softmax layer intuitly supports the situation that the target belongs to one class only.
    Use '[bce]' instead of '[softmax]' in the last layer.
    Use 'classifier train2' instead of 'classifier train'.
    'train2' reads labels of samples from the same folder with extension of image file replaced by '.txt'.
    Labels are stored as vectors in 'xxx.txt' files.
    While 'train' is the original function, it reads labels by matching specific phrase whether appearing in the name of image( or path ).


   




   
