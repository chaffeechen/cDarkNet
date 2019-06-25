# current experiment training, need revising before each testing
model_no="_ww_aug"
weights_name="yolov3_ww_aug_v1_9000.weights"

# names should follow the style or need changing the code below for corrent directories
YOLO_DIR="/home/ubuntu/darknet"
# DATASET_DIR="/home/ubuntu/CV/data/wework_activity/Scene_4"
# TESTSET_DIR="${DATASET_DIR}/test_2"
BACKUP_DIR="${YOLO_DIR}/backup/ww_aug" # "${model_no}"
# TEST_DATA_DIR="${TESTSET_DIR}/test.txt"

# create prediction directory if not exists
PRED_DIR="${YOLO_DIR}/predict_v${model_no}"
mkdir -p "${PRED_DIR}"

cfg_file=${YOLO_DIR}/cfg/yolov3_ww_608/yolov3_ww_608_test.cfg
weights_file="${BACKUP_DIR}/${weights_name}"

# create positions of predicted bounding boxes and save to result.txt under ${PRED_DIR}
#${YOLO_DIR}/darknet detector test ${YOLO_DIR}/cfg/activity_wework_tinyv1_Scene4.data ${cfg_file} ${weights_file} -thresh 0.25 -dont_show -ext_output < ${TEST_DATA_DIR} > ${PRED_DIR}/result.txt

# create images of predictions oand save images under ${PRED_DIR}
# for f in ${TESTSET_DIR}/*.jpg; do
#        ${YOLO_DIR}/darknet detector test ${YOLO_DIR}/cfg/activity_wework_tinyv1_Scene4.data ${cfg_file} ${weights_file} "$f" -dont_show
#    new_f_name="${f##*/}"
#    echo ${new_f_name}
#    mv ${YOLO_DIR}/predictions.jpg ${PRED_DIR}/${new_f_name}
# done

# create mAP of testing set and save to result.txt
${YOLO_DIR}/darknet detector map ${YOLO_DIR}/cfg/yolov3_ww_aug_v1/activity_wework_leaveout.data ${cfg_file} ${weights_file} -thresh 0.25 -iou_thresh 0.25 -dont_show -ext_output >> ${PRED_DIR}/result.txt


