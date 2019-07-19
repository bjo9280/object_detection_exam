# object_detection_exam

### install

<https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md> 

### prepare

  python -u create_tf_record_exam.py \
    --label_map_path=label_map_exam.pbtxt \
    --data_dir=. \
    --output_dir=./tfrecords



### train

PIPELINE_CONFIG_PATH=./faster_rcnn_resnet101_datasetname.config
MODEL_DIR=save
NUM_TRAIN_STEPS=${1-50000}
NUM_EVAL_STEPS=2000

  python ../object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --num_eval_steps=${NUM_EVAL_STEPS} \
    --alsologtostderr

### eval

PIPELINE_CONFIG_PATH=./faster_rcnn_resnet101_datasetname.config
MODEL_DIR=save
CHECKPOINT_DIR=save
CHECKPOINT=save/model.ckpt

  python ../object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --checkpoint_dir=${CHECKPOINT_DIR} \
    --run_once=True



### export

OBJECT_DETECTION_CONFIG=./faster_rcnn_resnet101_datasetname.config
YOUR_LOCAL_CHK_DIR=save
YOUR_LOCAL_EXPORT_DIR=export

  python ../object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${OBJECT_DETECTION_CONFIG} \
    --trained_checkpoint_prefix ${YOUR_LOCAL_CHK_DIR}/model.ckpt-50000 \
    --output_directory ${YOUR_LOCAL_EXPORT_DIR}



