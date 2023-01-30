#!/bin/bash

DEVICE_TYPE='gpu'
NUM_EPOCHS=10
HEATMAP_BATCH_SIZE=100
GPU_NUMBER=0

PATCH_MODEL_PATH='models/sample_patch_model.p'
IMAGE_MODEL_PATH='models/ImageOnly__ModeImage_weights.p'
IMAGEHEATMAPS_MODEL_PATH='models/ImageHeatmaps__ModeImage_weights.p'

SAMPLE_SINGLE_OUTPUT_PATH='/kaggle/working/sample_single_output'
export PYTHONPATH=$(pwd):$PYTHONPATH


echo 'Stage 1: Crop Mammograms'
python3 src/cropping/crop_single.py \
    --mammogram-path $1 \
    --view $2 \
    --cropped-mammogram-path ${SAMPLE_SINGLE_OUTPUT_PATH}/cropped.png \
    --metadata-path ${SAMPLE_SINGLE_OUTPUT_PATH}/cropped_metadata.pkl

echo 'Stage 2: Extract Centers'
python3 src/optimal_centers/get_optimal_center_single.py \
    --cropped-mammogram-path ${SAMPLE_SINGLE_OUTPUT_PATH}/cropped.png \
    --metadata-path ${SAMPLE_SINGLE_OUTPUT_PATH}/cropped_metadata.pkl
    
echo 'Stage 4a: Run Classifier (Image)'
python3 src/modeling/run_model_single.py \
    --view $2 \
    --model-path ${IMAGE_MODEL_PATH} \
    --cropped-mammogram-path ${SAMPLE_SINGLE_OUTPUT_PATH}/cropped.png \
    --metadata-path ${SAMPLE_SINGLE_OUTPUT_PATH}/cropped_metadata.pkl \
    --use-augmentation \
    --num-epochs ${NUM_EPOCHS} \
    --device-type ${DEVICE_TYPE} \
    --gpu-number ${GPU_NUMBER}
