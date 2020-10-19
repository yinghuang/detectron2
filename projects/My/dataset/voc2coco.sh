#! /bin/bash
source activate mmdetpy37
python voc2coco.py \
--ann_dir /media/ai/Data/hy/datasets/pospal_hb/Annotations \
--ann_ids /media/ai/Data/hy/datasets/pospal_hb/sets/79210_20865_id.txt \
--labels /media/ai/Data/hy/datasets/pospal_hb/class_names.txt \
--ext xml \
--output /media/ai/Data/hy/datasets/pospal_hb/coco_annotations/79210_20865.json

