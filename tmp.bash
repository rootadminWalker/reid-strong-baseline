sleep 14400
python3 tools/train_lightning.py --config_file=configs/am-softmax_triplet_with_center.yml MODEL.NAME "resnet50_ibn_a" MODEL.DEVICE_ID "'0'" DATASETS.ROOT_DIR "/media/root_walker/DATA/datasets" DATASETS.NAMES "('market1501')" OUTPUT_DIR "/media/root_walker/DATA/outputs/Market1501_outputs/2023_03_17/AM-Softmax-no-center" SOLVER.CENTER_LOSS_WEIGHT 0.0000001
