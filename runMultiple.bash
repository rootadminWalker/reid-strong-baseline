## Increase LR
#echo "AM-Softmax-m-0.4-triplet-margin-8.0-baseLR-0.00045" | figlet
#python3 tools/train_lightning.py --config_file=configs/am-softmax_triplet_with_center.yml MODEL.NAME "resnet50_ibn_a" MODEL.DEVICE_ID "'0'" DATASETS.ROOT_DIR "/media/root_walker/DATA/datasets" DATASETS.NAMES "('market1501')" OUTPUT_DIR "/media/root_walker/DATA/outputs/Market1501_outputs/2023_03_06~11/AM-Softmax-m-0.4-triplet-margin-8.0-baseLR-0.00045" SOLVER.AM_M 0.4 SOLVER.MARGIN 8.0 SOLVER.BASE_LR 0.00045
#echo "AM-Softmax-m-0.4-triplet-margin-8.0-baseLR-0.0005" | figlet
#python3 tools/train_lightning.py --config_file=configs/am-softmax_triplet_with_center.yml MODEL.NAME "resnet50_ibn_a" MODEL.DEVICE_ID "'0'" DATASETS.ROOT_DIR "/media/root_walker/DATA/datasets" DATASETS.NAMES "('market1501')" OUTPUT_DIR "/media/root_walker/DATA/outputs/Market1501_outputs/2023_03_06~11/AM-Softmax-m-0.4-triplet-margin-8.0-baseLR-0.0005" SOLVER.AM_M 0.4 SOLVER.MARGIN 8.0 SOLVER.BASE_LR 0.0005
## Lower Weight Decay
#echo "AM-Softmax-m-0.4-triplet-margin-8.0-weightdecay-0.0001" | figlet
#python3 tools/train_lightning.py --config_file=configs/am-softmax_triplet_with_center.yml MODEL.NAME "resnet50_ibn_a" MODEL.DEVICE_ID "'0'" DATASETS.ROOT_DIR "/media/root_walker/DATA/datasets" DATASETS.NAMES "('market1501')" OUTPUT_DIR "/media/root_walker/DATA/outputs/Market1501_outputs/2023_03_06~11/AM-Softmax-m-0.4-triplet-margin-8.0-weightdecay-0.0001" SOLVER.AM_M 0.4 SOLVER.MARGIN 8.0 SOLVER.WEIGHT_DECAY 0.0001
# Apply both
#echo "AM-Softmax-m-0.4-triplet-margin-8.0-baseLR-0.00045-weightdecay-0.0001" | figlet
#python3 tools/train_lightning.py --config_file=configs/am-softmax_triplet_with_center.yml MODEL.NAME "resnet50_ibn_a" MODEL.DEVICE_ID "'0'" DATASETS.ROOT_DIR "/media/root_walker/DATA/datasets" DATASETS.NAMES "('market1501')" OUTPUT_DIR "/media/root_walker/DATA/outputs/Market1501_outputs/2023_03_06~11/AM-Softmax-m-0.4-triplet-margin-8.0-baseLR-0.00045-weightdecay-0.0001" SOLVER.AM_M 0.4 SOLVER.MARGIN 8.0 SOLVER.BASE_LR 0.00045 SOLVER.WEIGHT_DECAY 0.0001
#echo "AM-Softmax-m-0.4-triplet-margin-8.0-baseLR-0.0005-weightdecay-0.0001" | figlet
#python3 tools/train_lightning.py --config_file=configs/am-softmax_triplet_with_center.yml MODEL.NAME "resnet50_ibn_a" MODEL.DEVICE_ID "'0'" DATASETS.ROOT_DIR "/media/root_walker/DATA/datasets" DATASETS.NAMES "('market1501')" OUTPUT_DIR "/media/root_walker/DATA/outputs/Market1501_outputs/2023_03_06~11/AM-Softmax-m-0.4-triplet-margin-8.0-baseLR-0.0005-weightdecay-0.0001" SOLVER.AM_M 0.4 SOLVER.MARGIN 8.0 SOLVER.BASE_LR 0.0005 SOLVER.WEIGHT_DECAY 0.0001

# Increase margin steadily
#echo "AM-Softmax-triplet-margin-6.5" | figlet
#python3 tools/train_lightning.py --config_file=configs/am-softmax_triplet_with_center.yml MODEL.NAME "resnet50_ibn_a" MODEL.DEVICE_ID "'0'" DATASETS.ROOT_DIR "/media/root_walker/DATA/datasets" DATASETS.NAMES "('market1501')" OUTPUT_DIR "/media/root_walker/DATA/outputs/Market1501_outputs/2023_03_06~11/AM-Softmax-triplet-margin-6.5" SOLVER.MARGIN 6.5
echo "AM-Softmax-triplet-margin-4.5" | figlet
python3 tools/train_lightning.py --config_file=configs/am-softmax_triplet_with_center.yml MODEL.NAME "resnet50_ibn_a" MODEL.DEVICE_ID "'0'" DATASETS.ROOT_DIR "/media/root_walker/DATA/datasets" DATASETS.NAMES "('market1501')" OUTPUT_DIR "/media/root_walker/DATA/outputs/Market1501_outputs/2023_03_06~11/AM-Softmax-triplet-margin-4.5" SOLVER.MARGIN 4.5
echo "AM-Softmax-triplet-margin-5.0" | figlet
python3 tools/train_lightning.py --config_file=configs/am-softmax_triplet_with_center.yml MODEL.NAME "resnet50_ibn_a" MODEL.DEVICE_ID "'0'" DATASETS.ROOT_DIR "/media/root_walker/DATA/datasets" DATASETS.NAMES "('market1501')" OUTPUT_DIR "/media/root_walker/DATA/outputs/Market1501_outputs/2023_03_06~11/AM-Softmax-triplet-margin-5.0" SOLVER.MARGIN 5.0
echo "AM-Softmax-triplet-margin-5.5" | figlet
python3 tools/train_lightning.py --config_file=configs/am-softmax_triplet_with_center.yml MODEL.NAME "resnet50_ibn_a" MODEL.DEVICE_ID "'0'" DATASETS.ROOT_DIR "/media/root_walker/DATA/datasets" DATASETS.NAMES "('market1501')" OUTPUT_DIR "/media/root_walker/DATA/outputs/Market1501_outputs/2023_03_06~11/AM-Softmax-triplet-margin-5.5" SOLVER.MARGIN 5.5
