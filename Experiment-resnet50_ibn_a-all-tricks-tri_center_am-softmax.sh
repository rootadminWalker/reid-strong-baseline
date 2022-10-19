# Experiment all tricks with center loss : 256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005
# Dataset 1: market1501
# imagesize: 256x128
# batchsize: 16x4
# warmup_step 10
# random erase prob 0.5
# labelsmooth: on
# last stride 1
# bnneck on
# with center loss
# AM-softmax s = 30, m = 0.35

python3 tools/train.py --config_file='configs/am-softmax_triplet_with_center.yml' MODEL.PRETRAIN_PATH "('pretrained_weights/r50_ibn_a.pth')" MODEL.DEVICE_ID "('1')" DATASETS.NAMES "('market1501')" MODEL.NAME  "('resnet50_ibn_a')" OUTPUT_DIR "('/home/matthew/outputs/Experiment-resnet50_ibn_a-all-tricks-tri_center_am-softmax-256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005')"
