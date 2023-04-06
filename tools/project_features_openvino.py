import os
import sys
from collections import defaultdict

import numpy as np
import plotly.graph_objects as go
import torch
import torchvision.transforms as T
from sklearn.decomposition import PCA

from evaluate_openvino import OpenVINOModule

sys.path.append('.')
from data.datasets import init_dataset
from data import make_val_dataset
from utils import setup_cli, setup_loggers


def get_image_label(image_name):
    return image_name.split('_')[0]


def main(cfg):
    assert cfg.TEST.IMS_PER_BATCH == 1, "Not supported"

    random_labels = 10
    samples_count = 50
    full_count = 0
    axis_limit = 15

    pca = PCA(n_components=3)
    tmp_descriptors_dic = defaultdict(list)
    descriptors_dic = defaultdict(list)
    omit_pids = []
    val_dataset = init_dataset(
        cfg.DATASETS.VAL_NAMES,
        root=cfg.DATASETS.VAL_ROOT,
        aug_per_image=cfg.SOLVER.AUG_PER_IMG
    )
    _, val_loader, val_num_queries, val_num_classes = make_val_dataset(cfg, val_dataset)
    val_loader.dataset.transform = T.Compose([
        T.Resize(size=cfg.INPUT.SIZE_TRAIN),
        T.Lambda(lambda x: torch.tensor(np.array(x)))
    ])

    module = OpenVINOModule(cfg, val_num_classes, val_num_queries)

    for batch in val_loader:
        if full_count >= random_labels:
            break

        img, pid, _ = batch
        pid = pid[0]
        if pid in omit_pids:
            continue
        img = img.detach().numpy()[0]
        descriptor = module(img)
        if len(tmp_descriptors_dic[pid]) < samples_count:
            tmp_descriptors_dic[pid].append(descriptor)
        else:
            descriptors_dic[pid] = tmp_descriptors_dic[pid]
            omit_pids.append(pid)
            full_count += 1
    keys, descriptors = list(descriptors_dic.keys()), np.array(list(descriptors_dic.values()))

    print(descriptors.shape)
    descriptors = descriptors.reshape(random_labels * samples_count, descriptors.shape[-1])
    pca_features = pca.fit_transform(descriptors)
    marker_data = []
    for idx in range(random_labels):
        start_idx = idx * samples_count
        end_idx = (idx + 1) * samples_count
        marker_data.append(go.Scatter3d(
            x=pca_features[start_idx:end_idx, 0],
            y=pca_features[start_idx:end_idx, 1],
            z=pca_features[start_idx:end_idx, 2],
            marker=go.scatter3d.Marker(size=3),
            name=keys[idx],
            opacity=0.8,
            mode='markers'
        ))

    title = f"{cfg.MODEL.PRETRAIN_PATH.split(os.sep)[-2]}(--){cfg.MODEL.PRETRAIN_PATH.split(os.sep)[-1].split('.')[0]}"
    fig = go.Figure(
        data=marker_data,
        layout=go.Layout(
            title=title,
            scene=go.layout.Scene(
                xaxis={'range': [-axis_limit, axis_limit]},
                yaxis={'range': [-axis_limit, axis_limit]},
                zaxis={'range': [-axis_limit, axis_limit]}
            )
        )
    )
    fig.write_html(f'{cfg.OUTPUT_DIR}/{title}.html')
    fig.show()


if __name__ == '__main__':
    cfg, args = setup_cli()
    logger = setup_loggers(args)
    main(cfg)
