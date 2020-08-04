import json
import sys
import time
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torch.distributed as dist

from spine_code.core.disease.data_loader import DisDataSet, DisDataLoader
from spine_code.core.disease.evaluation import Evaluator
from spine_code.core.disease.model import DiseaseModelBase
from spine_code.core.key_point import KeyPointModel, NullLoss
from spine_code.core.structure import construct_studies

sys.path.append('../nn_tools/')
from nn_tools import torch_utils

parser = argparse.ArgumentParser(description='Tianchi Model Testing')
parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://10.239.45.18:7689', type=str, help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--bf16', action='store_true', default=False, help='enable bf16 operator')

if __name__ == '__main__':
    args = parser.parse_args()
    start_time = time.time()
    train_studies, train_annotation, train_counter = construct_studies(
        'data/lumbar_train150', 'data/lumbar_train150_annotation.json', multiprocessing=True)
    valid_studies, valid_annotation, valid_counter = construct_studies(
        'data/train/', 'data/lumbar_train51_annotation.json', multiprocessing=True)

    # set model parameters
    train_images = {}
    for study_uid, study in train_studies.items():
        frame = study.t2_sagittal_middle_frame
        train_images[(study_uid, frame.series_uid, frame.instance_uid)] = frame.image
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    backbone = resnet_fpn_backbone('resnet50', True)
    kp_model = KeyPointModel(backbone)
    dis_model = DiseaseModelBase(kp_model, sagittal_size=(512, 512))
    print(dis_model)

    # set training parameters
    train_dataset = DisDataSet(
        train_studies, train_annotation, prob_rotate=1, max_angel=180, num_rep=10, max_dist=8, 
        sagittal_size=dis_model.sagittal_size, transverse_size=dis_model.sagittal_size, k_nearest=0
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=2, num_workers=3, shuffle=False, sampler=train_sampler, collate_fn=train_dataset.collate_fn)

    valid_evaluator = Evaluator(
        dis_model, valid_studies, 'data/lumbar_train51_annotation.json', num_rep=20, max_dist=6,
    )
    
    dis_model = torch.nn.parallel.DistributedDataParallel(dis_model, find_unused_parameters=True)
    step_per_batch = len(train_dataloader)
    optimizer = torch.optim.AdamW(dis_model.parameters(), lr=1e-5)
    max_step = 20 * step_per_batch
    print(max_step)
    fit_result = torch_utils.fit(
        dis_model,
        train_data=train_dataloader,
        valid_data=None,
        optimizer=optimizer,
        max_step=max_step,
        loss=NullLoss(),
        metrics=[valid_evaluator.metric],
        is_higher_better=True,
        evaluate_per_steps=step_per_batch,
        evaluate_fn=valid_evaluator,
    )

    torch.save(dis_model.cpu().state_dict(), 'models/baseline.dis_model')
    
    testA_studies = construct_studies('data/lumbar_testA50/')

    result = []
    for study in testA_studies.values():
        result.append(dis_model.eval()(study, True))

    with open('predictions/baseline.json', 'w') as file:
        json.dump(result, file)
    print('task completed, {} seconds used'.format(time.time() - start_time))
