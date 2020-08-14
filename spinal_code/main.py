import json
import sys
import time

import torch
import argparse

from torch.utils.data import DataLoader
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from spinal_code.core.disease.data_loader import DisDataSet
from spinal_code.core.disease.evaluation import Evaluator
from spinal_code.core.disease.model import DiseaseModelBase
from spinal_code.core.key_point import KeyPointModel, NullLoss
from spinal_code.core.structure import construct_studies

from zoo.pipeline.api.torch import TorchModel, TorchLoss
from zoo.common.nncontext import *
from torch.utils.data import DataLoader
from zoo.pipeline.estimator import *
from zoo.pipeline.api.keras.optimizers import Adam
from bigdl.optim.optimizer import MaxEpoch, EveryEpoch
from zoo.feature.common import FeatureSet
from zoo.ray import RayContext

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, required=True,
                help='The directory of the data.')
    parser.add_argument('--num_workers', '-n', type=int, default=1,
                help="The number of Horovod workers launched for distributed training.")
    parser.add_argument('--worker_cores', '-c', type=int, default=4,
                help='The number of cores allocated for each worker.')
    parser.add_argument('--epochs', '-e', type=int, default=20,
                help='The number of epochs to train the model.')
    parser.add_argument('--master', '-m', type=str,
                help='The Spark master address of a standalone cluster if any.')
    parser.add_argument('--use_bf16', type=bool, default=False,
                help='Whether to use BF16 for model training if you are running on a server with BF16 support.')

    opt, _ = parser.parse_known_args()
    start_time = time.time()
    # sc = init_spark_on_local(4, conf={"spark.driver.memory": "40g"})
    sc = init_spark_standalone(
            master=opt.master,
            num_executors=opt.num_workers,
            executor_cores=opt.worker_cores,
            driver_memory="40g")

    backbone = resnet_fpn_backbone('resnet50', False)
    kp_model = KeyPointModel(backbone)
    dis_model = DiseaseModelBase(kp_model, sagittal_size=(512, 512))
    for name, parameter in dis_model.named_parameters():
        if 'layer_blocks.1' in name or 'layer_blocks.2' in name or 'layer_blocks.3' in name:
            parameter.requires_grad_(False)
    print(dis_model)

    def train_dataloader():
        train_studies, train_annotation, train_counter = construct_studies(
            opt.data + '/lumbar_train150', opt.data + '/lumbar_train150_annotation.json', multiprocessing=False)

        train_images = {}
        for study_uid, study in train_studies.items():
            frame = study.t2_sagittal_middle_frame
            train_images[(study_uid, frame.series_uid, frame.instance_uid)] = frame.image
        train_dataset = DisDataSet(studies=train_studies, annotations=train_annotation, sagittal_size=dis_model.sagittal_size,
                                   transverse_size=dis_model.sagittal_size, k_nearest=0, prob_rotate=1,
                                   max_angel=180, num_rep=10, max_dist=8)
        return DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=3,
                          pin_memory=False, collate_fn=train_dataset.collate_fn)
        
    valid_studies, valid_annotation, valid_counter = construct_studies(
            opt.data + '/train/', opt.data + '/lumbar_train51_annotation.json', multiprocessing=False)
    valid_evaluator = Evaluator(
        dis_model, valid_studies, opt.data + '/lumbar_train51_annotation.json', num_rep=20, max_dist=6,
    )
    metrics_values = valid_evaluator(dis_model, None, valid_evaluator.metric)
    for a, b in metrics_values:
        print('valid {}: {}'.format(a, b))

    az_model = TorchModel.from_pytorch(dis_model)
    zoo_loss = TorchLoss.from_pytorch(NullLoss()) 

    train_featureset = FeatureSet.pytorch_dataloader(train_dataloader, "", "")
    zooOptimizer = Adam(lr=1e-5)
    estimator = Estimator(az_model, optim_methods=zooOptimizer)
    estimator.train_minibatch(train_featureset, zoo_loss, end_trigger=MaxEpoch(1),
                              checkpoint_trigger=EveryEpoch())

    valid_evaluator = Evaluator(
        az_model.to_pytorch(), valid_studies, opt.data + '/lumbar_train51_annotation.json', num_rep=20, max_dist=6,
    )
    metrics_values = valid_evaluator(az_model.to_pytorch(), None, valid_evaluator.metric)
    for a, b in metrics_values:
        print('valid {}: {}'.format(a, b))

    # 预测
    testA_studies = construct_studies(opt.data + '/lumbar_testA50/', multiprocessing=False)

    result = []
    for study in testA_studies.values():
        result.append(az_model.to_pytorch().eval()(study))

    with open('predictions/baseline.json', 'w') as file:
        json.dump(result, file)
    print('task completed, {} seconds used'.format(time.time() - start_time))
    sc.stop()
    if not opt.master:
        stop_spark_standalone()
