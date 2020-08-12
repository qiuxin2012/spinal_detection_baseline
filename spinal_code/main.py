import json
import sys
import time

import torch
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

if __name__ == '__main__':
    start_time = time.time()
    sc = init_spark_on_local(4, conf={"spark.driver.memory": "40g"})

    backbone = resnet_fpn_backbone('resnet50', False)
    kp_model = KeyPointModel(backbone)
    dis_model = DiseaseModelBase(kp_model, sagittal_size=(512, 512))
    for name, parameter in dis_model.named_parameters():
        if 'layer_blocks.1' in name or 'layer_blocks.2' in name or 'layer_blocks.3' in name:
            parameter.requires_grad_(False)
    print(dis_model)

    def train_dataloader():
        train_studies, train_annotation, train_counter = construct_studies(
            '/home/cpx/qiuxin/ali/spinal_detection_baseline/data/lumbar_train150', '/home/cpx/qiuxin/ali/spinal_detection_baseline/data/lumbar_train150_annotation.json', multiprocessing=False)
        valid_studies, valid_annotation, valid_counter = construct_studies(
            '/home/cpx/qiuxin/ali/spinal_detection_baseline/data/lumbar_train51/', '/home/cpx/qiuxin/ali/spinal_detection_baseline/data/lumbar_train51_annotation.json', multiprocessing=False)

        train_images = {}
        for study_uid, study in train_studies.items():
            frame = study.t2_sagittal_middle_frame
            train_images[(study_uid, frame.series_uid, frame.instance_uid)] = frame.image
        train_dataset = DisDataSet(studies=train_studies, annotations=train_annotation, sagittal_size=dis_model.sagittal_size,
                                   transverse_size=dis_model.sagittal_size, k_nearest=0, prob_rotate=1,
                                   max_angel=180, num_rep=10, max_dist=8)
        return DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=3,
                          pin_memory=False, collate_fn=train_dataset.collate_fn)

    az_model = TorchModel.from_pytorch(dis_model)
    zoo_loss = TorchLoss.from_pytorch(NullLoss()) 

    train_featureset = FeatureSet.pytorch_dataloader(train_dataloader, "", "")
    zooOptimizer = Adam(lr=1e-5)
    estimator = Estimator(az_model, optim_methods=zooOptimizer)
    estimator.train_minibatch(train_featureset, zoo_loss, end_trigger=MaxEpoch(20),
                              checkpoint_trigger=EveryEpoch())

    # 预测
    testA_studies = construct_studies('/home/cpx/qiuxin/ali/spinal_detection_baseline/data/lumbar_testA50/', multiprocessing=False)

    result = []
    for study in testA_studies.values():
        result.append(az_model.to_pytorch().eval()(study))

    with open('predictions/baseline.json', 'w') as file:
        json.dump(result, file)
    print('task completed, {} seconds used'.format(time.time() - start_time))
