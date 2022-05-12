import os
import pathlib
import pickle
import shutil
import time
from functools import partial
import copy
import fire
import numpy as np
import torch, gc
from torch import nn
from torch.nn import functional as F
gc.collect()
torch.cuda.empty_cache()
from google.protobuf import text_format
from tensorboardX import SummaryWriter
from torchsummary import summary
import torchplus
import second.data.kitti_common as kitti
from second.builder import target_assigner_builder, voxel_builder
from second.data.preprocess import merge_second_batch
from second.protos import pipeline_pb2
from second.pytorch.builder import (box_coder_builder, input_reader_builder,
                                      lr_scheduler_builder, optimizer_builder,
                                      second_builder)
from second.utils.eval import get_coco_eval_result, get_official_eval_result
from second.utils.progress_bar import ProgressBar

class distillModel:
    class student(nn.Module):
        def __init__(self, Model):
            super().__init__()
            self.name = 'StudentNet'
            self.layers = []
            for i in Model.modules():
                self.layers.append(i)
    def __init__(self, Model):
        self.name = 'TeacherNet'
        self.teacher = Model

def example_convert_to_torch(example, dtype=torch.float32,
                             device=None) -> dict:
    device = device or torch.device("cuda:0")
    example_torch = {}
    float_names = [
        "voxels", "anchors", "reg_targets", "reg_weights", "bev_map", "rect",
        "Trv2c", "P2"
    ]

    for k, v in example.items():
        if k in float_names:
            example_torch[k] = torch.as_tensor(v, dtype=dtype, device=device)
        elif k in ["coordinates", "labels", "num_points"]:
            example_torch[k] = torch.as_tensor(
                v, dtype=torch.int32, device=device)
        elif k in ["anchors_mask"]:
            example_torch[k] = torch.as_tensor(
                v, dtype=torch.uint8, device=device)
        else:
            example_torch[k] = v
    return example_torch
def split(config_path, model_dir,
          result_path=None,
          create_folder=False,
          display_step=50,
          summary_step=5,
          pickle_result=True):
    """train a VoxelNet model specified by a config file.
    """
    if create_folder:
        if pathlib.Path(model_dir).exists():
            model_dir = torchplus.train.create_folder(model_dir)

    model_dir = pathlib.Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    eval_checkpoint_dir = model_dir / 'eval_checkpoints'
    eval_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if result_path is None:
        result_path = model_dir / 'results'
    config_file_bkp = "pipeline.config"
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    shutil.copyfile(config_path, str(model_dir / config_file_bkp))
    input_cfg = config.train_input_reader
    eval_input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config
    class_names = list(input_cfg.class_names)
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    ######################
    # BUILD VOXEL GENERATOR
    ######################
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    
    ######################
    # BUILD TARGET ASSIGNER
    ######################
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)
    ######################
    # BUILD NET
    ######################
    center_limit_range = model_cfg.post_center_limit_range
    net = second_builder.build(model_cfg, voxel_generator, target_assigner)
    net.cuda()
    ######################
    # LOAD CHECKPOINT
    ######################
    torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    gstep = net.get_global_step() - 1
    optimizer_cfg = train_cfg.optimizer
    if train_cfg.enable_mixed_precision:
        net.half()
        net.metrics_to_float()
        net.convert_norm_to_float(net)
    optimizer = optimizer_builder.build(optimizer_cfg, net.parameters())
    if train_cfg.enable_mixed_precision:
        loss_scale = train_cfg.loss_scale_factor
        mixed_optimizer = torchplus.train.MixedPrecisionWrapper(
            optimizer, loss_scale)
    else:
        mixed_optimizer = optimizer
    # must restore optimizer AFTER using MixedPrecisionWrapper
    torchplus.train.try_restore_latest_checkpoints(model_dir,
                                                   [mixed_optimizer])
    lr_scheduler = lr_scheduler_builder.build(optimizer_cfg, optimizer, gstep)
    if train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32
    ######################
    # PRINT
    ######################
    #net_train = torch.nn.DataParallel(net).cuda()
    #print("num_trainable parameters:", len(list(net.parameters())))
    i = 0
    layers = []
    for n, p in net.named_parameters():
        print(i,n, p.shape)
        i+=1
    layers = [i for i in net.modules()]
    
    #print(model_cfg)
    #print(mixed_optimizer)
    #for param in net.parameters():
    #    print(param)
    #for n,p in layers[0].named_parameters():
    #    print(n,p.shape)
    #for i in range(len(layers)):
    #    print(layers[i].named_parameters())
    #print(net)
    #
    #########################################
    # LOAD DATA
    #########################################
    dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=True,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)
    eval_dataset = input_reader_builder.build(
        eval_input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)

    def _worker_init_fn(worker_id):
        time_seed = np.array(time.time(), dtype=np.int32)
        np.random.seed(time_seed + worker_id)
        print(f"WORKER {worker_id} seed:", np.random.get_state()[1][0])
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=input_cfg.batch_size,
        shuffle=True,
        num_workers=input_cfg.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch,
        worker_init_fn=_worker_init_fn)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_input_cfg.batch_size,
        shuffle=False,
        num_workers=eval_input_cfg.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch)
    data_iter = iter(dataloader)
    #########################################
    # Find Max
    #########################################
    #data_iter = iter(dataloader)
    #mx = [0,0,0]
    #for i in data_iter:
    #    tmp = [len(i['voxels']),len(i['voxels'][0]),len(i['voxels'][0][1])]
    #    if mx[0] < tmp[0]:
    #        mx = tmp
    #print(mx)

    #########################################
    # Student Net
    #########################################
    print(model_cfg)
    student = second_builder.build(model_cfg, voxel_generator, target_assigner, False)
    student.cuda()
    print(student.teacher) 

    #########################################
    # Load parameter to Student Net
    #########################################
    print(student.head_encoder)
    student.rpn = copy.deepcopy(net.rpn)
    student.voxel_feature_extractor = copy.deepcopy(net.voxel_feature_extractor)
    student.middle_feature_extractor = copy.deepcopy(net.middle_feature_extractor)
    for n,p in student.named_parameters():
        print(n,p.shape)
    #print(next(student.named_parameters()))
    #print(next(net.named_parameters()))
    #########################################
    # Validation
    #########################################
    #print(torch.eq(next(net.named_parameters())[1],next(student.named_parameters())[1]))
    #print(id(next(net.named_parameters())[1]),id(next(student.named_parameters())[1]))
    #netdict = { n:p for n,p in net.named_parameters()}
    #studentdict = {n:p for n,p in student.named_parameters()}
    #for i in netdict:
    #    print(i,"; same tensor: ",torch.equal(netdict[i],studentdict[i]))
    #########################################
    # Copy Teacher
    #########################################
    teacher_model_dir = (model_dir).joinpath('teacher')
    if not teacher_model_dir.exists():
        teacher_model_dir.mkdir()
        extensions = ["*.tckpt","*.json","*.txt"]
        all_files = []
        for ext in extensions:
            all_files.extend(model_dir.glob(ext))
        #files = get_files(('*.txt', '*.json', '*.tckpt'))
        for files in all_files:
            shutil.copy(files,teacher_model_dir)
    #########################################
    # Load Teacher
    #########################################
    teacher = second_builder.build(model_cfg, voxel_generator, target_assigner,True)
    torchplus.train.try_restore_latest_checkpoints(teacher_model_dir, [teacher])
    #########################################
    # Create Student
    #########################################
    student_model_dir = (model_dir).joinpath('student')
    if not student_model_dir.exists():
        student_model_dir.mkdir()
    torchplus.train.save_models(student_model_dir, [student, optimizer], net.get_global_step())
    #########################################
    # Load Student
    #########################################
    studentTmp = second_builder.build(model_cfg, voxel_generator, target_assigner,False) 
    torchplus.train.try_restore_latest_checkpoints(student_model_dir, [studentTmp])
    #########################################
    # Training
    #########################################
    student.freezeAll()
    for name,param in student.named_parameters():
        print(name,param.requires_grad)
    try:
        example = next(data_iter)
    except StopIteration:
        print("end epoch")
        if clear_metrics_every_epoch:
            net.clear_metrics()
        data_iter = iter(dataloader)
        example = next(data_iter)
    example_torch = example_convert_to_torch(example, float_dtype)

    batch_size = example["anchors"].shape[0]
    print(example_torch)
    ret_dict = net(example_torch)
    print(ret_dict)
if __name__ == '__main__':
    fire.Fire()
