_base_ = [
    '../_base_/models/cascade_mask_rcnn_wsfvit_fpn_giou_4conv1f.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_3x.py', '../_base_/default_runtime.py'
]

pretrained = '/path/to/pretrained/checkpoint'  

model = dict(
    backbone=dict(
        type='WSFVit_feat',
        model_name='wsfvit_m_224',
        pretrained_path=pretrained,
        drop_rte=0.,
        drop_path_rate=0.6),
    neck=dict(in_channels=[80, 160, 320, 640]))

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)

# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=4,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
