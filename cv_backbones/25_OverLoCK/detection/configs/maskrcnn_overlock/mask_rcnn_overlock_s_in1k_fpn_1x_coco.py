_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
dims = [64, 128, 448, 640]
model = dict(
    backbone=dict(
        _delete_=True,
        type='overlock_s',
        pretrained=True,
        drop_path_rate=0.4
    ),
    neck=dict(
        type='FPN',
        in_channels=dims,
        out_channels=256,
        num_outs=5))

###########################################################################################################
# https://github.com/Sense-X/UniFormer/blob/main/object_detection/exp/mask_rcnn_1x_hybrid_small/config.py
# We follow uniformer's optimizer and lr schedule
optimizer = dict(_delete_=True, type='AdamW', lr=0.0002, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

evaluation = dict(save_best='auto')
checkpoint_config = dict(interval=1, max_keep_ckpts=1, save_last=True)