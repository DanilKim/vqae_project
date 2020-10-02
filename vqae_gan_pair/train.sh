python pretrain.py --pretrain 'VQA' --epochs 20 --train_set 'VQA2' --val_set 'VQA2' --output 'saved_models/semi_pA20_pE20_GAN_VQAE'
python pretrain.py --pretrain 'VQE' --epochs 20 --train_set 'VQAE' --val_set 'VQAE' --pretrained 'pretrained_VQA.pth' --output 'saved_models/semi_pA20_pE20_GAN_VQAE'
python main.py --epochs 30 --train_set 'VQAS' --val_set 'VQAF' --output 'saved_models/semi_pA20_pE20_GAN_VQAE' --pretrained 'pretrained_VQE.pth'
