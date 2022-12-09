CKPT_SIMCLR0=save/SimCLR/cifar10_models/cifar10_resnet18_lr_0.05_decay_0.0001_bsz_256_temp_0.07_cosine_seed_0/model_200.pth
CKPT_SIMCLR10=save/SimCLR/cifar10_models/cifar10_resnet18_lr_0.05_decay_0.0001_bsz_256_temp_0.07_cosine_seed_10/model_200.pth
CKPT_SIMCLR20=save/SimCLR/cifar10_models/cifar10_resnet18_lr_0.05_decay_0.0001_bsz_256_temp_0.07_cosine_seed_20/model_200.pth
CKPT_SIMCLR30=save/SimCLR/cifar10_models/cifar10_resnet18_lr_0.05_decay_0.0001_bsz_256_temp_0.07_cosine_seed_30/model_200.pth


CKPT_SUPCON0=save/SupCon/cifar10_models/cifar10_resnet18_lr_0.05_decay_0.0001_bsz_256_temp_0.07_cosine_seed_0/model_200.pth
CKPT_SUPCON10=save/SupCon/cifar10_models/cifar10_resnet18_lr_0.05_decay_0.0001_bsz_256_temp_0.07_cosine_seed_10/model_200.pth
CKPT_SUPCON20=save/SupCon/cifar10_models/cifar10_resnet18_lr_0.05_decay_0.0001_bsz_256_temp_0.07_cosine_seed_20/model_200.pth
CKPT_SUPCON30=save/SupCon/cifar10_models/cifar10_resnet18_lr_0.05_decay_0.0001_bsz_256_temp_0.07_cosine_seed_30/model_200.pth

# Task 1 (0,1,2,3,4 -- aquatic mammals)
# Task 2 (95,96,97,98,99 -- vehicles 2)
# Task 3 (6,11,16,21,26 -- all superclasses different)
# Task 4 (56,58,62,66,68 -- 3 superclasses)

tasks=("0,1,2,3,4" "95,96,97,98,99" "6,11,16,21,26" "56,58,62,66,68")

for task in "${tasks[@]}"
do
    python3 main_linear.py --cosine --method SimCLR --dataset cifar100 --task_labels $task --seed 0 --ckpt $CKPT_SIMCLR0 --pretrained_tag simclr_on_cifar10
    python3 main_linear.py --cosine --method SimCLR --dataset cifar100 --task_labels $task --seed 10 --ckpt $CKPT_SIMCLR10 --pretrained_tag simclr_on_cifar10
    python3 main_linear.py --cosine --method SimCLR --dataset cifar100 --task_labels $task --seed 20 --ckpt $CKPT_SIMCLR20 --pretrained_tag simclr_on_cifar10
    python3 main_linear.py --cosine --method SimCLR --dataset cifar100 --task_labels $task --seed 30 --ckpt $CKPT_SIMCLR30 --pretrained_tag simclr_on_cifar10


    python3 main_linear.py --cosine --method SupCon --dataset cifar100 --task_labels $task --seed 0 --ckpt $CKPT_SUPCON0 --pretrained_tag supcon_on_cifar10
    python3 main_linear.py --cosine --method SupCon --dataset cifar100 --task_labels $task --seed 10 --ckpt $CKPT_SUPCON10 --pretrained_tag supcon_on_cifar10
    python3 main_linear.py --cosine --method SupCon --dataset cifar100 --task_labels $task --seed 20 --ckpt $CKPT_SUPCON20 --pretrained_tag supcon_on_cifar10
    python3 main_linear.py --cosine --method SupCon --dataset cifar100 --task_labels $task --seed 30 --ckpt $CKPT_SUPCON30 --pretrained_tag supcon_on_cifar10
done    

