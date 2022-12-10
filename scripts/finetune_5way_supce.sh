CKPT_SUPCE0=save/SupCE/cifar10_models/cifar10_resnet18_lr_0.2_decay_0.0001_bsz_256_cosine_seed_0/model_200.pth
CKPT_SUPCE10=save/SupCE/cifar10_models/cifar10_resnet18_lr_0.2_decay_0.0001_bsz_256_cosine_seed_10/model_200.pth
CKPT_SUPCE20=save/SupCE/cifar10_models/cifar10_resnet18_lr_0.2_decay_0.0001_bsz_256_cosine_seed_20/model_200.pth
CKPT_SUPCE30=save/SupCE/cifar10_models/cifar10_resnet18_lr_0.2_decay_0.0001_bsz_256_cosine_seed_30/model_200.pth


# Task 1 (0,1,2,3,4 -- aquatic mammals)
# Task 2 (95,96,97,98,99 -- vehicles 2)
# Task 3 (6,11,16,21,26 -- all superclasses different)
# Task 4 (56,58,62,66,68 -- 3 superclasses)

tasks=("0,1,2,3,4" "95,96,97,98,99" "6,11,16,21,26" "56,58,62,66,68")

for task in "${tasks[@]}"
do
    python3 main_linear.py --cosine --method SupCE --dataset cifar100 --task_labels $task --seed 0 --ckpt $CKPT_SUPCE0 --pretrained_tag supce_on_cifar10
    python3 main_linear.py --cosine --method SupCE --dataset cifar100 --task_labels $task --seed 10 --ckpt $CKPT_SUPCE10 --pretrained_tag supce_on_cifar10
    python3 main_linear.py --cosine --method SupCE --dataset cifar100 --task_labels $task --seed 20 --ckpt $CKPT_SUPCE20 --pretrained_tag supce_on_cifar10
    python3 main_linear.py --cosine --method SupCE --dataset cifar100 --task_labels $task --seed 30 --ckpt $CKPT_SUPCE30 --pretrained_tag supce_on_cifar10
done    

