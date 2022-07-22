# python tools/test.py configs/resnet/resnet18_b16x8_cifar10.py \
# chek/resnet/resnet18_b16x8_cifar10_20210528-bd6371c8.pth --metrics accuracy

# python tools/test.py configs/resnet/resnet34_8xb16_cifar10.py \
# chek/resnet/resnet34_b16x8_cifar10_20210528-a8aa36a6.pth --metrics accuracy

# python tools/test.py configs/resnet/resnet50_8xb16_cifar10.py \
# chek/resnet/resnet50_b16x8_cifar10_20210528-f54bfad9.pth --metrics accuracy

# python tools/test.py configs/resnet/resnet101_8xb16_cifar10.py  \
# chek/resnet/resnet101_b16x8_cifar10_20210528-2d29e936.pth --metrics accuracy

python tools/test.py configs/resnet/resnet152_8xb16_cifar10.py  \
chek/resnet/resnet152_b16x8_cifar10_20210528-3e8e9178.pth --metrics accuracy

python tools/test.py configs/mobilenet_v2/mobilenet_v2_8xb16_cifar10.py  \
/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/maxin/work/classification-performance/work_dirs/mobilenet_v2_8xb16_cifar10/epoch_199.pth --metrics accuracy


python tools/test.py configs/densenet/densenet121_8xb16_cifar10.py  \
/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/maxin/work/classification-performance/work_dirs/densenet121_8xb16_cifar10/epoch_89.pth --metrics accuracy


python tools/test.py configs/regnet/regnetx_1.6gf_b32x8_cifar10.py  \
/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/maxin/work/classification-performance/work_dirs/regnetx_1.6gf_b32x8_cifar10/epoch_199.pth --metrics accuracy

python tools/test.py configs/seresnet/seresnet_8xb16_cifar10.py  \
/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/maxin/work/classification-performance/work_dirs/seresnet_8xb16_cifar10/epoch_199.pth --metrics accuracy





docker run -d \
    -v /share/Container/docker/chineseSubFinder/config:/config    \
    -v /share/Life:/media     \
    -e PUID=1026 \
    -e PGID=100 \
    -e PERMS=true        \
    -e TZ=Asia/Shanghai  \
    -e UMASK=022         \
    -p 19035:19035 \
    -p 19037:19037 \
    --name chinesesubfinder \
    --hostname chinesesubfinder \
    --log-driver "json-file" \
    --log-opt "max-size=100m" \
    allanpk716/chinesesubfinder