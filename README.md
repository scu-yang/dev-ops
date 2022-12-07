# Dev-Ops Index

## dicom

## docker

+ kafka-single-docker-compose.yaml

```shell
docker-compose -f kafka-single-docker-compose.yaml up -d
```

+ centos-java11-opencv460-dockerfile

```shell
# 打包镜像
docker build  -f ubuntu-java11-opecv460-dockerfile -t java11-opencv460:v20221207
# 上传到中央仓库
docker tag b32320b21028  loorr/java11-opencv460:v20221217
docker push loorr/java11-opencv460:v20221217
```