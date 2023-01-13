NAME="detection-and-classification-server"
version="v"$(date "+%Y%m%d-%H%M%S")
docker build  -f  Dockerfile-GPU -t $NAME:$version .

IS_RUNNING=$(docker ps -a | grep $NAME | wc -l)
if [[ $IS_RUNNING == 1 ]]; then
    echo "container $NAME is running..."
    docker stop $NAME
    echo "container $NAME is stop ..."
    docker rm $NAME
    echo "container $NAME is remove ..."
fi

echo "container $NAME is remove ..."
docker run -i -t -d -p 5012:8080 \
 --restart=always  --gpus all --name $NAME  $NAME:$version
echo "container $NAME is start ..."
