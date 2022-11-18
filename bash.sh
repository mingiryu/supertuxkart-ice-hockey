export DOCKER_BUILDKIT=1

docker run -it \
    -v $(pwd):/code \
    $1 stk bash