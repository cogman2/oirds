#!/bin/bash
service docker stop
ifconfig docker0 down
iptables -t nat -F
brctl delbr docker0
#docker daemon --storage-opt=dm.basesize=20G &
service docker start
 docker rm -v $( docker ps -a -q -f status=exited)
 docker rmi $( docker images -f "dangling=true" -q)
 docker run -v $(which docker):/bin/docker -v /var/run/docker.sock:/var/run/docker.sock -v $(readlink -f /var/lib/docker):/var/lib/docker --rm martin/docker-cleanup-volumes
