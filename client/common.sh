export WORLD_SIZE=2
export RENDEZVOUS=env://
export MASTER_ADDR=  #The Server's IP address, not this machine
export MASTER_PORT=
export RANK=1 #This is the Client
export GLOO_SOCKET_IFNAME=
export OMP_NUM_THREADS=$(nproc)
export KMP_AFFINITY=granularity=fine,compact,1,0
