export WORLD_SIZE=2
export RENDEZVOUS=env://
export MASTER_ADDR= #IP address of Server (not client)
export MASTER_PORT=
export RANK=0 # This is the Server
export GLOO_SOCKET_IFNAME=
export OMP_NUM_THREADS=$(nproc)
export KMP_AFFINITY=granularity=fine,compact,1,0

