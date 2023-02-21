#!/usr/bin/bash

# Set IP address of the target board
export IP=192.168.8.21

# Copy files to the target board
scp -r ../build/aarch64 root@$IP:/tmp
scp -r ../test root@$IP:/tmp

# Connect to the target board and run the executable
ssh -tt root@$IP << remotessh
export LD_LIBRARY_PATH=/tmp/aarch64/lib
/tmp/aarch64/bin/hmap_generator_test /tmp/test/hmap-v3-e99-upv1-uint8.tmfile /tmp/test/barcode-test-1.png
exit
remotessh