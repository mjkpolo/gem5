# This builds in the gem5 directory and assumes that gem5-resources is checked out in the gem5 directory
# PANNOTIA
docker run --rm -u ${UID}:${GID} -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID gcr.io/gem5-test/gcn-gpu:latest bash -c 'cd gem5-resources/src/gpu/pannotia/bc ;  make clean '

docker run --rm -u ${UID}:${GID} -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID gcr.io/gem5-test/gcn-gpu:latest bash -c 'cd gem5-resources/src/gpu/pannotia/color ;  make clean '

docker run --rm -u ${UID}:${GID} -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID gcr.io/gem5-test/gcn-gpu:latest bash -c 'cd gem5-resources/src/gpu/pannotia/fw ;  make clean '

docker run --rm -u ${UID}:${GID} -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID gcr.io/gem5-test/gcn-gpu:latest bash -c 'cd gem5-resources/src/gpu/pannotia/mis ;  make clean '

docker run --rm -u ${UID}:${GID} -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID gcr.io/gem5-test/gcn-gpu:latest bash -c 'cd gem5-resources/src/gpu/pannotia/pagerank ;  make clean '

docker run --rm -u ${UID}:${GID} -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID gcr.io/gem5-test/gcn-gpu:latest bash -c 'cd gem5-resources/src/gpu/pannotia/sssp ;  make clean '

# Nightly tests (heterosync, square) - There is a bug here for heterosync. Trying to rm -f a directory (missing -r)
#docker run --rm -u ${UID}:${GID} -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID gcr.io/gem5-test/gcn-gpu:latest bash -c 'cd gem5-resources/src/gpu/heterosync ; GEM5_PATH=../../../../  make clean '
rm -rf gem5-resources/src/gpu/heterosync/bin

docker run --rm -u ${UID}:${GID} -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID gcr.io/gem5-test/gcn-gpu:latest bash -c 'cd gem5-resources/src/gpu/square ; GEM5_PATH=../../../../  make clean '

# HACC
rm -rf gem5-resources/src/gpu/halo-finder/src/hip

# PENNANT
docker run --rm -u ${UID}:${GID} -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID gcr.io/gem5-test/gcn-gpu:latest bash -c 'cd gem5-resources/src/gpu/pennant ; make clean '

# LULESH
docker run --rm -u ${UID}:${GID} -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID gcr.io/gem5-test/gcn-gpu:latest bash -c 'cd gem5-resources/src/gpu/lulesh ; make clean '

# DNNMark - Multiple steps
rm -rf gem5-resources/src/gpu/DNNMark/build
rm -rf gem5-resources/src/gpu/DNNMark/cachefiles
rm -rf gem5-resources/src/gpu/DNNMark/generate_rand_data

# Not tested but exist : hip-samples
docker run --rm -u ${UID}:${GID} -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID gcr.io/gem5-test/gcn-gpu:latest bash -c 'cd gem5-resources/src/gpu/hip-samples ; GEM5_PATH=../../../../  make clean '
