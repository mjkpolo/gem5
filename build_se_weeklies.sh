# This builds in the gem5 directory and assumes that gem5-resources is checked out in the gem5 directory
# Note: These uses m5ops so we specify GEM5_PATH for the make file.  I think only pannotia uses them.
# Similar Note: Assumes m5ops has already been built!
# PANNOTIA
docker run --rm -u ${UID}:${GID} -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID gcr.io/gem5-test/gcn-gpu:latest bash -c 'cd gem5-resources/src/gpu/pannotia/bc ;  GEM5_PATH=../../../../../  make gem5-fusion '

docker run --rm -u ${UID}:${GID} -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID gcr.io/gem5-test/gcn-gpu:latest bash -c 'cd gem5-resources/src/gpu/pannotia/color ;  VARIANT=MAX GEM5_PATH=../../../../../  make gem5-fusion '
docker run --rm -u ${UID}:${GID} -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID gcr.io/gem5-test/gcn-gpu:latest bash -c 'cd gem5-resources/src/gpu/pannotia/color ;  VARIANT=MAXMIN GEM5_PATH=../../../../../  make gem5-fusion '

# Note: two makes here, one for the mmap generator, one for the gem5 runner
docker run --rm -u ${UID}:${GID} -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID gcr.io/gem5-test/gcn-gpu:latest bash -c 'cd gem5-resources/src/gpu/pannotia/fw ;  make default ;  GEM5_PATH=../../../../../  make gem5-fusion '

docker run --rm -u ${UID}:${GID} -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID gcr.io/gem5-test/gcn-gpu:latest bash -c 'cd gem5-resources/src/gpu/pannotia/mis ;  GEM5_PATH=../../../../../  make gem5-fusion '

docker run --rm -u ${UID}:${GID} -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID gcr.io/gem5-test/gcn-gpu:latest bash -c 'cd gem5-resources/src/gpu/pannotia/pagerank ;  GEM5_PATH=../../../../../  make gem5-fusion '
docker run --rm -u ${UID}:${GID} -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID gcr.io/gem5-test/gcn-gpu:latest bash -c 'cd gem5-resources/src/gpu/pannotia/pagerank ;  VARIANT=SPMV GEM5_PATH=../../../../../  make gem5-fusion '

docker run --rm -u ${UID}:${GID} -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID gcr.io/gem5-test/gcn-gpu:latest bash -c 'cd gem5-resources/src/gpu/pannotia/sssp ;  GEM5_PATH=../../../../../  make gem5-fusion '
docker run --rm -u ${UID}:${GID} -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID gcr.io/gem5-test/gcn-gpu:latest bash -c 'cd gem5-resources/src/gpu/pannotia/sssp ;  VARIANT=ELL GEM5_PATH=../../../../../  make gem5-fusion '

# Nightly tests (heterosync, square)
docker run --rm -u ${UID}:${GID} -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID gcr.io/gem5-test/gcn-gpu:latest bash -c 'cd gem5-resources/src/gpu/heterosync ; GEM5_PATH=../../../../  make release '

docker run --rm -u ${UID}:${GID} -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID gcr.io/gem5-test/gcn-gpu:latest bash -c 'cd gem5-resources/src/gpu/square ; GEM5_PATH=../../../../  make '

# HACC
# Note: HCC_AMDGPU_TARGET does not include gfx902 in the Dockerfile, so override it here
docker run --rm -u ${UID}:${GID} -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID gcr.io/gem5-test/gcn-gpu:latest bash -c 'cd gem5-resources/src/gpu/halo-finder/src ; HCC_AMDGPU_TARGET=gfx801,gfx803,gfx900,gfx902 GEM5_PATH=../../../../  make hip/ForceTreeTest '

# PENNANT - Similar to HACC need to specify HCC_AMDGPU_TARGET for gfx902.  There is some issue with the Makefile
# trying to add it there. The warning message about gfx902 gets placed into a file when 'creating dependencies'
# and this causes an error
docker run --rm -u ${UID}:${GID} -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID gcr.io/gem5-test/gcn-gpu:latest bash -c 'cd gem5-resources/src/gpu/pennant ; HCC_AMDGPU_TARGET=gfx900,gfx902 GEM5_PATH=../../../../  make '

# LULESH
docker run --rm -u ${UID}:${GID} -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID gcr.io/gem5-test/gcn-gpu:latest bash -c 'cd gem5-resources/src/gpu/lulesh ; GEM5_PATH=../../../../  make '

# DNNMark - Multiple steps
docker run --rm -u ${UID}:${GID} -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID gcr.io/gem5-test/gcn-gpu:latest bash -c 'cd gem5-resources/src/gpu/DNNMark ; ./setup.sh HIP'
docker run --rm -u ${UID}:${GID} -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID gcr.io/gem5-test/gcn-gpu:latest bash -c 'cd gem5-resources/src/gpu/DNNMark/build ; make -j16'
docker run --rm -v ${PWD}:${PWD} -w ${PWD}/gem5-resources/src/gpu/DNNMark -v ${PWD}/gem5-resources/src/gpu/DNNMark/cachefiles:/root/.cache/miopen/2.9.0 --memory="24g" gcr.io/gem5-test/gcn-gpu:latest bash -c 'python3 generate_cachefiles.py cachefiles.csv --gfx-version=gfx902 --num-cus=4'
docker run --rm -u ${UID}:${GID} -v ${PWD}:${PWD} -w ${PWD}/gem5-resources/src/gpu/DNNMark/ -u $UID:$GID gcr.io/gem5-test/gcn-gpu:latest bash -c 'g++ -std=c++0x generate_rand_data.cpp -o  generate_rand_data'
docker run --rm -u ${UID}:${GID} -v ${PWD}:${PWD} -w ${PWD}/gem5-resources/src/gpu/DNNMark/ -u $UID:$GID gcr.io/gem5-test/gcn-gpu:latest bash -c './generate_rand_data'

# Not tested but exist : hip-samples
docker run --rm -u ${UID}:${GID} -v ${PWD}:${PWD} -w ${PWD} -u $UID:$GID gcr.io/gem5-test/gcn-gpu:latest bash -c 'cd gem5-resources/src/gpu/hip-samples ; GEM5_PATH=../../../../  make '

# Generate mmap file for fw - This takes about 20 minutes ?
docker run -u $UID:$GID --volume ${PWD}:${PWD} -w ${PWD} gcr.io/gem5-test/gcn-gpu:latest build/VEGA_X86/gem5.opt configs/example/apu_se.py -n8  -c ./gem5-resources/src/gpu/pannotia/fw/bin/fw_hip  --options="-f 1k_128k.gr -m 1" --gfx-version=gfx902
