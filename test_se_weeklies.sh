# Note: Assumes VEGA_X86 is already built and the pannotia inputs 1k_128k.gr / coAuthorsDBLP.graph are already downloaded and in the gem5 root directory
# PANNOTIA
docker run -u $UID:$GID --volume ${PWD}:${PWD} -w ${PWD} gcr.io/gem5-test/gcn-gpu:latest build/VEGA_X86/gem5.opt -re -d vega_testing/bc_apu --debug-flags=GPUCommandProc configs/example/apu_se.py -n8  -c ./gem5-resources/src/gpu/pannotia/bc/bin/bc.gem5  --options="1k_128k.gr"  --gfx-version=gfx902 &

docker run -u $UID:$GID --volume ${PWD}:${PWD} -w ${PWD} gcr.io/gem5-test/gcn-gpu:latest build/VEGA_X86/gem5.opt -re -d vega_testing/color_maxmin_apu --debug-flags=GPUCommandProc configs/example/apu_se.py -n8  -c ./gem5-resources/src/gpu/pannotia/color/bin/color_maxmin.gem5  --options="1k_128k.gr 0"  --gfx-version=gfx902 &

docker run -u $UID:$GID --volume ${PWD}:${PWD} -w ${PWD} gcr.io/gem5-test/gcn-gpu:latest build/VEGA_X86/gem5.opt -re -d vega_testing/color_max_apu --debug-flags=GPUCommandProc configs/example/apu_se.py -n8  -c ./gem5-resources/src/gpu/pannotia/color/bin/color_max.gem5  --options="1k_128k.gr 0"  --gfx-version=gfx902 &

docker run -u $UID:$GID --volume ${PWD}:${PWD} -w ${PWD} gcr.io/gem5-test/gcn-gpu:latest build/VEGA_X86/gem5.opt -re -d vega_testing/mis_apu --debug-flags=GPUCommandProc configs/example/apu_se.py -n8  -c ./gem5-resources/src/gpu/pannotia/mis/bin/mis_hip.gem5  --options="1k_128k.gr 0"  --gfx-version=gfx902 &

docker run -u $UID:$GID --memory="24g" --volume ${PWD}:${PWD} -w ${PWD} gcr.io/gem5-test/gcn-gpu:latest build/VEGA_X86/gem5.opt -re -d vega_testing/fw_apu --debug-flags=GPUCommandProc configs/example/apu_se.py -n8 --mem-size=8GB  -c ./gem5-resources/src/gpu/pannotia/fw/bin/fw_hip.gem5  --options="-f 1k_128k.gr -m 2"  --gfx-version=gfx902 &

docker run -u $UID:$GID --volume ${PWD}:${PWD} -w ${PWD} gcr.io/gem5-test/gcn-gpu:latest build/VEGA_X86/gem5.opt -re -d vega_testing/pagerank_apu --debug-flags=GPUCommandProc configs/example/apu_se.py -n8  -c ./gem5-resources/src/gpu/pannotia/pagerank/bin/pagerank.gem5  --options="coAuthorsDBLP.graph 1"  --gfx-version=gfx902 &

docker run -u $UID:$GID --volume ${PWD}:${PWD} -w ${PWD} gcr.io/gem5-test/gcn-gpu:latest build/VEGA_X86/gem5.opt -re -d vega_testing/pagerank_spmv_apu --debug-flags=GPUCommandProc configs/example/apu_se.py -n8  -c ./gem5-resources/src/gpu/pannotia/pagerank/bin/pagerank_spmv.gem5  --options="coAuthorsDBLP.graph 1"  --gfx-version=gfx902 &

docker run -u $UID:$GID --volume ${PWD}:${PWD} -w ${PWD} gcr.io/gem5-test/gcn-gpu:latest build/VEGA_X86/gem5.opt -re -d vega_testing/sssp_apu --debug-flags=GPUCommandProc configs/example/apu_se.py -n8  -c ./gem5-resources/src/gpu/pannotia/sssp/bin/sssp.gem5  --options="1k_128k.gr 0"  --gfx-version=gfx902 &

docker run -u $UID:$GID --volume ${PWD}:${PWD} -w ${PWD} gcr.io/gem5-test/gcn-gpu:latest build/VEGA_X86/gem5.opt -re -d vega_testing/sssp_ell_apu --debug-flags=GPUCommandProc configs/example/apu_se.py -n8  -c ./gem5-resources/src/gpu/pannotia/sssp/bin/sssp_ell.gem5  --options="1k_128k.gr 0"  --gfx-version=gfx902 &

# Nightly tests (heterosync, square)
docker run -u $UID:$GID --volume ${PWD}:${PWD} -w ${PWD} gcr.io/gem5-test/gcn-gpu:latest build/VEGA_X86/gem5.opt -re -d vega_testing/lfTreeBarrUniq_apu --debug-flags=GPUDriver,GPUShader,GPUCommandProc configs/example/apu_se.py -n8  -c  ./gem5-resources/src/gpu/heterosync/bin/allSyncPrims-1kernel  --options="lfTreeBarrUniq 10 16 4" --gfx-version=gfx902  &

docker run -u $UID:$GID --volume ${PWD}:${PWD} -w ${PWD} gcr.io/gem5-test/gcn-gpu:latest build/VEGA_X86/gem5.opt -re -d vega_testing/sleepMutex_apu --debug-flags=GPUDriver,GPUShader,GPUCommandProc configs/example/apu_se.py -n8  -c  ./gem5-resources/src/gpu/heterosync/bin/allSyncPrims-1kernel  --options="sleepMutex 10 16 4" --gfx-version=gfx902  &

docker run -u $UID:$GID --volume ${PWD}:${PWD} -w ${PWD} gcr.io/gem5-test/gcn-gpu:latest build/VEGA_X86/gem5.opt -re -d vega_testing/square --debug-flags=GPUCommandProc configs/example/apu_se.py -n8  -c gem5-resources/src/gpu/square/bin/square --gfx-version=gfx902 &

# HACC
docker run -u $UID:$GID -v ${PWD}:${PWD} -w ${PWD}   gcr.io/gem5-test/gcn-gpu:latest build/VEGA_X86/gem5.opt -d vega_testing/hacc_apu -re --debug-flags=GPUCommandProc,GPUDriver,GPUShader configs/example/apu_se.py -n8  -c ./gem5-resources/src/gpu/halo-finder/src/hip/ForceTreeTest  --options="0.5 0.1 64 0.1 1 N 12 rcb"    --gfx-version=gfx902  &

# PENNANT
docker run -u $UID:$GID --volume ${PWD}:${PWD} -w ${PWD} gcr.io/gem5-test/gcn-gpu:latest build/VEGA_X86/gem5.opt -d vega_testing/pennant_nohsmall_apu -re --debug-flags=GPUCommandProc configs/example/apu_se.py -n8  -c ./gem5-resources/src/gpu/pennant/build/pennant --options="./bin/nohsmall.pnt"  --gfx-version=gfx902   &

# LULESH
docker run -u $UID:$GID --memory="24g" --volume ${PWD}:${PWD} -w ${PWD} gcr.io/gem5-test/gcn-gpu:latest build/VEGA_X86/gem5.opt -d vega_testing/lulesh_apu -re --debug-flags=GPUCommandProc configs/example/apu_se.py -n8  -c ./bin/lulesh --gfx-version=gfx902 --mem-size=8GB &

# DNNMark
docker run --rm -v ${PWD}:${PWD} -w ${PWD}/gem5-resources/src/gpu/DNNMark -v ${PWD}/gem5-resources/src/gpu/DNNMark/cachefiles:/root/.cache/miopen/2.9.0 --memory="24g" gcr.io/gem5-test/gcn-gpu:latest ${PWD}/build/VEGA_X86/gem5.opt -d ../../../../vega_testing/softmax_apu -re --debug-flags=GPUCommandProc ${PWD}/configs/example/apu_se.py -n8  --reg-alloc-policy=dynamic --benchmark-root="${PWD}/gem5-resources/src/gpu/DNNMark/build/benchmarks/test_fwd_softmax"  -c dnnmark_test_fwd_softmax   --options="-config ${PWD}/gem5-resources/src/gpu/DNNMark/config_example/softmax_config.dnnmark  -mmap ${PWD}/gem5-resources/src/gpu/DNNMark/mmap.bin" --gfx-version=gfx902 &

docker run --rm -v ${PWD}:${PWD} -w ${PWD}/gem5-resources/src/gpu/DNNMark -v ${PWD}/gem5-resources/src/gpu/DNNMark/cachefiles:/root/.cache/miopen/2.9.0 --memory="24g" gcr.io/gem5-test/gcn-gpu:latest ${PWD}/build/VEGA_X86/gem5.opt -d ../../../../vega_testing/pool_apu -re --debug-flags=GPUCommandProc ${PWD}/configs/example/apu_se.py -n8  --reg-alloc-policy=dynamic --benchmark-root="${PWD}/gem5-resources/src/gpu/DNNMark/build/benchmarks/test_fwd_pool"  -c dnnmark_test_fwd_pool   --options="-config ${PWD}/gem5-resources/src/gpu/DNNMark/config_example/pool_config.dnnmark  -mmap ${PWD}/gem5-resources/src/gpu/DNNMark/mmap.bin" --gfx-version=gfx902 &

docker run --rm -v ${PWD}:${PWD} -w ${PWD}/gem5-resources/src/gpu/DNNMark -v ${PWD}/gem5-resources/src/gpu/DNNMark/cachefiles:/root/.cache/miopen/2.9.0 --memory="24g" gcr.io/gem5-test/gcn-gpu:latest ${PWD}/build/VEGA_X86/gem5.opt -d ../../../../vega_testing/bn_apu -re --debug-flags=GPUCommandProc ${PWD}/configs/example/apu_se.py -n8  --reg-alloc-policy=dynamic --benchmark-root="${PWD}/gem5-resources/src/gpu/DNNMark/build/benchmarks/test_bwd_bn"  -c dnnmark_test_bwd_bn   --options="-config ${PWD}/gem5-resources/src/gpu/DNNMark/config_example/bn_config.dnnmark  -mmap ${PWD}/gem5-resources/src/gpu/DNNMark/mmap.bin" --gfx-version=gfx902 &

# Not tested but exist : hip-samples
docker run -u $UID:$GID --volume ${PWD}:${PWD} -w ${PWD} gcr.io/gem5-test/gcn-gpu:latest build/VEGA_X86/gem5.opt -re -d vega_testing/2dshfl --debug-flags=GPUCommandProc configs/example/apu_se.py -n8  -c gem5-resources/src/gpu/hip-samples/bin/2dshfl --gfx-version=gfx902 &

docker run -u $UID:$GID --volume ${PWD}:${PWD} -w ${PWD} gcr.io/gem5-test/gcn-gpu:latest build/VEGA_X86/gem5.opt -re -d vega_testing/dynamic_shared --debug-flags=GPUCommandProc configs/example/apu_se.py -n8  -c gem5-resources/src/gpu/hip-samples/bin/dynamic_shared --gfx-version=gfx902 &

docker run -u $UID:$GID --volume ${PWD}:${PWD} -w ${PWD} gcr.io/gem5-test/gcn-gpu:latest build/VEGA_X86/gem5.opt -re -d vega_testing/inline_asm  --debug-flags=GPUCommandProc configs/example/apu_se.py -n8  -c gem5-resources/src/gpu/hip-samples/bin/inline_asm --gfx-version=gfx902 &

docker run -u $UID:$GID --volume ${PWD}:${PWD} -w ${PWD} gcr.io/gem5-test/gcn-gpu:latest build/VEGA_X86/gem5.opt -re -d vega_testing/MatrixTranspose  --debug-flags=GPUCommandProc configs/example/apu_se.py -n8  -c gem5-resources/src/gpu/hip-samples/bin/MatrixTranspose  --gfx-version=gfx902 &

docker run -u $UID:$GID --volume ${PWD}:${PWD} -w ${PWD} gcr.io/gem5-test/gcn-gpu:latest build/VEGA_X86/gem5.opt -re -d vega_testing/sharedMemory  --debug-flags=GPUCommandProc configs/example/apu_se.py -n8  -c gem5-resources/src/gpu/hip-samples/bin/sharedMemory  --gfx-version=gfx902 &

docker run -u $UID:$GID --volume ${PWD}:${PWD} -w ${PWD} gcr.io/gem5-test/gcn-gpu:latest build/VEGA_X86/gem5.opt -re -d vega_testing/shfl --debug-flags=GPUCommandProc configs/example/apu_se.py -n8  -c gem5-resources/src/gpu/hip-samples/bin/shfl --gfx-version=gfx902 &

docker run -u $UID:$GID --volume ${PWD}:${PWD} -w ${PWD} gcr.io/gem5-test/gcn-gpu:latest build/VEGA_X86/gem5.opt -re -d vega_testing/stream --debug-flags=GPUCommandProc configs/example/apu_se.py -n8  -c gem5-resources/src/gpu/hip-samples/bin/stream --gfx-version=gfx902 &

docker run -u $UID:$GID --volume ${PWD}:${PWD} -w ${PWD} gcr.io/gem5-test/gcn-gpu:latest build/VEGA_X86/gem5.opt -re -d vega_testing/unroll --debug-flags=GPUCommandProc configs/example/apu_se.py -n8  -c gem5-resources/src/gpu/hip-samples/bin/unroll --gfx-version=gfx902 &
