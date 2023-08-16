#!/usr/bin/env python3
import sys
import json
import test_create
import argparse
import gzip

header = """
#include <iostream>
#include <vector>
#include <cstring>

#include "hip/hip_runtime.h"

#define WAVE_SIZE 64   // Wave 64
#define MAX_REGS 10    // Max # of src or dests a *single* inst could have

"""

main_func_top = """
int main()
{
    // Common vector inputs -- Allocate once and recopy per-test
    uint32_t *voutbufs_h, *vinbufs_h;

    voutbufs_h = (uint32_t*)malloc(WAVE_SIZE * sizeof(uint32_t) * MAX_REGS);
    vinbufs_h = (uint32_t*)malloc(WAVE_SIZE * sizeof(uint32_t) * MAX_REGS);

    uint32_t *voutbufs_d, *vinbufs_d;

    hipMalloc((void**)&voutbufs_d, WAVE_SIZE * sizeof(uint32_t) * MAX_REGS);
    hipMalloc((void**)&vinbufs_d, WAVE_SIZE * sizeof(uint32_t) * MAX_REGS);

    // For ABI init
    uint32_t *out_d, *in_d;

    hipMalloc((void**)&out_d, WAVE_SIZE * sizeof(uint32_t));
    hipMalloc((void**)&in_d, WAVE_SIZE * sizeof(uint32_t));

    // For test verification pretty-printing
    bool passed = true;
"""

main_func_bot = """

    return 0;
}
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input", type=str, default=None, help="Input trace file"
    )
    parser.add_argument(
        "-s", "--start", type=int, default=0, help="Start seqNum"
    )
    parser.add_argument("-e", "--end", type=int, default=0, help="End seqNum")
    parser.add_argument(
        "-w", "--wfDynId", type=int, default=0, help="Specific WF to test"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file for CPP code",
    )

    args = parser.parse_args()

    if args.input is None:
        print(
            "No input given. Use -i or --input to specify gem5 gpu instruction trace file"
        )
        sys.exit(1)

    print(
        f"Running tests on wavefront ID {args.wfDynId} seqNums [{args.start}, {args.end})"
    )

    kernel_codes = ""
    host_codes = ""

    line_num = 0
    with gzip.open(args.input, "r") as inst_trace:
        for line in inst_trace:
            line_num += 1
            line_dict = json.loads(line.decode().rstrip())
            wfDynId = line_dict["wfDynId"]
            seqNum = line_dict["sn"]
            if wfDynId == args.wfDynId:
                if seqNum >= args.start and seqNum < args.end:
                    if line_dict["test_type"] == "v":
                        print(f"Testing sn {seqNum} on wf {wfDynId}")
                        print(line.rstrip())
                        (kcode, hcode) = test_create.create_test(line_dict)
                        kernel_codes += kcode
                        host_codes += hcode
                    elif (
                        line_dict["test_type"] == "s"
                        or line_dict["test_type"] == "vf"
                        or line_dict["test_type"] == "sf"
                        or line_dict["test_type"] == "vxf"
                        or line_dict["test_type"] == "sxf"
                    ):
                        print(
                            f"Skipping sn {seqNum} on wf {wfDynId} of type {line_dict['test_type']}"
                        )
                    elif line_dict["test_type"] == "do_not_test":
                        print(
                            f"Not testing sn {seqNum} on wf {wfDynId} -- Marked not testable"
                        )
                    else:
                        print(
                            f"Unknown test_type {line_dict['test_type']} -- Ignoring"
                        )

    # Dump CPP
    if args.output is None:
        print(header)
        print(kernel_codes)
        print(main_func_top)
        print(host_codes)
        print(main_func_bot)
    else:
        with open(args.output, "w") as out_code:
            out_code.write(header)
            out_code.write(kernel_codes)
            out_code.write(main_func_top)
            out_code.write(host_codes)
            out_code.write(main_func_bot)
