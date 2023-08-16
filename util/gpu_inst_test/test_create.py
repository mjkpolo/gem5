#!/usr/bin/env python3


def special_reg(reg):
    # VCC_LO / VCC_HI
    if reg == "s106" or reg == "s107" or reg == "t106" or reg == "t107":
        return True
    # EXEC_LO / EXEC_HI
    elif reg == "s126" or reg == "s127" or reg == "t126" or reg == "t127":
        return True
    # SCC
    elif reg == "s253" or reg == "t253":
        return True
    # VCCZ
    elif reg == "s251" or reg == "t251":
        return True
    # EXECZ
    elif reg == "s252" or reg == "t252":
        return True

    return False


def load_reg(reg, varname):
    if reg[0] == "v":
        cpp_code = f'asm volatile("global_load_dword {reg}, %0, %1" '
        cpp_code += f': : "v"(abi_v0), "s"({varname}) : "{reg}");\n'
        cpp_code += '    asm volatile("s_waitcnt vmcnt(0)");\n'
    elif reg[0] == "s":
        cpp_code = f'    asm volatile("s_mov_b32 {reg}, %0" : : "s"({varname}) : "{reg}");\n'
    else:
        assert False, f"Unknown source reg type {reg}"

    return cpp_code


def store_reg(reg, varname):
    # Vector dest store
    if reg[0] == "v":
        cpp_code = f'    asm volatile("global_store_dword %0, {reg}, %1" '
        cpp_code += f': : "v"(abi_v0), "s"({varname}));'
    # Scalar dest store
    elif reg[0] == "s":
        # TODO
        assert False, "Scalar register dests unimplemented"
    else:
        assert False, f"Unknown reg type {reg}"

    return cpp_code


def create_test(kvargs):
    # Figure out dest and sources. To differentiate between src/dest the trace
    # output uses 'w' for vector register dests and 't' for scalar. The kvdests
    # list contains the names with w/t so that we can look up things from the
    # input dict using them, and the dests has w/t changed to v/s.
    kvdests = []
    dests = []
    srcs = []

    # Collect each vector/scalar register ignoring special registers like EXEC,
    # VCC, SCC, VCCZ, and SCCZ.
    for reg in kvargs["regs"]:
        if reg[0] == "v" or reg[0] == "s":
            if not special_reg(reg):
                srcs.append(reg)
        elif reg[0] == "w" or reg[0] == "t":
            if not special_reg(reg):
                kvdests.append(reg)
                reg = reg.replace("w", "v")
                reg = reg.replace("t", "s")
                dests.append(reg)

    # Generate our kernel. There will be one 32-bit array for vector and scalar
    # destinations and one 32-bit array for vector and 32-bit ordinal for scalar
    # sources.
    cpp_code = f'__global__ void test_sn_{kvargs["sn"]} (uint32_t *out, uint32_t *in, '

    for reg in dests:
        cpp_code += f"uint32_t *{reg}_dst, "
    for reg in srcs:
        if reg[0] == "v":
            cpp_code += f"uint32_t *{reg}_src, "
        else:
            cpp_code += f"uint32_t {reg}_src, "

    # Remove last comma, wheverever it is and end parameter list
    cpp_code = cpp_code[:-2] + ")"

    # This code is common to all tests. It (1) ensure ABI inits v0 with the
    # thread ID and then copies it to a variable, which lets the compiler
    # pick any register it wants. This is needed in case the test uses v0.
    cpp_code += """
{
    out[threadIdx.x] = in[threadIdx.x];
    uint32_t abi_v0;
    asm volatile("v_add_u32 %0, v0, 0" : "=v"(abi_v0));

    """

    # Load in all of the sources to their registers
    for reg in srcs:
        cpp_code += load_reg(reg, reg + "_src") + "\n"

    # Add the instruction to test -- This relies on gem5's disassembly
    # being correct and in LLVM's format.
    cpp_code += f'    asm volatile("{kvargs["asm"]}" : : : "{dests[0]}");\n\n'

    # Write all of the output dests
    for reg in dests:
        cpp_code += store_reg(reg, reg + "_dst") + "\n"

    # End of kernel
    cpp_code += """
}
    """

    # Host side code -- Put each test in a new scope to avoid variable redefinition
    main_code = """
    {

"""

    # Begin building the kernel launch code as we create variables/memcpy for regs
    launch_code = f'test_sn_{kvargs["sn"]}<<<1, 64, 0, 0>>>(out_d, in_d, '

    main_code += """
        // Output vectors / scalars
"""

    # Collect all dest regs and the gem5 values
    for i, reg in enumerate(dests):
        if reg[0] == "v":
            # Replace JSON list brackets with C++ initializer list braces
            data = (
                str(kvargs["regs"][kvdests[i]])
                .replace("[", "{")
                .replace("]", "}")
            )
            main_code += (
                f"        std::vector<uint32_t> {reg}_dst_h = {data};\n"
            )
            launch_code += f"&voutbufs_d[WAVE_SIZE*{i}], "
        elif reg[0] == "s":
            assert False, "No scalar dest support yet"

    main_code += """

        // Input vectors / scalars
"""

    # Collect all source regs and their input
    for i, reg in enumerate(srcs):
        if reg[0] == "v":
            # Replace JSON list brackets with C++ initializer list braces
            data = str(kvargs["regs"][reg]).replace("[", "{").replace("]", "}")
            main_code += (
                f"        std::vector<uint32_t> {reg}_src_h = {data};\n"
            )
            main_code += f"        std::memcpy(&vinbufs_h[WAVE_SIZE*{i}], {reg}_src_h.data(), sizeof(uint32_t) * WAVE_SIZE);\n"
            launch_code += f"&vinbufs_d[WAVE_SIZE*{i}], "
        elif reg[0] == "s":
            data = kvargs["regs"][reg]
            main_code += f"        uint32_t {reg}_src_h = {data};\n"
            launch_code += f"{reg}_src_h, "

    # Remove last comma and end launch code
    launch_code = launch_code[:-2] + ");\n"

    # Copy data region containing input vectors
    main_code += """
        hipMemcpy(vinbufs_d, vinbufs_h, WAVE_SIZE * sizeof(uint32_t) * MAX_REGS, hipMemcpyHostToDevice);

        """

    # Launch the kernel -- Should we force a sync here?
    main_code += launch_code + "\n"

    # Copy the output back
    main_code += """
        hipMemcpy(voutbufs_h, voutbufs_d, WAVE_SIZE * sizeof(uint32_t) * MAX_REGS, hipMemcpyDeviceToHost);

"""

    # Do verification - We consider the exec mask here instead of trying to set it in the
    # GPU kernel as that is tricky and can impact non-tested instructions.
    for idx, reg in enumerate(dests):
        if reg[0] == "v":
            main_code += "        passed = true;\n"
            main_code += (
                f'        uint64_t exec_mask = {kvargs["regs"]["preExec"]};\n'
            )
            main_code += "        for (int i = 0; i < WAVE_SIZE; ++i) {\n"
            main_code += f"            if (voutbufs_h[WAVE_SIZE*{idx}+i] != {reg}_dst_h[i] && exec_mask & 0x1) {{\n"
            main_code += f'                std::cout << "Error in sn {kvargs["sn"]}! Lane " << i << ": "'
            main_code += f'<< {reg}_dst_h[i] << " should be " << voutbufs_h[WAVE_SIZE*{idx}+i] << "\\n";\n'
            main_code += "                passed = false;\n"
            main_code += "            }\n"
            main_code += "            exec_mask >>= 1;\n"
            main_code += "        }\n\n"
            main_code += f'        std::cout << "test_sn_{kvargs["sn"]} " << (passed ? "PASSED" : "FAILED") << "\\n";'

    # End of scope for test
    main_code += """
    }
"""

    return (cpp_code, main_code)
