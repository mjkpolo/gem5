/*
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef __GEM5_AMDGPU_OPS_H__
#define __GEM5_AMDGPU_OPS_H__

// This function sets up all of the macros needed for an GPU pseudo ops so
// that the definitions of the functions themselves do not need to do this
// each time. It must, however, be called once by any GPU kernel using pseudo
// ops.
__device__ void
gem5_pseudo_op_init()
{
    // VOP1 instructions have one source operand and one VGPR dest. These are
    // primarily for unary operations.
    //
    // There are three macros defined to create VOP1 instructions with a VGPR
    // source. The format of VOP1 is import in explaining what these do, so
    // here is the format of VOP1. The number in () is the bit count.
    //
    // | 0111111b(7) | vdst(8) | opcode(8) | src(9) |
    //
    // We eventually use the .byte assembler directive, however the fact that
    // the VOP1 fields are not simply four 8-bit fields complicates things.
    // We need to somehow place the upper most src bit in the second byte,
    // which means to upper most opcode bit must be in the third byte, and
    // the upper most vdst bit must be in the fourth byte. We solve this as
    // follows: (1) Make a VGPR and SGPR version of this macro. Luckily, any
    // src >= 0x100 is a VGPR otherwise SGPR. (2) The opcodes for VOP1 at the
    // time of writing go up to 82. This means we can force the upper opcode
    // bit to be 1 since the others can't be used anyways. This makes it more
    // simple *and* still allows for 128 VOP1 pseudo ops, which should be
    // enough! (3) The limit vdst to be v0-v127. It's not clear if these
    // should be that flexible right now, but the first macro,
    // vop1_{s|v}_gem5_raw, allows for any vdst.
    //
    // The macro vop1_{v|s}_gem5_op calls vop1_{v|s}_gem5_raw. In both cases
    // we call the _raw macro with vdst+vdst, basically multiplying by two or
    // bit shifting by one to move it into the correct place. Otherwise, there
    // would be restrictions that the _raw vdst and opcode must be multiples
    // of two. We similarly bit shift opcode by adding with itself. In the
    // VGPR version we also add 1 to make the top bit of src be 1.
    //
    // Finally, the vop1_{v|s} macro provides a simpler interface to the _op
    // macros and forces vdst to v1 and src to v2|s2.
    asm volatile("\n\
      .macro vop1_v_gem5_raw opcode, vdst, vsrc\n\
        .byte \\vsrc, \\opcode, \\vdst, 0x7e\n\
      .endm\n\
      \n\
      .macro vop1_v_gem5_op opcode, vdst, vsrc\n\
        vop1_v_gem5_raw \"\\opcode+\\opcode+1\", \"\\vdst+\\vdst+1\", \\vsrc\n\
      .endm\n\
      \n\
      .macro vop1_v opcode\n\
        vop1_v_gem5_op \\opcode, 1, 2\n\
      .endm");

    asm volatile("\n\
      .macro vop1_s_gem5_raw opcode, vdst, ssrc\n\
        .byte \\ssrc, \\opcode, \\vdst, 0x7e\n\
      .endm\n\
      \n\
      .macro vop1_s_gem5_op opcode, vdst, ssrc\n\
        vop1_s_gem5_raw \"\\opcode+\\opcode\", \"\\vdst+\\vdst+1\", \\ssrc\n\
      .endm");

    // FLAT instructions can go to global memory or LDS and as a pseudo op can
    // be used to implement new types of memory instructions. For example, an
    // load/store instruction that has a different mtype than what is allowed
    // right now.
    //
    // To simplify things we assume GCN3 style where only a VGPR pair can be
    // used for the address (rather than SGPR that can be used in Vega). Only
    // the opcode, address, data, and dest registers can be modified using
    // this macro.
    asm volatile("\n\
      .macro flat_gem5_op opcode, outreg, inaddr, indata\n\
        .byte 0, 0, \\opcode, 0xdc, \\inaddr, \\indata, 0x7f, \\outreg\n\
      .endm");
}

// It is difficult without compiler support to specify which registers will be
// clobbered by the pseudo instructions and how to pass in a "named" C++
// variable to the pseudo op (i.e., using %0, %1, ...). These macros simplify
// this by first moving into a temporary register which is the same as the
// register hardcoded by the vop1_{v|s} macros.
#define GEM5_VOP1_X1_VGPR(opcode, reg1)                     \
    asm volatile("v_mov_b32 v2, %0" : : "v"(reg1) : "v2");  \
    asm volatile("vop1_v_gem5_op 0, 1, 2\n");

#define GEM5_VOP1_X1_SGPR(opcode, reg1)                     \
    asm volatile("s_mov_b32 s2, %0" : : "s"(reg1) : "s2");  \
    asm volatile("vop1_s_gem5_op 0, 1, 2\n");

// Many instructions in the GPU encode a single register number and use that
// as the *starting* register for multiple registers. A flat_load_dwordx4,
// for example, uses a single starting register and uses the next three for
// output data and also a single starting register for addr and uses the
// next register to create a full 64-bit address. The simulator could in
// theory use as many as it wants. We simply need to tell the compiler
// what registers are clobbered and move data into temporary registers.
#define GEM5_VOP1_X2_VGPR(opcode, reg1)                                      \
    asm volatile("v_mov_b32 v2, %0" : : "v"((uint32_t)reg1) : "v2");         \
    asm volatile("v_mov_b32 v3, %0" : : "v"((uint32_t)(reg1 >> 32)) : "v3"); \
    asm volatile("vop1_v_gem5_op 0, 1, 2\n");

#define GEM5_VOP1_X2_SGPR(opcode, reg1)                               \
    asm volatile("s_mov_b64 s[2:3], %0" : : "s"(reg1) : "s2", "s3");  \
    asm volatile("vop1_s_gem5_op 0, 1, 2\n");


// This is simply a list of pseudo op opcodes. As described in the macros
// above, it is implied that this opcode is fixed up to ignore real opcodes.
#define v_gem5_print_reg_b32 0

// Assumes dword
__device__ void
gem5_print_vreg(int reg)
{
    GEM5_VOP1_X1_VGPR(v_gem5_print_reg_b32, reg);
}

// Assumes dword
__device__ void
gem5_print_sreg(uint32_t reg)
{
    GEM5_VOP1_X1_SGPR(v_gem5_print_reg_b32, reg);
}

#endif // __GEM5_AMDGPU_OPS_H__
