	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.protected	_Z8micro_tk13micro_globals ; -- Begin function _Z8micro_tk13micro_globals
	.globl	_Z8micro_tk13micro_globals
	.p2align	8
	.type	_Z8micro_tk13micro_globals,@function
_Z8micro_tk13micro_globals:             ; @_Z8micro_tk13micro_globals
; %bb.0:                                ; %entry
	s_cmp_lg_u32 0, -1
	s_cselect_b32 s6, 0, 0
	s_load_dwordx2 s[4:5], s[0:1], 0x0
	s_load_dword s33, s[0:1], 0x20
	s_load_dwordx2 s[12:13], s[0:1], 0x30
	s_load_dwordx2 s[14:15], s[0:1], 0x50
	s_and_b32 s2, s6, -16
	s_mov_b32 s9, 0
	s_and_b32 s8, s6, 15
	s_add_u32 s7, s2, 16
	s_cmp_eq_u64 s[8:9], 0
	s_load_dwordx2 s[16:17], s[0:1], 0xe0
	s_waitcnt lgkmcnt(0)
	s_cselect_b32 s15, s6, s7
	s_lshl_b32 s6, s33, 7
	v_mbcnt_lo_u32_b32 v1, -1, 0
	v_mbcnt_hi_u32_b32 v56, -1, v1
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, .str.1@rel32@lo+4
	s_addc_u32 s1, s1, .str.1@rel32@hi+12
	v_mov_b32_e32 v5, 0
	s_cmp_lg_u64 s[0:1], 0
	v_lshlrev_b32_e32 v58, 6, v56
	s_mov_b64 s[2:3], 0
	s_mov_b32 s7, 0x110000
	s_cselect_b64 s[18:19], -1, 0
	v_mov_b32_e32 v1, v5
	v_mov_b32_e32 v34, v58
	v_mov_b32_e32 v35, v5
	s_mov_b32 s8, s9
	s_mov_b32 s10, s9
	s_mov_b32 s11, s9
	s_movk_i32 s36, 0xff1f
	s_movk_i32 s37, 0xff1d
	s_movk_i32 s38, 0x1bf
	v_mov_b32_e32 v8, 2
	v_mov_b32_e32 v9, 1
	v_mov_b32_e32 v10, 33
	v_mov_b32_e32 v11, v5
	v_mov_b32_e32 v12, v5
	v_mov_b32_e32 v13, v5
	v_mov_b32_e32 v37, v0
	s_branch .LBB0_2
.LBB0_1:                                ; %__ockl_printf_append_args.exit62.i
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[22:23]
	v_lshlrev_b32_e32 v2, 7, v4
	v_lshlrev_b32_e32 v3, 1, v16
	v_add3_u32 v2, s15, v2, v3
	v_cmp_lt_u32_e32 vcc, s38, v37
	v_readfirstlane_b32 s0, v2
	s_mov_b32 m0, s0
	v_add_u32_e32 v2, 64, v37
	buffer_load_dwordx4 v20, s[4:7], 0 offen lds
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b32_e32 v37, v2
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execz .LBB0_271
.LBB0_2:                                ; %for.body.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_5 Depth 2
                                        ;     Child Loop BB0_13 Depth 2
                                        ;     Child Loop BB0_22 Depth 2
                                        ;     Child Loop BB0_27 Depth 2
                                        ;     Child Loop BB0_117 Depth 2
                                        ;     Child Loop BB0_125 Depth 2
                                        ;     Child Loop BB0_134 Depth 2
                                        ;     Child Loop BB0_139 Depth 2
                                        ;     Child Loop BB0_31 Depth 2
                                        ;       Child Loop BB0_34 Depth 3
                                        ;       Child Loop BB0_41 Depth 3
                                        ;       Child Loop BB0_49 Depth 3
                                        ;       Child Loop BB0_57 Depth 3
                                        ;       Child Loop BB0_65 Depth 3
                                        ;       Child Loop BB0_73 Depth 3
                                        ;       Child Loop BB0_81 Depth 3
                                        ;       Child Loop BB0_89 Depth 3
                                        ;       Child Loop BB0_97 Depth 3
                                        ;       Child Loop BB0_106 Depth 3
                                        ;       Child Loop BB0_111 Depth 3
                                        ;     Child Loop BB0_144 Depth 2
                                        ;     Child Loop BB0_152 Depth 2
                                        ;     Child Loop BB0_161 Depth 2
                                        ;     Child Loop BB0_166 Depth 2
                                        ;     Child Loop BB0_170 Depth 2
                                        ;     Child Loop BB0_178 Depth 2
                                        ;     Child Loop BB0_187 Depth 2
                                        ;     Child Loop BB0_192 Depth 2
                                        ;     Child Loop BB0_196 Depth 2
                                        ;     Child Loop BB0_204 Depth 2
                                        ;     Child Loop BB0_213 Depth 2
                                        ;     Child Loop BB0_218 Depth 2
                                        ;     Child Loop BB0_222 Depth 2
                                        ;     Child Loop BB0_230 Depth 2
                                        ;     Child Loop BB0_239 Depth 2
                                        ;     Child Loop BB0_244 Depth 2
                                        ;     Child Loop BB0_248 Depth 2
                                        ;     Child Loop BB0_256 Depth 2
                                        ;     Child Loop BB0_265 Depth 2
                                        ;     Child Loop BB0_270 Depth 2
	v_readfirstlane_b32 s0, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v56
	s_and_saveexec_b64 s[20:21], s[0:1]
	s_cbranch_execz .LBB0_8
; %bb.3:                                ;   in Loop: Header=BB0_2 Depth=1
	global_load_dwordx2 v[16:17], v5, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v5, s[16:17] offset:40
	global_load_dwordx2 v[6:7], v5, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v16
	v_and_b32_e32 v3, v3, v17
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v4, v2, 24
	v_add_u32_e32 v3, v4, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[6:7], 0, v[2:3]
	global_load_dwordx2 v[14:15], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v5, v[14:17], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[16:17]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_7
; %bb.4:                                ; %.preheader3.i.i.i.i.preheader
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_5:                                ; %.preheader3.i.i.i.i
                                        ;   Parent Loop BB0_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[6:7], v5, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v5, s[16:17]
	v_mov_b64_e32 v[16:17], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v6, v16
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[26:27], v2, 24, v[14:15]
	v_and_b32_e32 v7, v7, v17
	v_mov_b32_e32 v4, v3
	v_mad_u64_u32 v[6:7], s[26:27], v7, 24, v[4:5]
	v_mov_b32_e32 v3, v6
	global_load_dwordx2 v[14:15], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v5, v[14:17], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[16:17]
	s_or_b64 s[24:25], vcc, s[24:25]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_5
; %bb.6:                                ; %Flow4503
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[24:25]
.LBB0_7:                                ; %Flow4505
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_8:                                ; %.loopexit4.i.i.i.i
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[20:21]
	global_load_dwordx2 v[6:7], v5, s[16:17] offset:40
	global_load_dwordx4 v[14:17], v5, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[22:23], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s24, v6
	v_readfirstlane_b32 s25, v7
	s_and_b64 s[24:25], s[20:21], s[24:25]
	s_mul_i32 s26, s25, 24
	s_mul_hi_u32 s27, s24, 24
	s_add_i32 s27, s27, s26
	s_mul_i32 s26, s24, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, s[26:27]
	s_and_saveexec_b64 s[26:27], s[0:1]
	s_cbranch_execz .LBB0_10
; %bb.9:                                ;   in Loop: Header=BB0_2 Depth=1
	v_mov_b64_e32 v[6:7], s[22:23]
	global_store_dwordx4 v[2:3], v[6:9], off offset:8
.LBB0_10:                               ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[26:27]
	s_lshl_b64 s[22:23], s[24:25], 12
	v_lshl_add_u64 v[6:7], v[16:17], 0, s[22:23]
	v_mov_b64_e32 v[18:19], s[10:11]
	v_readfirstlane_b32 s22, v6
	v_readfirstlane_b32 s23, v7
	v_mov_b64_e32 v[16:17], s[8:9]
	s_nop 3
	global_store_dwordx4 v58, v[10:13], s[22:23]
	global_store_dwordx4 v58, v[16:19], s[22:23] offset:16
	global_store_dwordx4 v58, v[16:19], s[22:23] offset:32
	global_store_dwordx4 v58, v[16:19], s[22:23] offset:48
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_18
; %bb.11:                               ;   in Loop: Header=BB0_2 Depth=1
	global_load_dwordx2 v[22:23], v5, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[16:17], v5, s[16:17] offset:40
	v_mov_b32_e32 v20, s20
	v_mov_b32_e32 v21, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s24, v16
	v_readfirstlane_b32 s25, v17
	s_and_b64 s[24:25], s[24:25], s[20:21]
	s_mul_i32 s25, s25, 24
	s_mul_hi_u32 s26, s24, 24
	s_mul_i32 s24, s24, 24
	s_add_i32 s25, s26, s25
	v_lshl_add_u64 v[18:19], v[14:15], 0, s[24:25]
	global_store_dwordx2 v[18:19], v[22:23], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[16:17], v5, v[20:23], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[16:17], v[22:23]
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB0_14
; %bb.12:                               ; %.preheader1.i.i.i.i.preheader
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_mov_b64 s[26:27], 0
.LBB0_13:                               ; %.preheader1.i.i.i.i
                                        ;   Parent Loop BB0_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[18:19], v[16:17], off
	v_mov_b32_e32 v14, s20
	v_mov_b32_e32 v15, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[14:15], v5, v[14:17], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[14:15], v[16:17]
	s_or_b64 s[26:27], vcc, s[26:27]
	v_mov_b64_e32 v[16:17], v[14:15]
	s_andn2_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB0_13
.LBB0_14:                               ; %Flow4501
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[24:25]
	global_load_dwordx2 v[14:15], v5, s[16:17] offset:16
	s_mov_b64 s[26:27], exec
	v_mbcnt_lo_u32_b32 v4, s26, 0
	v_mbcnt_hi_u32_b32 v4, s27, v4
	v_cmp_eq_u32_e32 vcc, 0, v4
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB0_16
; %bb.15:                               ;   in Loop: Header=BB0_2 Depth=1
	s_bcnt1_i32_b64 s26, s[26:27]
	v_mov_b32_e32 v4, s26
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[4:5], off offset:8 sc1
.LBB0_16:                               ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[16:17], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[16:17]
	s_cbranch_vccnz .LBB0_18
; %bb.17:                               ;   in Loop: Header=BB0_2 Depth=1
	global_load_dword v4, v[14:15], off offset:24
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_store_dwordx2 v[16:17], v[4:5], off sc0 sc1
	v_and_b32_e32 v4, 0xffffff, v4
	s_nop 0
	v_readfirstlane_b32 s24, v4
	s_mov_b32 m0, s24
	s_nop 0
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_18:                               ; %Flow4502
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[22:23]
	v_lshl_add_u64 v[6:7], v[6:7], 0, v[34:35]
	s_branch .LBB0_22
.LBB0_19:                               ;   in Loop: Header=BB0_22 Depth=2
	s_or_b64 exec, exec, s[22:23]
	v_readfirstlane_b32 s22, v4
	s_cmp_eq_u32 s22, 0
	s_cbranch_scc1 .LBB0_21
; %bb.20:                               ;   in Loop: Header=BB0_22 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_22
	s_branch .LBB0_24
.LBB0_21:                               ;   in Loop: Header=BB0_2 Depth=1
	s_branch .LBB0_24
.LBB0_22:                               ;   Parent Loop BB0_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v4, 1
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_19
; %bb.23:                               ;   in Loop: Header=BB0_22 Depth=2
	global_load_dword v4, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v4, 1, v4
	s_branch .LBB0_19
.LBB0_24:                               ;   in Loop: Header=BB0_2 Depth=1
	global_load_dwordx2 v[2:3], v[6:7], off
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_28
; %bb.25:                               ;   in Loop: Header=BB0_2 Depth=1
	global_load_dwordx2 v[6:7], v5, s[16:17] offset:40
	global_load_dwordx2 v[18:19], v5, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[20:21], v5, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[14:15], v[6:7], 0, 1
	v_lshl_add_u64 v[22:23], v[14:15], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v16, v18
	v_mov_b32_e32 v17, v19
	v_cndmask_b32_e32 v15, v23, v15, vcc
	v_cndmask_b32_e32 v14, v22, v14, vcc
	v_and_b32_e32 v4, v15, v7
	v_and_b32_e32 v6, v14, v6
	v_mul_lo_u32 v4, v4, 24
	v_mul_hi_u32 v7, v6, 24
	v_mul_lo_u32 v6, v6, 24
	v_add_u32_e32 v7, v7, v4
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[6:7], v[20:21], 0, v[6:7]
	global_store_dwordx2 v[6:7], v[18:19], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[16:17], v5, v[14:17], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[16:17], v[18:19]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_28
; %bb.26:                               ; %.preheader.i.i.i.i.preheader
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_mov_b64 s[0:1], 0
.LBB0_27:                               ; %.preheader.i.i.i.i
                                        ;   Parent Loop BB0_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[6:7], v[16:17], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v5, v[14:17], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[16:17]
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b64_e32 v[16:17], v[18:19]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_27
.LBB0_28:                               ; %__ockl_printf_begin.exit.i
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_and_b64 vcc, exec, s[18:19]
	s_cbranch_vccz .LBB0_113
; %bb.29:                               ;   in Loop: Header=BB0_2 Depth=1
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v36, 2, v2
	v_and_b32_e32 v14, -3, v2
	v_mov_b32_e32 v15, v3
	s_mov_b64 s[22:23], 0x59
	s_getpc_b64 s[20:21]
	s_add_u32 s20, s20, .str.1@rel32@lo+4
	s_addc_u32 s21, s21, .str.1@rel32@hi+12
	s_branch .LBB0_31
.LBB0_30:                               ; %__ockl_hostcall_preview.exit19.i.i
                                        ;   in Loop: Header=BB0_31 Depth=2
	s_or_b64 exec, exec, s[28:29]
	s_sub_u32 s22, s22, s24
	s_subb_u32 s23, s23, s25
	s_add_u32 s20, s20, s24
	s_addc_u32 s21, s21, s25
	s_cmp_lg_u64 s[22:23], 0
	s_cbranch_scc0 .LBB0_112
.LBB0_31:                               ;   Parent Loop BB0_2 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_34 Depth 3
                                        ;       Child Loop BB0_41 Depth 3
                                        ;       Child Loop BB0_49 Depth 3
                                        ;       Child Loop BB0_57 Depth 3
                                        ;       Child Loop BB0_65 Depth 3
                                        ;       Child Loop BB0_73 Depth 3
                                        ;       Child Loop BB0_81 Depth 3
                                        ;       Child Loop BB0_89 Depth 3
                                        ;       Child Loop BB0_97 Depth 3
                                        ;       Child Loop BB0_106 Depth 3
                                        ;       Child Loop BB0_111 Depth 3
	v_cmp_lt_u64_e64 s[0:1], s[22:23], 56
	s_and_b64 s[0:1], s[0:1], exec
	v_cmp_gt_u64_e64 s[0:1], s[22:23], 7
	s_cselect_b32 s25, s23, 0
	s_cselect_b32 s24, s22, 56
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB0_36
; %bb.32:                               ;   in Loop: Header=BB0_31 Depth=2
	s_mov_b64 s[0:1], 0
	s_cmp_eq_u64 s[22:23], 0
	s_waitcnt vmcnt(0)
	v_mov_b64_e32 v[16:17], 0
	s_cbranch_scc1 .LBB0_35
; %bb.33:                               ; %.preheader30.i.i.preheader
                                        ;   in Loop: Header=BB0_31 Depth=2
	s_lshl_b64 s[26:27], s[24:25], 3
	s_mov_b64 s[28:29], 0
	v_mov_b64_e32 v[16:17], 0
	s_mov_b64 s[30:31], s[20:21]
.LBB0_34:                               ; %.preheader30.i.i
                                        ;   Parent Loop BB0_2 Depth=1
                                        ;     Parent Loop BB0_31 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	global_load_ubyte v4, v5, s[30:31]
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v4, 0xffff, v4
	v_lshlrev_b64 v[6:7], s28, v[4:5]
	s_add_u32 s28, s28, 8
	s_addc_u32 s29, s29, 0
	s_add_u32 s30, s30, 1
	s_addc_u32 s31, s31, 0
	v_or_b32_e32 v16, v6, v16
	s_cmp_lg_u32 s26, s28
	v_or_b32_e32 v17, v7, v17
	s_cbranch_scc1 .LBB0_34
.LBB0_35:                               ; %Flow4470
                                        ;   in Loop: Header=BB0_31 Depth=2
	s_mov_b32 s30, 0
	s_andn2_b64 vcc, exec, s[0:1]
	s_mov_b64 s[0:1], s[20:21]
	s_cbranch_vccz .LBB0_37
	s_branch .LBB0_38
.LBB0_36:                               ;   in Loop: Header=BB0_31 Depth=2
                                        ; implicit-def: $vgpr16_vgpr17
                                        ; implicit-def: $sgpr30
	s_mov_b64 s[0:1], s[20:21]
.LBB0_37:                               ;   in Loop: Header=BB0_31 Depth=2
	global_load_dwordx2 v[16:17], v5, s[20:21]
	s_add_i32 s30, s24, -8
	s_add_u32 s0, s20, 8
	s_addc_u32 s1, s21, 0
.LBB0_38:                               ; %.loopexit31.i.i
                                        ;   in Loop: Header=BB0_31 Depth=2
	s_cmp_gt_u32 s30, 7
	s_cbranch_scc1 .LBB0_42
; %bb.39:                               ;   in Loop: Header=BB0_31 Depth=2
	s_cmp_eq_u32 s30, 0
	s_cbranch_scc1 .LBB0_43
; %bb.40:                               ; %.preheader28.i.i.preheader
                                        ;   in Loop: Header=BB0_31 Depth=2
	s_mov_b64 s[26:27], 0
	v_mov_b64_e32 v[18:19], 0
	s_mov_b64 s[28:29], 0
.LBB0_41:                               ; %.preheader28.i.i
                                        ;   Parent Loop BB0_2 Depth=1
                                        ;     Parent Loop BB0_31 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s34, s0, s28
	s_addc_u32 s35, s1, s29
	global_load_ubyte v4, v5, s[34:35]
	s_add_u32 s28, s28, 1
	s_addc_u32 s29, s29, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v4, 0xffff, v4
	v_lshlrev_b64 v[6:7], s26, v[4:5]
	s_add_u32 s26, s26, 8
	s_addc_u32 s27, s27, 0
	v_or_b32_e32 v18, v6, v18
	s_cmp_lg_u32 s30, s28
	v_or_b32_e32 v19, v7, v19
	s_cbranch_scc1 .LBB0_41
	s_branch .LBB0_44
.LBB0_42:                               ;   in Loop: Header=BB0_31 Depth=2
                                        ; implicit-def: $vgpr18_vgpr19
                                        ; implicit-def: $sgpr31
	s_branch .LBB0_45
.LBB0_43:                               ;   in Loop: Header=BB0_31 Depth=2
	v_mov_b64_e32 v[18:19], 0
.LBB0_44:                               ; %Flow4465
                                        ;   in Loop: Header=BB0_31 Depth=2
	s_mov_b32 s31, 0
	s_cbranch_execnz .LBB0_46
.LBB0_45:                               ;   in Loop: Header=BB0_31 Depth=2
	global_load_dwordx2 v[18:19], v5, s[0:1]
	s_add_i32 s31, s30, -8
	s_add_u32 s0, s0, 8
	s_addc_u32 s1, s1, 0
.LBB0_46:                               ; %.loopexit29.i.i
                                        ;   in Loop: Header=BB0_31 Depth=2
	s_cmp_gt_u32 s31, 7
	s_cbranch_scc1 .LBB0_50
; %bb.47:                               ;   in Loop: Header=BB0_31 Depth=2
	s_cmp_eq_u32 s31, 0
	s_cbranch_scc1 .LBB0_51
; %bb.48:                               ; %.preheader26.i.i.preheader
                                        ;   in Loop: Header=BB0_31 Depth=2
	s_mov_b64 s[26:27], 0
	v_mov_b64_e32 v[20:21], 0
	s_mov_b64 s[28:29], 0
.LBB0_49:                               ; %.preheader26.i.i
                                        ;   Parent Loop BB0_2 Depth=1
                                        ;     Parent Loop BB0_31 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s34, s0, s28
	s_addc_u32 s35, s1, s29
	global_load_ubyte v4, v5, s[34:35]
	s_add_u32 s28, s28, 1
	s_addc_u32 s29, s29, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v4, 0xffff, v4
	v_lshlrev_b64 v[6:7], s26, v[4:5]
	s_add_u32 s26, s26, 8
	s_addc_u32 s27, s27, 0
	v_or_b32_e32 v20, v6, v20
	s_cmp_lg_u32 s31, s28
	v_or_b32_e32 v21, v7, v21
	s_cbranch_scc1 .LBB0_49
	s_branch .LBB0_52
.LBB0_50:                               ;   in Loop: Header=BB0_31 Depth=2
                                        ; implicit-def: $sgpr30
	s_branch .LBB0_53
.LBB0_51:                               ;   in Loop: Header=BB0_31 Depth=2
	v_mov_b64_e32 v[20:21], 0
.LBB0_52:                               ; %Flow4460
                                        ;   in Loop: Header=BB0_31 Depth=2
	s_mov_b32 s30, 0
	s_cbranch_execnz .LBB0_54
.LBB0_53:                               ;   in Loop: Header=BB0_31 Depth=2
	global_load_dwordx2 v[20:21], v5, s[0:1]
	s_add_i32 s30, s31, -8
	s_add_u32 s0, s0, 8
	s_addc_u32 s1, s1, 0
.LBB0_54:                               ; %.loopexit27.i.i
                                        ;   in Loop: Header=BB0_31 Depth=2
	s_cmp_gt_u32 s30, 7
	s_cbranch_scc1 .LBB0_58
; %bb.55:                               ;   in Loop: Header=BB0_31 Depth=2
	s_cmp_eq_u32 s30, 0
	s_cbranch_scc1 .LBB0_59
; %bb.56:                               ; %.preheader24.i.i.preheader
                                        ;   in Loop: Header=BB0_31 Depth=2
	s_mov_b64 s[26:27], 0
	v_mov_b64_e32 v[22:23], 0
	s_mov_b64 s[28:29], 0
.LBB0_57:                               ; %.preheader24.i.i
                                        ;   Parent Loop BB0_2 Depth=1
                                        ;     Parent Loop BB0_31 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s34, s0, s28
	s_addc_u32 s35, s1, s29
	global_load_ubyte v4, v5, s[34:35]
	s_add_u32 s28, s28, 1
	s_addc_u32 s29, s29, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v4, 0xffff, v4
	v_lshlrev_b64 v[6:7], s26, v[4:5]
	s_add_u32 s26, s26, 8
	s_addc_u32 s27, s27, 0
	v_or_b32_e32 v22, v6, v22
	s_cmp_lg_u32 s30, s28
	v_or_b32_e32 v23, v7, v23
	s_cbranch_scc1 .LBB0_57
	s_branch .LBB0_60
.LBB0_58:                               ;   in Loop: Header=BB0_31 Depth=2
                                        ; implicit-def: $vgpr22_vgpr23
                                        ; implicit-def: $sgpr31
	s_branch .LBB0_61
.LBB0_59:                               ;   in Loop: Header=BB0_31 Depth=2
	v_mov_b64_e32 v[22:23], 0
.LBB0_60:                               ; %Flow4455
                                        ;   in Loop: Header=BB0_31 Depth=2
	s_mov_b32 s31, 0
	s_cbranch_execnz .LBB0_62
.LBB0_61:                               ;   in Loop: Header=BB0_31 Depth=2
	global_load_dwordx2 v[22:23], v5, s[0:1]
	s_add_i32 s31, s30, -8
	s_add_u32 s0, s0, 8
	s_addc_u32 s1, s1, 0
.LBB0_62:                               ; %.loopexit25.i.i
                                        ;   in Loop: Header=BB0_31 Depth=2
	s_cmp_gt_u32 s31, 7
	s_cbranch_scc1 .LBB0_66
; %bb.63:                               ;   in Loop: Header=BB0_31 Depth=2
	s_cmp_eq_u32 s31, 0
	s_cbranch_scc1 .LBB0_67
; %bb.64:                               ; %.preheader22.i.i.preheader
                                        ;   in Loop: Header=BB0_31 Depth=2
	s_mov_b64 s[26:27], 0
	v_mov_b64_e32 v[24:25], 0
	s_mov_b64 s[28:29], 0
.LBB0_65:                               ; %.preheader22.i.i
                                        ;   Parent Loop BB0_2 Depth=1
                                        ;     Parent Loop BB0_31 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s34, s0, s28
	s_addc_u32 s35, s1, s29
	global_load_ubyte v4, v5, s[34:35]
	s_add_u32 s28, s28, 1
	s_addc_u32 s29, s29, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v4, 0xffff, v4
	v_lshlrev_b64 v[6:7], s26, v[4:5]
	s_add_u32 s26, s26, 8
	s_addc_u32 s27, s27, 0
	v_or_b32_e32 v24, v6, v24
	s_cmp_lg_u32 s31, s28
	v_or_b32_e32 v25, v7, v25
	s_cbranch_scc1 .LBB0_65
	s_branch .LBB0_68
.LBB0_66:                               ;   in Loop: Header=BB0_31 Depth=2
                                        ; implicit-def: $sgpr30
	s_branch .LBB0_69
.LBB0_67:                               ;   in Loop: Header=BB0_31 Depth=2
	v_mov_b64_e32 v[24:25], 0
.LBB0_68:                               ; %Flow4450
                                        ;   in Loop: Header=BB0_31 Depth=2
	s_mov_b32 s30, 0
	s_cbranch_execnz .LBB0_70
.LBB0_69:                               ;   in Loop: Header=BB0_31 Depth=2
	global_load_dwordx2 v[24:25], v5, s[0:1]
	s_add_i32 s30, s31, -8
	s_add_u32 s0, s0, 8
	s_addc_u32 s1, s1, 0
.LBB0_70:                               ; %.loopexit23.i.i
                                        ;   in Loop: Header=BB0_31 Depth=2
	s_cmp_gt_u32 s30, 7
	s_cbranch_scc1 .LBB0_74
; %bb.71:                               ;   in Loop: Header=BB0_31 Depth=2
	s_cmp_eq_u32 s30, 0
	s_cbranch_scc1 .LBB0_75
; %bb.72:                               ; %.preheader20.i.i.preheader
                                        ;   in Loop: Header=BB0_31 Depth=2
	s_mov_b64 s[26:27], 0
	v_mov_b64_e32 v[26:27], 0
	s_mov_b64 s[28:29], 0
.LBB0_73:                               ; %.preheader20.i.i
                                        ;   Parent Loop BB0_2 Depth=1
                                        ;     Parent Loop BB0_31 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s34, s0, s28
	s_addc_u32 s35, s1, s29
	global_load_ubyte v4, v5, s[34:35]
	s_add_u32 s28, s28, 1
	s_addc_u32 s29, s29, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v4, 0xffff, v4
	v_lshlrev_b64 v[6:7], s26, v[4:5]
	s_add_u32 s26, s26, 8
	s_addc_u32 s27, s27, 0
	v_or_b32_e32 v26, v6, v26
	s_cmp_lg_u32 s30, s28
	v_or_b32_e32 v27, v7, v27
	s_cbranch_scc1 .LBB0_73
	s_branch .LBB0_76
.LBB0_74:                               ;   in Loop: Header=BB0_31 Depth=2
                                        ; implicit-def: $vgpr26_vgpr27
                                        ; implicit-def: $sgpr31
	s_branch .LBB0_77
.LBB0_75:                               ;   in Loop: Header=BB0_31 Depth=2
	v_mov_b64_e32 v[26:27], 0
.LBB0_76:                               ; %Flow4445
                                        ;   in Loop: Header=BB0_31 Depth=2
	s_mov_b32 s31, 0
	s_cbranch_execnz .LBB0_78
.LBB0_77:                               ;   in Loop: Header=BB0_31 Depth=2
	global_load_dwordx2 v[26:27], v5, s[0:1]
	s_add_i32 s31, s30, -8
	s_add_u32 s0, s0, 8
	s_addc_u32 s1, s1, 0
.LBB0_78:                               ; %.loopexit21.i.i
                                        ;   in Loop: Header=BB0_31 Depth=2
	s_cmp_gt_u32 s31, 7
	s_cbranch_scc1 .LBB0_82
; %bb.79:                               ;   in Loop: Header=BB0_31 Depth=2
	s_cmp_eq_u32 s31, 0
	s_cbranch_scc1 .LBB0_83
; %bb.80:                               ; %.preheader.i.i.preheader
                                        ;   in Loop: Header=BB0_31 Depth=2
	s_mov_b64 s[26:27], 0
	v_mov_b64_e32 v[28:29], 0
	s_mov_b64 s[28:29], s[0:1]
.LBB0_81:                               ; %.preheader.i.i
                                        ;   Parent Loop BB0_2 Depth=1
                                        ;     Parent Loop BB0_31 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	global_load_ubyte v4, v5, s[28:29]
	s_add_i32 s31, s31, -1
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v4, 0xffff, v4
	v_lshlrev_b64 v[6:7], s26, v[4:5]
	s_add_u32 s26, s26, 8
	s_addc_u32 s27, s27, 0
	s_add_u32 s28, s28, 1
	s_addc_u32 s29, s29, 0
	v_or_b32_e32 v28, v6, v28
	s_cmp_lg_u32 s31, 0
	v_or_b32_e32 v29, v7, v29
	s_cbranch_scc1 .LBB0_81
	s_branch .LBB0_84
.LBB0_82:                               ;   in Loop: Header=BB0_31 Depth=2
	s_branch .LBB0_85
.LBB0_83:                               ;   in Loop: Header=BB0_31 Depth=2
	v_mov_b64_e32 v[28:29], 0
.LBB0_84:                               ; %Flow4440
                                        ;   in Loop: Header=BB0_31 Depth=2
	s_cbranch_execnz .LBB0_86
.LBB0_85:                               ;   in Loop: Header=BB0_31 Depth=2
	global_load_dwordx2 v[28:29], v5, s[0:1]
.LBB0_86:                               ; %.loopexit.i.i
                                        ;   in Loop: Header=BB0_31 Depth=2
	v_readfirstlane_b32 s0, v56
	v_mov_b64_e32 v[6:7], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v56
	s_and_saveexec_b64 s[26:27], s[0:1]
	s_cbranch_execz .LBB0_92
; %bb.87:                               ;   in Loop: Header=BB0_31 Depth=2
	global_load_dwordx2 v[32:33], v5, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[6:7], v5, s[16:17] offset:40
	global_load_dwordx2 v[30:31], v5, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v4, v6, v32
	v_and_b32_e32 v6, v7, v33
	v_mul_lo_u32 v6, v6, 24
	v_mul_hi_u32 v7, v4, 24
	v_add_u32_e32 v7, v7, v6
	v_mul_lo_u32 v6, v4, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[6:7], v[30:31], 0, v[6:7]
	global_load_dwordx2 v[30:31], v[6:7], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v5, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[6:7], v[32:33]
	s_and_saveexec_b64 s[28:29], vcc
	s_cbranch_execz .LBB0_91
; %bb.88:                               ; %.preheader3.i.i18.i.i.preheader
                                        ;   in Loop: Header=BB0_31 Depth=2
	s_mov_b64 s[30:31], 0
.LBB0_89:                               ; %.preheader3.i.i18.i.i
                                        ;   Parent Loop BB0_2 Depth=1
                                        ;     Parent Loop BB0_31 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_sleep 1
	global_load_dwordx2 v[30:31], v5, s[16:17] offset:40
	global_load_dwordx2 v[38:39], v5, s[16:17]
	v_mov_b64_e32 v[32:33], v[6:7]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v4, v30, v32
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[6:7], s[34:35], v4, 24, v[38:39]
	v_and_b32_e32 v31, v31, v33
	v_mov_b32_e32 v4, v7
	v_mad_u64_u32 v[30:31], s[34:35], v31, 24, v[4:5]
	v_mov_b32_e32 v7, v30
	global_load_dwordx2 v[30:31], v[6:7], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v5, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[6:7], v[32:33]
	s_or_b64 s[30:31], vcc, s[30:31]
	s_andn2_b64 exec, exec, s[30:31]
	s_cbranch_execnz .LBB0_89
; %bb.90:                               ; %Flow4435
                                        ;   in Loop: Header=BB0_31 Depth=2
	s_or_b64 exec, exec, s[30:31]
.LBB0_91:                               ; %Flow4437
                                        ;   in Loop: Header=BB0_31 Depth=2
	s_or_b64 exec, exec, s[28:29]
.LBB0_92:                               ; %.loopexit4.i.i13.i.i
                                        ;   in Loop: Header=BB0_31 Depth=2
	s_or_b64 exec, exec, s[26:27]
	global_load_dwordx2 v[38:39], v5, s[16:17] offset:40
	global_load_dwordx4 v[30:33], v5, s[16:17]
	v_readfirstlane_b32 s26, v6
	v_readfirstlane_b32 s27, v7
	s_mov_b64 s[28:29], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s30, v38
	v_readfirstlane_b32 s31, v39
	s_and_b64 s[30:31], s[26:27], s[30:31]
	s_mul_i32 s34, s31, 24
	s_mul_hi_u32 s35, s30, 24
	s_add_i32 s35, s35, s34
	s_mul_i32 s34, s30, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[38:39], v[30:31], 0, s[34:35]
	s_and_saveexec_b64 s[34:35], s[0:1]
	s_cbranch_execz .LBB0_94
; %bb.93:                               ;   in Loop: Header=BB0_31 Depth=2
	v_mov_b64_e32 v[6:7], s[28:29]
	global_store_dwordx4 v[38:39], v[6:9], off offset:8
.LBB0_94:                               ;   in Loop: Header=BB0_31 Depth=2
	s_or_b64 exec, exec, s[34:35]
	s_lshl_b64 s[28:29], s[30:31], 12
	v_lshl_add_u64 v[6:7], v[32:33], 0, s[28:29]
	v_or_b32_e32 v4, 0, v15
	v_or_b32_e32 v32, v14, v36
	v_cmp_gt_u64_e64 vcc, s[22:23], 56
	s_lshl_b32 s28, s24, 2
	s_add_i32 s28, s28, 28
	v_cndmask_b32_e32 v15, v4, v15, vcc
	v_cndmask_b32_e32 v4, v32, v14, vcc
	s_and_b32 s28, s28, 0x1e0
	v_and_b32_e32 v4, 0xffffff1f, v4
	v_or_b32_e32 v14, s28, v4
	v_readfirstlane_b32 s28, v6
	v_readfirstlane_b32 s29, v7
	s_nop 4
	global_store_dwordx4 v58, v[14:17], s[28:29]
	global_store_dwordx4 v58, v[18:21], s[28:29] offset:16
	global_store_dwordx4 v58, v[22:25], s[28:29] offset:32
	global_store_dwordx4 v58, v[26:29], s[28:29] offset:48
	s_and_saveexec_b64 s[28:29], s[0:1]
	s_cbranch_execz .LBB0_102
; %bb.95:                               ;   in Loop: Header=BB0_31 Depth=2
	global_load_dwordx2 v[22:23], v5, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v5, s[16:17] offset:40
	v_mov_b32_e32 v20, s26
	v_mov_b32_e32 v21, s27
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s30, v14
	v_readfirstlane_b32 s31, v15
	s_and_b64 s[30:31], s[30:31], s[26:27]
	s_mul_i32 s31, s31, 24
	s_mul_hi_u32 s34, s30, 24
	s_mul_i32 s30, s30, 24
	s_add_i32 s31, s34, s31
	v_lshl_add_u64 v[18:19], v[30:31], 0, s[30:31]
	global_store_dwordx2 v[18:19], v[22:23], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[16:17], v5, v[20:23], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[16:17], v[22:23]
	s_and_saveexec_b64 s[30:31], vcc
	s_cbranch_execz .LBB0_98
; %bb.96:                               ; %.preheader1.i.i16.i.i.preheader
                                        ;   in Loop: Header=BB0_31 Depth=2
	s_mov_b64 s[34:35], 0
.LBB0_97:                               ; %.preheader1.i.i16.i.i
                                        ;   Parent Loop BB0_2 Depth=1
                                        ;     Parent Loop BB0_31 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_sleep 1
	global_store_dwordx2 v[18:19], v[16:17], off
	v_mov_b32_e32 v14, s26
	v_mov_b32_e32 v15, s27
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[14:15], v5, v[14:17], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[14:15], v[16:17]
	s_or_b64 s[34:35], vcc, s[34:35]
	v_mov_b64_e32 v[16:17], v[14:15]
	s_andn2_b64 exec, exec, s[34:35]
	s_cbranch_execnz .LBB0_97
.LBB0_98:                               ; %Flow4433
                                        ;   in Loop: Header=BB0_31 Depth=2
	s_or_b64 exec, exec, s[30:31]
	global_load_dwordx2 v[14:15], v5, s[16:17] offset:16
	s_mov_b64 s[34:35], exec
	v_mbcnt_lo_u32_b32 v4, s34, 0
	v_mbcnt_hi_u32_b32 v4, s35, v4
	v_cmp_eq_u32_e32 vcc, 0, v4
	s_and_saveexec_b64 s[30:31], vcc
	s_cbranch_execz .LBB0_100
; %bb.99:                               ;   in Loop: Header=BB0_31 Depth=2
	s_bcnt1_i32_b64 s34, s[34:35]
	v_mov_b32_e32 v4, s34
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[4:5], off offset:8 sc1
.LBB0_100:                              ;   in Loop: Header=BB0_31 Depth=2
	s_or_b64 exec, exec, s[30:31]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[16:17], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[16:17]
	s_cbranch_vccnz .LBB0_102
; %bb.101:                              ;   in Loop: Header=BB0_31 Depth=2
	global_load_dword v4, v[14:15], off offset:24
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_store_dwordx2 v[16:17], v[4:5], off sc0 sc1
	v_and_b32_e32 v4, 0xffffff, v4
	s_nop 0
	v_readfirstlane_b32 s30, v4
	s_mov_b32 m0, s30
	s_nop 0
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_102:                              ; %Flow4434
                                        ;   in Loop: Header=BB0_31 Depth=2
	s_or_b64 exec, exec, s[28:29]
	v_mov_b32_e32 v59, v5
	v_lshl_add_u64 v[6:7], v[6:7], 0, v[58:59]
	s_branch .LBB0_106
.LBB0_103:                              ;   in Loop: Header=BB0_106 Depth=3
	s_or_b64 exec, exec, s[28:29]
	v_readfirstlane_b32 s28, v4
	s_cmp_eq_u32 s28, 0
	s_cbranch_scc1 .LBB0_105
; %bb.104:                              ;   in Loop: Header=BB0_106 Depth=3
	s_sleep 1
	s_cbranch_execnz .LBB0_106
	s_branch .LBB0_108
.LBB0_105:                              ;   in Loop: Header=BB0_31 Depth=2
	s_branch .LBB0_108
.LBB0_106:                              ;   Parent Loop BB0_2 Depth=1
                                        ;     Parent Loop BB0_31 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_mov_b32_e32 v4, 1
	s_and_saveexec_b64 s[28:29], s[0:1]
	s_cbranch_execz .LBB0_103
; %bb.107:                              ;   in Loop: Header=BB0_106 Depth=3
	global_load_dword v4, v[38:39], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v4, 1, v4
	s_branch .LBB0_103
.LBB0_108:                              ;   in Loop: Header=BB0_31 Depth=2
	global_load_dwordx4 v[14:17], v[6:7], off
	s_and_saveexec_b64 s[28:29], s[0:1]
	s_cbranch_execz .LBB0_30
; %bb.109:                              ;   in Loop: Header=BB0_31 Depth=2
	global_load_dwordx2 v[6:7], v5, s[16:17] offset:40
	global_load_dwordx2 v[20:21], v5, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[22:23], v5, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[16:17], v[6:7], 0, 1
	v_lshl_add_u64 v[24:25], v[16:17], 0, s[26:27]
	v_cmp_eq_u64_e32 vcc, 0, v[24:25]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v18, v20
	v_mov_b32_e32 v19, v21
	v_cndmask_b32_e32 v17, v25, v17, vcc
	v_cndmask_b32_e32 v16, v24, v16, vcc
	v_and_b32_e32 v4, v17, v7
	v_and_b32_e32 v6, v16, v6
	v_mul_lo_u32 v4, v4, 24
	v_mul_hi_u32 v7, v6, 24
	v_mul_lo_u32 v6, v6, 24
	v_add_u32_e32 v7, v7, v4
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[6:7], v[22:23], 0, v[6:7]
	global_store_dwordx2 v[6:7], v[20:21], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v5, v[16:19], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[18:19], v[20:21]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_30
; %bb.110:                              ; %.preheader.i.i15.i.i.preheader
                                        ;   in Loop: Header=BB0_31 Depth=2
	s_mov_b64 s[0:1], 0
.LBB0_111:                              ; %.preheader.i.i15.i.i
                                        ;   Parent Loop BB0_2 Depth=1
                                        ;     Parent Loop BB0_31 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_sleep 1
	global_store_dwordx2 v[6:7], v[18:19], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[20:21], v5, v[16:19], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[20:21], v[18:19]
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b64_e32 v[18:19], v[20:21]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_111
	s_branch .LBB0_30
.LBB0_112:                              ; %Flow4473
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_branch .LBB0_141
.LBB0_113:                              ;   in Loop: Header=BB0_2 Depth=1
                                        ; implicit-def: $vgpr14_vgpr15
	s_cbranch_execz .LBB0_141
; %bb.114:                              ;   in Loop: Header=BB0_2 Depth=1
	v_readfirstlane_b32 s0, v56
	v_mov_b64_e32 v[6:7], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v56
	s_and_saveexec_b64 s[20:21], s[0:1]
	s_cbranch_execz .LBB0_120
; %bb.115:                              ;   in Loop: Header=BB0_2 Depth=1
	global_load_dwordx2 v[16:17], v5, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[6:7], v5, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v5, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v4, v6, v16
	v_and_b32_e32 v6, v7, v17
	v_mul_lo_u32 v6, v6, 24
	v_mul_hi_u32 v7, v4, 24
	v_add_u32_e32 v7, v7, v6
	v_mul_lo_u32 v6, v4, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[6:7], v[14:15], 0, v[6:7]
	global_load_dwordx2 v[14:15], v[6:7], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v5, v[14:17], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[6:7], v[16:17]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_119
; %bb.116:                              ; %.preheader3.i.i.i23.i.preheader
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_117:                              ; %.preheader3.i.i.i23.i
                                        ;   Parent Loop BB0_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[14:15], v5, s[16:17] offset:40
	global_load_dwordx2 v[18:19], v5, s[16:17]
	v_mov_b64_e32 v[16:17], v[6:7]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v4, v14, v16
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[6:7], s[26:27], v4, 24, v[18:19]
	v_and_b32_e32 v15, v15, v17
	v_mov_b32_e32 v4, v7
	v_mad_u64_u32 v[14:15], s[26:27], v15, 24, v[4:5]
	v_mov_b32_e32 v7, v14
	global_load_dwordx2 v[14:15], v[6:7], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v5, v[14:17], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[6:7], v[16:17]
	s_or_b64 s[24:25], vcc, s[24:25]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_117
; %bb.118:                              ; %Flow4486
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[24:25]
.LBB0_119:                              ; %Flow4488
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_120:                              ; %.loopexit4.i.i.i18.i
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[20:21]
	global_load_dwordx2 v[18:19], v5, s[16:17] offset:40
	global_load_dwordx4 v[14:17], v5, s[16:17]
	v_readfirstlane_b32 s20, v6
	v_readfirstlane_b32 s21, v7
	s_mov_b64 s[22:23], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s24, v18
	v_readfirstlane_b32 s25, v19
	s_and_b64 s[24:25], s[20:21], s[24:25]
	s_mul_i32 s26, s25, 24
	s_mul_hi_u32 s27, s24, 24
	s_add_i32 s27, s27, s26
	s_mul_i32 s26, s24, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[18:19], v[14:15], 0, s[26:27]
	s_and_saveexec_b64 s[26:27], s[0:1]
	s_cbranch_execz .LBB0_122
; %bb.121:                              ;   in Loop: Header=BB0_2 Depth=1
	v_mov_b64_e32 v[6:7], s[22:23]
	global_store_dwordx4 v[18:19], v[6:9], off offset:8
.LBB0_122:                              ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[26:27]
	s_lshl_b64 s[22:23], s[24:25], 12
	v_lshl_add_u64 v[6:7], v[16:17], 0, s[22:23]
	v_mov_b64_e32 v[22:23], s[10:11]
	v_and_or_b32 v2, v2, s36, 32
	v_mov_b32_e32 v4, v5
	v_readfirstlane_b32 s22, v6
	v_readfirstlane_b32 s23, v7
	v_mov_b64_e32 v[20:21], s[8:9]
	s_nop 3
	global_store_dwordx4 v58, v[2:5], s[22:23]
	global_store_dwordx4 v58, v[20:23], s[22:23] offset:16
	global_store_dwordx4 v58, v[20:23], s[22:23] offset:32
	global_store_dwordx4 v58, v[20:23], s[22:23] offset:48
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_130
; %bb.123:                              ;   in Loop: Header=BB0_2 Depth=1
	global_load_dwordx2 v[22:23], v5, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[2:3], v5, s[16:17] offset:40
	v_mov_b32_e32 v20, s20
	v_mov_b32_e32 v21, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s24, v2
	v_readfirstlane_b32 s25, v3
	s_and_b64 s[24:25], s[24:25], s[20:21]
	s_mul_i32 s25, s25, 24
	s_mul_hi_u32 s26, s24, 24
	s_mul_i32 s24, s24, 24
	s_add_i32 s25, s26, s25
	v_lshl_add_u64 v[2:3], v[14:15], 0, s[24:25]
	global_store_dwordx2 v[2:3], v[22:23], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[16:17], v5, v[20:23], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[16:17], v[22:23]
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB0_126
; %bb.124:                              ; %.preheader1.i.i.i21.i.preheader
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_mov_b64 s[26:27], 0
.LBB0_125:                              ; %.preheader1.i.i.i21.i
                                        ;   Parent Loop BB0_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[16:17], off
	v_mov_b32_e32 v14, s20
	v_mov_b32_e32 v15, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[14:15], v5, v[14:17], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[14:15], v[16:17]
	s_or_b64 s[26:27], vcc, s[26:27]
	v_mov_b64_e32 v[16:17], v[14:15]
	s_andn2_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB0_125
.LBB0_126:                              ; %Flow4484
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[24:25]
	global_load_dwordx2 v[2:3], v5, s[16:17] offset:16
	s_mov_b64 s[26:27], exec
	v_mbcnt_lo_u32_b32 v4, s26, 0
	v_mbcnt_hi_u32_b32 v4, s27, v4
	v_cmp_eq_u32_e32 vcc, 0, v4
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB0_128
; %bb.127:                              ;   in Loop: Header=BB0_2 Depth=1
	s_bcnt1_i32_b64 s26, s[26:27]
	v_mov_b32_e32 v4, s26
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[2:3], v[4:5], off offset:8 sc1
.LBB0_128:                              ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[14:15], v[2:3], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[14:15]
	s_cbranch_vccnz .LBB0_130
; %bb.129:                              ;   in Loop: Header=BB0_2 Depth=1
	global_load_dword v4, v[2:3], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v2, 0xffffff, v4
	s_nop 0
	v_readfirstlane_b32 s24, v2
	s_mov_b32 m0, s24
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[14:15], v[4:5], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_130:                              ; %Flow4485
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[22:23]
	v_mov_b32_e32 v59, v5
	v_lshl_add_u64 v[2:3], v[6:7], 0, v[58:59]
	s_branch .LBB0_134
.LBB0_131:                              ;   in Loop: Header=BB0_134 Depth=2
	s_or_b64 exec, exec, s[22:23]
	v_readfirstlane_b32 s22, v4
	s_cmp_eq_u32 s22, 0
	s_cbranch_scc1 .LBB0_133
; %bb.132:                              ;   in Loop: Header=BB0_134 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_134
	s_branch .LBB0_136
.LBB0_133:                              ;   in Loop: Header=BB0_2 Depth=1
	s_branch .LBB0_136
.LBB0_134:                              ;   Parent Loop BB0_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v4, 1
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_131
; %bb.135:                              ;   in Loop: Header=BB0_134 Depth=2
	global_load_dword v4, v[18:19], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v4, 1, v4
	s_branch .LBB0_131
.LBB0_136:                              ;   in Loop: Header=BB0_2 Depth=1
	global_load_dwordx2 v[14:15], v[2:3], off
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_140
; %bb.137:                              ;   in Loop: Header=BB0_2 Depth=1
	global_load_dwordx2 v[2:3], v5, s[16:17] offset:40
	global_load_dwordx2 v[6:7], v5, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[20:21], v5, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[16:17], v[2:3], 0, 1
	v_lshl_add_u64 v[22:23], v[16:17], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v18, v6
	v_mov_b32_e32 v19, v7
	v_cndmask_b32_e32 v17, v23, v17, vcc
	v_cndmask_b32_e32 v16, v22, v16, vcc
	v_and_b32_e32 v3, v17, v3
	v_and_b32_e32 v2, v16, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v4, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v4, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[20:21], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[6:7], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v5, v[16:19], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[18:19], v[6:7]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_140
; %bb.138:                              ; %.preheader.i.i.i20.i.preheader
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_mov_b64 s[0:1], 0
.LBB0_139:                              ; %.preheader.i.i.i20.i
                                        ;   Parent Loop BB0_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[18:19], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v5, v[16:19], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[6:7], v[18:19]
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b64_e32 v[18:19], v[6:7]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_139
.LBB0_140:                              ; %Flow4477
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_141:                              ; %__ockl_printf_append_string_n.exit.i
                                        ;   in Loop: Header=BB0_2 Depth=1
	v_readfirstlane_b32 s0, v56
	s_waitcnt vmcnt(0)
	v_mov_b64_e32 v[2:3], 0
	v_cmp_eq_u32_e64 s[0:1], s0, v56
	s_and_saveexec_b64 s[20:21], s[0:1]
	s_cbranch_execz .LBB0_147
; %bb.142:                              ;   in Loop: Header=BB0_2 Depth=1
	global_load_dwordx2 v[18:19], v5, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v5, s[16:17] offset:40
	global_load_dwordx2 v[6:7], v5, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v18
	v_and_b32_e32 v3, v3, v19
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v4, v2, 24
	v_add_u32_e32 v3, v4, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[6:7], 0, v[2:3]
	global_load_dwordx2 v[16:17], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v5, v[16:19], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[18:19]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_146
; %bb.143:                              ; %.preheader3.i.i.i30.i.preheader
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_144:                              ; %.preheader3.i.i.i30.i
                                        ;   Parent Loop BB0_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[6:7], v5, s[16:17] offset:40
	global_load_dwordx2 v[16:17], v5, s[16:17]
	v_mov_b64_e32 v[18:19], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v6, v18
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[26:27], v2, 24, v[16:17]
	v_and_b32_e32 v7, v7, v19
	v_mov_b32_e32 v4, v3
	v_mad_u64_u32 v[6:7], s[26:27], v7, 24, v[4:5]
	v_mov_b32_e32 v3, v6
	global_load_dwordx2 v[16:17], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v5, v[16:19], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[18:19]
	s_or_b64 s[24:25], vcc, s[24:25]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_144
; %bb.145:                              ; %Flow4421
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[24:25]
.LBB0_146:                              ; %Flow4423
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_147:                              ; %.loopexit4.i.i.i24.i
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[20:21]
	global_load_dwordx2 v[6:7], v5, s[16:17] offset:40
	global_load_dwordx4 v[18:21], v5, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[22:23], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s24, v6
	v_readfirstlane_b32 s25, v7
	s_and_b64 s[24:25], s[20:21], s[24:25]
	s_mul_i32 s26, s25, 24
	s_mul_hi_u32 s27, s24, 24
	s_add_i32 s27, s27, s26
	s_mul_i32 s26, s24, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[18:19], 0, s[26:27]
	s_and_saveexec_b64 s[26:27], s[0:1]
	s_cbranch_execz .LBB0_149
; %bb.148:                              ;   in Loop: Header=BB0_2 Depth=1
	v_mov_b64_e32 v[6:7], s[22:23]
	global_store_dwordx4 v[2:3], v[6:9], off offset:8
.LBB0_149:                              ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[26:27]
	s_lshl_b64 s[22:23], s[24:25], 12
	v_lshl_add_u64 v[6:7], v[20:21], 0, s[22:23]
	v_and_or_b32 v14, v14, s36, 32
	v_mov_b32_e32 v16, v0
	v_mov_b32_e32 v17, v1
	v_readfirstlane_b32 s22, v6
	v_readfirstlane_b32 s23, v7
	s_nop 4
	global_store_dwordx4 v58, v[14:17], s[22:23]
	s_nop 1
	v_mov_b64_e32 v[16:17], s[10:11]
	v_mov_b64_e32 v[14:15], s[8:9]
	global_store_dwordx4 v58, v[14:17], s[22:23] offset:16
	global_store_dwordx4 v58, v[14:17], s[22:23] offset:32
	global_store_dwordx4 v58, v[14:17], s[22:23] offset:48
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_157
; %bb.150:                              ;   in Loop: Header=BB0_2 Depth=1
	global_load_dwordx2 v[22:23], v5, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v5, s[16:17] offset:40
	v_mov_b32_e32 v20, s20
	v_mov_b32_e32 v21, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s24, v14
	v_readfirstlane_b32 s25, v15
	s_and_b64 s[24:25], s[24:25], s[20:21]
	s_mul_i32 s25, s25, 24
	s_mul_hi_u32 s26, s24, 24
	s_mul_i32 s24, s24, 24
	s_add_i32 s25, s26, s25
	v_lshl_add_u64 v[18:19], v[18:19], 0, s[24:25]
	global_store_dwordx2 v[18:19], v[22:23], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[16:17], v5, v[20:23], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[16:17], v[22:23]
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB0_153
; %bb.151:                              ; %.preheader1.i.i.i28.i.preheader
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_mov_b64 s[26:27], 0
.LBB0_152:                              ; %.preheader1.i.i.i28.i
                                        ;   Parent Loop BB0_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[18:19], v[16:17], off
	v_mov_b32_e32 v14, s20
	v_mov_b32_e32 v15, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[14:15], v5, v[14:17], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[14:15], v[16:17]
	s_or_b64 s[26:27], vcc, s[26:27]
	v_mov_b64_e32 v[16:17], v[14:15]
	s_andn2_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB0_152
.LBB0_153:                              ; %Flow4419
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[24:25]
	global_load_dwordx2 v[14:15], v5, s[16:17] offset:16
	s_mov_b64 s[26:27], exec
	v_mbcnt_lo_u32_b32 v4, s26, 0
	v_mbcnt_hi_u32_b32 v4, s27, v4
	v_cmp_eq_u32_e32 vcc, 0, v4
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB0_155
; %bb.154:                              ;   in Loop: Header=BB0_2 Depth=1
	s_bcnt1_i32_b64 s26, s[26:27]
	v_mov_b32_e32 v4, s26
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[4:5], off offset:8 sc1
.LBB0_155:                              ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[16:17], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[16:17]
	s_cbranch_vccnz .LBB0_157
; %bb.156:                              ;   in Loop: Header=BB0_2 Depth=1
	global_load_dword v4, v[14:15], off offset:24
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_store_dwordx2 v[16:17], v[4:5], off sc0 sc1
	v_and_b32_e32 v4, 0xffffff, v4
	s_nop 0
	v_readfirstlane_b32 s24, v4
	s_mov_b32 m0, s24
	s_nop 0
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_157:                              ; %Flow4420
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[22:23]
	v_mov_b32_e32 v59, v5
	v_lshl_add_u64 v[6:7], v[6:7], 0, v[58:59]
	s_branch .LBB0_161
.LBB0_158:                              ;   in Loop: Header=BB0_161 Depth=2
	s_or_b64 exec, exec, s[22:23]
	v_readfirstlane_b32 s22, v4
	s_cmp_eq_u32 s22, 0
	s_cbranch_scc1 .LBB0_160
; %bb.159:                              ;   in Loop: Header=BB0_161 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_161
	s_branch .LBB0_163
.LBB0_160:                              ;   in Loop: Header=BB0_2 Depth=1
	s_branch .LBB0_163
.LBB0_161:                              ;   Parent Loop BB0_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v4, 1
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_158
; %bb.162:                              ;   in Loop: Header=BB0_161 Depth=2
	global_load_dword v4, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v4, 1, v4
	s_branch .LBB0_158
.LBB0_163:                              ;   in Loop: Header=BB0_2 Depth=1
	global_load_dwordx2 v[2:3], v[6:7], off
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_167
; %bb.164:                              ;   in Loop: Header=BB0_2 Depth=1
	global_load_dwordx2 v[6:7], v5, s[16:17] offset:40
	global_load_dwordx2 v[18:19], v5, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[20:21], v5, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[14:15], v[6:7], 0, 1
	v_lshl_add_u64 v[22:23], v[14:15], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v16, v18
	v_mov_b32_e32 v17, v19
	v_cndmask_b32_e32 v15, v23, v15, vcc
	v_cndmask_b32_e32 v14, v22, v14, vcc
	v_and_b32_e32 v4, v15, v7
	v_and_b32_e32 v6, v14, v6
	v_mul_lo_u32 v4, v4, 24
	v_mul_hi_u32 v7, v6, 24
	v_mul_lo_u32 v6, v6, 24
	v_add_u32_e32 v7, v7, v4
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[6:7], v[20:21], 0, v[6:7]
	global_store_dwordx2 v[6:7], v[18:19], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[16:17], v5, v[14:17], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[16:17], v[18:19]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_167
; %bb.165:                              ; %.preheader.i.i.i27.i.preheader
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_mov_b64 s[0:1], 0
.LBB0_166:                              ; %.preheader.i.i.i27.i
                                        ;   Parent Loop BB0_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[6:7], v[16:17], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v5, v[14:17], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[16:17]
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b64_e32 v[16:17], v[18:19]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_166
.LBB0_167:                              ; %__ockl_printf_append_args.exit.i
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[22:23]
	v_readfirstlane_b32 s0, v56
	v_mov_b64_e32 v[6:7], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v56
	s_and_saveexec_b64 s[20:21], s[0:1]
	s_cbranch_execz .LBB0_173
; %bb.168:                              ;   in Loop: Header=BB0_2 Depth=1
	global_load_dwordx2 v[16:17], v5, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[6:7], v5, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v5, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v4, v6, v16
	v_and_b32_e32 v6, v7, v17
	v_mul_lo_u32 v6, v6, 24
	v_mul_hi_u32 v7, v4, 24
	v_add_u32_e32 v7, v7, v6
	v_mul_lo_u32 v6, v4, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[6:7], v[14:15], 0, v[6:7]
	global_load_dwordx2 v[14:15], v[6:7], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v5, v[14:17], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[6:7], v[16:17]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_172
; %bb.169:                              ; %.preheader3.i.i.i37.i.preheader
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_170:                              ; %.preheader3.i.i.i37.i
                                        ;   Parent Loop BB0_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[14:15], v5, s[16:17] offset:40
	global_load_dwordx2 v[18:19], v5, s[16:17]
	v_mov_b64_e32 v[16:17], v[6:7]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v4, v14, v16
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[6:7], s[26:27], v4, 24, v[18:19]
	v_and_b32_e32 v15, v15, v17
	v_mov_b32_e32 v4, v7
	v_mad_u64_u32 v[14:15], s[26:27], v15, 24, v[4:5]
	v_mov_b32_e32 v7, v14
	global_load_dwordx2 v[14:15], v[6:7], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v5, v[14:17], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[6:7], v[16:17]
	s_or_b64 s[24:25], vcc, s[24:25]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_170
; %bb.171:                              ; %Flow4407
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[24:25]
.LBB0_172:                              ; %Flow4409
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_173:                              ; %.loopexit4.i.i.i31.i
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[20:21]
	global_load_dwordx2 v[18:19], v5, s[16:17] offset:40
	global_load_dwordx4 v[14:17], v5, s[16:17]
	v_readfirstlane_b32 s20, v6
	v_readfirstlane_b32 s21, v7
	s_mov_b64 s[22:23], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s24, v18
	v_readfirstlane_b32 s25, v19
	s_and_b64 s[24:25], s[20:21], s[24:25]
	s_mul_i32 s26, s25, 24
	s_mul_hi_u32 s27, s24, 24
	s_add_i32 s27, s27, s26
	s_mul_i32 s26, s24, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[18:19], v[14:15], 0, s[26:27]
	s_and_saveexec_b64 s[26:27], s[0:1]
	s_cbranch_execz .LBB0_175
; %bb.174:                              ;   in Loop: Header=BB0_2 Depth=1
	v_mov_b64_e32 v[6:7], s[22:23]
	global_store_dwordx4 v[18:19], v[6:9], off offset:8
.LBB0_175:                              ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[26:27]
	s_lshl_b64 s[22:23], s[24:25], 12
	v_lshl_add_u64 v[6:7], v[16:17], 0, s[22:23]
	v_mov_b64_e32 v[22:23], s[10:11]
	v_lshrrev_b32_e32 v4, 3, v37
	v_and_or_b32 v2, v2, s36, 32
	v_readfirstlane_b32 s22, v6
	v_readfirstlane_b32 s23, v7
	v_mov_b64_e32 v[20:21], s[8:9]
	s_nop 3
	global_store_dwordx4 v58, v[2:5], s[22:23]
	global_store_dwordx4 v58, v[20:23], s[22:23] offset:16
	global_store_dwordx4 v58, v[20:23], s[22:23] offset:32
	global_store_dwordx4 v58, v[20:23], s[22:23] offset:48
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_183
; %bb.176:                              ;   in Loop: Header=BB0_2 Depth=1
	global_load_dwordx2 v[22:23], v5, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[2:3], v5, s[16:17] offset:40
	v_mov_b32_e32 v20, s20
	v_mov_b32_e32 v21, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s24, v2
	v_readfirstlane_b32 s25, v3
	s_and_b64 s[24:25], s[24:25], s[20:21]
	s_mul_i32 s25, s25, 24
	s_mul_hi_u32 s26, s24, 24
	s_mul_i32 s24, s24, 24
	s_add_i32 s25, s26, s25
	v_lshl_add_u64 v[2:3], v[14:15], 0, s[24:25]
	global_store_dwordx2 v[2:3], v[22:23], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[16:17], v5, v[20:23], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[16:17], v[22:23]
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB0_179
; %bb.177:                              ; %.preheader1.i.i.i35.i.preheader
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_mov_b64 s[26:27], 0
.LBB0_178:                              ; %.preheader1.i.i.i35.i
                                        ;   Parent Loop BB0_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[16:17], off
	v_mov_b32_e32 v14, s20
	v_mov_b32_e32 v15, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[14:15], v5, v[14:17], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[14:15], v[16:17]
	s_or_b64 s[26:27], vcc, s[26:27]
	v_mov_b64_e32 v[16:17], v[14:15]
	s_andn2_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB0_178
.LBB0_179:                              ; %Flow4405
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[24:25]
	global_load_dwordx2 v[2:3], v5, s[16:17] offset:16
	s_mov_b64 s[26:27], exec
	v_mbcnt_lo_u32_b32 v14, s26, 0
	v_mbcnt_hi_u32_b32 v14, s27, v14
	v_cmp_eq_u32_e32 vcc, 0, v14
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB0_181
; %bb.180:                              ;   in Loop: Header=BB0_2 Depth=1
	s_bcnt1_i32_b64 s26, s[26:27]
	v_mov_b32_e32 v14, s26
	v_mov_b32_e32 v15, v5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[2:3], v[14:15], off offset:8 sc1
.LBB0_181:                              ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[14:15], v[2:3], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[14:15]
	s_cbranch_vccnz .LBB0_183
; %bb.182:                              ;   in Loop: Header=BB0_2 Depth=1
	global_load_dword v2, v[2:3], off offset:24
	v_mov_b32_e32 v3, v5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_store_dwordx2 v[14:15], v[2:3], off sc0 sc1
	v_and_b32_e32 v2, 0xffffff, v2
	s_nop 0
	v_readfirstlane_b32 s24, v2
	s_mov_b32 m0, s24
	s_nop 0
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_183:                              ; %Flow4406
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[22:23]
	v_mov_b32_e32 v59, v5
	v_lshl_add_u64 v[2:3], v[6:7], 0, v[58:59]
	s_branch .LBB0_187
.LBB0_184:                              ;   in Loop: Header=BB0_187 Depth=2
	s_or_b64 exec, exec, s[22:23]
	v_readfirstlane_b32 s22, v6
	s_cmp_eq_u32 s22, 0
	s_cbranch_scc1 .LBB0_186
; %bb.185:                              ;   in Loop: Header=BB0_187 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_187
	s_branch .LBB0_189
.LBB0_186:                              ;   in Loop: Header=BB0_2 Depth=1
	s_branch .LBB0_189
.LBB0_187:                              ;   Parent Loop BB0_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v6, 1
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_184
; %bb.188:                              ;   in Loop: Header=BB0_187 Depth=2
	global_load_dword v6, v[18:19], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v6, 1, v6
	s_branch .LBB0_184
.LBB0_189:                              ;   in Loop: Header=BB0_2 Depth=1
	global_load_dwordx2 v[14:15], v[2:3], off
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_193
; %bb.190:                              ;   in Loop: Header=BB0_2 Depth=1
	global_load_dwordx2 v[2:3], v5, s[16:17] offset:40
	global_load_dwordx2 v[6:7], v5, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[20:21], v5, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[16:17], v[2:3], 0, 1
	v_lshl_add_u64 v[22:23], v[16:17], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v18, v6
	v_cndmask_b32_e32 v17, v23, v17, vcc
	v_cndmask_b32_e32 v16, v22, v16, vcc
	v_and_b32_e32 v3, v17, v3
	v_and_b32_e32 v2, v16, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v19, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v19, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[20:21], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[6:7], off
	v_mov_b32_e32 v19, v7
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v5, v[16:19], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[18:19], v[6:7]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_193
; %bb.191:                              ; %.preheader.i.i.i34.i.preheader
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_mov_b64 s[0:1], 0
.LBB0_192:                              ; %.preheader.i.i.i34.i
                                        ;   Parent Loop BB0_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[18:19], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v5, v[16:19], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[6:7], v[18:19]
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b64_e32 v[18:19], v[6:7]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_192
.LBB0_193:                              ; %__ockl_printf_append_args.exit38.i
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[22:23]
	v_readfirstlane_b32 s0, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v56
	s_and_saveexec_b64 s[20:21], s[0:1]
	s_cbranch_execz .LBB0_199
; %bb.194:                              ;   in Loop: Header=BB0_2 Depth=1
	global_load_dwordx2 v[18:19], v5, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v5, s[16:17] offset:40
	global_load_dwordx2 v[6:7], v5, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v18
	v_and_b32_e32 v3, v3, v19
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v16, v2, 24
	v_add_u32_e32 v3, v16, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[6:7], 0, v[2:3]
	global_load_dwordx2 v[16:17], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v5, v[16:19], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[18:19]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_198
; %bb.195:                              ; %.preheader3.i.i.i45.i.preheader
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_196:                              ; %.preheader3.i.i.i45.i
                                        ;   Parent Loop BB0_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[6:7], v5, s[16:17] offset:40
	global_load_dwordx2 v[16:17], v5, s[16:17]
	v_mov_b64_e32 v[18:19], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v6, v18
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[26:27], v2, 24, v[16:17]
	v_and_b32_e32 v7, v7, v19
	v_mov_b32_e32 v6, v3
	v_mad_u64_u32 v[6:7], s[26:27], v7, 24, v[6:7]
	v_mov_b32_e32 v3, v6
	global_load_dwordx2 v[16:17], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v5, v[16:19], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[18:19]
	s_or_b64 s[24:25], vcc, s[24:25]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_196
; %bb.197:                              ; %Flow4393
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[24:25]
.LBB0_198:                              ; %Flow4395
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_199:                              ; %.loopexit4.i.i.i39.i
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[20:21]
	global_load_dwordx2 v[6:7], v5, s[16:17] offset:40
	global_load_dwordx4 v[18:21], v5, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[22:23], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s24, v6
	v_readfirstlane_b32 s25, v7
	s_and_b64 s[24:25], s[20:21], s[24:25]
	s_mul_i32 s26, s25, 24
	s_mul_hi_u32 s27, s24, 24
	s_add_i32 s27, s27, s26
	s_mul_i32 s26, s24, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[18:19], 0, s[26:27]
	s_and_saveexec_b64 s[26:27], s[0:1]
	s_cbranch_execz .LBB0_201
; %bb.200:                              ;   in Loop: Header=BB0_2 Depth=1
	v_mov_b64_e32 v[6:7], s[22:23]
	global_store_dwordx4 v[2:3], v[6:9], off offset:8
.LBB0_201:                              ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[26:27]
	s_lshl_b64 s[22:23], s[24:25], 12
	v_lshl_add_u64 v[6:7], v[20:21], 0, s[22:23]
	v_lshlrev_b32_e32 v16, 3, v37
	v_mov_b64_e32 v[22:23], s[10:11]
	v_and_b32_e32 v16, 56, v16
	v_and_or_b32 v14, v14, s36, 32
	v_mov_b32_e32 v17, v5
	v_readfirstlane_b32 s22, v6
	v_readfirstlane_b32 s23, v7
	v_mov_b64_e32 v[20:21], s[8:9]
	s_nop 3
	global_store_dwordx4 v58, v[14:17], s[22:23]
	global_store_dwordx4 v58, v[20:23], s[22:23] offset:16
	global_store_dwordx4 v58, v[20:23], s[22:23] offset:32
	global_store_dwordx4 v58, v[20:23], s[22:23] offset:48
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_209
; %bb.202:                              ;   in Loop: Header=BB0_2 Depth=1
	global_load_dwordx2 v[22:23], v5, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v5, s[16:17] offset:40
	v_mov_b32_e32 v20, s20
	v_mov_b32_e32 v21, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s24, v14
	v_readfirstlane_b32 s25, v15
	s_and_b64 s[24:25], s[24:25], s[20:21]
	s_mul_i32 s25, s25, 24
	s_mul_hi_u32 s26, s24, 24
	s_mul_i32 s24, s24, 24
	s_add_i32 s25, s26, s25
	v_lshl_add_u64 v[14:15], v[18:19], 0, s[24:25]
	global_store_dwordx2 v[14:15], v[22:23], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[20:21], v5, v[20:23], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[20:21], v[22:23]
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB0_205
; %bb.203:                              ; %.preheader1.i.i.i43.i.preheader
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_mov_b64 s[26:27], 0
.LBB0_204:                              ; %.preheader1.i.i.i43.i
                                        ;   Parent Loop BB0_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[20:21], off
	v_mov_b32_e32 v18, s20
	v_mov_b32_e32 v19, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v5, v[18:21], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[20:21]
	s_or_b64 s[26:27], vcc, s[26:27]
	v_mov_b64_e32 v[20:21], v[18:19]
	s_andn2_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB0_204
.LBB0_205:                              ; %Flow4391
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[24:25]
	global_load_dwordx2 v[14:15], v5, s[16:17] offset:16
	s_mov_b64 s[26:27], exec
	v_mbcnt_lo_u32_b32 v17, s26, 0
	v_mbcnt_hi_u32_b32 v17, s27, v17
	v_cmp_eq_u32_e32 vcc, 0, v17
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB0_207
; %bb.206:                              ;   in Loop: Header=BB0_2 Depth=1
	s_bcnt1_i32_b64 s26, s[26:27]
	v_mov_b32_e32 v18, s26
	v_mov_b32_e32 v19, v5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[18:19], off offset:8 sc1
.LBB0_207:                              ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_209
; %bb.208:                              ;   in Loop: Header=BB0_2 Depth=1
	global_load_dword v14, v[14:15], off offset:24
	v_mov_b32_e32 v15, v5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_store_dwordx2 v[18:19], v[14:15], off sc0 sc1
	v_and_b32_e32 v14, 0xffffff, v14
	s_nop 0
	v_readfirstlane_b32 s24, v14
	s_mov_b32 m0, s24
	s_nop 0
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_209:                              ; %Flow4392
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[22:23]
	v_mov_b32_e32 v59, v5
	v_lshl_add_u64 v[6:7], v[6:7], 0, v[58:59]
	s_branch .LBB0_213
.LBB0_210:                              ;   in Loop: Header=BB0_213 Depth=2
	s_or_b64 exec, exec, s[22:23]
	v_readfirstlane_b32 s22, v14
	s_cmp_eq_u32 s22, 0
	s_cbranch_scc1 .LBB0_212
; %bb.211:                              ;   in Loop: Header=BB0_213 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_213
	s_branch .LBB0_215
.LBB0_212:                              ;   in Loop: Header=BB0_2 Depth=1
	s_branch .LBB0_215
.LBB0_213:                              ;   Parent Loop BB0_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v14, 1
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_210
; %bb.214:                              ;   in Loop: Header=BB0_213 Depth=2
	global_load_dword v14, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v14, 1, v14
	s_branch .LBB0_210
.LBB0_215:                              ;   in Loop: Header=BB0_2 Depth=1
	global_load_dwordx2 v[18:19], v[6:7], off
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_219
; %bb.216:                              ;   in Loop: Header=BB0_2 Depth=1
	global_load_dwordx2 v[2:3], v5, s[16:17] offset:40
	global_load_dwordx2 v[6:7], v5, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v5, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[20:21], v[2:3], 0, 1
	v_lshl_add_u64 v[24:25], v[20:21], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[24:25]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v22, v6
	v_mov_b32_e32 v23, v7
	v_cndmask_b32_e32 v21, v25, v21, vcc
	v_cndmask_b32_e32 v20, v24, v20, vcc
	v_and_b32_e32 v3, v21, v3
	v_and_b32_e32 v2, v20, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v17, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v17, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[6:7], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[22:23], v5, v[20:23], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[22:23], v[6:7]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_219
; %bb.217:                              ; %.preheader.i.i.i42.i.preheader
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_mov_b64 s[0:1], 0
.LBB0_218:                              ; %.preheader.i.i.i42.i
                                        ;   Parent Loop BB0_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[22:23], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v5, v[20:23], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[6:7], v[22:23]
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b64_e32 v[22:23], v[6:7]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_218
.LBB0_219:                              ; %__ockl_printf_append_args.exit46.i
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[22:23]
	v_readfirstlane_b32 s0, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v56
	s_and_saveexec_b64 s[20:21], s[0:1]
	s_cbranch_execz .LBB0_225
; %bb.220:                              ;   in Loop: Header=BB0_2 Depth=1
	global_load_dwordx2 v[22:23], v5, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v5, s[16:17] offset:40
	global_load_dwordx2 v[6:7], v5, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v22
	v_and_b32_e32 v3, v3, v23
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v14, v2, 24
	v_add_u32_e32 v3, v14, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[6:7], 0, v[2:3]
	global_load_dwordx2 v[20:21], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v5, v[20:23], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[22:23]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_224
; %bb.221:                              ; %.preheader3.i.i.i53.i.preheader
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_222:                              ; %.preheader3.i.i.i53.i
                                        ;   Parent Loop BB0_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[6:7], v5, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v5, s[16:17]
	v_mov_b64_e32 v[22:23], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v6, v22
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[26:27], v2, 24, v[14:15]
	v_and_b32_e32 v7, v7, v23
	v_mov_b32_e32 v6, v3
	v_mad_u64_u32 v[6:7], s[26:27], v7, 24, v[6:7]
	v_mov_b32_e32 v3, v6
	global_load_dwordx2 v[20:21], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v5, v[20:23], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[22:23]
	s_or_b64 s[24:25], vcc, s[24:25]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_222
; %bb.223:                              ; %Flow4379
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[24:25]
.LBB0_224:                              ; %Flow4381
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_225:                              ; %.loopexit4.i.i.i47.i
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[20:21]
	global_load_dwordx2 v[6:7], v5, s[16:17] offset:40
	global_load_dwordx4 v[22:25], v5, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[22:23], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s24, v6
	v_readfirstlane_b32 s25, v7
	s_and_b64 s[24:25], s[20:21], s[24:25]
	s_mul_i32 s26, s25, 24
	s_mul_hi_u32 s27, s24, 24
	s_add_i32 s27, s27, s26
	s_mul_i32 s26, s24, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[22:23], 0, s[26:27]
	s_and_saveexec_b64 s[26:27], s[0:1]
	s_cbranch_execz .LBB0_227
; %bb.226:                              ;   in Loop: Header=BB0_2 Depth=1
	v_mov_b64_e32 v[6:7], s[22:23]
	global_store_dwordx4 v[2:3], v[6:9], off offset:8
.LBB0_227:                              ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[26:27]
	s_lshl_b64 s[22:23], s[24:25], 12
	v_lshl_add_u64 v[6:7], v[24:25], 0, s[22:23]
	v_mul_lo_u32 v14, v4, s33
	v_mov_b64_e32 v[26:27], s[10:11]
	v_add_lshl_u32 v20, v14, v16, 1
	v_and_or_b32 v18, v18, s36, 32
	v_mov_b32_e32 v21, v5
	v_readfirstlane_b32 s22, v6
	v_readfirstlane_b32 s23, v7
	v_mov_b64_e32 v[24:25], s[8:9]
	s_nop 3
	global_store_dwordx4 v58, v[18:21], s[22:23]
	global_store_dwordx4 v58, v[24:27], s[22:23] offset:16
	global_store_dwordx4 v58, v[24:27], s[22:23] offset:32
	global_store_dwordx4 v58, v[24:27], s[22:23] offset:48
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_235
; %bb.228:                              ;   in Loop: Header=BB0_2 Depth=1
	global_load_dwordx2 v[26:27], v5, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v5, s[16:17] offset:40
	v_mov_b32_e32 v24, s20
	v_mov_b32_e32 v25, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s24, v14
	v_readfirstlane_b32 s25, v15
	s_and_b64 s[24:25], s[24:25], s[20:21]
	s_mul_i32 s25, s25, 24
	s_mul_hi_u32 s26, s24, 24
	s_mul_i32 s24, s24, 24
	s_add_i32 s25, s26, s25
	v_lshl_add_u64 v[14:15], v[22:23], 0, s[24:25]
	global_store_dwordx2 v[14:15], v[26:27], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[24:25], v5, v[24:27], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[24:25], v[26:27]
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB0_231
; %bb.229:                              ; %.preheader1.i.i.i51.i.preheader
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_mov_b64 s[26:27], 0
.LBB0_230:                              ; %.preheader1.i.i.i51.i
                                        ;   Parent Loop BB0_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[24:25], off
	v_mov_b32_e32 v22, s20
	v_mov_b32_e32 v23, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v5, v[22:25], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[24:25]
	s_or_b64 s[26:27], vcc, s[26:27]
	v_mov_b64_e32 v[24:25], v[18:19]
	s_andn2_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB0_230
.LBB0_231:                              ; %Flow4377
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[24:25]
	global_load_dwordx2 v[14:15], v5, s[16:17] offset:16
	s_mov_b64 s[26:27], exec
	v_mbcnt_lo_u32_b32 v17, s26, 0
	v_mbcnt_hi_u32_b32 v17, s27, v17
	v_cmp_eq_u32_e32 vcc, 0, v17
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB0_233
; %bb.232:                              ;   in Loop: Header=BB0_2 Depth=1
	s_bcnt1_i32_b64 s26, s[26:27]
	v_mov_b32_e32 v18, s26
	v_mov_b32_e32 v19, v5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[18:19], off offset:8 sc1
.LBB0_233:                              ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_235
; %bb.234:                              ;   in Loop: Header=BB0_2 Depth=1
	global_load_dword v14, v[14:15], off offset:24
	v_mov_b32_e32 v15, v5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_store_dwordx2 v[18:19], v[14:15], off sc0 sc1
	v_and_b32_e32 v14, 0xffffff, v14
	s_nop 0
	v_readfirstlane_b32 s24, v14
	s_mov_b32 m0, s24
	s_nop 0
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_235:                              ; %Flow4378
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[22:23]
	v_mov_b32_e32 v59, v5
	v_lshl_add_u64 v[6:7], v[6:7], 0, v[58:59]
	s_branch .LBB0_239
.LBB0_236:                              ;   in Loop: Header=BB0_239 Depth=2
	s_or_b64 exec, exec, s[22:23]
	v_readfirstlane_b32 s22, v14
	s_cmp_eq_u32 s22, 0
	s_cbranch_scc1 .LBB0_238
; %bb.237:                              ;   in Loop: Header=BB0_239 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_239
	s_branch .LBB0_241
.LBB0_238:                              ;   in Loop: Header=BB0_2 Depth=1
	s_branch .LBB0_241
.LBB0_239:                              ;   Parent Loop BB0_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v14, 1
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_236
; %bb.240:                              ;   in Loop: Header=BB0_239 Depth=2
	global_load_dword v14, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v14, 1, v14
	s_branch .LBB0_236
.LBB0_241:                              ;   in Loop: Header=BB0_2 Depth=1
	global_load_dwordx2 v[18:19], v[6:7], off
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_245
; %bb.242:                              ;   in Loop: Header=BB0_2 Depth=1
	global_load_dwordx2 v[2:3], v5, s[16:17] offset:40
	global_load_dwordx2 v[6:7], v5, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v5, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[22:23], v[2:3], 0, 1
	v_lshl_add_u64 v[26:27], v[22:23], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[26:27]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v24, v6
	v_mov_b32_e32 v25, v7
	v_cndmask_b32_e32 v23, v27, v23, vcc
	v_cndmask_b32_e32 v22, v26, v22, vcc
	v_and_b32_e32 v3, v23, v3
	v_and_b32_e32 v2, v22, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v17, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v17, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[6:7], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[24:25], v5, v[22:25], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[24:25], v[6:7]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_245
; %bb.243:                              ; %.preheader.i.i.i50.i.preheader
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_mov_b64 s[0:1], 0
.LBB0_244:                              ; %.preheader.i.i.i50.i
                                        ;   Parent Loop BB0_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[24:25], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v5, v[22:25], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[6:7], v[24:25]
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b64_e32 v[24:25], v[6:7]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_244
.LBB0_245:                              ; %__ockl_printf_append_args.exit54.i
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[22:23]
	v_readfirstlane_b32 s0, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v56
	s_and_saveexec_b64 s[20:21], s[0:1]
	s_cbranch_execz .LBB0_251
; %bb.246:                              ;   in Loop: Header=BB0_2 Depth=1
	global_load_dwordx2 v[24:25], v5, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v5, s[16:17] offset:40
	global_load_dwordx2 v[6:7], v5, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v24
	v_and_b32_e32 v3, v3, v25
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v14, v2, 24
	v_add_u32_e32 v3, v14, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[6:7], 0, v[2:3]
	global_load_dwordx2 v[22:23], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v5, v[22:25], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[24:25]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_250
; %bb.247:                              ; %.preheader3.i.i.i61.i.preheader
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_248:                              ; %.preheader3.i.i.i61.i
                                        ;   Parent Loop BB0_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[6:7], v5, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v5, s[16:17]
	v_mov_b64_e32 v[24:25], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v6, v24
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[26:27], v2, 24, v[14:15]
	v_and_b32_e32 v7, v7, v25
	v_mov_b32_e32 v6, v3
	v_mad_u64_u32 v[6:7], s[26:27], v7, 24, v[6:7]
	v_mov_b32_e32 v3, v6
	global_load_dwordx2 v[22:23], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v5, v[22:25], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[24:25]
	s_or_b64 s[24:25], vcc, s[24:25]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_248
; %bb.249:                              ; %Flow4365
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[24:25]
.LBB0_250:                              ; %Flow4367
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_251:                              ; %.loopexit4.i.i.i55.i
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[20:21]
	global_load_dwordx2 v[6:7], v5, s[16:17] offset:40
	global_load_dwordx4 v[22:25], v5, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[22:23], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s24, v6
	v_readfirstlane_b32 s25, v7
	s_and_b64 s[24:25], s[20:21], s[24:25]
	s_mul_i32 s26, s25, 24
	s_mul_hi_u32 s27, s24, 24
	s_add_i32 s27, s27, s26
	s_mul_i32 s26, s24, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[22:23], 0, s[26:27]
	s_and_saveexec_b64 s[26:27], s[0:1]
	s_cbranch_execz .LBB0_253
; %bb.252:                              ;   in Loop: Header=BB0_2 Depth=1
	v_mov_b64_e32 v[6:7], s[22:23]
	global_store_dwordx4 v[2:3], v[6:9], off offset:8
.LBB0_253:                              ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[26:27]
	v_ashrrev_i32_e32 v14, 31, v20
	v_lshrrev_b32_e32 v14, 22, v14
	v_add_u32_e32 v14, v20, v14
	v_and_b32_e32 v14, 0xfffffc00, v14
	s_lshl_b64 s[22:23], s[24:25], 12
	v_sub_u32_e32 v14, v20, v14
	v_lshl_add_u64 v[6:7], v[24:25], 0, s[22:23]
	v_ashrrev_i32_e32 v14, 3, v14
	v_mov_b64_e32 v[26:27], s[10:11]
	v_bitop3_b32 v20, v14, v20, -16 bitop3:0x6c
	v_and_or_b32 v18, v18, s37, 34
	v_mov_b32_e32 v21, v5
	v_readfirstlane_b32 s22, v6
	v_readfirstlane_b32 s23, v7
	v_mov_b64_e32 v[24:25], s[8:9]
	s_nop 3
	global_store_dwordx4 v58, v[18:21], s[22:23]
	global_store_dwordx4 v58, v[24:27], s[22:23] offset:16
	global_store_dwordx4 v58, v[24:27], s[22:23] offset:32
	global_store_dwordx4 v58, v[24:27], s[22:23] offset:48
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_261
; %bb.254:                              ;   in Loop: Header=BB0_2 Depth=1
	global_load_dwordx2 v[26:27], v5, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[6:7], v5, s[16:17] offset:40
	v_mov_b32_e32 v24, s20
	v_mov_b32_e32 v25, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s24, v6
	v_readfirstlane_b32 s25, v7
	s_and_b64 s[24:25], s[24:25], s[20:21]
	s_mul_i32 s25, s25, 24
	s_mul_hi_u32 s26, s24, 24
	s_mul_i32 s24, s24, 24
	s_add_i32 s25, s26, s25
	v_lshl_add_u64 v[6:7], v[22:23], 0, s[24:25]
	global_store_dwordx2 v[6:7], v[26:27], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[24:25], v5, v[24:27], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[24:25], v[26:27]
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB0_257
; %bb.255:                              ; %.preheader1.i.i.i59.i.preheader
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_mov_b64 s[26:27], 0
.LBB0_256:                              ; %.preheader1.i.i.i59.i
                                        ;   Parent Loop BB0_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[6:7], v[24:25], off
	v_mov_b32_e32 v22, s20
	v_mov_b32_e32 v23, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[14:15], v5, v[22:25], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[14:15], v[24:25]
	s_or_b64 s[26:27], vcc, s[26:27]
	v_mov_b64_e32 v[24:25], v[14:15]
	s_andn2_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB0_256
.LBB0_257:                              ; %Flow4363
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[24:25]
	global_load_dwordx2 v[6:7], v5, s[16:17] offset:16
	s_mov_b64 s[26:27], exec
	v_mbcnt_lo_u32_b32 v14, s26, 0
	v_mbcnt_hi_u32_b32 v14, s27, v14
	v_cmp_eq_u32_e32 vcc, 0, v14
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB0_259
; %bb.258:                              ;   in Loop: Header=BB0_2 Depth=1
	s_bcnt1_i32_b64 s26, s[26:27]
	v_mov_b32_e32 v14, s26
	v_mov_b32_e32 v15, v5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[6:7], v[14:15], off offset:8 sc1
.LBB0_259:                              ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[14:15], v[6:7], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[14:15]
	s_cbranch_vccnz .LBB0_261
; %bb.260:                              ;   in Loop: Header=BB0_2 Depth=1
	global_load_dword v6, v[6:7], off offset:24
	v_mov_b32_e32 v7, v5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_store_dwordx2 v[14:15], v[6:7], off sc0 sc1
	v_and_b32_e32 v6, 0xffffff, v6
	s_nop 0
	v_readfirstlane_b32 s24, v6
	s_mov_b32 m0, s24
	s_nop 0
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_261:                              ; %Flow4364
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_branch .LBB0_265
.LBB0_262:                              ;   in Loop: Header=BB0_265 Depth=2
	s_or_b64 exec, exec, s[22:23]
	v_readfirstlane_b32 s22, v6
	s_cmp_eq_u32 s22, 0
	s_cbranch_scc1 .LBB0_264
; %bb.263:                              ;   in Loop: Header=BB0_265 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_265
	s_branch .LBB0_267
.LBB0_264:                              ;   in Loop: Header=BB0_2 Depth=1
	s_branch .LBB0_267
.LBB0_265:                              ;   Parent Loop BB0_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v6, 1
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_262
; %bb.266:                              ;   in Loop: Header=BB0_265 Depth=2
	global_load_dword v6, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v6, 1, v6
	s_branch .LBB0_262
.LBB0_267:                              ;   in Loop: Header=BB0_2 Depth=1
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_1
; %bb.268:                              ;   in Loop: Header=BB0_2 Depth=1
	global_load_dwordx2 v[2:3], v5, s[16:17] offset:40
	global_load_dwordx2 v[6:7], v5, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v5, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[18:19], v[2:3], 0, 1
	v_lshl_add_u64 v[22:23], v[18:19], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v24, v6
	v_mov_b32_e32 v25, v7
	v_cndmask_b32_e32 v23, v23, v19, vcc
	v_cndmask_b32_e32 v22, v22, v18, vcc
	v_and_b32_e32 v3, v23, v3
	v_and_b32_e32 v2, v22, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v17, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v17, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[6:7], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[24:25], v5, v[22:25], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[24:25], v[6:7]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_1
; %bb.269:                              ; %.preheader.i.i.i58.i.preheader
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_mov_b64 s[0:1], 0
.LBB0_270:                              ; %.preheader.i.i.i58.i
                                        ;   Parent Loop BB0_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[24:25], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v5, v[22:25], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[6:7], v[24:25]
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b64_e32 v[24:25], v[6:7]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_270
	s_branch .LBB0_1
.LBB0_271:                              ; %_Z28load_global_to_shared_directILi2ELb0ETkN7kittens5ducks2st3allENS0_2stI14__hip_bfloat16Li64ELi64EEETkNS1_2gl3allENS0_2glIS4_Lin1ELin1ELin1ELin1EJEEETkNS1_5coord4tileENS0_5coordIS5_EELi64ELi1EEvRKT2_RKT3_RT1_.exit
	s_or_b64 exec, exec, s[2:3]
	s_waitcnt lgkmcnt(0)
	; wave barrier
	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	; wave barrier
	; sched_barrier mask(0x00000000)
	s_getpc_b64 s[2:3]
	s_add_u32 s2, s2, .str.3@rel32@lo+4
	s_addc_u32 s3, s3, .str.3@rel32@hi+12
	v_lshrrev_b32_e32 v2, 1, v0
	s_cmp_lg_u64 s[2:3], 0
	v_and_b32_e32 v4, 24, v2
	s_cselect_b64 s[8:9], -1, 0
	s_getpc_b64 s[2:3]
	s_add_u32 s2, s2, .str.2@rel32@lo+4
	s_addc_u32 s3, s3, .str.2@rel32@hi+12
	s_cmp_lg_u64 s[2:3], 0
	v_mov_b32_e32 v9, 0
	v_or_b32_e32 v8, 32, v4
	v_and_b32_e32 v57, 15, v0
	v_cmp_gt_u32_e64 s[0:1], 3, v0
	s_cselect_b64 s[10:11], -1, 0
	v_lshlrev_b32_e32 v60, 1, v4
	s_mov_b32 s4, 0
	v_mov_b32_e32 v5, v9
	v_lshlrev_b32_e32 v61, 1, v8
	v_mov_b64_e32 v[12:13], v[8:9]
	s_movk_i32 s33, 0xff1f
	v_mov_b32_e32 v16, 0x80
	v_mov_b32_e32 v20, 64
	s_movk_i32 s34, 0xff1d
	s_movk_i32 s35, 0x70
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v24, 2
	v_mov_b32_e32 v25, 1
	v_mov_b32_e32 v6, 33
	s_mov_b32 s36, 0
	s_waitcnt lgkmcnt(0)
	; wave barrier
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_273
.LBB0_272:                              ; %Flow3323
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[18:19]
	s_add_i32 s36, s36, 2
	s_cmp_lg_u32 s36, 4
	;;#ASMSTART
	ds_read_b128 v[26:29], v30
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	scratch_store_dwordx4 v63, v[26:29], off offset:48
	s_cbranch_scc0 .LBB0_2329
.LBB0_273:                              ; %for.body.i85
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_277 Depth 2
                                        ;     Child Loop BB0_285 Depth 2
                                        ;     Child Loop BB0_294 Depth 2
                                        ;     Child Loop BB0_299 Depth 2
                                        ;     Child Loop BB0_389 Depth 2
                                        ;     Child Loop BB0_397 Depth 2
                                        ;     Child Loop BB0_406 Depth 2
                                        ;     Child Loop BB0_411 Depth 2
                                        ;     Child Loop BB0_303 Depth 2
                                        ;       Child Loop BB0_306 Depth 3
                                        ;       Child Loop BB0_313 Depth 3
                                        ;       Child Loop BB0_321 Depth 3
                                        ;       Child Loop BB0_329 Depth 3
                                        ;       Child Loop BB0_337 Depth 3
                                        ;       Child Loop BB0_345 Depth 3
                                        ;       Child Loop BB0_353 Depth 3
                                        ;       Child Loop BB0_361 Depth 3
                                        ;       Child Loop BB0_369 Depth 3
                                        ;       Child Loop BB0_378 Depth 3
                                        ;       Child Loop BB0_383 Depth 3
                                        ;     Child Loop BB0_416 Depth 2
                                        ;     Child Loop BB0_424 Depth 2
                                        ;     Child Loop BB0_433 Depth 2
                                        ;     Child Loop BB0_438 Depth 2
                                        ;     Child Loop BB0_442 Depth 2
                                        ;     Child Loop BB0_450 Depth 2
                                        ;     Child Loop BB0_459 Depth 2
                                        ;     Child Loop BB0_464 Depth 2
                                        ;     Child Loop BB0_468 Depth 2
                                        ;     Child Loop BB0_476 Depth 2
                                        ;     Child Loop BB0_485 Depth 2
                                        ;     Child Loop BB0_490 Depth 2
                                        ;     Child Loop BB0_494 Depth 2
                                        ;     Child Loop BB0_502 Depth 2
                                        ;     Child Loop BB0_511 Depth 2
                                        ;     Child Loop BB0_516 Depth 2
                                        ;     Child Loop BB0_520 Depth 2
                                        ;     Child Loop BB0_528 Depth 2
                                        ;     Child Loop BB0_537 Depth 2
                                        ;     Child Loop BB0_542 Depth 2
                                        ;     Child Loop BB0_547 Depth 2
                                        ;     Child Loop BB0_555 Depth 2
                                        ;     Child Loop BB0_564 Depth 2
                                        ;     Child Loop BB0_569 Depth 2
                                        ;     Child Loop BB0_659 Depth 2
                                        ;     Child Loop BB0_667 Depth 2
                                        ;     Child Loop BB0_676 Depth 2
                                        ;     Child Loop BB0_681 Depth 2
                                        ;     Child Loop BB0_573 Depth 2
                                        ;       Child Loop BB0_576 Depth 3
                                        ;       Child Loop BB0_583 Depth 3
                                        ;       Child Loop BB0_591 Depth 3
                                        ;       Child Loop BB0_599 Depth 3
                                        ;       Child Loop BB0_607 Depth 3
                                        ;       Child Loop BB0_615 Depth 3
                                        ;       Child Loop BB0_623 Depth 3
                                        ;       Child Loop BB0_631 Depth 3
                                        ;       Child Loop BB0_639 Depth 3
                                        ;       Child Loop BB0_648 Depth 3
                                        ;       Child Loop BB0_653 Depth 3
                                        ;     Child Loop BB0_686 Depth 2
                                        ;     Child Loop BB0_694 Depth 2
                                        ;     Child Loop BB0_703 Depth 2
                                        ;     Child Loop BB0_708 Depth 2
                                        ;     Child Loop BB0_712 Depth 2
                                        ;     Child Loop BB0_720 Depth 2
                                        ;     Child Loop BB0_729 Depth 2
                                        ;     Child Loop BB0_734 Depth 2
                                        ;     Child Loop BB0_738 Depth 2
                                        ;     Child Loop BB0_746 Depth 2
                                        ;     Child Loop BB0_755 Depth 2
                                        ;     Child Loop BB0_760 Depth 2
                                        ;     Child Loop BB0_764 Depth 2
                                        ;     Child Loop BB0_772 Depth 2
                                        ;     Child Loop BB0_781 Depth 2
                                        ;     Child Loop BB0_786 Depth 2
                                        ;     Child Loop BB0_791 Depth 2
                                        ;     Child Loop BB0_799 Depth 2
                                        ;     Child Loop BB0_808 Depth 2
                                        ;     Child Loop BB0_813 Depth 2
                                        ;     Child Loop BB0_903 Depth 2
                                        ;     Child Loop BB0_911 Depth 2
                                        ;     Child Loop BB0_920 Depth 2
                                        ;     Child Loop BB0_925 Depth 2
                                        ;     Child Loop BB0_817 Depth 2
                                        ;       Child Loop BB0_820 Depth 3
                                        ;       Child Loop BB0_827 Depth 3
                                        ;       Child Loop BB0_835 Depth 3
                                        ;       Child Loop BB0_843 Depth 3
                                        ;       Child Loop BB0_851 Depth 3
                                        ;       Child Loop BB0_859 Depth 3
                                        ;       Child Loop BB0_867 Depth 3
                                        ;       Child Loop BB0_875 Depth 3
                                        ;       Child Loop BB0_883 Depth 3
                                        ;       Child Loop BB0_892 Depth 3
                                        ;       Child Loop BB0_897 Depth 3
                                        ;     Child Loop BB0_930 Depth 2
                                        ;     Child Loop BB0_938 Depth 2
                                        ;     Child Loop BB0_947 Depth 2
                                        ;     Child Loop BB0_952 Depth 2
                                        ;     Child Loop BB0_956 Depth 2
                                        ;     Child Loop BB0_964 Depth 2
                                        ;     Child Loop BB0_973 Depth 2
                                        ;     Child Loop BB0_978 Depth 2
                                        ;     Child Loop BB0_982 Depth 2
                                        ;     Child Loop BB0_990 Depth 2
                                        ;     Child Loop BB0_999 Depth 2
                                        ;     Child Loop BB0_1004 Depth 2
                                        ;     Child Loop BB0_1008 Depth 2
                                        ;     Child Loop BB0_1016 Depth 2
                                        ;     Child Loop BB0_1025 Depth 2
                                        ;     Child Loop BB0_1030 Depth 2
                                        ;     Child Loop BB0_1034 Depth 2
                                        ;     Child Loop BB0_1042 Depth 2
                                        ;     Child Loop BB0_1051 Depth 2
                                        ;     Child Loop BB0_1056 Depth 2
                                        ;     Child Loop BB0_1061 Depth 2
                                        ;     Child Loop BB0_1069 Depth 2
                                        ;     Child Loop BB0_1078 Depth 2
                                        ;     Child Loop BB0_1083 Depth 2
                                        ;     Child Loop BB0_1173 Depth 2
                                        ;     Child Loop BB0_1181 Depth 2
                                        ;     Child Loop BB0_1190 Depth 2
                                        ;     Child Loop BB0_1195 Depth 2
                                        ;     Child Loop BB0_1087 Depth 2
                                        ;       Child Loop BB0_1090 Depth 3
                                        ;       Child Loop BB0_1097 Depth 3
                                        ;       Child Loop BB0_1105 Depth 3
                                        ;       Child Loop BB0_1113 Depth 3
                                        ;       Child Loop BB0_1121 Depth 3
                                        ;       Child Loop BB0_1129 Depth 3
                                        ;       Child Loop BB0_1137 Depth 3
                                        ;       Child Loop BB0_1145 Depth 3
                                        ;       Child Loop BB0_1153 Depth 3
                                        ;       Child Loop BB0_1162 Depth 3
                                        ;       Child Loop BB0_1167 Depth 3
                                        ;     Child Loop BB0_1200 Depth 2
                                        ;     Child Loop BB0_1208 Depth 2
                                        ;     Child Loop BB0_1217 Depth 2
                                        ;     Child Loop BB0_1222 Depth 2
                                        ;     Child Loop BB0_1226 Depth 2
                                        ;     Child Loop BB0_1234 Depth 2
                                        ;     Child Loop BB0_1243 Depth 2
                                        ;     Child Loop BB0_1248 Depth 2
                                        ;     Child Loop BB0_1252 Depth 2
                                        ;     Child Loop BB0_1260 Depth 2
                                        ;     Child Loop BB0_1269 Depth 2
                                        ;     Child Loop BB0_1274 Depth 2
                                        ;     Child Loop BB0_1278 Depth 2
                                        ;     Child Loop BB0_1286 Depth 2
                                        ;     Child Loop BB0_1295 Depth 2
                                        ;     Child Loop BB0_1300 Depth 2
                                        ;     Child Loop BB0_1305 Depth 2
                                        ;     Child Loop BB0_1313 Depth 2
                                        ;     Child Loop BB0_1322 Depth 2
                                        ;     Child Loop BB0_1327 Depth 2
                                        ;     Child Loop BB0_1417 Depth 2
                                        ;     Child Loop BB0_1425 Depth 2
                                        ;     Child Loop BB0_1434 Depth 2
                                        ;     Child Loop BB0_1439 Depth 2
                                        ;     Child Loop BB0_1331 Depth 2
                                        ;       Child Loop BB0_1334 Depth 3
                                        ;       Child Loop BB0_1341 Depth 3
                                        ;       Child Loop BB0_1349 Depth 3
                                        ;       Child Loop BB0_1357 Depth 3
                                        ;       Child Loop BB0_1365 Depth 3
                                        ;       Child Loop BB0_1373 Depth 3
                                        ;       Child Loop BB0_1381 Depth 3
                                        ;       Child Loop BB0_1389 Depth 3
                                        ;       Child Loop BB0_1397 Depth 3
                                        ;       Child Loop BB0_1406 Depth 3
                                        ;       Child Loop BB0_1411 Depth 3
                                        ;     Child Loop BB0_1444 Depth 2
                                        ;     Child Loop BB0_1452 Depth 2
                                        ;     Child Loop BB0_1461 Depth 2
                                        ;     Child Loop BB0_1466 Depth 2
                                        ;     Child Loop BB0_1470 Depth 2
                                        ;     Child Loop BB0_1478 Depth 2
                                        ;     Child Loop BB0_1487 Depth 2
                                        ;     Child Loop BB0_1492 Depth 2
                                        ;     Child Loop BB0_1496 Depth 2
                                        ;     Child Loop BB0_1504 Depth 2
                                        ;     Child Loop BB0_1513 Depth 2
                                        ;     Child Loop BB0_1518 Depth 2
                                        ;     Child Loop BB0_1522 Depth 2
                                        ;     Child Loop BB0_1530 Depth 2
                                        ;     Child Loop BB0_1539 Depth 2
                                        ;     Child Loop BB0_1544 Depth 2
                                        ;     Child Loop BB0_1548 Depth 2
                                        ;     Child Loop BB0_1556 Depth 2
                                        ;     Child Loop BB0_1565 Depth 2
                                        ;     Child Loop BB0_1570 Depth 2
                                        ;     Child Loop BB0_1575 Depth 2
                                        ;     Child Loop BB0_1583 Depth 2
                                        ;     Child Loop BB0_1592 Depth 2
                                        ;     Child Loop BB0_1597 Depth 2
                                        ;     Child Loop BB0_1687 Depth 2
                                        ;     Child Loop BB0_1695 Depth 2
                                        ;     Child Loop BB0_1704 Depth 2
                                        ;     Child Loop BB0_1709 Depth 2
                                        ;     Child Loop BB0_1601 Depth 2
                                        ;       Child Loop BB0_1604 Depth 3
                                        ;       Child Loop BB0_1611 Depth 3
                                        ;       Child Loop BB0_1619 Depth 3
                                        ;       Child Loop BB0_1627 Depth 3
                                        ;       Child Loop BB0_1635 Depth 3
                                        ;       Child Loop BB0_1643 Depth 3
                                        ;       Child Loop BB0_1651 Depth 3
                                        ;       Child Loop BB0_1659 Depth 3
                                        ;       Child Loop BB0_1667 Depth 3
                                        ;       Child Loop BB0_1676 Depth 3
                                        ;       Child Loop BB0_1681 Depth 3
                                        ;     Child Loop BB0_1714 Depth 2
                                        ;     Child Loop BB0_1722 Depth 2
                                        ;     Child Loop BB0_1731 Depth 2
                                        ;     Child Loop BB0_1736 Depth 2
                                        ;     Child Loop BB0_1740 Depth 2
                                        ;     Child Loop BB0_1748 Depth 2
                                        ;     Child Loop BB0_1757 Depth 2
                                        ;     Child Loop BB0_1762 Depth 2
                                        ;     Child Loop BB0_1766 Depth 2
                                        ;     Child Loop BB0_1774 Depth 2
                                        ;     Child Loop BB0_1783 Depth 2
                                        ;     Child Loop BB0_1788 Depth 2
                                        ;     Child Loop BB0_1792 Depth 2
                                        ;     Child Loop BB0_1800 Depth 2
                                        ;     Child Loop BB0_1809 Depth 2
                                        ;     Child Loop BB0_1814 Depth 2
                                        ;     Child Loop BB0_1819 Depth 2
                                        ;     Child Loop BB0_1827 Depth 2
                                        ;     Child Loop BB0_1836 Depth 2
                                        ;     Child Loop BB0_1841 Depth 2
                                        ;     Child Loop BB0_1931 Depth 2
                                        ;     Child Loop BB0_1939 Depth 2
                                        ;     Child Loop BB0_1948 Depth 2
                                        ;     Child Loop BB0_1953 Depth 2
                                        ;     Child Loop BB0_1845 Depth 2
                                        ;       Child Loop BB0_1848 Depth 3
                                        ;       Child Loop BB0_1855 Depth 3
                                        ;       Child Loop BB0_1863 Depth 3
                                        ;       Child Loop BB0_1871 Depth 3
                                        ;       Child Loop BB0_1879 Depth 3
                                        ;       Child Loop BB0_1887 Depth 3
                                        ;       Child Loop BB0_1895 Depth 3
                                        ;       Child Loop BB0_1903 Depth 3
                                        ;       Child Loop BB0_1911 Depth 3
                                        ;       Child Loop BB0_1920 Depth 3
                                        ;       Child Loop BB0_1925 Depth 3
                                        ;     Child Loop BB0_1958 Depth 2
                                        ;     Child Loop BB0_1966 Depth 2
                                        ;     Child Loop BB0_1975 Depth 2
                                        ;     Child Loop BB0_1980 Depth 2
                                        ;     Child Loop BB0_1984 Depth 2
                                        ;     Child Loop BB0_1992 Depth 2
                                        ;     Child Loop BB0_2001 Depth 2
                                        ;     Child Loop BB0_2006 Depth 2
                                        ;     Child Loop BB0_2010 Depth 2
                                        ;     Child Loop BB0_2018 Depth 2
                                        ;     Child Loop BB0_2027 Depth 2
                                        ;     Child Loop BB0_2032 Depth 2
                                        ;     Child Loop BB0_2036 Depth 2
                                        ;     Child Loop BB0_2044 Depth 2
                                        ;     Child Loop BB0_2053 Depth 2
                                        ;     Child Loop BB0_2058 Depth 2
                                        ;     Child Loop BB0_2062 Depth 2
                                        ;     Child Loop BB0_2070 Depth 2
                                        ;     Child Loop BB0_2079 Depth 2
                                        ;     Child Loop BB0_2084 Depth 2
                                        ;     Child Loop BB0_2089 Depth 2
                                        ;     Child Loop BB0_2097 Depth 2
                                        ;     Child Loop BB0_2106 Depth 2
                                        ;     Child Loop BB0_2111 Depth 2
                                        ;     Child Loop BB0_2201 Depth 2
                                        ;     Child Loop BB0_2209 Depth 2
                                        ;     Child Loop BB0_2218 Depth 2
                                        ;     Child Loop BB0_2223 Depth 2
                                        ;     Child Loop BB0_2115 Depth 2
                                        ;       Child Loop BB0_2118 Depth 3
                                        ;       Child Loop BB0_2125 Depth 3
                                        ;       Child Loop BB0_2133 Depth 3
                                        ;       Child Loop BB0_2141 Depth 3
                                        ;       Child Loop BB0_2149 Depth 3
                                        ;       Child Loop BB0_2157 Depth 3
                                        ;       Child Loop BB0_2165 Depth 3
                                        ;       Child Loop BB0_2173 Depth 3
                                        ;       Child Loop BB0_2181 Depth 3
                                        ;       Child Loop BB0_2190 Depth 3
                                        ;       Child Loop BB0_2195 Depth 3
                                        ;     Child Loop BB0_2228 Depth 2
                                        ;     Child Loop BB0_2236 Depth 2
                                        ;     Child Loop BB0_2245 Depth 2
                                        ;     Child Loop BB0_2250 Depth 2
                                        ;     Child Loop BB0_2254 Depth 2
                                        ;     Child Loop BB0_2262 Depth 2
                                        ;     Child Loop BB0_2271 Depth 2
                                        ;     Child Loop BB0_2276 Depth 2
                                        ;     Child Loop BB0_2280 Depth 2
                                        ;     Child Loop BB0_2288 Depth 2
                                        ;     Child Loop BB0_2297 Depth 2
                                        ;     Child Loop BB0_2302 Depth 2
                                        ;     Child Loop BB0_2306 Depth 2
                                        ;     Child Loop BB0_2314 Depth 2
                                        ;     Child Loop BB0_2323 Depth 2
                                        ;     Child Loop BB0_2328 Depth 2
	s_nop 0
	v_lshl_or_b32 v28, s36, 4, v57
	v_lshl_add_u32 v63, v28, 7, s15
	v_add_u32_e32 v26, v63, v60
	s_and_saveexec_b64 s[18:19], s[0:1]
	s_cbranch_execz .LBB0_543
; %bb.274:                              ; %if.then.i.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_280
; %bb.275:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[32:33], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v32
	v_and_b32_e32 v3, v3, v33
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[32:33]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_279
; %bb.276:                              ; %.preheader3.i.i.i.i.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_277:                              ; %.preheader3.i.i.i.i.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[32:33], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v32
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v33
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[32:33]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_277
; %bb.278:                              ; %Flow4350
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_279:                              ; %Flow4352
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_280:                              ; %.loopexit4.i.i.i.i.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[30:33], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[30:31], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_282
; %bb.281:                              ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_282:                              ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[32:33], 0, s[6:7]
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	v_mov_b64_e32 v[34:35], s[6:7]
	v_mov_b32_e32 v7, v9
	v_mov_b32_e32 v8, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	v_mov_b64_e32 v[32:33], s[4:5]
	s_nop 3
	global_store_dwordx4 v58, v[6:9], s[22:23]
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:16
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:32
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_290
; %bb.283:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[34:35], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v32, s20
	v_mov_b32_e32 v33, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[30:31], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[34:35], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[32:35], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[34:35]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_286
; %bb.284:                              ; %.preheader1.i.i.i.i.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_285:                              ; %.preheader1.i.i.i.i.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[32:33], off
	v_mov_b32_e32 v30, s20
	v_mov_b32_e32 v31, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[30:33], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[32:33]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[32:33], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_285
.LBB0_286:                              ; %Flow4348
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_288
; %bb.287:                              ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_288:                              ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_290
; %bb.289:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_290:                              ; %Flow4349
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_294
.LBB0_291:                              ;   in Loop: Header=BB0_294 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_293
; %bb.292:                              ;   in Loop: Header=BB0_294 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_294
	s_branch .LBB0_296
.LBB0_293:                              ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_296
.LBB0_294:                              ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_291
; %bb.295:                              ;   in Loop: Header=BB0_294 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_291
.LBB0_296:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[30:31], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_300
; %bb.297:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[18:19], v[2:3], 0, 1
	v_lshl_add_u64 v[22:23], v[18:19], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v34, v10
	v_mov_b32_e32 v35, v11
	v_cndmask_b32_e32 v33, v23, v19, vcc
	v_cndmask_b32_e32 v32, v22, v18, vcc
	v_and_b32_e32 v3, v33, v3
	v_and_b32_e32 v2, v32, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[34:35], v9, v[32:35], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[34:35], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_300
; %bb.298:                              ; %.preheader.i.i.i.i.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_299:                              ; %.preheader.i.i.i.i.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[34:35], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[32:35], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[34:35]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[34:35], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_299
.LBB0_300:                              ; %__ockl_printf_begin.exit.i.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	s_and_b64 vcc, exec, s[8:9]
	s_cbranch_vccz .LBB0_385
; %bb.301:                              ;   in Loop: Header=BB0_273 Depth=1
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v2, 2, v30
	v_and_b32_e32 v32, -3, v30
	v_mov_b32_e32 v33, v31
	s_mov_b64 s[20:21], 0x43
	s_getpc_b64 s[6:7]
	s_add_u32 s6, s6, .str.3@rel32@lo+4
	s_addc_u32 s7, s7, .str.3@rel32@hi+12
	s_branch .LBB0_303
.LBB0_302:                              ; %__ockl_hostcall_preview.exit19.i.i.i
                                        ;   in Loop: Header=BB0_303 Depth=2
	s_or_b64 exec, exec, s[26:27]
	s_sub_u32 s20, s20, s22
	s_subb_u32 s21, s21, s23
	s_add_u32 s6, s6, s22
	s_addc_u32 s7, s7, s23
	s_cmp_lg_u64 s[20:21], 0
	s_cbranch_scc0 .LBB0_384
.LBB0_303:                              ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_306 Depth 3
                                        ;       Child Loop BB0_313 Depth 3
                                        ;       Child Loop BB0_321 Depth 3
                                        ;       Child Loop BB0_329 Depth 3
                                        ;       Child Loop BB0_337 Depth 3
                                        ;       Child Loop BB0_345 Depth 3
                                        ;       Child Loop BB0_353 Depth 3
                                        ;       Child Loop BB0_361 Depth 3
                                        ;       Child Loop BB0_369 Depth 3
                                        ;       Child Loop BB0_378 Depth 3
                                        ;       Child Loop BB0_383 Depth 3
	v_cmp_lt_u64_e64 s[2:3], s[20:21], 56
	s_and_b64 s[2:3], s[2:3], exec
	v_cmp_gt_u64_e64 s[2:3], s[20:21], 7
	s_cselect_b32 s23, s21, 0
	s_cselect_b32 s22, s20, 56
	s_and_b64 vcc, exec, s[2:3]
	s_cbranch_vccnz .LBB0_308
; %bb.304:                              ;   in Loop: Header=BB0_303 Depth=2
	s_mov_b64 s[2:3], 0
	s_cmp_eq_u64 s[20:21], 0
	s_waitcnt vmcnt(0)
	v_mov_b64_e32 v[34:35], 0
	s_cbranch_scc1 .LBB0_307
; %bb.305:                              ; %.preheader30.i.i.i.preheader
                                        ;   in Loop: Header=BB0_303 Depth=2
	s_lshl_b64 s[24:25], s[22:23], 3
	s_mov_b64 s[26:27], 0
	v_mov_b64_e32 v[34:35], 0
	s_mov_b64 s[28:29], s[6:7]
.LBB0_306:                              ; %.preheader30.i.i.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_303 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	global_load_ubyte v3, v9, s[28:29]
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s26, v[8:9]
	s_add_u32 s26, s26, 8
	s_addc_u32 s27, s27, 0
	s_add_u32 s28, s28, 1
	s_addc_u32 s29, s29, 0
	v_or_b32_e32 v34, v10, v34
	s_cmp_lg_u32 s24, s26
	v_or_b32_e32 v35, v11, v35
	s_cbranch_scc1 .LBB0_306
.LBB0_307:                              ; %Flow4318
                                        ;   in Loop: Header=BB0_303 Depth=2
	s_mov_b32 s5, 0
	s_andn2_b64 vcc, exec, s[2:3]
	s_mov_b64 s[2:3], s[6:7]
	s_cbranch_vccz .LBB0_309
	s_branch .LBB0_310
.LBB0_308:                              ;   in Loop: Header=BB0_303 Depth=2
                                        ; implicit-def: $vgpr34_vgpr35
                                        ; implicit-def: $sgpr5
	s_mov_b64 s[2:3], s[6:7]
.LBB0_309:                              ;   in Loop: Header=BB0_303 Depth=2
	global_load_dwordx2 v[34:35], v9, s[6:7]
	s_add_i32 s5, s22, -8
	s_add_u32 s2, s6, 8
	s_addc_u32 s3, s7, 0
.LBB0_310:                              ; %.loopexit31.i.i.i
                                        ;   in Loop: Header=BB0_303 Depth=2
	s_cmp_gt_u32 s5, 7
	s_cbranch_scc1 .LBB0_314
; %bb.311:                              ;   in Loop: Header=BB0_303 Depth=2
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_315
; %bb.312:                              ; %.preheader28.i.i.i.preheader
                                        ;   in Loop: Header=BB0_303 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[36:37], 0
	s_mov_b64 s[26:27], 0
.LBB0_313:                              ; %.preheader28.i.i.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_303 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s28, s2, s26
	s_addc_u32 s29, s3, s27
	global_load_ubyte v3, v9, s[28:29]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v36, v10, v36
	s_cmp_lg_u32 s5, s26
	v_or_b32_e32 v37, v11, v37
	s_cbranch_scc1 .LBB0_313
	s_branch .LBB0_316
.LBB0_314:                              ;   in Loop: Header=BB0_303 Depth=2
                                        ; implicit-def: $vgpr36_vgpr37
                                        ; implicit-def: $sgpr28
	s_branch .LBB0_317
.LBB0_315:                              ;   in Loop: Header=BB0_303 Depth=2
	v_mov_b64_e32 v[36:37], 0
.LBB0_316:                              ; %Flow4313
                                        ;   in Loop: Header=BB0_303 Depth=2
	s_mov_b32 s28, 0
	s_cbranch_execnz .LBB0_318
.LBB0_317:                              ;   in Loop: Header=BB0_303 Depth=2
	global_load_dwordx2 v[36:37], v9, s[2:3]
	s_add_i32 s28, s5, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_318:                              ; %.loopexit29.i.i.i
                                        ;   in Loop: Header=BB0_303 Depth=2
	s_cmp_gt_u32 s28, 7
	s_cbranch_scc1 .LBB0_322
; %bb.319:                              ;   in Loop: Header=BB0_303 Depth=2
	s_cmp_eq_u32 s28, 0
	s_cbranch_scc1 .LBB0_323
; %bb.320:                              ; %.preheader26.i.i.i.preheader
                                        ;   in Loop: Header=BB0_303 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[38:39], 0
	s_mov_b64 s[26:27], 0
.LBB0_321:                              ; %.preheader26.i.i.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_303 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s30, s2, s26
	s_addc_u32 s31, s3, s27
	global_load_ubyte v3, v9, s[30:31]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v38, v10, v38
	s_cmp_lg_u32 s28, s26
	v_or_b32_e32 v39, v11, v39
	s_cbranch_scc1 .LBB0_321
	s_branch .LBB0_324
.LBB0_322:                              ;   in Loop: Header=BB0_303 Depth=2
                                        ; implicit-def: $sgpr5
	s_branch .LBB0_325
.LBB0_323:                              ;   in Loop: Header=BB0_303 Depth=2
	v_mov_b64_e32 v[38:39], 0
.LBB0_324:                              ; %Flow4308
                                        ;   in Loop: Header=BB0_303 Depth=2
	s_mov_b32 s5, 0
	s_cbranch_execnz .LBB0_326
.LBB0_325:                              ;   in Loop: Header=BB0_303 Depth=2
	global_load_dwordx2 v[38:39], v9, s[2:3]
	s_add_i32 s5, s28, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_326:                              ; %.loopexit27.i.i.i
                                        ;   in Loop: Header=BB0_303 Depth=2
	s_cmp_gt_u32 s5, 7
	s_cbranch_scc1 .LBB0_330
; %bb.327:                              ;   in Loop: Header=BB0_303 Depth=2
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_331
; %bb.328:                              ; %.preheader24.i.i.i.preheader
                                        ;   in Loop: Header=BB0_303 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[40:41], 0
	s_mov_b64 s[26:27], 0
.LBB0_329:                              ; %.preheader24.i.i.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_303 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s28, s2, s26
	s_addc_u32 s29, s3, s27
	global_load_ubyte v3, v9, s[28:29]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v40, v10, v40
	s_cmp_lg_u32 s5, s26
	v_or_b32_e32 v41, v11, v41
	s_cbranch_scc1 .LBB0_329
	s_branch .LBB0_332
.LBB0_330:                              ;   in Loop: Header=BB0_303 Depth=2
                                        ; implicit-def: $vgpr40_vgpr41
                                        ; implicit-def: $sgpr28
	s_branch .LBB0_333
.LBB0_331:                              ;   in Loop: Header=BB0_303 Depth=2
	v_mov_b64_e32 v[40:41], 0
.LBB0_332:                              ; %Flow4303
                                        ;   in Loop: Header=BB0_303 Depth=2
	s_mov_b32 s28, 0
	s_cbranch_execnz .LBB0_334
.LBB0_333:                              ;   in Loop: Header=BB0_303 Depth=2
	global_load_dwordx2 v[40:41], v9, s[2:3]
	s_add_i32 s28, s5, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_334:                              ; %.loopexit25.i.i.i
                                        ;   in Loop: Header=BB0_303 Depth=2
	s_cmp_gt_u32 s28, 7
	s_cbranch_scc1 .LBB0_338
; %bb.335:                              ;   in Loop: Header=BB0_303 Depth=2
	s_cmp_eq_u32 s28, 0
	s_cbranch_scc1 .LBB0_339
; %bb.336:                              ; %.preheader22.i.i.i.preheader
                                        ;   in Loop: Header=BB0_303 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[42:43], 0
	s_mov_b64 s[26:27], 0
.LBB0_337:                              ; %.preheader22.i.i.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_303 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s30, s2, s26
	s_addc_u32 s31, s3, s27
	global_load_ubyte v3, v9, s[30:31]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v42, v10, v42
	s_cmp_lg_u32 s28, s26
	v_or_b32_e32 v43, v11, v43
	s_cbranch_scc1 .LBB0_337
	s_branch .LBB0_340
.LBB0_338:                              ;   in Loop: Header=BB0_303 Depth=2
                                        ; implicit-def: $sgpr5
	s_branch .LBB0_341
.LBB0_339:                              ;   in Loop: Header=BB0_303 Depth=2
	v_mov_b64_e32 v[42:43], 0
.LBB0_340:                              ; %Flow4298
                                        ;   in Loop: Header=BB0_303 Depth=2
	s_mov_b32 s5, 0
	s_cbranch_execnz .LBB0_342
.LBB0_341:                              ;   in Loop: Header=BB0_303 Depth=2
	global_load_dwordx2 v[42:43], v9, s[2:3]
	s_add_i32 s5, s28, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_342:                              ; %.loopexit23.i.i.i
                                        ;   in Loop: Header=BB0_303 Depth=2
	s_cmp_gt_u32 s5, 7
	s_cbranch_scc1 .LBB0_346
; %bb.343:                              ;   in Loop: Header=BB0_303 Depth=2
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_347
; %bb.344:                              ; %.preheader20.i.i.i.preheader
                                        ;   in Loop: Header=BB0_303 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[44:45], 0
	s_mov_b64 s[26:27], 0
.LBB0_345:                              ; %.preheader20.i.i.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_303 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s28, s2, s26
	s_addc_u32 s29, s3, s27
	global_load_ubyte v3, v9, s[28:29]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v44, v10, v44
	s_cmp_lg_u32 s5, s26
	v_or_b32_e32 v45, v11, v45
	s_cbranch_scc1 .LBB0_345
	s_branch .LBB0_348
.LBB0_346:                              ;   in Loop: Header=BB0_303 Depth=2
                                        ; implicit-def: $vgpr44_vgpr45
                                        ; implicit-def: $sgpr28
	s_branch .LBB0_349
.LBB0_347:                              ;   in Loop: Header=BB0_303 Depth=2
	v_mov_b64_e32 v[44:45], 0
.LBB0_348:                              ; %Flow4293
                                        ;   in Loop: Header=BB0_303 Depth=2
	s_mov_b32 s28, 0
	s_cbranch_execnz .LBB0_350
.LBB0_349:                              ;   in Loop: Header=BB0_303 Depth=2
	global_load_dwordx2 v[44:45], v9, s[2:3]
	s_add_i32 s28, s5, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_350:                              ; %.loopexit21.i.i.i
                                        ;   in Loop: Header=BB0_303 Depth=2
	s_cmp_gt_u32 s28, 7
	s_cbranch_scc1 .LBB0_354
; %bb.351:                              ;   in Loop: Header=BB0_303 Depth=2
	s_cmp_eq_u32 s28, 0
	s_cbranch_scc1 .LBB0_355
; %bb.352:                              ; %.preheader.i.i.i.preheader
                                        ;   in Loop: Header=BB0_303 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[46:47], 0
	s_mov_b64 s[26:27], s[2:3]
.LBB0_353:                              ; %.preheader.i.i.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_303 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	global_load_ubyte v3, v9, s[26:27]
	s_add_i32 s28, s28, -1
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	v_or_b32_e32 v46, v10, v46
	s_cmp_lg_u32 s28, 0
	v_or_b32_e32 v47, v11, v47
	s_cbranch_scc1 .LBB0_353
	s_branch .LBB0_356
.LBB0_354:                              ;   in Loop: Header=BB0_303 Depth=2
	s_branch .LBB0_357
.LBB0_355:                              ;   in Loop: Header=BB0_303 Depth=2
	v_mov_b64_e32 v[46:47], 0
.LBB0_356:                              ; %Flow4288
                                        ;   in Loop: Header=BB0_303 Depth=2
	s_cbranch_execnz .LBB0_358
.LBB0_357:                              ;   in Loop: Header=BB0_303 Depth=2
	global_load_dwordx2 v[46:47], v9, s[2:3]
.LBB0_358:                              ; %.loopexit.i.i.i
                                        ;   in Loop: Header=BB0_303 Depth=2
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[10:11], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_364
; %bb.359:                              ;   in Loop: Header=BB0_303 Depth=2
	global_load_dwordx2 v[50:51], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v3, v10, v50
	v_and_b32_e32 v7, v11, v51
	v_mul_lo_u32 v7, v7, 24
	v_mul_hi_u32 v8, v3, 24
	v_add_u32_e32 v11, v8, v7
	v_mul_lo_u32 v10, v3, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[14:15], 0, v[10:11]
	global_load_dwordx2 v[48:49], v[10:11], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[48:51], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[10:11], v[50:51]
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB0_363
; %bb.360:                              ; %.preheader3.i.i18.i.i.i.preheader
                                        ;   in Loop: Header=BB0_303 Depth=2
	s_mov_b64 s[28:29], 0
.LBB0_361:                              ; %.preheader3.i.i18.i.i.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_303 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_sleep 1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	global_load_dwordx2 v[18:19], v9, s[16:17]
	v_mov_b64_e32 v[50:51], v[10:11]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v7, v14, v50
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[10:11], s[30:31], v7, 24, v[18:19]
	v_and_b32_e32 v3, v15, v51
	v_mov_b32_e32 v8, v11
	v_mad_u64_u32 v[14:15], s[30:31], v3, 24, v[8:9]
	v_mov_b32_e32 v11, v14
	global_load_dwordx2 v[48:49], v[10:11], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[48:51], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[10:11], v[50:51]
	s_or_b64 s[28:29], vcc, s[28:29]
	s_andn2_b64 exec, exec, s[28:29]
	s_cbranch_execnz .LBB0_361
; %bb.362:                              ; %Flow4283
                                        ;   in Loop: Header=BB0_303 Depth=2
	s_or_b64 exec, exec, s[28:29]
.LBB0_363:                              ; %Flow4285
                                        ;   in Loop: Header=BB0_303 Depth=2
	s_or_b64 exec, exec, s[26:27]
.LBB0_364:                              ; %.loopexit4.i.i13.i.i.i
                                        ;   in Loop: Header=BB0_303 Depth=2
	s_or_b64 exec, exec, s[24:25]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	global_load_dwordx4 v[48:51], v9, s[16:17]
	v_readfirstlane_b32 s24, v10
	v_readfirstlane_b32 s25, v11
	s_mov_b64 s[26:27], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s28, v14
	v_readfirstlane_b32 s29, v15
	s_and_b64 s[28:29], s[24:25], s[28:29]
	s_mul_i32 s5, s29, 24
	s_mul_hi_u32 s30, s28, 24
	s_add_i32 s31, s30, s5
	s_mul_i32 s30, s28, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[48:49], 0, s[30:31]
	s_and_saveexec_b64 s[30:31], s[2:3]
	s_cbranch_execz .LBB0_366
; %bb.365:                              ;   in Loop: Header=BB0_303 Depth=2
	v_mov_b64_e32 v[22:23], s[26:27]
	global_store_dwordx4 v[10:11], v[22:25], off offset:8
.LBB0_366:                              ;   in Loop: Header=BB0_303 Depth=2
	s_or_b64 exec, exec, s[30:31]
	v_or_b32_e32 v3, 0, v33
	v_or_b32_e32 v7, v32, v2
	v_cmp_gt_u64_e64 vcc, s[20:21], 56
	s_lshl_b32 s5, s22, 2
	s_lshl_b64 s[26:27], s[28:29], 12
	v_cndmask_b32_e32 v33, v3, v33, vcc
	v_cndmask_b32_e32 v3, v7, v32, vcc
	s_add_i32 s5, s5, 28
	v_lshl_add_u64 v[14:15], v[50:51], 0, s[26:27]
	s_and_b32 s5, s5, 0x1e0
	v_and_b32_e32 v3, 0xffffff1f, v3
	v_or_b32_e32 v32, s5, v3
	v_readfirstlane_b32 s26, v14
	v_readfirstlane_b32 s27, v15
	s_nop 4
	global_store_dwordx4 v58, v[32:35], s[26:27]
	global_store_dwordx4 v58, v[36:39], s[26:27] offset:16
	global_store_dwordx4 v58, v[40:43], s[26:27] offset:32
	global_store_dwordx4 v58, v[44:47], s[26:27] offset:48
	s_and_saveexec_b64 s[26:27], s[2:3]
	s_cbranch_execz .LBB0_374
; %bb.367:                              ;   in Loop: Header=BB0_303 Depth=2
	global_load_dwordx2 v[36:37], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[18:19], v9, s[16:17] offset:40
	v_mov_b32_e32 v34, s24
	v_mov_b32_e32 v35, s25
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s28, v18
	v_readfirstlane_b32 s29, v19
	s_and_b64 s[28:29], s[28:29], s[24:25]
	s_mul_i32 s5, s29, 24
	s_mul_hi_u32 s29, s28, 24
	s_mul_i32 s28, s28, 24
	s_add_i32 s29, s29, s5
	v_lshl_add_u64 v[18:19], v[48:49], 0, s[28:29]
	global_store_dwordx2 v[18:19], v[36:37], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[34:35], v9, v[34:37], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[34:35], v[36:37]
	s_and_saveexec_b64 s[28:29], vcc
	s_cbranch_execz .LBB0_370
; %bb.368:                              ; %.preheader1.i.i16.i.i.i.preheader
                                        ;   in Loop: Header=BB0_303 Depth=2
	s_mov_b64 s[30:31], 0
.LBB0_369:                              ; %.preheader1.i.i16.i.i.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_303 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_sleep 1
	global_store_dwordx2 v[18:19], v[34:35], off
	v_mov_b32_e32 v32, s24
	v_mov_b32_e32 v33, s25
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[22:23], v9, v[32:35], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[22:23], v[34:35]
	s_or_b64 s[30:31], vcc, s[30:31]
	v_mov_b64_e32 v[34:35], v[22:23]
	s_andn2_b64 exec, exec, s[30:31]
	s_cbranch_execnz .LBB0_369
.LBB0_370:                              ; %Flow4281
                                        ;   in Loop: Header=BB0_303 Depth=2
	s_or_b64 exec, exec, s[28:29]
	global_load_dwordx2 v[18:19], v9, s[16:17] offset:16
	s_mov_b64 s[30:31], exec
	v_mbcnt_lo_u32_b32 v3, s30, 0
	v_mbcnt_hi_u32_b32 v3, s31, v3
	v_cmp_eq_u32_e32 vcc, 0, v3
	s_and_saveexec_b64 s[28:29], vcc
	s_cbranch_execz .LBB0_372
; %bb.371:                              ;   in Loop: Header=BB0_303 Depth=2
	s_bcnt1_i32_b64 s5, s[30:31]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[18:19], v[8:9], off offset:8 sc1
.LBB0_372:                              ;   in Loop: Header=BB0_303 Depth=2
	s_or_b64 exec, exec, s[28:29]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[22:23], v[18:19], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_cbranch_vccnz .LBB0_374
; %bb.373:                              ;   in Loop: Header=BB0_303 Depth=2
	global_load_dword v8, v[18:19], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v3, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v3
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[22:23], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_374:                              ; %Flow4282
                                        ;   in Loop: Header=BB0_303 Depth=2
	s_or_b64 exec, exec, s[26:27]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[14:15], v[14:15], 0, v[58:59]
	s_branch .LBB0_378
.LBB0_375:                              ;   in Loop: Header=BB0_378 Depth=3
	s_or_b64 exec, exec, s[26:27]
	v_readfirstlane_b32 s5, v3
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_377
; %bb.376:                              ;   in Loop: Header=BB0_378 Depth=3
	s_sleep 1
	s_cbranch_execnz .LBB0_378
	s_branch .LBB0_380
.LBB0_377:                              ;   in Loop: Header=BB0_303 Depth=2
	s_branch .LBB0_380
.LBB0_378:                              ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_303 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_mov_b32_e32 v3, 1
	s_and_saveexec_b64 s[26:27], s[2:3]
	s_cbranch_execz .LBB0_375
; %bb.379:                              ;   in Loop: Header=BB0_378 Depth=3
	global_load_dword v3, v[10:11], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v3, 1, v3
	s_branch .LBB0_375
.LBB0_380:                              ;   in Loop: Header=BB0_303 Depth=2
	global_load_dwordx4 v[32:35], v[14:15], off
	s_and_saveexec_b64 s[26:27], s[2:3]
	s_cbranch_execz .LBB0_302
; %bb.381:                              ;   in Loop: Header=BB0_303 Depth=2
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[18:19], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[22:23], v[10:11], 0, 1
	v_lshl_add_u64 v[34:35], v[22:23], 0, s[24:25]
	v_cmp_eq_u64_e32 vcc, 0, v[34:35]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v36, v14
	v_mov_b32_e32 v37, v15
	v_cndmask_b32_e32 v35, v35, v23, vcc
	v_cndmask_b32_e32 v34, v34, v22, vcc
	v_and_b32_e32 v3, v35, v11
	v_and_b32_e32 v7, v34, v10
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v8, v7, 24
	v_mul_lo_u32 v10, v7, 24
	v_add_u32_e32 v11, v8, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[18:19], 0, v[10:11]
	global_store_dwordx2 v[10:11], v[14:15], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[14:15]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_302
; %bb.382:                              ; %.preheader.i.i15.i.i.i.preheader
                                        ;   in Loop: Header=BB0_303 Depth=2
	s_mov_b64 s[2:3], 0
.LBB0_383:                              ; %.preheader.i.i15.i.i.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_303 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_sleep 1
	global_store_dwordx2 v[10:11], v[36:37], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[14:15], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[14:15], v[36:37]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[36:37], v[14:15]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_383
	s_branch .LBB0_302
.LBB0_384:                              ; %Flow4321
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_413
.LBB0_385:                              ;   in Loop: Header=BB0_273 Depth=1
                                        ; implicit-def: $vgpr32_vgpr33
	s_cbranch_execz .LBB0_413
; %bb.386:                              ;   in Loop: Header=BB0_273 Depth=1
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_392
; %bb.387:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[34:35], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v34
	v_and_b32_e32 v3, v3, v35
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[32:33], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[32:35], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[34:35]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_391
; %bb.388:                              ; %.preheader3.i.i.i9.i.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_389:                              ; %.preheader3.i.i.i9.i.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[34:35], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v34
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v35
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[32:33], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[32:35], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[34:35]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_389
; %bb.390:                              ; %Flow4334
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_391:                              ; %Flow4336
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_392:                              ; %.loopexit4.i.i.i4.i.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[34:37], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[34:35], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_394
; %bb.393:                              ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_394:                              ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[36:37], 0, s[6:7]
	v_and_or_b32 v30, v30, s33, 32
	v_mov_b32_e32 v32, v9
	v_mov_b32_e32 v33, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	s_nop 1
	global_store_dwordx4 v58, v[30:33], s[22:23]
	s_nop 1
	v_mov_b64_e32 v[32:33], s[6:7]
	v_mov_b64_e32 v[30:31], s[4:5]
	global_store_dwordx4 v58, v[30:33], s[22:23] offset:16
	global_store_dwordx4 v58, v[30:33], s[22:23] offset:32
	global_store_dwordx4 v58, v[30:33], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_402
; %bb.395:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[38:39], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v36, s20
	v_mov_b32_e32 v37, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[34:35], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[38:39], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[36:39], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[38:39]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_398
; %bb.396:                              ; %.preheader1.i.i.i7.i.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_397:                              ; %.preheader1.i.i.i7.i.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[32:33], off
	v_mov_b32_e32 v30, s20
	v_mov_b32_e32 v31, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[30:33], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[32:33]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[32:33], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_397
.LBB0_398:                              ; %Flow4332
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_400
; %bb.399:                              ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_400:                              ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_402
; %bb.401:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_402:                              ; %Flow4333
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_406
.LBB0_403:                              ;   in Loop: Header=BB0_406 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_405
; %bb.404:                              ;   in Loop: Header=BB0_406 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_406
	s_branch .LBB0_408
.LBB0_405:                              ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_408
.LBB0_406:                              ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_403
; %bb.407:                              ;   in Loop: Header=BB0_406 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_403
.LBB0_408:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[32:33], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_412
; %bb.409:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[18:19], v[2:3], 0, 1
	v_lshl_add_u64 v[22:23], v[18:19], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v36, v10
	v_mov_b32_e32 v37, v11
	v_cndmask_b32_e32 v35, v23, v19, vcc
	v_cndmask_b32_e32 v34, v22, v18, vcc
	v_and_b32_e32 v3, v35, v3
	v_and_b32_e32 v2, v34, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_412
; %bb.410:                              ; %.preheader.i.i.i6.i.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_411:                              ; %.preheader.i.i.i6.i.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[36:37], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[36:37]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[36:37], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_411
.LBB0_412:                              ; %Flow4325
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
.LBB0_413:                              ; %__ockl_printf_append_string_n.exit.i.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_419
; %bb.414:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[36:37], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v36
	v_and_b32_e32 v3, v3, v37
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[34:35], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[36:37]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_418
; %bb.415:                              ; %.preheader3.i.i.i16.i.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_416:                              ; %.preheader3.i.i.i16.i.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[36:37], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v36
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v37
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[34:35], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[36:37]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_416
; %bb.417:                              ; %Flow4269
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_418:                              ; %Flow4271
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_419:                              ; %.loopexit4.i.i.i10.i.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[36:39], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[36:37], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_421
; %bb.420:                              ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_421:                              ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[38:39], 0, s[6:7]
	v_and_or_b32 v32, v32, s33, 32
	v_mov_b32_e32 v34, v26
	v_mov_b32_e32 v35, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	s_nop 1
	global_store_dwordx4 v58, v[32:35], s[22:23]
	s_nop 1
	v_mov_b64_e32 v[32:33], s[6:7]
	v_mov_b64_e32 v[30:31], s[4:5]
	global_store_dwordx4 v58, v[30:33], s[22:23] offset:16
	global_store_dwordx4 v58, v[30:33], s[22:23] offset:32
	global_store_dwordx4 v58, v[30:33], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_429
; %bb.422:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[34:35], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v32, s20
	v_mov_b32_e32 v33, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[36:37], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[34:35], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[32:35], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[34:35]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_425
; %bb.423:                              ; %.preheader1.i.i.i14.i.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_424:                              ; %.preheader1.i.i.i14.i.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[32:33], off
	v_mov_b32_e32 v30, s20
	v_mov_b32_e32 v31, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[30:33], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[32:33]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[32:33], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_424
.LBB0_425:                              ; %Flow4267
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_427
; %bb.426:                              ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_427:                              ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_429
; %bb.428:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_429:                              ; %Flow4268
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_433
.LBB0_430:                              ;   in Loop: Header=BB0_433 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_432
; %bb.431:                              ;   in Loop: Header=BB0_433 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_433
	s_branch .LBB0_435
.LBB0_432:                              ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_435
.LBB0_433:                              ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_430
; %bb.434:                              ;   in Loop: Header=BB0_433 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_430
.LBB0_435:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[14:15], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_439
; %bb.436:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[18:19], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[22:23], v[2:3], 0, 1
	v_lshl_add_u64 v[30:31], v[22:23], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[30:31]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v32, v10
	v_mov_b32_e32 v33, v11
	v_cndmask_b32_e32 v31, v31, v23, vcc
	v_cndmask_b32_e32 v30, v30, v22, vcc
	v_and_b32_e32 v3, v31, v3
	v_and_b32_e32 v2, v30, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[18:19], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_439
; %bb.437:                              ; %.preheader.i.i.i13.i.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_438:                              ; %.preheader.i.i.i13.i.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[32:33], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[32:33]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[32:33], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_438
.LBB0_439:                              ; %__ockl_printf_append_args.exit.i.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_445
; %bb.440:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[32:33], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v32
	v_and_b32_e32 v3, v3, v33
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[32:33]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_444
; %bb.441:                              ; %.preheader3.i.i.i23.i.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_442:                              ; %.preheader3.i.i.i23.i.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[18:19], v9, s[16:17]
	v_mov_b64_e32 v[32:33], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v32
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[18:19]
	v_and_b32_e32 v7, v11, v33
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[32:33]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_442
; %bb.443:                              ; %Flow4255
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_444:                              ; %Flow4257
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_445:                              ; %.loopexit4.i.i.i17.i.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[30:33], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[30:31], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_447
; %bb.446:                              ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_447:                              ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[32:33], 0, s[6:7]
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	v_mov_b64_e32 v[34:35], s[6:7]
	v_and_or_b32 v14, v14, s33, 32
	v_mov_b32_e32 v17, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	v_mov_b64_e32 v[32:33], s[4:5]
	s_nop 3
	global_store_dwordx4 v58, v[14:17], s[22:23]
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:16
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:32
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_455
; %bb.448:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[34:35], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v32, s20
	v_mov_b32_e32 v33, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[30:31], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[34:35], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[32:35], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[34:35]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_451
; %bb.449:                              ; %.preheader1.i.i.i21.i.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_450:                              ; %.preheader1.i.i.i21.i.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[32:33], off
	v_mov_b32_e32 v30, s20
	v_mov_b32_e32 v31, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[30:33], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[32:33]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[32:33], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_450
.LBB0_451:                              ; %Flow4253
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_453
; %bb.452:                              ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_453:                              ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_455
; %bb.454:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_455:                              ; %Flow4254
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_459
.LBB0_456:                              ;   in Loop: Header=BB0_459 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_458
; %bb.457:                              ;   in Loop: Header=BB0_459 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_459
	s_branch .LBB0_461
.LBB0_458:                              ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_461
.LBB0_459:                              ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_456
; %bb.460:                              ;   in Loop: Header=BB0_459 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_456
.LBB0_461:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[18:19], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_465
; %bb.462:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[22:23], v[2:3], 0, 1
	v_lshl_add_u64 v[30:31], v[22:23], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[30:31]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v32, v10
	v_mov_b32_e32 v33, v11
	v_cndmask_b32_e32 v31, v31, v23, vcc
	v_cndmask_b32_e32 v30, v30, v22, vcc
	v_and_b32_e32 v3, v31, v3
	v_and_b32_e32 v2, v30, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_465
; %bb.463:                              ; %.preheader.i.i.i20.i.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_464:                              ; %.preheader.i.i.i20.i.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[32:33], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[32:33]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[32:33], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_464
.LBB0_465:                              ; %__ockl_printf_append_args.exit24.i.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_471
; %bb.466:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[32:33], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v32
	v_and_b32_e32 v3, v3, v33
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[32:33]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_470
; %bb.467:                              ; %.preheader3.i.i.i31.i.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_468:                              ; %.preheader3.i.i.i31.i.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[32:33], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v32
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v33
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[32:33]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_468
; %bb.469:                              ; %Flow4241
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_470:                              ; %Flow4243
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_471:                              ; %.loopexit4.i.i.i25.i.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[30:33], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[30:31], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_473
; %bb.472:                              ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_473:                              ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[32:33], 0, s[6:7]
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	v_mov_b64_e32 v[34:35], s[6:7]
	v_and_or_b32 v18, v18, s33, 32
	v_mov_b32_e32 v21, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	v_mov_b64_e32 v[32:33], s[4:5]
	s_nop 3
	global_store_dwordx4 v58, v[18:21], s[22:23]
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:16
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:32
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_481
; %bb.474:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[34:35], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v32, s20
	v_mov_b32_e32 v33, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[30:31], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[34:35], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[32:35], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[34:35]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_477
; %bb.475:                              ; %.preheader1.i.i.i29.i.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_476:                              ; %.preheader1.i.i.i29.i.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[32:33], off
	v_mov_b32_e32 v30, s20
	v_mov_b32_e32 v31, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[30:33], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[32:33]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[32:33], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_476
.LBB0_477:                              ; %Flow4239
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_479
; %bb.478:                              ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_479:                              ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_481
; %bb.480:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_481:                              ; %Flow4240
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_485
.LBB0_482:                              ;   in Loop: Header=BB0_485 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_484
; %bb.483:                              ;   in Loop: Header=BB0_485 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_485
	s_branch .LBB0_487
.LBB0_484:                              ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_487
.LBB0_485:                              ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_482
; %bb.486:                              ;   in Loop: Header=BB0_485 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_482
.LBB0_487:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[18:19], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_491
; %bb.488:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[22:23], v[2:3], 0, 1
	v_lshl_add_u64 v[30:31], v[22:23], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[30:31]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v32, v10
	v_mov_b32_e32 v33, v11
	v_cndmask_b32_e32 v31, v31, v23, vcc
	v_cndmask_b32_e32 v30, v30, v22, vcc
	v_and_b32_e32 v3, v31, v3
	v_and_b32_e32 v2, v30, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_491
; %bb.489:                              ; %.preheader.i.i.i28.i.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_490:                              ; %.preheader.i.i.i28.i.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[32:33], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[32:33]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[32:33], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_490
.LBB0_491:                              ; %__ockl_printf_append_args.exit32.i.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_497
; %bb.492:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[32:33], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v32
	v_and_b32_e32 v3, v3, v33
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[32:33]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_496
; %bb.493:                              ; %.preheader3.i.i.i39.i.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_494:                              ; %.preheader3.i.i.i39.i.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[32:33], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v32
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v33
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[32:33]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_494
; %bb.495:                              ; %Flow4227
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_496:                              ; %Flow4229
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_497:                              ; %.loopexit4.i.i.i33.i.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[30:33], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[30:31], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_499
; %bb.498:                              ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_499:                              ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[32:33], 0, s[6:7]
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	v_mov_b64_e32 v[34:35], s[6:7]
	v_and_or_b32 v18, v18, s33, 32
	v_mov_b32_e32 v21, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	v_mov_b64_e32 v[32:33], s[4:5]
	s_nop 3
	global_store_dwordx4 v58, v[18:21], s[22:23]
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:16
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:32
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_507
; %bb.500:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[34:35], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v32, s20
	v_mov_b32_e32 v33, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[30:31], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[34:35], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[32:35], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[34:35]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_503
; %bb.501:                              ; %.preheader1.i.i.i37.i.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_502:                              ; %.preheader1.i.i.i37.i.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[32:33], off
	v_mov_b32_e32 v30, s20
	v_mov_b32_e32 v31, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[30:33], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[32:33]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[32:33], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_502
.LBB0_503:                              ; %Flow4225
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_505
; %bb.504:                              ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_505:                              ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_507
; %bb.506:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_507:                              ; %Flow4226
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_511
.LBB0_508:                              ;   in Loop: Header=BB0_511 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_510
; %bb.509:                              ;   in Loop: Header=BB0_511 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_511
	s_branch .LBB0_513
.LBB0_510:                              ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_513
.LBB0_511:                              ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_508
; %bb.512:                              ;   in Loop: Header=BB0_511 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_508
.LBB0_513:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[18:19], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_517
; %bb.514:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[22:23], v[2:3], 0, 1
	v_lshl_add_u64 v[30:31], v[22:23], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[30:31]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v32, v10
	v_mov_b32_e32 v33, v11
	v_cndmask_b32_e32 v31, v31, v23, vcc
	v_cndmask_b32_e32 v30, v30, v22, vcc
	v_and_b32_e32 v3, v31, v3
	v_and_b32_e32 v2, v30, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_517
; %bb.515:                              ; %.preheader.i.i.i36.i.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_516:                              ; %.preheader.i.i.i36.i.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[32:33], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[32:33]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[32:33], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_516
.LBB0_517:                              ; %__ockl_printf_append_args.exit40.i.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_523
; %bb.518:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[32:33], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v32
	v_and_b32_e32 v3, v3, v33
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[32:33]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_522
; %bb.519:                              ; %.preheader3.i.i.i47.i.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_520:                              ; %.preheader3.i.i.i47.i.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[32:33], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v32
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v33
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[32:33]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_520
; %bb.521:                              ; %Flow4213
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_522:                              ; %Flow4215
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_523:                              ; %.loopexit4.i.i.i41.i.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[30:33], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[30:31], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_525
; %bb.524:                              ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_525:                              ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[32:33], 0, s[6:7]
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	v_mov_b64_e32 v[34:35], s[6:7]
	v_and_or_b32 v18, v18, s34, 34
	v_mov_b32_e32 v21, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	v_mov_b64_e32 v[32:33], s[4:5]
	s_nop 3
	global_store_dwordx4 v58, v[18:21], s[22:23]
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:16
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:32
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_533
; %bb.526:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[34:35], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	v_mov_b32_e32 v32, s20
	v_mov_b32_e32 v33, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[10:11], v[30:31], 0, s[22:23]
	global_store_dwordx2 v[10:11], v[34:35], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[32:35], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[34:35]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_529
; %bb.527:                              ; %.preheader1.i.i.i45.i.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_528:                              ; %.preheader1.i.i.i45.i.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[10:11], v[32:33], off
	v_mov_b32_e32 v30, s20
	v_mov_b32_e32 v31, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[14:15], v9, v[30:33], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[14:15], v[32:33]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[32:33], v[14:15]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_528
.LBB0_529:                              ; %Flow4211
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_531
; %bb.530:                              ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[10:11], v[8:9], off offset:8 sc1
.LBB0_531:                              ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[14:15], v[10:11], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[14:15]
	s_cbranch_vccnz .LBB0_533
; %bb.532:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[10:11], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[14:15], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_533:                              ; %Flow4212
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	s_branch .LBB0_537
.LBB0_534:                              ;   in Loop: Header=BB0_537 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_536
; %bb.535:                              ;   in Loop: Header=BB0_537 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_537
	s_branch .LBB0_539
.LBB0_536:                              ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_539
.LBB0_537:                              ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_534
; %bb.538:                              ;   in Loop: Header=BB0_537 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_534
.LBB0_539:                              ;   in Loop: Header=BB0_273 Depth=1
	s_and_b64 exec, exec, s[2:3]
	s_cbranch_execz .LBB0_543
; %bb.540:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[18:19], v[2:3], 0, 1
	v_lshl_add_u64 v[22:23], v[18:19], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v32, v10
	v_mov_b32_e32 v33, v11
	v_cndmask_b32_e32 v31, v23, v19, vcc
	v_cndmask_b32_e32 v30, v22, v18, vcc
	v_and_b32_e32 v3, v31, v3
	v_and_b32_e32 v2, v30, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_543
; %bb.541:                              ; %.preheader.i.i.i44.i.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_542:                              ; %.preheader.i.i.i44.i.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[32:33], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[32:33]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[32:33], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_542
.LBB0_543:                              ; %Flow4353
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[18:19]
	v_lshrrev_b32_e32 v2, 3, v26
	v_bitop3_b32 v32, v2, v26, s35 bitop3:0x6c
	s_and_saveexec_b64 s[18:19], s[0:1]
	s_cbranch_execz .LBB0_787
; %bb.544:                              ; %if.then.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_550
; %bb.545:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[36:37], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v36
	v_and_b32_e32 v3, v3, v37
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[34:35], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[36:37]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_549
; %bb.546:                              ; %.preheader3.i.i.i.i123.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_547:                              ; %.preheader3.i.i.i.i123
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[36:37], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v36
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v37
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[34:35], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[36:37]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_547
; %bb.548:                              ; %Flow4197
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_549:                              ; %Flow4199
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_550:                              ; %.loopexit4.i.i.i.i87
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[34:37], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[34:35], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_552
; %bb.551:                              ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_552:                              ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[36:37], 0, s[6:7]
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	v_mov_b64_e32 v[38:39], s[6:7]
	v_mov_b32_e32 v7, v9
	v_mov_b32_e32 v8, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	v_mov_b64_e32 v[36:37], s[4:5]
	s_nop 3
	global_store_dwordx4 v58, v[6:9], s[22:23]
	global_store_dwordx4 v58, v[36:39], s[22:23] offset:16
	global_store_dwordx4 v58, v[36:39], s[22:23] offset:32
	global_store_dwordx4 v58, v[36:39], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_560
; %bb.553:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[38:39], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v36, s20
	v_mov_b32_e32 v37, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[34:35], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[38:39], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[36:39], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[38:39]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_556
; %bb.554:                              ; %.preheader1.i.i.i.i121.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_555:                              ; %.preheader1.i.i.i.i121
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[36:37], off
	v_mov_b32_e32 v34, s20
	v_mov_b32_e32 v35, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[34:37], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[36:37]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[36:37], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_555
.LBB0_556:                              ; %Flow4195
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_558
; %bb.557:                              ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_558:                              ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_560
; %bb.559:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_560:                              ; %Flow4196
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_564
.LBB0_561:                              ;   in Loop: Header=BB0_564 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_563
; %bb.562:                              ;   in Loop: Header=BB0_564 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_564
	s_branch .LBB0_566
.LBB0_563:                              ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_566
.LBB0_564:                              ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_561
; %bb.565:                              ;   in Loop: Header=BB0_564 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_561
.LBB0_566:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[34:35], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_570
; %bb.567:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[18:19], v[2:3], 0, 1
	v_lshl_add_u64 v[22:23], v[18:19], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v38, v10
	v_mov_b32_e32 v39, v11
	v_cndmask_b32_e32 v37, v23, v19, vcc
	v_cndmask_b32_e32 v36, v22, v18, vcc
	v_and_b32_e32 v3, v37, v3
	v_and_b32_e32 v2, v36, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v9, v[36:39], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_570
; %bb.568:                              ; %.preheader.i.i.i.i120.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_569:                              ; %.preheader.i.i.i.i120
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[38:39], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[36:39], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[38:39]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[38:39], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_569
.LBB0_570:                              ; %__ockl_printf_begin.exit.i89
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	s_and_b64 vcc, exec, s[10:11]
	s_cbranch_vccz .LBB0_655
; %bb.571:                              ;   in Loop: Header=BB0_273 Depth=1
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v2, 2, v34
	v_and_b32_e32 v36, -3, v34
	v_mov_b32_e32 v37, v35
	s_mov_b64 s[20:21], 49
	s_getpc_b64 s[6:7]
	s_add_u32 s6, s6, .str.2@rel32@lo+4
	s_addc_u32 s7, s7, .str.2@rel32@hi+12
	s_branch .LBB0_573
.LBB0_572:                              ; %__ockl_hostcall_preview.exit19.i.i106
                                        ;   in Loop: Header=BB0_573 Depth=2
	s_or_b64 exec, exec, s[26:27]
	s_sub_u32 s20, s20, s22
	s_subb_u32 s21, s21, s23
	s_add_u32 s6, s6, s22
	s_addc_u32 s7, s7, s23
	s_cmp_lg_u64 s[20:21], 0
	s_cbranch_scc0 .LBB0_654
.LBB0_573:                              ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_576 Depth 3
                                        ;       Child Loop BB0_583 Depth 3
                                        ;       Child Loop BB0_591 Depth 3
                                        ;       Child Loop BB0_599 Depth 3
                                        ;       Child Loop BB0_607 Depth 3
                                        ;       Child Loop BB0_615 Depth 3
                                        ;       Child Loop BB0_623 Depth 3
                                        ;       Child Loop BB0_631 Depth 3
                                        ;       Child Loop BB0_639 Depth 3
                                        ;       Child Loop BB0_648 Depth 3
                                        ;       Child Loop BB0_653 Depth 3
	v_cmp_lt_u64_e64 s[2:3], s[20:21], 56
	s_and_b64 s[2:3], s[2:3], exec
	v_cmp_gt_u64_e64 s[2:3], s[20:21], 7
	s_cselect_b32 s23, s21, 0
	s_cselect_b32 s22, s20, 56
	s_and_b64 vcc, exec, s[2:3]
	s_cbranch_vccnz .LBB0_578
; %bb.574:                              ;   in Loop: Header=BB0_573 Depth=2
	s_mov_b64 s[2:3], 0
	s_cmp_eq_u64 s[20:21], 0
	s_waitcnt vmcnt(0)
	v_mov_b64_e32 v[38:39], 0
	s_cbranch_scc1 .LBB0_577
; %bb.575:                              ; %.preheader30.i.i90.preheader
                                        ;   in Loop: Header=BB0_573 Depth=2
	s_lshl_b64 s[24:25], s[22:23], 3
	s_mov_b64 s[26:27], 0
	v_mov_b64_e32 v[38:39], 0
	s_mov_b64 s[28:29], s[6:7]
.LBB0_576:                              ; %.preheader30.i.i90
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_573 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	global_load_ubyte v3, v9, s[28:29]
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s26, v[8:9]
	s_add_u32 s26, s26, 8
	s_addc_u32 s27, s27, 0
	s_add_u32 s28, s28, 1
	s_addc_u32 s29, s29, 0
	v_or_b32_e32 v38, v10, v38
	s_cmp_lg_u32 s24, s26
	v_or_b32_e32 v39, v11, v39
	s_cbranch_scc1 .LBB0_576
.LBB0_577:                              ; %Flow4165
                                        ;   in Loop: Header=BB0_573 Depth=2
	s_mov_b32 s5, 0
	s_andn2_b64 vcc, exec, s[2:3]
	s_mov_b64 s[2:3], s[6:7]
	s_cbranch_vccz .LBB0_579
	s_branch .LBB0_580
.LBB0_578:                              ;   in Loop: Header=BB0_573 Depth=2
                                        ; implicit-def: $vgpr38_vgpr39
                                        ; implicit-def: $sgpr5
	s_mov_b64 s[2:3], s[6:7]
.LBB0_579:                              ;   in Loop: Header=BB0_573 Depth=2
	global_load_dwordx2 v[38:39], v9, s[6:7]
	s_add_i32 s5, s22, -8
	s_add_u32 s2, s6, 8
	s_addc_u32 s3, s7, 0
.LBB0_580:                              ; %.loopexit31.i.i91
                                        ;   in Loop: Header=BB0_573 Depth=2
	s_cmp_gt_u32 s5, 7
	s_cbranch_scc1 .LBB0_584
; %bb.581:                              ;   in Loop: Header=BB0_573 Depth=2
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_585
; %bb.582:                              ; %.preheader28.i.i92.preheader
                                        ;   in Loop: Header=BB0_573 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[40:41], 0
	s_mov_b64 s[26:27], 0
.LBB0_583:                              ; %.preheader28.i.i92
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_573 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s28, s2, s26
	s_addc_u32 s29, s3, s27
	global_load_ubyte v3, v9, s[28:29]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v40, v10, v40
	s_cmp_lg_u32 s5, s26
	v_or_b32_e32 v41, v11, v41
	s_cbranch_scc1 .LBB0_583
	s_branch .LBB0_586
.LBB0_584:                              ;   in Loop: Header=BB0_573 Depth=2
                                        ; implicit-def: $vgpr40_vgpr41
                                        ; implicit-def: $sgpr28
	s_branch .LBB0_587
.LBB0_585:                              ;   in Loop: Header=BB0_573 Depth=2
	v_mov_b64_e32 v[40:41], 0
.LBB0_586:                              ; %Flow4160
                                        ;   in Loop: Header=BB0_573 Depth=2
	s_mov_b32 s28, 0
	s_cbranch_execnz .LBB0_588
.LBB0_587:                              ;   in Loop: Header=BB0_573 Depth=2
	global_load_dwordx2 v[40:41], v9, s[2:3]
	s_add_i32 s28, s5, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_588:                              ; %.loopexit29.i.i93
                                        ;   in Loop: Header=BB0_573 Depth=2
	s_cmp_gt_u32 s28, 7
	s_cbranch_scc1 .LBB0_592
; %bb.589:                              ;   in Loop: Header=BB0_573 Depth=2
	s_cmp_eq_u32 s28, 0
	s_cbranch_scc1 .LBB0_593
; %bb.590:                              ; %.preheader26.i.i94.preheader
                                        ;   in Loop: Header=BB0_573 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[42:43], 0
	s_mov_b64 s[26:27], 0
.LBB0_591:                              ; %.preheader26.i.i94
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_573 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s30, s2, s26
	s_addc_u32 s31, s3, s27
	global_load_ubyte v3, v9, s[30:31]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v42, v10, v42
	s_cmp_lg_u32 s28, s26
	v_or_b32_e32 v43, v11, v43
	s_cbranch_scc1 .LBB0_591
	s_branch .LBB0_594
.LBB0_592:                              ;   in Loop: Header=BB0_573 Depth=2
                                        ; implicit-def: $sgpr5
	s_branch .LBB0_595
.LBB0_593:                              ;   in Loop: Header=BB0_573 Depth=2
	v_mov_b64_e32 v[42:43], 0
.LBB0_594:                              ; %Flow4155
                                        ;   in Loop: Header=BB0_573 Depth=2
	s_mov_b32 s5, 0
	s_cbranch_execnz .LBB0_596
.LBB0_595:                              ;   in Loop: Header=BB0_573 Depth=2
	global_load_dwordx2 v[42:43], v9, s[2:3]
	s_add_i32 s5, s28, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_596:                              ; %.loopexit27.i.i95
                                        ;   in Loop: Header=BB0_573 Depth=2
	s_cmp_gt_u32 s5, 7
	s_cbranch_scc1 .LBB0_600
; %bb.597:                              ;   in Loop: Header=BB0_573 Depth=2
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_601
; %bb.598:                              ; %.preheader24.i.i96.preheader
                                        ;   in Loop: Header=BB0_573 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[44:45], 0
	s_mov_b64 s[26:27], 0
.LBB0_599:                              ; %.preheader24.i.i96
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_573 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s28, s2, s26
	s_addc_u32 s29, s3, s27
	global_load_ubyte v3, v9, s[28:29]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v44, v10, v44
	s_cmp_lg_u32 s5, s26
	v_or_b32_e32 v45, v11, v45
	s_cbranch_scc1 .LBB0_599
	s_branch .LBB0_602
.LBB0_600:                              ;   in Loop: Header=BB0_573 Depth=2
                                        ; implicit-def: $vgpr44_vgpr45
                                        ; implicit-def: $sgpr28
	s_branch .LBB0_603
.LBB0_601:                              ;   in Loop: Header=BB0_573 Depth=2
	v_mov_b64_e32 v[44:45], 0
.LBB0_602:                              ; %Flow4150
                                        ;   in Loop: Header=BB0_573 Depth=2
	s_mov_b32 s28, 0
	s_cbranch_execnz .LBB0_604
.LBB0_603:                              ;   in Loop: Header=BB0_573 Depth=2
	global_load_dwordx2 v[44:45], v9, s[2:3]
	s_add_i32 s28, s5, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_604:                              ; %.loopexit25.i.i97
                                        ;   in Loop: Header=BB0_573 Depth=2
	s_cmp_gt_u32 s28, 7
	s_cbranch_scc1 .LBB0_608
; %bb.605:                              ;   in Loop: Header=BB0_573 Depth=2
	s_cmp_eq_u32 s28, 0
	s_cbranch_scc1 .LBB0_609
; %bb.606:                              ; %.preheader22.i.i98.preheader
                                        ;   in Loop: Header=BB0_573 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[46:47], 0
	s_mov_b64 s[26:27], 0
.LBB0_607:                              ; %.preheader22.i.i98
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_573 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s30, s2, s26
	s_addc_u32 s31, s3, s27
	global_load_ubyte v3, v9, s[30:31]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v46, v10, v46
	s_cmp_lg_u32 s28, s26
	v_or_b32_e32 v47, v11, v47
	s_cbranch_scc1 .LBB0_607
	s_branch .LBB0_610
.LBB0_608:                              ;   in Loop: Header=BB0_573 Depth=2
                                        ; implicit-def: $sgpr5
	s_branch .LBB0_611
.LBB0_609:                              ;   in Loop: Header=BB0_573 Depth=2
	v_mov_b64_e32 v[46:47], 0
.LBB0_610:                              ; %Flow4145
                                        ;   in Loop: Header=BB0_573 Depth=2
	s_mov_b32 s5, 0
	s_cbranch_execnz .LBB0_612
.LBB0_611:                              ;   in Loop: Header=BB0_573 Depth=2
	global_load_dwordx2 v[46:47], v9, s[2:3]
	s_add_i32 s5, s28, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_612:                              ; %.loopexit23.i.i99
                                        ;   in Loop: Header=BB0_573 Depth=2
	s_cmp_gt_u32 s5, 7
	s_cbranch_scc1 .LBB0_616
; %bb.613:                              ;   in Loop: Header=BB0_573 Depth=2
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_617
; %bb.614:                              ; %.preheader20.i.i100.preheader
                                        ;   in Loop: Header=BB0_573 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[48:49], 0
	s_mov_b64 s[26:27], 0
.LBB0_615:                              ; %.preheader20.i.i100
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_573 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s28, s2, s26
	s_addc_u32 s29, s3, s27
	global_load_ubyte v3, v9, s[28:29]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v48, v10, v48
	s_cmp_lg_u32 s5, s26
	v_or_b32_e32 v49, v11, v49
	s_cbranch_scc1 .LBB0_615
	s_branch .LBB0_618
.LBB0_616:                              ;   in Loop: Header=BB0_573 Depth=2
                                        ; implicit-def: $vgpr48_vgpr49
                                        ; implicit-def: $sgpr28
	s_branch .LBB0_619
.LBB0_617:                              ;   in Loop: Header=BB0_573 Depth=2
	v_mov_b64_e32 v[48:49], 0
.LBB0_618:                              ; %Flow4140
                                        ;   in Loop: Header=BB0_573 Depth=2
	s_mov_b32 s28, 0
	s_cbranch_execnz .LBB0_620
.LBB0_619:                              ;   in Loop: Header=BB0_573 Depth=2
	global_load_dwordx2 v[48:49], v9, s[2:3]
	s_add_i32 s28, s5, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_620:                              ; %.loopexit21.i.i101
                                        ;   in Loop: Header=BB0_573 Depth=2
	s_cmp_gt_u32 s28, 7
	s_cbranch_scc1 .LBB0_624
; %bb.621:                              ;   in Loop: Header=BB0_573 Depth=2
	s_cmp_eq_u32 s28, 0
	s_cbranch_scc1 .LBB0_625
; %bb.622:                              ; %.preheader.i.i102.preheader
                                        ;   in Loop: Header=BB0_573 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[50:51], 0
	s_mov_b64 s[26:27], s[2:3]
.LBB0_623:                              ; %.preheader.i.i102
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_573 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	global_load_ubyte v3, v9, s[26:27]
	s_add_i32 s28, s28, -1
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	v_or_b32_e32 v50, v10, v50
	s_cmp_lg_u32 s28, 0
	v_or_b32_e32 v51, v11, v51
	s_cbranch_scc1 .LBB0_623
	s_branch .LBB0_626
.LBB0_624:                              ;   in Loop: Header=BB0_573 Depth=2
	s_branch .LBB0_627
.LBB0_625:                              ;   in Loop: Header=BB0_573 Depth=2
	v_mov_b64_e32 v[50:51], 0
.LBB0_626:                              ; %Flow4135
                                        ;   in Loop: Header=BB0_573 Depth=2
	s_cbranch_execnz .LBB0_628
.LBB0_627:                              ;   in Loop: Header=BB0_573 Depth=2
	global_load_dwordx2 v[50:51], v9, s[2:3]
.LBB0_628:                              ; %.loopexit.i.i103
                                        ;   in Loop: Header=BB0_573 Depth=2
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[10:11], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_634
; %bb.629:                              ;   in Loop: Header=BB0_573 Depth=2
	global_load_dwordx2 v[54:55], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v3, v10, v54
	v_and_b32_e32 v7, v11, v55
	v_mul_lo_u32 v7, v7, 24
	v_mul_hi_u32 v8, v3, 24
	v_add_u32_e32 v11, v8, v7
	v_mul_lo_u32 v10, v3, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[14:15], 0, v[10:11]
	global_load_dwordx2 v[52:53], v[10:11], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[52:55], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[10:11], v[54:55]
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB0_633
; %bb.630:                              ; %.preheader3.i.i18.i.i118.preheader
                                        ;   in Loop: Header=BB0_573 Depth=2
	s_mov_b64 s[28:29], 0
.LBB0_631:                              ; %.preheader3.i.i18.i.i118
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_573 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_sleep 1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	global_load_dwordx2 v[18:19], v9, s[16:17]
	v_mov_b64_e32 v[54:55], v[10:11]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v7, v14, v54
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[10:11], s[30:31], v7, 24, v[18:19]
	v_and_b32_e32 v3, v15, v55
	v_mov_b32_e32 v8, v11
	v_mad_u64_u32 v[14:15], s[30:31], v3, 24, v[8:9]
	v_mov_b32_e32 v11, v14
	global_load_dwordx2 v[52:53], v[10:11], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[52:55], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[10:11], v[54:55]
	s_or_b64 s[28:29], vcc, s[28:29]
	s_andn2_b64 exec, exec, s[28:29]
	s_cbranch_execnz .LBB0_631
; %bb.632:                              ; %Flow4130
                                        ;   in Loop: Header=BB0_573 Depth=2
	s_or_b64 exec, exec, s[28:29]
.LBB0_633:                              ; %Flow4132
                                        ;   in Loop: Header=BB0_573 Depth=2
	s_or_b64 exec, exec, s[26:27]
.LBB0_634:                              ; %.loopexit4.i.i13.i.i104
                                        ;   in Loop: Header=BB0_573 Depth=2
	s_or_b64 exec, exec, s[24:25]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	global_load_dwordx4 v[52:55], v9, s[16:17]
	v_readfirstlane_b32 s24, v10
	v_readfirstlane_b32 s25, v11
	s_mov_b64 s[26:27], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s28, v14
	v_readfirstlane_b32 s29, v15
	s_and_b64 s[28:29], s[24:25], s[28:29]
	s_mul_i32 s5, s29, 24
	s_mul_hi_u32 s30, s28, 24
	s_add_i32 s31, s30, s5
	s_mul_i32 s30, s28, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[52:53], 0, s[30:31]
	s_and_saveexec_b64 s[30:31], s[2:3]
	s_cbranch_execz .LBB0_636
; %bb.635:                              ;   in Loop: Header=BB0_573 Depth=2
	v_mov_b64_e32 v[22:23], s[26:27]
	global_store_dwordx4 v[10:11], v[22:25], off offset:8
.LBB0_636:                              ;   in Loop: Header=BB0_573 Depth=2
	s_or_b64 exec, exec, s[30:31]
	v_or_b32_e32 v3, 0, v37
	v_or_b32_e32 v7, v36, v2
	v_cmp_gt_u64_e64 vcc, s[20:21], 56
	s_lshl_b32 s5, s22, 2
	s_lshl_b64 s[26:27], s[28:29], 12
	v_cndmask_b32_e32 v37, v3, v37, vcc
	v_cndmask_b32_e32 v3, v7, v36, vcc
	s_add_i32 s5, s5, 28
	v_lshl_add_u64 v[14:15], v[54:55], 0, s[26:27]
	s_and_b32 s5, s5, 0x1e0
	v_and_b32_e32 v3, 0xffffff1f, v3
	v_or_b32_e32 v36, s5, v3
	v_readfirstlane_b32 s26, v14
	v_readfirstlane_b32 s27, v15
	s_nop 4
	global_store_dwordx4 v58, v[36:39], s[26:27]
	global_store_dwordx4 v58, v[40:43], s[26:27] offset:16
	global_store_dwordx4 v58, v[44:47], s[26:27] offset:32
	global_store_dwordx4 v58, v[48:51], s[26:27] offset:48
	s_and_saveexec_b64 s[26:27], s[2:3]
	s_cbranch_execz .LBB0_644
; %bb.637:                              ;   in Loop: Header=BB0_573 Depth=2
	global_load_dwordx2 v[40:41], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[18:19], v9, s[16:17] offset:40
	v_mov_b32_e32 v38, s24
	v_mov_b32_e32 v39, s25
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s28, v18
	v_readfirstlane_b32 s29, v19
	s_and_b64 s[28:29], s[28:29], s[24:25]
	s_mul_i32 s5, s29, 24
	s_mul_hi_u32 s29, s28, 24
	s_mul_i32 s28, s28, 24
	s_add_i32 s29, s29, s5
	v_lshl_add_u64 v[18:19], v[52:53], 0, s[28:29]
	global_store_dwordx2 v[18:19], v[40:41], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v9, v[38:41], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[40:41]
	s_and_saveexec_b64 s[28:29], vcc
	s_cbranch_execz .LBB0_640
; %bb.638:                              ; %.preheader1.i.i16.i.i116.preheader
                                        ;   in Loop: Header=BB0_573 Depth=2
	s_mov_b64 s[30:31], 0
.LBB0_639:                              ; %.preheader1.i.i16.i.i116
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_573 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_sleep 1
	global_store_dwordx2 v[18:19], v[38:39], off
	v_mov_b32_e32 v36, s24
	v_mov_b32_e32 v37, s25
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[22:23], v9, v[36:39], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[22:23], v[38:39]
	s_or_b64 s[30:31], vcc, s[30:31]
	v_mov_b64_e32 v[38:39], v[22:23]
	s_andn2_b64 exec, exec, s[30:31]
	s_cbranch_execnz .LBB0_639
.LBB0_640:                              ; %Flow4128
                                        ;   in Loop: Header=BB0_573 Depth=2
	s_or_b64 exec, exec, s[28:29]
	global_load_dwordx2 v[18:19], v9, s[16:17] offset:16
	s_mov_b64 s[30:31], exec
	v_mbcnt_lo_u32_b32 v3, s30, 0
	v_mbcnt_hi_u32_b32 v3, s31, v3
	v_cmp_eq_u32_e32 vcc, 0, v3
	s_and_saveexec_b64 s[28:29], vcc
	s_cbranch_execz .LBB0_642
; %bb.641:                              ;   in Loop: Header=BB0_573 Depth=2
	s_bcnt1_i32_b64 s5, s[30:31]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[18:19], v[8:9], off offset:8 sc1
.LBB0_642:                              ;   in Loop: Header=BB0_573 Depth=2
	s_or_b64 exec, exec, s[28:29]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[22:23], v[18:19], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_cbranch_vccnz .LBB0_644
; %bb.643:                              ;   in Loop: Header=BB0_573 Depth=2
	global_load_dword v8, v[18:19], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v3, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v3
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[22:23], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_644:                              ; %Flow4129
                                        ;   in Loop: Header=BB0_573 Depth=2
	s_or_b64 exec, exec, s[26:27]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[14:15], v[14:15], 0, v[58:59]
	s_branch .LBB0_648
.LBB0_645:                              ;   in Loop: Header=BB0_648 Depth=3
	s_or_b64 exec, exec, s[26:27]
	v_readfirstlane_b32 s5, v3
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_647
; %bb.646:                              ;   in Loop: Header=BB0_648 Depth=3
	s_sleep 1
	s_cbranch_execnz .LBB0_648
	s_branch .LBB0_650
.LBB0_647:                              ;   in Loop: Header=BB0_573 Depth=2
	s_branch .LBB0_650
.LBB0_648:                              ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_573 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_mov_b32_e32 v3, 1
	s_and_saveexec_b64 s[26:27], s[2:3]
	s_cbranch_execz .LBB0_645
; %bb.649:                              ;   in Loop: Header=BB0_648 Depth=3
	global_load_dword v3, v[10:11], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v3, 1, v3
	s_branch .LBB0_645
.LBB0_650:                              ;   in Loop: Header=BB0_573 Depth=2
	global_load_dwordx4 v[36:39], v[14:15], off
	s_and_saveexec_b64 s[26:27], s[2:3]
	s_cbranch_execz .LBB0_572
; %bb.651:                              ;   in Loop: Header=BB0_573 Depth=2
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[18:19], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[22:23], v[10:11], 0, 1
	v_lshl_add_u64 v[26:27], v[22:23], 0, s[24:25]
	v_cmp_eq_u64_e32 vcc, 0, v[26:27]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v40, v14
	v_mov_b32_e32 v41, v15
	v_cndmask_b32_e32 v39, v27, v23, vcc
	v_cndmask_b32_e32 v38, v26, v22, vcc
	v_and_b32_e32 v3, v39, v11
	v_and_b32_e32 v7, v38, v10
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v8, v7, 24
	v_mul_lo_u32 v10, v7, 24
	v_add_u32_e32 v11, v8, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[18:19], 0, v[10:11]
	global_store_dwordx2 v[10:11], v[14:15], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[40:41], v9, v[38:41], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[40:41], v[14:15]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_572
; %bb.652:                              ; %.preheader.i.i15.i.i115.preheader
                                        ;   in Loop: Header=BB0_573 Depth=2
	s_mov_b64 s[2:3], 0
.LBB0_653:                              ; %.preheader.i.i15.i.i115
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_573 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_sleep 1
	global_store_dwordx2 v[10:11], v[40:41], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[14:15], v9, v[38:41], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[14:15], v[40:41]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[40:41], v[14:15]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_653
	s_branch .LBB0_572
.LBB0_654:                              ; %Flow4168
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_683
.LBB0_655:                              ;   in Loop: Header=BB0_273 Depth=1
                                        ; implicit-def: $vgpr36_vgpr37
	s_cbranch_execz .LBB0_683
; %bb.656:                              ;   in Loop: Header=BB0_273 Depth=1
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_662
; %bb.657:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[38:39], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v38
	v_and_b32_e32 v3, v3, v39
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[36:37], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[36:39], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[38:39]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_661
; %bb.658:                              ; %.preheader3.i.i.i38.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_659:                              ; %.preheader3.i.i.i38.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[38:39], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v38
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v39
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[36:37], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[36:39], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[38:39]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_659
; %bb.660:                              ; %Flow4181
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_661:                              ; %Flow4183
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_662:                              ; %.loopexit4.i.i.i33.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[38:41], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[38:39], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_664
; %bb.663:                              ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_664:                              ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[40:41], 0, s[6:7]
	v_and_or_b32 v34, v34, s33, 32
	v_mov_b32_e32 v36, v9
	v_mov_b32_e32 v37, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	s_nop 1
	global_store_dwordx4 v58, v[34:37], s[22:23]
	s_nop 1
	v_mov_b64_e32 v[36:37], s[6:7]
	v_mov_b64_e32 v[34:35], s[4:5]
	global_store_dwordx4 v58, v[34:37], s[22:23] offset:16
	global_store_dwordx4 v58, v[34:37], s[22:23] offset:32
	global_store_dwordx4 v58, v[34:37], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_672
; %bb.665:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[42:43], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v40, s20
	v_mov_b32_e32 v41, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[38:39], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[42:43], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[40:43], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[42:43]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_668
; %bb.666:                              ; %.preheader1.i.i.i36.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_667:                              ; %.preheader1.i.i.i36.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[36:37], off
	v_mov_b32_e32 v34, s20
	v_mov_b32_e32 v35, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[34:37], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[36:37]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[36:37], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_667
.LBB0_668:                              ; %Flow4179
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_670
; %bb.669:                              ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_670:                              ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_672
; %bb.671:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_672:                              ; %Flow4180
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_676
.LBB0_673:                              ;   in Loop: Header=BB0_676 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_675
; %bb.674:                              ;   in Loop: Header=BB0_676 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_676
	s_branch .LBB0_678
.LBB0_675:                              ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_678
.LBB0_676:                              ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_673
; %bb.677:                              ;   in Loop: Header=BB0_676 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_673
.LBB0_678:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[36:37], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_682
; %bb.679:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[18:19], v[2:3], 0, 1
	v_lshl_add_u64 v[22:23], v[18:19], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v40, v10
	v_mov_b32_e32 v41, v11
	v_cndmask_b32_e32 v39, v23, v19, vcc
	v_cndmask_b32_e32 v38, v22, v18, vcc
	v_and_b32_e32 v3, v39, v3
	v_and_b32_e32 v2, v38, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[40:41], v9, v[38:41], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[40:41], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_682
; %bb.680:                              ; %.preheader.i.i.i35.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_681:                              ; %.preheader.i.i.i35.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[40:41], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[38:41], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[40:41]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[40:41], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_681
.LBB0_682:                              ; %Flow4172
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
.LBB0_683:                              ; %__ockl_printf_append_string_n.exit.i107
                                        ;   in Loop: Header=BB0_273 Depth=1
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_689
; %bb.684:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[40:41], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v40
	v_and_b32_e32 v3, v3, v41
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[38:39], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[38:41], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[40:41]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_688
; %bb.685:                              ; %.preheader3.i.i.i45.i114.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_686:                              ; %.preheader3.i.i.i45.i114
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[40:41], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v40
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v41
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[38:39], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[38:41], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[40:41]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_686
; %bb.687:                              ; %Flow4116
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_688:                              ; %Flow4118
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_689:                              ; %.loopexit4.i.i.i39.i108
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[40:43], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[40:41], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_691
; %bb.690:                              ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_691:                              ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[42:43], 0, s[6:7]
	v_and_or_b32 v36, v36, s33, 32
	v_mov_b32_e32 v38, v0
	v_mov_b32_e32 v39, v1
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	s_nop 1
	global_store_dwordx4 v58, v[36:39], s[22:23]
	s_nop 1
	v_mov_b64_e32 v[36:37], s[6:7]
	v_mov_b64_e32 v[34:35], s[4:5]
	global_store_dwordx4 v58, v[34:37], s[22:23] offset:16
	global_store_dwordx4 v58, v[34:37], s[22:23] offset:32
	global_store_dwordx4 v58, v[34:37], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_699
; %bb.692:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[38:39], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v36, s20
	v_mov_b32_e32 v37, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[40:41], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[38:39], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[36:39], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[38:39]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_695
; %bb.693:                              ; %.preheader1.i.i.i43.i112.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_694:                              ; %.preheader1.i.i.i43.i112
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[36:37], off
	v_mov_b32_e32 v34, s20
	v_mov_b32_e32 v35, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[34:37], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[36:37]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[36:37], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_694
.LBB0_695:                              ; %Flow4114
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_697
; %bb.696:                              ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_697:                              ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_699
; %bb.698:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_699:                              ; %Flow4115
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_703
.LBB0_700:                              ;   in Loop: Header=BB0_703 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_702
; %bb.701:                              ;   in Loop: Header=BB0_703 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_703
	s_branch .LBB0_705
.LBB0_702:                              ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_705
.LBB0_703:                              ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_700
; %bb.704:                              ;   in Loop: Header=BB0_703 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_700
.LBB0_705:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[26:27], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_709
; %bb.706:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[18:19], v[2:3], 0, 1
	v_lshl_add_u64 v[22:23], v[18:19], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v36, v10
	v_mov_b32_e32 v37, v11
	v_cndmask_b32_e32 v35, v23, v19, vcc
	v_cndmask_b32_e32 v34, v22, v18, vcc
	v_and_b32_e32 v3, v35, v3
	v_and_b32_e32 v2, v34, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_709
; %bb.707:                              ; %.preheader.i.i.i42.i111.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_708:                              ; %.preheader.i.i.i42.i111
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[36:37], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[36:37]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[36:37], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_708
.LBB0_709:                              ; %__ockl_printf_append_args.exit.i110
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_715
; %bb.710:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[36:37], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v36
	v_and_b32_e32 v3, v3, v37
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[34:35], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[36:37]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_714
; %bb.711:                              ; %.preheader3.i.i.i52.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_712:                              ; %.preheader3.i.i.i52.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[36:37], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v36
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v37
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[34:35], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[36:37]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_712
; %bb.713:                              ; %Flow4102
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_714:                              ; %Flow4104
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_715:                              ; %.loopexit4.i.i.i46.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[34:37], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[34:35], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_717
; %bb.716:                              ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_717:                              ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[36:37], 0, s[6:7]
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	v_mov_b64_e32 v[38:39], s[6:7]
	v_and_or_b32 v26, v26, s33, 32
	v_mov_b32_e32 v29, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	v_mov_b64_e32 v[36:37], s[4:5]
	s_nop 3
	global_store_dwordx4 v58, v[26:29], s[22:23]
	global_store_dwordx4 v58, v[36:39], s[22:23] offset:16
	global_store_dwordx4 v58, v[36:39], s[22:23] offset:32
	global_store_dwordx4 v58, v[36:39], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_725
; %bb.718:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[38:39], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v36, s20
	v_mov_b32_e32 v37, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[34:35], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[38:39], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[36:39], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[38:39]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_721
; %bb.719:                              ; %.preheader1.i.i.i50.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_720:                              ; %.preheader1.i.i.i50.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[36:37], off
	v_mov_b32_e32 v34, s20
	v_mov_b32_e32 v35, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[34:37], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[36:37]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[36:37], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_720
.LBB0_721:                              ; %Flow4100
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_723
; %bb.722:                              ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_723:                              ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_725
; %bb.724:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_725:                              ; %Flow4101
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_729
.LBB0_726:                              ;   in Loop: Header=BB0_729 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_728
; %bb.727:                              ;   in Loop: Header=BB0_729 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_729
	s_branch .LBB0_731
.LBB0_728:                              ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_731
.LBB0_729:                              ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_726
; %bb.730:                              ;   in Loop: Header=BB0_729 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_726
.LBB0_731:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_735
; %bb.732:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[18:19], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[22:23], v[10:11], 0, 1
	v_lshl_add_u64 v[26:27], v[22:23], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[26:27]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v36, v14
	v_mov_b32_e32 v37, v15
	v_cndmask_b32_e32 v35, v27, v23, vcc
	v_cndmask_b32_e32 v34, v26, v22, vcc
	v_and_b32_e32 v7, v35, v11
	v_and_b32_e32 v8, v34, v10
	v_mul_lo_u32 v7, v7, 24
	v_mul_hi_u32 v11, v8, 24
	v_mul_lo_u32 v10, v8, 24
	v_add_u32_e32 v11, v11, v7
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[18:19], 0, v[10:11]
	global_store_dwordx2 v[10:11], v[14:15], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[14:15]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_735
; %bb.733:                              ; %.preheader.i.i.i49.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_734:                              ; %.preheader.i.i.i49.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[10:11], v[36:37], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[14:15], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[14:15], v[36:37]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[36:37], v[14:15]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_734
.LBB0_735:                              ; %__ockl_printf_append_args.exit53.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[10:11], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_741
; %bb.736:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[36:37], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v7, v10, v36
	v_and_b32_e32 v8, v11, v37
	v_mul_lo_u32 v8, v8, 24
	v_mul_hi_u32 v10, v7, 24
	v_add_u32_e32 v11, v10, v8
	v_mul_lo_u32 v10, v7, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[14:15], 0, v[10:11]
	global_load_dwordx2 v[34:35], v[10:11], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[10:11], v[36:37]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_740
; %bb.737:                              ; %.preheader3.i.i.i60.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_738:                              ; %.preheader3.i.i.i60.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	global_load_dwordx2 v[18:19], v9, s[16:17]
	v_mov_b64_e32 v[36:37], v[10:11]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v8, v14, v36
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[10:11], s[24:25], v8, 24, v[18:19]
	v_and_b32_e32 v7, v15, v37
	v_mov_b32_e32 v8, v11
	v_mad_u64_u32 v[14:15], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v11, v14
	global_load_dwordx2 v[34:35], v[10:11], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[10:11], v[36:37]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_738
; %bb.739:                              ; %Flow4088
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_740:                              ; %Flow4090
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_741:                              ; %.loopexit4.i.i.i54.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	global_load_dwordx4 v[34:37], v9, s[16:17]
	v_readfirstlane_b32 s20, v10
	v_readfirstlane_b32 s21, v11
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[34:35], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_743
; %bb.742:                              ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[10:11], v[22:25], off offset:8
.LBB0_743:                              ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[14:15], v[36:37], 0, s[6:7]
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	v_mov_b64_e32 v[38:39], s[6:7]
	v_and_or_b32 v2, v2, s33, 32
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	v_mov_b64_e32 v[36:37], s[4:5]
	s_nop 3
	global_store_dwordx4 v58, v[2:5], s[22:23]
	global_store_dwordx4 v58, v[36:39], s[22:23] offset:16
	global_store_dwordx4 v58, v[36:39], s[22:23] offset:32
	global_store_dwordx4 v58, v[36:39], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_751
; %bb.744:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[38:39], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	v_mov_b32_e32 v36, s20
	v_mov_b32_e32 v37, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v2
	v_readfirstlane_b32 s23, v3
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[2:3], v[34:35], 0, s[22:23]
	global_store_dwordx2 v[2:3], v[38:39], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[36:39], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[38:39]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_747
; %bb.745:                              ; %.preheader1.i.i.i58.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_746:                              ; %.preheader1.i.i.i58.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[36:37], off
	v_mov_b32_e32 v34, s20
	v_mov_b32_e32 v35, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[34:37], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[36:37]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[36:37], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_746
.LBB0_747:                              ; %Flow4086
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_749
; %bb.748:                              ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[2:3], v[8:9], off offset:8 sc1
.LBB0_749:                              ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[2:3], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_751
; %bb.750:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[2:3], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v2, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v2
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_751:                              ; %Flow4087
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[58:59]
	s_branch .LBB0_755
.LBB0_752:                              ;   in Loop: Header=BB0_755 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_754
; %bb.753:                              ;   in Loop: Header=BB0_755 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_755
	s_branch .LBB0_757
.LBB0_754:                              ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_757
.LBB0_755:                              ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_752
; %bb.756:                              ;   in Loop: Header=BB0_755 Depth=2
	global_load_dword v7, v[10:11], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_752
.LBB0_757:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[30:31], v[2:3], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_761
; %bb.758:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[18:19], v[2:3], 0, 1
	v_lshl_add_u64 v[22:23], v[18:19], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v36, v10
	v_mov_b32_e32 v37, v11
	v_cndmask_b32_e32 v35, v23, v19, vcc
	v_cndmask_b32_e32 v34, v22, v18, vcc
	v_and_b32_e32 v3, v35, v3
	v_and_b32_e32 v2, v34, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_761
; %bb.759:                              ; %.preheader.i.i.i57.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_760:                              ; %.preheader.i.i.i57.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[36:37], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[36:37]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[36:37], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_760
.LBB0_761:                              ; %__ockl_printf_append_args.exit61.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_767
; %bb.762:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[36:37], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v36
	v_and_b32_e32 v3, v3, v37
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[34:35], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[36:37]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_766
; %bb.763:                              ; %.preheader3.i.i.i68.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_764:                              ; %.preheader3.i.i.i68.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[36:37], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v36
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v37
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[34:35], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[36:37]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_764
; %bb.765:                              ; %Flow4074
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_766:                              ; %Flow4076
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_767:                              ; %.loopexit4.i.i.i62.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[34:37], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[34:35], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_769
; %bb.768:                              ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_769:                              ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[36:37], 0, s[6:7]
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	v_mov_b64_e32 v[38:39], s[6:7]
	v_and_or_b32 v30, v30, s34, 34
	v_mov_b32_e32 v33, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	v_mov_b64_e32 v[36:37], s[4:5]
	s_nop 3
	global_store_dwordx4 v58, v[30:33], s[22:23]
	global_store_dwordx4 v58, v[36:39], s[22:23] offset:16
	global_store_dwordx4 v58, v[36:39], s[22:23] offset:32
	global_store_dwordx4 v58, v[36:39], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_777
; %bb.770:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[38:39], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	v_mov_b32_e32 v36, s20
	v_mov_b32_e32 v37, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[10:11], v[34:35], 0, s[22:23]
	global_store_dwordx2 v[10:11], v[38:39], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[36:39], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[38:39]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_773
; %bb.771:                              ; %.preheader1.i.i.i66.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_772:                              ; %.preheader1.i.i.i66.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[10:11], v[36:37], off
	v_mov_b32_e32 v34, s20
	v_mov_b32_e32 v35, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[14:15], v9, v[34:37], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[14:15], v[36:37]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[36:37], v[14:15]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_772
.LBB0_773:                              ; %Flow4072
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_775
; %bb.774:                              ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[10:11], v[8:9], off offset:8 sc1
.LBB0_775:                              ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[14:15], v[10:11], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[14:15]
	s_cbranch_vccnz .LBB0_777
; %bb.776:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[10:11], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[14:15], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_777:                              ; %Flow4073
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	s_branch .LBB0_781
.LBB0_778:                              ;   in Loop: Header=BB0_781 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_780
; %bb.779:                              ;   in Loop: Header=BB0_781 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_781
	s_branch .LBB0_783
.LBB0_780:                              ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_783
.LBB0_781:                              ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_778
; %bb.782:                              ;   in Loop: Header=BB0_781 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_778
.LBB0_783:                              ;   in Loop: Header=BB0_273 Depth=1
	s_and_b64 exec, exec, s[2:3]
	s_cbranch_execz .LBB0_787
; %bb.784:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[18:19], v[2:3], 0, 1
	v_lshl_add_u64 v[22:23], v[18:19], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v36, v10
	v_mov_b32_e32 v37, v11
	v_cndmask_b32_e32 v35, v23, v19, vcc
	v_cndmask_b32_e32 v34, v22, v18, vcc
	v_and_b32_e32 v3, v35, v3
	v_and_b32_e32 v2, v34, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_787
; %bb.785:                              ; %.preheader.i.i.i65.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_786:                              ; %.preheader.i.i.i65.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[36:37], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[36:37]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[36:37], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_786
.LBB0_787:                              ; %Flow4200
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[18:19]
	s_lshl_b32 s37, s36, 5
	v_add_u32_e32 v26, v63, v61
	;;#ASMSTART
	ds_read_b128 v[30:33], v32
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	scratch_store_dwordx4 off, v[30:33], s37
	s_and_saveexec_b64 s[18:19], s[0:1]
	s_cbranch_execz .LBB0_1057
; %bb.788:                              ; %if.then.i.1.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_794
; %bb.789:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[32:33], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v32
	v_and_b32_e32 v3, v3, v33
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[32:33]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_793
; %bb.790:                              ; %.preheader3.i.i.i.i.1.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_791:                              ; %.preheader3.i.i.i.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[32:33], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v32
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v33
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[32:33]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_791
; %bb.792:                              ; %Flow4058
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_793:                              ; %Flow4060
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_794:                              ; %.loopexit4.i.i.i.i.1.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[30:33], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[30:31], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_796
; %bb.795:                              ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_796:                              ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[32:33], 0, s[6:7]
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	v_mov_b64_e32 v[34:35], s[6:7]
	v_mov_b32_e32 v7, v9
	v_mov_b32_e32 v8, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	v_mov_b64_e32 v[32:33], s[4:5]
	s_nop 3
	global_store_dwordx4 v58, v[6:9], s[22:23]
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:16
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:32
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_804
; %bb.797:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[34:35], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v32, s20
	v_mov_b32_e32 v33, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[30:31], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[34:35], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[32:35], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[34:35]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_800
; %bb.798:                              ; %.preheader1.i.i.i.i.1.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_799:                              ; %.preheader1.i.i.i.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[32:33], off
	v_mov_b32_e32 v30, s20
	v_mov_b32_e32 v31, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[30:33], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[32:33]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[32:33], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_799
.LBB0_800:                              ; %Flow4056
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_802
; %bb.801:                              ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_802:                              ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_804
; %bb.803:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_804:                              ; %Flow4057
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_808
.LBB0_805:                              ;   in Loop: Header=BB0_808 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_807
; %bb.806:                              ;   in Loop: Header=BB0_808 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_808
	s_branch .LBB0_810
.LBB0_807:                              ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_810
.LBB0_808:                              ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_805
; %bb.809:                              ;   in Loop: Header=BB0_808 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_805
.LBB0_810:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[30:31], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_814
; %bb.811:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[18:19], v[2:3], 0, 1
	v_lshl_add_u64 v[22:23], v[18:19], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v34, v10
	v_mov_b32_e32 v35, v11
	v_cndmask_b32_e32 v33, v23, v19, vcc
	v_cndmask_b32_e32 v32, v22, v18, vcc
	v_and_b32_e32 v3, v33, v3
	v_and_b32_e32 v2, v32, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[34:35], v9, v[32:35], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[34:35], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_814
; %bb.812:                              ; %.preheader.i.i.i.i.1.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_813:                              ; %.preheader.i.i.i.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[34:35], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[32:35], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[34:35]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[34:35], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_813
.LBB0_814:                              ; %__ockl_printf_begin.exit.i.1.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	s_andn2_b64 vcc, exec, s[8:9]
	s_cbranch_vccnz .LBB0_899
; %bb.815:                              ;   in Loop: Header=BB0_273 Depth=1
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v2, 2, v30
	v_and_b32_e32 v32, -3, v30
	v_mov_b32_e32 v33, v31
	s_mov_b64 s[20:21], 0x43
	s_getpc_b64 s[6:7]
	s_add_u32 s6, s6, .str.3@rel32@lo+4
	s_addc_u32 s7, s7, .str.3@rel32@hi+12
	s_branch .LBB0_817
.LBB0_816:                              ; %__ockl_hostcall_preview.exit19.i.i.1.i
                                        ;   in Loop: Header=BB0_817 Depth=2
	s_or_b64 exec, exec, s[26:27]
	s_sub_u32 s20, s20, s22
	s_subb_u32 s21, s21, s23
	s_add_u32 s6, s6, s22
	s_addc_u32 s7, s7, s23
	s_cmp_eq_u64 s[20:21], 0
	s_cbranch_scc1 .LBB0_898
.LBB0_817:                              ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_820 Depth 3
                                        ;       Child Loop BB0_827 Depth 3
                                        ;       Child Loop BB0_835 Depth 3
                                        ;       Child Loop BB0_843 Depth 3
                                        ;       Child Loop BB0_851 Depth 3
                                        ;       Child Loop BB0_859 Depth 3
                                        ;       Child Loop BB0_867 Depth 3
                                        ;       Child Loop BB0_875 Depth 3
                                        ;       Child Loop BB0_883 Depth 3
                                        ;       Child Loop BB0_892 Depth 3
                                        ;       Child Loop BB0_897 Depth 3
	v_cmp_lt_u64_e64 s[2:3], s[20:21], 56
	s_and_b64 s[2:3], s[2:3], exec
	v_cmp_gt_u64_e64 s[2:3], s[20:21], 7
	s_cselect_b32 s23, s21, 0
	s_cselect_b32 s22, s20, 56
	s_and_b64 vcc, exec, s[2:3]
	s_cbranch_vccnz .LBB0_822
; %bb.818:                              ;   in Loop: Header=BB0_817 Depth=2
	s_mov_b64 s[2:3], 0
	s_cmp_eq_u64 s[20:21], 0
	s_waitcnt vmcnt(0)
	v_mov_b64_e32 v[34:35], 0
	s_cbranch_scc1 .LBB0_821
; %bb.819:                              ; %.preheader30.i.i.1.i.preheader
                                        ;   in Loop: Header=BB0_817 Depth=2
	s_lshl_b64 s[24:25], s[22:23], 3
	s_mov_b64 s[26:27], 0
	v_mov_b64_e32 v[34:35], 0
	s_mov_b64 s[28:29], s[6:7]
.LBB0_820:                              ; %.preheader30.i.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_817 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	global_load_ubyte v3, v9, s[28:29]
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s26, v[8:9]
	s_add_u32 s26, s26, 8
	s_addc_u32 s27, s27, 0
	s_add_u32 s28, s28, 1
	s_addc_u32 s29, s29, 0
	v_or_b32_e32 v34, v10, v34
	s_cmp_eq_u32 s24, s26
	v_or_b32_e32 v35, v11, v35
	s_cbranch_scc0 .LBB0_820
.LBB0_821:                              ; %Flow4026
                                        ;   in Loop: Header=BB0_817 Depth=2
	s_mov_b32 s5, 0
	s_andn2_b64 vcc, exec, s[2:3]
	s_mov_b64 s[2:3], s[6:7]
	s_cbranch_vccz .LBB0_823
	s_branch .LBB0_824
.LBB0_822:                              ;   in Loop: Header=BB0_817 Depth=2
                                        ; implicit-def: $vgpr34_vgpr35
                                        ; implicit-def: $sgpr5
	s_mov_b64 s[2:3], s[6:7]
.LBB0_823:                              ;   in Loop: Header=BB0_817 Depth=2
	global_load_dwordx2 v[34:35], v9, s[6:7]
	s_add_i32 s5, s22, -8
	s_add_u32 s2, s6, 8
	s_addc_u32 s3, s7, 0
.LBB0_824:                              ; %.loopexit31.i.i.1.i
                                        ;   in Loop: Header=BB0_817 Depth=2
	s_cmp_gt_u32 s5, 7
	s_cbranch_scc1 .LBB0_828
; %bb.825:                              ;   in Loop: Header=BB0_817 Depth=2
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_829
; %bb.826:                              ; %.preheader28.i.i.1.i.preheader
                                        ;   in Loop: Header=BB0_817 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[36:37], 0
	s_mov_b64 s[26:27], 0
.LBB0_827:                              ; %.preheader28.i.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_817 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s28, s2, s26
	s_addc_u32 s29, s3, s27
	global_load_ubyte v3, v9, s[28:29]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v36, v10, v36
	s_cmp_eq_u32 s5, s26
	v_or_b32_e32 v37, v11, v37
	s_cbranch_scc0 .LBB0_827
	s_branch .LBB0_830
.LBB0_828:                              ;   in Loop: Header=BB0_817 Depth=2
                                        ; implicit-def: $vgpr36_vgpr37
                                        ; implicit-def: $sgpr28
	s_branch .LBB0_831
.LBB0_829:                              ;   in Loop: Header=BB0_817 Depth=2
	v_mov_b64_e32 v[36:37], 0
.LBB0_830:                              ; %Flow4021
                                        ;   in Loop: Header=BB0_817 Depth=2
	s_mov_b32 s28, 0
	s_cbranch_execnz .LBB0_832
.LBB0_831:                              ;   in Loop: Header=BB0_817 Depth=2
	global_load_dwordx2 v[36:37], v9, s[2:3]
	s_add_i32 s28, s5, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_832:                              ; %.loopexit29.i.i.1.i
                                        ;   in Loop: Header=BB0_817 Depth=2
	s_cmp_gt_u32 s28, 7
	s_cbranch_scc1 .LBB0_836
; %bb.833:                              ;   in Loop: Header=BB0_817 Depth=2
	s_cmp_eq_u32 s28, 0
	s_cbranch_scc1 .LBB0_837
; %bb.834:                              ; %.preheader26.i.i.1.i.preheader
                                        ;   in Loop: Header=BB0_817 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[38:39], 0
	s_mov_b64 s[26:27], 0
.LBB0_835:                              ; %.preheader26.i.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_817 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s30, s2, s26
	s_addc_u32 s31, s3, s27
	global_load_ubyte v3, v9, s[30:31]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v38, v10, v38
	s_cmp_eq_u32 s28, s26
	v_or_b32_e32 v39, v11, v39
	s_cbranch_scc0 .LBB0_835
	s_branch .LBB0_838
.LBB0_836:                              ;   in Loop: Header=BB0_817 Depth=2
                                        ; implicit-def: $sgpr5
	s_branch .LBB0_839
.LBB0_837:                              ;   in Loop: Header=BB0_817 Depth=2
	v_mov_b64_e32 v[38:39], 0
.LBB0_838:                              ; %Flow4016
                                        ;   in Loop: Header=BB0_817 Depth=2
	s_mov_b32 s5, 0
	s_cbranch_execnz .LBB0_840
.LBB0_839:                              ;   in Loop: Header=BB0_817 Depth=2
	global_load_dwordx2 v[38:39], v9, s[2:3]
	s_add_i32 s5, s28, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_840:                              ; %.loopexit27.i.i.1.i
                                        ;   in Loop: Header=BB0_817 Depth=2
	s_cmp_gt_u32 s5, 7
	s_cbranch_scc1 .LBB0_844
; %bb.841:                              ;   in Loop: Header=BB0_817 Depth=2
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_845
; %bb.842:                              ; %.preheader24.i.i.1.i.preheader
                                        ;   in Loop: Header=BB0_817 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[40:41], 0
	s_mov_b64 s[26:27], 0
.LBB0_843:                              ; %.preheader24.i.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_817 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s28, s2, s26
	s_addc_u32 s29, s3, s27
	global_load_ubyte v3, v9, s[28:29]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v40, v10, v40
	s_cmp_eq_u32 s5, s26
	v_or_b32_e32 v41, v11, v41
	s_cbranch_scc0 .LBB0_843
	s_branch .LBB0_846
.LBB0_844:                              ;   in Loop: Header=BB0_817 Depth=2
                                        ; implicit-def: $vgpr40_vgpr41
                                        ; implicit-def: $sgpr28
	s_branch .LBB0_847
.LBB0_845:                              ;   in Loop: Header=BB0_817 Depth=2
	v_mov_b64_e32 v[40:41], 0
.LBB0_846:                              ; %Flow4011
                                        ;   in Loop: Header=BB0_817 Depth=2
	s_mov_b32 s28, 0
	s_cbranch_execnz .LBB0_848
.LBB0_847:                              ;   in Loop: Header=BB0_817 Depth=2
	global_load_dwordx2 v[40:41], v9, s[2:3]
	s_add_i32 s28, s5, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_848:                              ; %.loopexit25.i.i.1.i
                                        ;   in Loop: Header=BB0_817 Depth=2
	s_cmp_gt_u32 s28, 7
	s_cbranch_scc1 .LBB0_852
; %bb.849:                              ;   in Loop: Header=BB0_817 Depth=2
	s_cmp_eq_u32 s28, 0
	s_cbranch_scc1 .LBB0_853
; %bb.850:                              ; %.preheader22.i.i.1.i.preheader
                                        ;   in Loop: Header=BB0_817 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[42:43], 0
	s_mov_b64 s[26:27], 0
.LBB0_851:                              ; %.preheader22.i.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_817 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s30, s2, s26
	s_addc_u32 s31, s3, s27
	global_load_ubyte v3, v9, s[30:31]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v42, v10, v42
	s_cmp_eq_u32 s28, s26
	v_or_b32_e32 v43, v11, v43
	s_cbranch_scc0 .LBB0_851
	s_branch .LBB0_854
.LBB0_852:                              ;   in Loop: Header=BB0_817 Depth=2
                                        ; implicit-def: $sgpr5
	s_branch .LBB0_855
.LBB0_853:                              ;   in Loop: Header=BB0_817 Depth=2
	v_mov_b64_e32 v[42:43], 0
.LBB0_854:                              ; %Flow4006
                                        ;   in Loop: Header=BB0_817 Depth=2
	s_mov_b32 s5, 0
	s_cbranch_execnz .LBB0_856
.LBB0_855:                              ;   in Loop: Header=BB0_817 Depth=2
	global_load_dwordx2 v[42:43], v9, s[2:3]
	s_add_i32 s5, s28, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_856:                              ; %.loopexit23.i.i.1.i
                                        ;   in Loop: Header=BB0_817 Depth=2
	s_cmp_gt_u32 s5, 7
	s_cbranch_scc1 .LBB0_860
; %bb.857:                              ;   in Loop: Header=BB0_817 Depth=2
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_861
; %bb.858:                              ; %.preheader20.i.i.1.i.preheader
                                        ;   in Loop: Header=BB0_817 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[44:45], 0
	s_mov_b64 s[26:27], 0
.LBB0_859:                              ; %.preheader20.i.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_817 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s28, s2, s26
	s_addc_u32 s29, s3, s27
	global_load_ubyte v3, v9, s[28:29]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v44, v10, v44
	s_cmp_eq_u32 s5, s26
	v_or_b32_e32 v45, v11, v45
	s_cbranch_scc0 .LBB0_859
	s_branch .LBB0_862
.LBB0_860:                              ;   in Loop: Header=BB0_817 Depth=2
                                        ; implicit-def: $vgpr44_vgpr45
                                        ; implicit-def: $sgpr28
	s_branch .LBB0_863
.LBB0_861:                              ;   in Loop: Header=BB0_817 Depth=2
	v_mov_b64_e32 v[44:45], 0
.LBB0_862:                              ; %Flow4001
                                        ;   in Loop: Header=BB0_817 Depth=2
	s_mov_b32 s28, 0
	s_cbranch_execnz .LBB0_864
.LBB0_863:                              ;   in Loop: Header=BB0_817 Depth=2
	global_load_dwordx2 v[44:45], v9, s[2:3]
	s_add_i32 s28, s5, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_864:                              ; %.loopexit21.i.i.1.i
                                        ;   in Loop: Header=BB0_817 Depth=2
	s_cmp_gt_u32 s28, 7
	s_cbranch_scc1 .LBB0_868
; %bb.865:                              ;   in Loop: Header=BB0_817 Depth=2
	s_cmp_eq_u32 s28, 0
	s_cbranch_scc1 .LBB0_869
; %bb.866:                              ; %.preheader.i.i.1.i.preheader
                                        ;   in Loop: Header=BB0_817 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[46:47], 0
	s_mov_b64 s[26:27], s[2:3]
.LBB0_867:                              ; %.preheader.i.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_817 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	global_load_ubyte v3, v9, s[26:27]
	s_add_i32 s28, s28, -1
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	v_or_b32_e32 v46, v10, v46
	s_cmp_eq_u32 s28, 0
	v_or_b32_e32 v47, v11, v47
	s_cbranch_scc0 .LBB0_867
	s_branch .LBB0_870
.LBB0_868:                              ;   in Loop: Header=BB0_817 Depth=2
	s_branch .LBB0_871
.LBB0_869:                              ;   in Loop: Header=BB0_817 Depth=2
	v_mov_b64_e32 v[46:47], 0
.LBB0_870:                              ; %Flow3996
                                        ;   in Loop: Header=BB0_817 Depth=2
	s_cbranch_execnz .LBB0_872
.LBB0_871:                              ;   in Loop: Header=BB0_817 Depth=2
	global_load_dwordx2 v[46:47], v9, s[2:3]
.LBB0_872:                              ; %.loopexit.i.i.1.i
                                        ;   in Loop: Header=BB0_817 Depth=2
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[10:11], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_878
; %bb.873:                              ;   in Loop: Header=BB0_817 Depth=2
	global_load_dwordx2 v[50:51], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v3, v10, v50
	v_and_b32_e32 v7, v11, v51
	v_mul_lo_u32 v7, v7, 24
	v_mul_hi_u32 v8, v3, 24
	v_add_u32_e32 v11, v8, v7
	v_mul_lo_u32 v10, v3, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[14:15], 0, v[10:11]
	global_load_dwordx2 v[48:49], v[10:11], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[48:51], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[10:11], v[50:51]
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB0_877
; %bb.874:                              ; %.preheader3.i.i18.i.i.1.i.preheader
                                        ;   in Loop: Header=BB0_817 Depth=2
	s_mov_b64 s[28:29], 0
.LBB0_875:                              ; %.preheader3.i.i18.i.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_817 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_sleep 1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	global_load_dwordx2 v[18:19], v9, s[16:17]
	v_mov_b64_e32 v[50:51], v[10:11]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v7, v14, v50
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[10:11], s[30:31], v7, 24, v[18:19]
	v_and_b32_e32 v3, v15, v51
	v_mov_b32_e32 v8, v11
	v_mad_u64_u32 v[14:15], s[30:31], v3, 24, v[8:9]
	v_mov_b32_e32 v11, v14
	global_load_dwordx2 v[48:49], v[10:11], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[48:51], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[10:11], v[50:51]
	s_or_b64 s[28:29], vcc, s[28:29]
	s_andn2_b64 exec, exec, s[28:29]
	s_cbranch_execnz .LBB0_875
; %bb.876:                              ; %Flow3991
                                        ;   in Loop: Header=BB0_817 Depth=2
	s_or_b64 exec, exec, s[28:29]
.LBB0_877:                              ; %Flow3993
                                        ;   in Loop: Header=BB0_817 Depth=2
	s_or_b64 exec, exec, s[26:27]
.LBB0_878:                              ; %.loopexit4.i.i13.i.i.1.i
                                        ;   in Loop: Header=BB0_817 Depth=2
	s_or_b64 exec, exec, s[24:25]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	global_load_dwordx4 v[48:51], v9, s[16:17]
	v_readfirstlane_b32 s24, v10
	v_readfirstlane_b32 s25, v11
	s_mov_b64 s[26:27], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s28, v14
	v_readfirstlane_b32 s29, v15
	s_and_b64 s[28:29], s[24:25], s[28:29]
	s_mul_i32 s5, s29, 24
	s_mul_hi_u32 s30, s28, 24
	s_add_i32 s31, s30, s5
	s_mul_i32 s30, s28, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[48:49], 0, s[30:31]
	s_and_saveexec_b64 s[30:31], s[2:3]
	s_cbranch_execz .LBB0_880
; %bb.879:                              ;   in Loop: Header=BB0_817 Depth=2
	v_mov_b64_e32 v[22:23], s[26:27]
	global_store_dwordx4 v[10:11], v[22:25], off offset:8
.LBB0_880:                              ;   in Loop: Header=BB0_817 Depth=2
	s_or_b64 exec, exec, s[30:31]
	v_or_b32_e32 v3, 0, v33
	v_or_b32_e32 v7, v32, v2
	v_cmp_gt_u64_e64 vcc, s[20:21], 56
	s_lshl_b32 s5, s22, 2
	s_lshl_b64 s[26:27], s[28:29], 12
	v_cndmask_b32_e32 v33, v3, v33, vcc
	v_cndmask_b32_e32 v3, v7, v32, vcc
	s_add_i32 s5, s5, 28
	v_lshl_add_u64 v[14:15], v[50:51], 0, s[26:27]
	s_and_b32 s5, s5, 0x1e0
	v_and_b32_e32 v3, 0xffffff1f, v3
	v_or_b32_e32 v32, s5, v3
	v_readfirstlane_b32 s26, v14
	v_readfirstlane_b32 s27, v15
	s_nop 4
	global_store_dwordx4 v58, v[32:35], s[26:27]
	global_store_dwordx4 v58, v[36:39], s[26:27] offset:16
	global_store_dwordx4 v58, v[40:43], s[26:27] offset:32
	global_store_dwordx4 v58, v[44:47], s[26:27] offset:48
	s_and_saveexec_b64 s[26:27], s[2:3]
	s_cbranch_execz .LBB0_888
; %bb.881:                              ;   in Loop: Header=BB0_817 Depth=2
	global_load_dwordx2 v[36:37], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[18:19], v9, s[16:17] offset:40
	v_mov_b32_e32 v34, s24
	v_mov_b32_e32 v35, s25
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s28, v18
	v_readfirstlane_b32 s29, v19
	s_and_b64 s[28:29], s[28:29], s[24:25]
	s_mul_i32 s5, s29, 24
	s_mul_hi_u32 s29, s28, 24
	s_mul_i32 s28, s28, 24
	s_add_i32 s29, s29, s5
	v_lshl_add_u64 v[18:19], v[48:49], 0, s[28:29]
	global_store_dwordx2 v[18:19], v[36:37], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[34:35], v9, v[34:37], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[34:35], v[36:37]
	s_and_saveexec_b64 s[28:29], vcc
	s_cbranch_execz .LBB0_884
; %bb.882:                              ; %.preheader1.i.i16.i.i.1.i.preheader
                                        ;   in Loop: Header=BB0_817 Depth=2
	s_mov_b64 s[30:31], 0
.LBB0_883:                              ; %.preheader1.i.i16.i.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_817 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_sleep 1
	global_store_dwordx2 v[18:19], v[34:35], off
	v_mov_b32_e32 v32, s24
	v_mov_b32_e32 v33, s25
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[22:23], v9, v[32:35], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[22:23], v[34:35]
	s_or_b64 s[30:31], vcc, s[30:31]
	v_mov_b64_e32 v[34:35], v[22:23]
	s_andn2_b64 exec, exec, s[30:31]
	s_cbranch_execnz .LBB0_883
.LBB0_884:                              ; %Flow3989
                                        ;   in Loop: Header=BB0_817 Depth=2
	s_or_b64 exec, exec, s[28:29]
	global_load_dwordx2 v[18:19], v9, s[16:17] offset:16
	s_mov_b64 s[30:31], exec
	v_mbcnt_lo_u32_b32 v3, s30, 0
	v_mbcnt_hi_u32_b32 v3, s31, v3
	v_cmp_eq_u32_e32 vcc, 0, v3
	s_and_saveexec_b64 s[28:29], vcc
	s_cbranch_execz .LBB0_886
; %bb.885:                              ;   in Loop: Header=BB0_817 Depth=2
	s_bcnt1_i32_b64 s5, s[30:31]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[18:19], v[8:9], off offset:8 sc1
.LBB0_886:                              ;   in Loop: Header=BB0_817 Depth=2
	s_or_b64 exec, exec, s[28:29]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[22:23], v[18:19], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_cbranch_vccnz .LBB0_888
; %bb.887:                              ;   in Loop: Header=BB0_817 Depth=2
	global_load_dword v8, v[18:19], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v3, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v3
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[22:23], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_888:                              ; %Flow3990
                                        ;   in Loop: Header=BB0_817 Depth=2
	s_or_b64 exec, exec, s[26:27]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[14:15], v[14:15], 0, v[58:59]
	s_branch .LBB0_892
.LBB0_889:                              ;   in Loop: Header=BB0_892 Depth=3
	s_or_b64 exec, exec, s[26:27]
	v_readfirstlane_b32 s5, v3
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_891
; %bb.890:                              ;   in Loop: Header=BB0_892 Depth=3
	s_sleep 1
	s_cbranch_execnz .LBB0_892
	s_branch .LBB0_894
.LBB0_891:                              ;   in Loop: Header=BB0_817 Depth=2
	s_branch .LBB0_894
.LBB0_892:                              ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_817 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_mov_b32_e32 v3, 1
	s_and_saveexec_b64 s[26:27], s[2:3]
	s_cbranch_execz .LBB0_889
; %bb.893:                              ;   in Loop: Header=BB0_892 Depth=3
	global_load_dword v3, v[10:11], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v3, 1, v3
	s_branch .LBB0_889
.LBB0_894:                              ;   in Loop: Header=BB0_817 Depth=2
	global_load_dwordx4 v[32:35], v[14:15], off
	s_and_saveexec_b64 s[26:27], s[2:3]
	s_cbranch_execz .LBB0_816
; %bb.895:                              ;   in Loop: Header=BB0_817 Depth=2
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[18:19], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[22:23], v[10:11], 0, 1
	v_lshl_add_u64 v[34:35], v[22:23], 0, s[24:25]
	v_cmp_eq_u64_e32 vcc, 0, v[34:35]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v36, v14
	v_mov_b32_e32 v37, v15
	v_cndmask_b32_e32 v35, v35, v23, vcc
	v_cndmask_b32_e32 v34, v34, v22, vcc
	v_and_b32_e32 v3, v35, v11
	v_and_b32_e32 v7, v34, v10
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v8, v7, 24
	v_mul_lo_u32 v10, v7, 24
	v_add_u32_e32 v11, v8, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[18:19], 0, v[10:11]
	global_store_dwordx2 v[10:11], v[14:15], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[14:15]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_816
; %bb.896:                              ; %.preheader.i.i15.i.i.1.i.preheader
                                        ;   in Loop: Header=BB0_817 Depth=2
	s_mov_b64 s[2:3], 0
.LBB0_897:                              ; %.preheader.i.i15.i.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_817 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_sleep 1
	global_store_dwordx2 v[10:11], v[36:37], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[14:15], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[14:15], v[36:37]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[36:37], v[14:15]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_897
	s_branch .LBB0_816
.LBB0_898:                              ; %Flow4029
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_927
.LBB0_899:                              ;   in Loop: Header=BB0_273 Depth=1
                                        ; implicit-def: $vgpr32_vgpr33
	s_cbranch_execz .LBB0_927
; %bb.900:                              ;   in Loop: Header=BB0_273 Depth=1
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_906
; %bb.901:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[34:35], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v34
	v_and_b32_e32 v3, v3, v35
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[32:33], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[32:35], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[34:35]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_905
; %bb.902:                              ; %.preheader3.i.i.i9.i.1.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_903:                              ; %.preheader3.i.i.i9.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[34:35], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v34
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v35
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[32:33], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[32:35], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[34:35]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_903
; %bb.904:                              ; %Flow4042
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_905:                              ; %Flow4044
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_906:                              ; %.loopexit4.i.i.i4.i.1.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[34:37], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[34:35], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_908
; %bb.907:                              ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_908:                              ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[36:37], 0, s[6:7]
	v_and_or_b32 v30, v30, s33, 32
	v_mov_b32_e32 v32, v9
	v_mov_b32_e32 v33, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	s_nop 1
	global_store_dwordx4 v58, v[30:33], s[22:23]
	s_nop 1
	v_mov_b64_e32 v[32:33], s[6:7]
	v_mov_b64_e32 v[30:31], s[4:5]
	global_store_dwordx4 v58, v[30:33], s[22:23] offset:16
	global_store_dwordx4 v58, v[30:33], s[22:23] offset:32
	global_store_dwordx4 v58, v[30:33], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_916
; %bb.909:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[38:39], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v36, s20
	v_mov_b32_e32 v37, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[34:35], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[38:39], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[36:39], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[38:39]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_912
; %bb.910:                              ; %.preheader1.i.i.i7.i.1.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_911:                              ; %.preheader1.i.i.i7.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[32:33], off
	v_mov_b32_e32 v30, s20
	v_mov_b32_e32 v31, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[30:33], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[32:33]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[32:33], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_911
.LBB0_912:                              ; %Flow4040
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_914
; %bb.913:                              ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_914:                              ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_916
; %bb.915:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_916:                              ; %Flow4041
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_920
.LBB0_917:                              ;   in Loop: Header=BB0_920 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_919
; %bb.918:                              ;   in Loop: Header=BB0_920 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_920
	s_branch .LBB0_922
.LBB0_919:                              ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_922
.LBB0_920:                              ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_917
; %bb.921:                              ;   in Loop: Header=BB0_920 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_917
.LBB0_922:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[32:33], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_926
; %bb.923:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[18:19], v[2:3], 0, 1
	v_lshl_add_u64 v[22:23], v[18:19], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v36, v10
	v_mov_b32_e32 v37, v11
	v_cndmask_b32_e32 v35, v23, v19, vcc
	v_cndmask_b32_e32 v34, v22, v18, vcc
	v_and_b32_e32 v3, v35, v3
	v_and_b32_e32 v2, v34, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_926
; %bb.924:                              ; %.preheader.i.i.i6.i.1.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_925:                              ; %.preheader.i.i.i6.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[36:37], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[36:37]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[36:37], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_925
.LBB0_926:                              ; %Flow4033
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
.LBB0_927:                              ; %__ockl_printf_append_string_n.exit.i.1.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_933
; %bb.928:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[36:37], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v36
	v_and_b32_e32 v3, v3, v37
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[34:35], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[36:37]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_932
; %bb.929:                              ; %.preheader3.i.i.i16.i.1.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_930:                              ; %.preheader3.i.i.i16.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[36:37], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v36
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v37
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[34:35], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[36:37]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_930
; %bb.931:                              ; %Flow3977
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_932:                              ; %Flow3979
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_933:                              ; %.loopexit4.i.i.i10.i.1.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[36:39], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[36:37], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_935
; %bb.934:                              ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_935:                              ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[38:39], 0, s[6:7]
	v_and_or_b32 v32, v32, s33, 32
	v_mov_b32_e32 v34, v26
	v_mov_b32_e32 v35, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	s_nop 1
	global_store_dwordx4 v58, v[32:35], s[22:23]
	s_nop 1
	v_mov_b64_e32 v[32:33], s[6:7]
	v_mov_b64_e32 v[30:31], s[4:5]
	global_store_dwordx4 v58, v[30:33], s[22:23] offset:16
	global_store_dwordx4 v58, v[30:33], s[22:23] offset:32
	global_store_dwordx4 v58, v[30:33], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_943
; %bb.936:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[34:35], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v32, s20
	v_mov_b32_e32 v33, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[36:37], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[34:35], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[32:35], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[34:35]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_939
; %bb.937:                              ; %.preheader1.i.i.i14.i.1.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_938:                              ; %.preheader1.i.i.i14.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[32:33], off
	v_mov_b32_e32 v30, s20
	v_mov_b32_e32 v31, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[30:33], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[32:33]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[32:33], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_938
.LBB0_939:                              ; %Flow3975
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_941
; %bb.940:                              ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_941:                              ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_943
; %bb.942:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_943:                              ; %Flow3976
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_947
.LBB0_944:                              ;   in Loop: Header=BB0_947 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_946
; %bb.945:                              ;   in Loop: Header=BB0_947 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_947
	s_branch .LBB0_949
.LBB0_946:                              ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_949
.LBB0_947:                              ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_944
; %bb.948:                              ;   in Loop: Header=BB0_947 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_944
.LBB0_949:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[14:15], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_953
; %bb.950:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[18:19], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[22:23], v[2:3], 0, 1
	v_lshl_add_u64 v[30:31], v[22:23], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[30:31]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v32, v10
	v_mov_b32_e32 v33, v11
	v_cndmask_b32_e32 v31, v31, v23, vcc
	v_cndmask_b32_e32 v30, v30, v22, vcc
	v_and_b32_e32 v3, v31, v3
	v_and_b32_e32 v2, v30, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[18:19], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_953
; %bb.951:                              ; %.preheader.i.i.i13.i.1.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_952:                              ; %.preheader.i.i.i13.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[32:33], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[32:33]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[32:33], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_952
.LBB0_953:                              ; %__ockl_printf_append_args.exit.i.1.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_959
; %bb.954:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[32:33], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v32
	v_and_b32_e32 v3, v3, v33
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[32:33]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_958
; %bb.955:                              ; %.preheader3.i.i.i23.i.1.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_956:                              ; %.preheader3.i.i.i23.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[18:19], v9, s[16:17]
	v_mov_b64_e32 v[32:33], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v32
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[18:19]
	v_and_b32_e32 v7, v11, v33
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[32:33]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_956
; %bb.957:                              ; %Flow3963
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_958:                              ; %Flow3965
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_959:                              ; %.loopexit4.i.i.i17.i.1.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[30:33], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[30:31], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_961
; %bb.960:                              ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_961:                              ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[32:33], 0, s[6:7]
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	v_mov_b64_e32 v[34:35], s[6:7]
	v_and_or_b32 v14, v14, s33, 32
	v_mov_b32_e32 v17, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	v_mov_b64_e32 v[32:33], s[4:5]
	s_nop 3
	global_store_dwordx4 v58, v[14:17], s[22:23]
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:16
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:32
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_969
; %bb.962:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[34:35], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v32, s20
	v_mov_b32_e32 v33, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[30:31], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[34:35], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[32:35], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[34:35]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_965
; %bb.963:                              ; %.preheader1.i.i.i21.i.1.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_964:                              ; %.preheader1.i.i.i21.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[32:33], off
	v_mov_b32_e32 v30, s20
	v_mov_b32_e32 v31, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[30:33], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[32:33]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[32:33], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_964
.LBB0_965:                              ; %Flow3961
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_967
; %bb.966:                              ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_967:                              ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_969
; %bb.968:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_969:                              ; %Flow3962
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_973
.LBB0_970:                              ;   in Loop: Header=BB0_973 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_972
; %bb.971:                              ;   in Loop: Header=BB0_973 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_973
	s_branch .LBB0_975
.LBB0_972:                              ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_975
.LBB0_973:                              ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_970
; %bb.974:                              ;   in Loop: Header=BB0_973 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_970
.LBB0_975:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[18:19], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_979
; %bb.976:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[22:23], v[2:3], 0, 1
	v_lshl_add_u64 v[30:31], v[22:23], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[30:31]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v32, v10
	v_mov_b32_e32 v33, v11
	v_cndmask_b32_e32 v31, v31, v23, vcc
	v_cndmask_b32_e32 v30, v30, v22, vcc
	v_and_b32_e32 v3, v31, v3
	v_and_b32_e32 v2, v30, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_979
; %bb.977:                              ; %.preheader.i.i.i20.i.1.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_978:                              ; %.preheader.i.i.i20.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[32:33], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[32:33]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[32:33], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_978
.LBB0_979:                              ; %__ockl_printf_append_args.exit24.i.1.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_985
; %bb.980:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[32:33], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v32
	v_and_b32_e32 v3, v3, v33
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[32:33]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_984
; %bb.981:                              ; %.preheader3.i.i.i31.i.1.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_982:                              ; %.preheader3.i.i.i31.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[32:33], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v32
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v33
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[32:33]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_982
; %bb.983:                              ; %Flow3949
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_984:                              ; %Flow3951
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_985:                              ; %.loopexit4.i.i.i25.i.1.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[30:33], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[30:31], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_987
; %bb.986:                              ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_987:                              ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[32:33], 0, s[6:7]
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	v_mov_b64_e32 v[34:35], s[6:7]
	v_and_or_b32 v18, v18, s33, 32
	v_mov_b32_e32 v21, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	v_mov_b64_e32 v[32:33], s[4:5]
	s_nop 3
	global_store_dwordx4 v58, v[18:21], s[22:23]
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:16
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:32
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_995
; %bb.988:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[34:35], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v32, s20
	v_mov_b32_e32 v33, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[30:31], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[34:35], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[32:35], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[34:35]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_991
; %bb.989:                              ; %.preheader1.i.i.i29.i.1.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_990:                              ; %.preheader1.i.i.i29.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[32:33], off
	v_mov_b32_e32 v30, s20
	v_mov_b32_e32 v31, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[30:33], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[32:33]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[32:33], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_990
.LBB0_991:                              ; %Flow3947
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_993
; %bb.992:                              ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_993:                              ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_995
; %bb.994:                              ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_995:                              ; %Flow3948
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_999
.LBB0_996:                              ;   in Loop: Header=BB0_999 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_998
; %bb.997:                              ;   in Loop: Header=BB0_999 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_999
	s_branch .LBB0_1001
.LBB0_998:                              ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_1001
.LBB0_999:                              ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_996
; %bb.1000:                             ;   in Loop: Header=BB0_999 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_996
.LBB0_1001:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[18:19], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1005
; %bb.1002:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[22:23], v[2:3], 0, 1
	v_lshl_add_u64 v[30:31], v[22:23], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[30:31]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v32, v10
	v_mov_b32_e32 v33, v11
	v_cndmask_b32_e32 v31, v31, v23, vcc
	v_cndmask_b32_e32 v30, v30, v22, vcc
	v_and_b32_e32 v3, v31, v3
	v_and_b32_e32 v2, v30, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_1005
; %bb.1003:                             ; %.preheader.i.i.i28.i.1.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_1004:                             ; %.preheader.i.i.i28.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[32:33], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[32:33]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[32:33], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_1004
.LBB0_1005:                             ; %__ockl_printf_append_args.exit32.i.1.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1011
; %bb.1006:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[32:33], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v32
	v_and_b32_e32 v3, v3, v33
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[32:33]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_1010
; %bb.1007:                             ; %.preheader3.i.i.i39.i.1.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_1008:                             ; %.preheader3.i.i.i39.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[32:33], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v32
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v33
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[32:33]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_1008
; %bb.1009:                             ; %Flow3935
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_1010:                             ; %Flow3937
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_1011:                             ; %.loopexit4.i.i.i33.i.1.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[30:33], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[30:31], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_1013
; %bb.1012:                             ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_1013:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[32:33], 0, s[6:7]
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	v_mov_b64_e32 v[34:35], s[6:7]
	v_and_or_b32 v18, v18, s33, 32
	v_mov_b32_e32 v21, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	v_mov_b64_e32 v[32:33], s[4:5]
	s_nop 3
	global_store_dwordx4 v58, v[18:21], s[22:23]
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:16
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:32
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1021
; %bb.1014:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[34:35], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v32, s20
	v_mov_b32_e32 v33, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[30:31], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[34:35], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[32:35], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[34:35]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1017
; %bb.1015:                             ; %.preheader1.i.i.i37.i.1.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_1016:                             ; %.preheader1.i.i.i37.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[32:33], off
	v_mov_b32_e32 v30, s20
	v_mov_b32_e32 v31, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[30:33], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[32:33]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[32:33], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_1016
.LBB0_1017:                             ; %Flow3933
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1019
; %bb.1018:                             ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_1019:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_1021
; %bb.1020:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_1021:                             ; %Flow3934
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_1025
.LBB0_1022:                             ;   in Loop: Header=BB0_1025 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1024
; %bb.1023:                             ;   in Loop: Header=BB0_1025 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_1025
	s_branch .LBB0_1027
.LBB0_1024:                             ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_1027
.LBB0_1025:                             ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1022
; %bb.1026:                             ;   in Loop: Header=BB0_1025 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_1022
.LBB0_1027:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[18:19], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1031
; %bb.1028:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[22:23], v[2:3], 0, 1
	v_lshl_add_u64 v[30:31], v[22:23], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[30:31]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v32, v10
	v_mov_b32_e32 v33, v11
	v_cndmask_b32_e32 v31, v31, v23, vcc
	v_cndmask_b32_e32 v30, v30, v22, vcc
	v_and_b32_e32 v3, v31, v3
	v_and_b32_e32 v2, v30, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_1031
; %bb.1029:                             ; %.preheader.i.i.i36.i.1.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_1030:                             ; %.preheader.i.i.i36.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[32:33], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[32:33]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[32:33], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_1030
.LBB0_1031:                             ; %__ockl_printf_append_args.exit40.i.1.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1037
; %bb.1032:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[32:33], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v32
	v_and_b32_e32 v3, v3, v33
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[32:33]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_1036
; %bb.1033:                             ; %.preheader3.i.i.i47.i.1.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_1034:                             ; %.preheader3.i.i.i47.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[32:33], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v32
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v33
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[32:33]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_1034
; %bb.1035:                             ; %Flow3921
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_1036:                             ; %Flow3923
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_1037:                             ; %.loopexit4.i.i.i41.i.1.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[30:33], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[30:31], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_1039
; %bb.1038:                             ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_1039:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[32:33], 0, s[6:7]
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	v_mov_b64_e32 v[34:35], s[6:7]
	v_and_or_b32 v18, v18, s34, 34
	v_mov_b32_e32 v21, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	v_mov_b64_e32 v[32:33], s[4:5]
	s_nop 3
	global_store_dwordx4 v58, v[18:21], s[22:23]
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:16
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:32
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1047
; %bb.1040:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[34:35], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	v_mov_b32_e32 v32, s20
	v_mov_b32_e32 v33, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[10:11], v[30:31], 0, s[22:23]
	global_store_dwordx2 v[10:11], v[34:35], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[32:35], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[34:35]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1043
; %bb.1041:                             ; %.preheader1.i.i.i45.i.1.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_1042:                             ; %.preheader1.i.i.i45.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[10:11], v[32:33], off
	v_mov_b32_e32 v30, s20
	v_mov_b32_e32 v31, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[14:15], v9, v[30:33], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[14:15], v[32:33]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[32:33], v[14:15]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_1042
.LBB0_1043:                             ; %Flow3919
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1045
; %bb.1044:                             ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[10:11], v[8:9], off offset:8 sc1
.LBB0_1045:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[14:15], v[10:11], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[14:15]
	s_cbranch_vccnz .LBB0_1047
; %bb.1046:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[10:11], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[14:15], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_1047:                             ; %Flow3920
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	s_branch .LBB0_1051
.LBB0_1048:                             ;   in Loop: Header=BB0_1051 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1050
; %bb.1049:                             ;   in Loop: Header=BB0_1051 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_1051
	s_branch .LBB0_1053
.LBB0_1050:                             ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_1053
.LBB0_1051:                             ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1048
; %bb.1052:                             ;   in Loop: Header=BB0_1051 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_1048
.LBB0_1053:                             ;   in Loop: Header=BB0_273 Depth=1
	s_and_b64 exec, exec, s[2:3]
	s_cbranch_execz .LBB0_1057
; %bb.1054:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[18:19], v[2:3], 0, 1
	v_lshl_add_u64 v[22:23], v[18:19], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v32, v10
	v_mov_b32_e32 v33, v11
	v_cndmask_b32_e32 v31, v23, v19, vcc
	v_cndmask_b32_e32 v30, v22, v18, vcc
	v_and_b32_e32 v3, v31, v3
	v_and_b32_e32 v2, v30, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_1057
; %bb.1055:                             ; %.preheader.i.i.i44.i.1.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_1056:                             ; %.preheader.i.i.i44.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[32:33], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[32:33]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[32:33], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_1056
.LBB0_1057:                             ; %Flow4061
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[18:19]
	v_lshrrev_b32_e32 v2, 3, v26
	v_add_u32_e32 v63, s37, v62
	v_bitop3_b32 v32, v2, v26, s35 bitop3:0x6c
	s_and_saveexec_b64 s[18:19], s[0:1]
	s_cbranch_execz .LBB0_1301
; %bb.1058:                             ; %if.then.1.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1064
; %bb.1059:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[36:37], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v36
	v_and_b32_e32 v3, v3, v37
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[34:35], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[36:37]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_1063
; %bb.1060:                             ; %.preheader3.i.i.i.1.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_1061:                             ; %.preheader3.i.i.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[36:37], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v36
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v37
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[34:35], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[36:37]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_1061
; %bb.1062:                             ; %Flow3905
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_1063:                             ; %Flow3907
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_1064:                             ; %.loopexit4.i.i.i.1.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[34:37], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[34:35], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_1066
; %bb.1065:                             ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_1066:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[36:37], 0, s[6:7]
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	v_mov_b64_e32 v[38:39], s[6:7]
	v_mov_b32_e32 v7, v9
	v_mov_b32_e32 v8, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	v_mov_b64_e32 v[36:37], s[4:5]
	s_nop 3
	global_store_dwordx4 v58, v[6:9], s[22:23]
	global_store_dwordx4 v58, v[36:39], s[22:23] offset:16
	global_store_dwordx4 v58, v[36:39], s[22:23] offset:32
	global_store_dwordx4 v58, v[36:39], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1074
; %bb.1067:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[38:39], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v36, s20
	v_mov_b32_e32 v37, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[34:35], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[38:39], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[36:39], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[38:39]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1070
; %bb.1068:                             ; %.preheader1.i.i.i.1.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_1069:                             ; %.preheader1.i.i.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[36:37], off
	v_mov_b32_e32 v34, s20
	v_mov_b32_e32 v35, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[34:37], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[36:37]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[36:37], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_1069
.LBB0_1070:                             ; %Flow3903
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1072
; %bb.1071:                             ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_1072:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_1074
; %bb.1073:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_1074:                             ; %Flow3904
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_1078
.LBB0_1075:                             ;   in Loop: Header=BB0_1078 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1077
; %bb.1076:                             ;   in Loop: Header=BB0_1078 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_1078
	s_branch .LBB0_1080
.LBB0_1077:                             ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_1080
.LBB0_1078:                             ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1075
; %bb.1079:                             ;   in Loop: Header=BB0_1078 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_1075
.LBB0_1080:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[34:35], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1084
; %bb.1081:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[18:19], v[2:3], 0, 1
	v_lshl_add_u64 v[22:23], v[18:19], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v38, v10
	v_mov_b32_e32 v39, v11
	v_cndmask_b32_e32 v37, v23, v19, vcc
	v_cndmask_b32_e32 v36, v22, v18, vcc
	v_and_b32_e32 v3, v37, v3
	v_and_b32_e32 v2, v36, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v9, v[36:39], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_1084
; %bb.1082:                             ; %.preheader.i.i.i.1.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_1083:                             ; %.preheader.i.i.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[38:39], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[36:39], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[38:39]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[38:39], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_1083
.LBB0_1084:                             ; %__ockl_printf_begin.exit.1.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	s_andn2_b64 vcc, exec, s[10:11]
	s_cbranch_vccnz .LBB0_1169
; %bb.1085:                             ;   in Loop: Header=BB0_273 Depth=1
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v2, 2, v34
	v_and_b32_e32 v36, -3, v34
	v_mov_b32_e32 v37, v35
	s_mov_b64 s[20:21], 49
	s_getpc_b64 s[6:7]
	s_add_u32 s6, s6, .str.2@rel32@lo+4
	s_addc_u32 s7, s7, .str.2@rel32@hi+12
	s_branch .LBB0_1087
.LBB0_1086:                             ; %__ockl_hostcall_preview.exit19.i.1.i
                                        ;   in Loop: Header=BB0_1087 Depth=2
	s_or_b64 exec, exec, s[26:27]
	s_sub_u32 s20, s20, s22
	s_subb_u32 s21, s21, s23
	s_add_u32 s6, s6, s22
	s_addc_u32 s7, s7, s23
	s_cmp_eq_u64 s[20:21], 0
	s_cbranch_scc1 .LBB0_1168
.LBB0_1087:                             ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_1090 Depth 3
                                        ;       Child Loop BB0_1097 Depth 3
                                        ;       Child Loop BB0_1105 Depth 3
                                        ;       Child Loop BB0_1113 Depth 3
                                        ;       Child Loop BB0_1121 Depth 3
                                        ;       Child Loop BB0_1129 Depth 3
                                        ;       Child Loop BB0_1137 Depth 3
                                        ;       Child Loop BB0_1145 Depth 3
                                        ;       Child Loop BB0_1153 Depth 3
                                        ;       Child Loop BB0_1162 Depth 3
                                        ;       Child Loop BB0_1167 Depth 3
	v_cmp_lt_u64_e64 s[2:3], s[20:21], 56
	s_and_b64 s[2:3], s[2:3], exec
	v_cmp_gt_u64_e64 s[2:3], s[20:21], 7
	s_cselect_b32 s23, s21, 0
	s_cselect_b32 s22, s20, 56
	s_and_b64 vcc, exec, s[2:3]
	s_cbranch_vccnz .LBB0_1092
; %bb.1088:                             ;   in Loop: Header=BB0_1087 Depth=2
	s_mov_b64 s[2:3], 0
	s_cmp_eq_u64 s[20:21], 0
	s_waitcnt vmcnt(0)
	v_mov_b64_e32 v[38:39], 0
	s_cbranch_scc1 .LBB0_1091
; %bb.1089:                             ; %.preheader30.i.1.i.preheader
                                        ;   in Loop: Header=BB0_1087 Depth=2
	s_lshl_b64 s[24:25], s[22:23], 3
	s_mov_b64 s[26:27], 0
	v_mov_b64_e32 v[38:39], 0
	s_mov_b64 s[28:29], s[6:7]
.LBB0_1090:                             ; %.preheader30.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1087 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	global_load_ubyte v3, v9, s[28:29]
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s26, v[8:9]
	s_add_u32 s26, s26, 8
	s_addc_u32 s27, s27, 0
	s_add_u32 s28, s28, 1
	s_addc_u32 s29, s29, 0
	v_or_b32_e32 v38, v10, v38
	s_cmp_eq_u32 s24, s26
	v_or_b32_e32 v39, v11, v39
	s_cbranch_scc0 .LBB0_1090
.LBB0_1091:                             ; %Flow3873
                                        ;   in Loop: Header=BB0_1087 Depth=2
	s_mov_b32 s5, 0
	s_andn2_b64 vcc, exec, s[2:3]
	s_mov_b64 s[2:3], s[6:7]
	s_cbranch_vccz .LBB0_1093
	s_branch .LBB0_1094
.LBB0_1092:                             ;   in Loop: Header=BB0_1087 Depth=2
                                        ; implicit-def: $vgpr38_vgpr39
                                        ; implicit-def: $sgpr5
	s_mov_b64 s[2:3], s[6:7]
.LBB0_1093:                             ;   in Loop: Header=BB0_1087 Depth=2
	global_load_dwordx2 v[38:39], v9, s[6:7]
	s_add_i32 s5, s22, -8
	s_add_u32 s2, s6, 8
	s_addc_u32 s3, s7, 0
.LBB0_1094:                             ; %.loopexit31.i.1.i
                                        ;   in Loop: Header=BB0_1087 Depth=2
	s_cmp_gt_u32 s5, 7
	s_cbranch_scc1 .LBB0_1098
; %bb.1095:                             ;   in Loop: Header=BB0_1087 Depth=2
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1099
; %bb.1096:                             ; %.preheader28.i.1.i.preheader
                                        ;   in Loop: Header=BB0_1087 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[40:41], 0
	s_mov_b64 s[26:27], 0
.LBB0_1097:                             ; %.preheader28.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1087 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s28, s2, s26
	s_addc_u32 s29, s3, s27
	global_load_ubyte v3, v9, s[28:29]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v40, v10, v40
	s_cmp_eq_u32 s5, s26
	v_or_b32_e32 v41, v11, v41
	s_cbranch_scc0 .LBB0_1097
	s_branch .LBB0_1100
.LBB0_1098:                             ;   in Loop: Header=BB0_1087 Depth=2
                                        ; implicit-def: $vgpr40_vgpr41
                                        ; implicit-def: $sgpr28
	s_branch .LBB0_1101
.LBB0_1099:                             ;   in Loop: Header=BB0_1087 Depth=2
	v_mov_b64_e32 v[40:41], 0
.LBB0_1100:                             ; %Flow3868
                                        ;   in Loop: Header=BB0_1087 Depth=2
	s_mov_b32 s28, 0
	s_cbranch_execnz .LBB0_1102
.LBB0_1101:                             ;   in Loop: Header=BB0_1087 Depth=2
	global_load_dwordx2 v[40:41], v9, s[2:3]
	s_add_i32 s28, s5, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_1102:                             ; %.loopexit29.i.1.i
                                        ;   in Loop: Header=BB0_1087 Depth=2
	s_cmp_gt_u32 s28, 7
	s_cbranch_scc1 .LBB0_1106
; %bb.1103:                             ;   in Loop: Header=BB0_1087 Depth=2
	s_cmp_eq_u32 s28, 0
	s_cbranch_scc1 .LBB0_1107
; %bb.1104:                             ; %.preheader26.i.1.i.preheader
                                        ;   in Loop: Header=BB0_1087 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[42:43], 0
	s_mov_b64 s[26:27], 0
.LBB0_1105:                             ; %.preheader26.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1087 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s30, s2, s26
	s_addc_u32 s31, s3, s27
	global_load_ubyte v3, v9, s[30:31]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v42, v10, v42
	s_cmp_eq_u32 s28, s26
	v_or_b32_e32 v43, v11, v43
	s_cbranch_scc0 .LBB0_1105
	s_branch .LBB0_1108
.LBB0_1106:                             ;   in Loop: Header=BB0_1087 Depth=2
                                        ; implicit-def: $sgpr5
	s_branch .LBB0_1109
.LBB0_1107:                             ;   in Loop: Header=BB0_1087 Depth=2
	v_mov_b64_e32 v[42:43], 0
.LBB0_1108:                             ; %Flow3863
                                        ;   in Loop: Header=BB0_1087 Depth=2
	s_mov_b32 s5, 0
	s_cbranch_execnz .LBB0_1110
.LBB0_1109:                             ;   in Loop: Header=BB0_1087 Depth=2
	global_load_dwordx2 v[42:43], v9, s[2:3]
	s_add_i32 s5, s28, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_1110:                             ; %.loopexit27.i.1.i
                                        ;   in Loop: Header=BB0_1087 Depth=2
	s_cmp_gt_u32 s5, 7
	s_cbranch_scc1 .LBB0_1114
; %bb.1111:                             ;   in Loop: Header=BB0_1087 Depth=2
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1115
; %bb.1112:                             ; %.preheader24.i.1.i.preheader
                                        ;   in Loop: Header=BB0_1087 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[44:45], 0
	s_mov_b64 s[26:27], 0
.LBB0_1113:                             ; %.preheader24.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1087 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s28, s2, s26
	s_addc_u32 s29, s3, s27
	global_load_ubyte v3, v9, s[28:29]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v44, v10, v44
	s_cmp_eq_u32 s5, s26
	v_or_b32_e32 v45, v11, v45
	s_cbranch_scc0 .LBB0_1113
	s_branch .LBB0_1116
.LBB0_1114:                             ;   in Loop: Header=BB0_1087 Depth=2
                                        ; implicit-def: $vgpr44_vgpr45
                                        ; implicit-def: $sgpr28
	s_branch .LBB0_1117
.LBB0_1115:                             ;   in Loop: Header=BB0_1087 Depth=2
	v_mov_b64_e32 v[44:45], 0
.LBB0_1116:                             ; %Flow3858
                                        ;   in Loop: Header=BB0_1087 Depth=2
	s_mov_b32 s28, 0
	s_cbranch_execnz .LBB0_1118
.LBB0_1117:                             ;   in Loop: Header=BB0_1087 Depth=2
	global_load_dwordx2 v[44:45], v9, s[2:3]
	s_add_i32 s28, s5, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_1118:                             ; %.loopexit25.i.1.i
                                        ;   in Loop: Header=BB0_1087 Depth=2
	s_cmp_gt_u32 s28, 7
	s_cbranch_scc1 .LBB0_1122
; %bb.1119:                             ;   in Loop: Header=BB0_1087 Depth=2
	s_cmp_eq_u32 s28, 0
	s_cbranch_scc1 .LBB0_1123
; %bb.1120:                             ; %.preheader22.i.1.i.preheader
                                        ;   in Loop: Header=BB0_1087 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[46:47], 0
	s_mov_b64 s[26:27], 0
.LBB0_1121:                             ; %.preheader22.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1087 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s30, s2, s26
	s_addc_u32 s31, s3, s27
	global_load_ubyte v3, v9, s[30:31]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v46, v10, v46
	s_cmp_eq_u32 s28, s26
	v_or_b32_e32 v47, v11, v47
	s_cbranch_scc0 .LBB0_1121
	s_branch .LBB0_1124
.LBB0_1122:                             ;   in Loop: Header=BB0_1087 Depth=2
                                        ; implicit-def: $sgpr5
	s_branch .LBB0_1125
.LBB0_1123:                             ;   in Loop: Header=BB0_1087 Depth=2
	v_mov_b64_e32 v[46:47], 0
.LBB0_1124:                             ; %Flow3853
                                        ;   in Loop: Header=BB0_1087 Depth=2
	s_mov_b32 s5, 0
	s_cbranch_execnz .LBB0_1126
.LBB0_1125:                             ;   in Loop: Header=BB0_1087 Depth=2
	global_load_dwordx2 v[46:47], v9, s[2:3]
	s_add_i32 s5, s28, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_1126:                             ; %.loopexit23.i.1.i
                                        ;   in Loop: Header=BB0_1087 Depth=2
	s_cmp_gt_u32 s5, 7
	s_cbranch_scc1 .LBB0_1130
; %bb.1127:                             ;   in Loop: Header=BB0_1087 Depth=2
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1131
; %bb.1128:                             ; %.preheader20.i.1.i.preheader
                                        ;   in Loop: Header=BB0_1087 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[48:49], 0
	s_mov_b64 s[26:27], 0
.LBB0_1129:                             ; %.preheader20.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1087 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s28, s2, s26
	s_addc_u32 s29, s3, s27
	global_load_ubyte v3, v9, s[28:29]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v48, v10, v48
	s_cmp_eq_u32 s5, s26
	v_or_b32_e32 v49, v11, v49
	s_cbranch_scc0 .LBB0_1129
	s_branch .LBB0_1132
.LBB0_1130:                             ;   in Loop: Header=BB0_1087 Depth=2
                                        ; implicit-def: $vgpr48_vgpr49
                                        ; implicit-def: $sgpr28
	s_branch .LBB0_1133
.LBB0_1131:                             ;   in Loop: Header=BB0_1087 Depth=2
	v_mov_b64_e32 v[48:49], 0
.LBB0_1132:                             ; %Flow3848
                                        ;   in Loop: Header=BB0_1087 Depth=2
	s_mov_b32 s28, 0
	s_cbranch_execnz .LBB0_1134
.LBB0_1133:                             ;   in Loop: Header=BB0_1087 Depth=2
	global_load_dwordx2 v[48:49], v9, s[2:3]
	s_add_i32 s28, s5, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_1134:                             ; %.loopexit21.i.1.i
                                        ;   in Loop: Header=BB0_1087 Depth=2
	s_cmp_gt_u32 s28, 7
	s_cbranch_scc1 .LBB0_1138
; %bb.1135:                             ;   in Loop: Header=BB0_1087 Depth=2
	s_cmp_eq_u32 s28, 0
	s_cbranch_scc1 .LBB0_1139
; %bb.1136:                             ; %.preheader.i.1.i.preheader
                                        ;   in Loop: Header=BB0_1087 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[50:51], 0
	s_mov_b64 s[26:27], s[2:3]
.LBB0_1137:                             ; %.preheader.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1087 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	global_load_ubyte v3, v9, s[26:27]
	s_add_i32 s28, s28, -1
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	v_or_b32_e32 v50, v10, v50
	s_cmp_eq_u32 s28, 0
	v_or_b32_e32 v51, v11, v51
	s_cbranch_scc0 .LBB0_1137
	s_branch .LBB0_1140
.LBB0_1138:                             ;   in Loop: Header=BB0_1087 Depth=2
	s_branch .LBB0_1141
.LBB0_1139:                             ;   in Loop: Header=BB0_1087 Depth=2
	v_mov_b64_e32 v[50:51], 0
.LBB0_1140:                             ; %Flow3843
                                        ;   in Loop: Header=BB0_1087 Depth=2
	s_cbranch_execnz .LBB0_1142
.LBB0_1141:                             ;   in Loop: Header=BB0_1087 Depth=2
	global_load_dwordx2 v[50:51], v9, s[2:3]
.LBB0_1142:                             ; %.loopexit.i.1.i
                                        ;   in Loop: Header=BB0_1087 Depth=2
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[10:11], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_1148
; %bb.1143:                             ;   in Loop: Header=BB0_1087 Depth=2
	global_load_dwordx2 v[54:55], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v3, v10, v54
	v_and_b32_e32 v7, v11, v55
	v_mul_lo_u32 v7, v7, 24
	v_mul_hi_u32 v8, v3, 24
	v_add_u32_e32 v11, v8, v7
	v_mul_lo_u32 v10, v3, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[14:15], 0, v[10:11]
	global_load_dwordx2 v[52:53], v[10:11], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[52:55], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[10:11], v[54:55]
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB0_1147
; %bb.1144:                             ; %.preheader3.i.i18.i.1.i.preheader
                                        ;   in Loop: Header=BB0_1087 Depth=2
	s_mov_b64 s[28:29], 0
.LBB0_1145:                             ; %.preheader3.i.i18.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1087 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_sleep 1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	global_load_dwordx2 v[18:19], v9, s[16:17]
	v_mov_b64_e32 v[54:55], v[10:11]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v7, v14, v54
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[10:11], s[30:31], v7, 24, v[18:19]
	v_and_b32_e32 v3, v15, v55
	v_mov_b32_e32 v8, v11
	v_mad_u64_u32 v[14:15], s[30:31], v3, 24, v[8:9]
	v_mov_b32_e32 v11, v14
	global_load_dwordx2 v[52:53], v[10:11], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[52:55], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[10:11], v[54:55]
	s_or_b64 s[28:29], vcc, s[28:29]
	s_andn2_b64 exec, exec, s[28:29]
	s_cbranch_execnz .LBB0_1145
; %bb.1146:                             ; %Flow3838
                                        ;   in Loop: Header=BB0_1087 Depth=2
	s_or_b64 exec, exec, s[28:29]
.LBB0_1147:                             ; %Flow3840
                                        ;   in Loop: Header=BB0_1087 Depth=2
	s_or_b64 exec, exec, s[26:27]
.LBB0_1148:                             ; %.loopexit4.i.i13.i.1.i
                                        ;   in Loop: Header=BB0_1087 Depth=2
	s_or_b64 exec, exec, s[24:25]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	global_load_dwordx4 v[52:55], v9, s[16:17]
	v_readfirstlane_b32 s24, v10
	v_readfirstlane_b32 s25, v11
	s_mov_b64 s[26:27], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s28, v14
	v_readfirstlane_b32 s29, v15
	s_and_b64 s[28:29], s[24:25], s[28:29]
	s_mul_i32 s5, s29, 24
	s_mul_hi_u32 s30, s28, 24
	s_add_i32 s31, s30, s5
	s_mul_i32 s30, s28, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[52:53], 0, s[30:31]
	s_and_saveexec_b64 s[30:31], s[2:3]
	s_cbranch_execz .LBB0_1150
; %bb.1149:                             ;   in Loop: Header=BB0_1087 Depth=2
	v_mov_b64_e32 v[22:23], s[26:27]
	global_store_dwordx4 v[10:11], v[22:25], off offset:8
.LBB0_1150:                             ;   in Loop: Header=BB0_1087 Depth=2
	s_or_b64 exec, exec, s[30:31]
	v_or_b32_e32 v3, 0, v37
	v_or_b32_e32 v7, v36, v2
	v_cmp_gt_u64_e64 vcc, s[20:21], 56
	s_lshl_b32 s5, s22, 2
	s_lshl_b64 s[26:27], s[28:29], 12
	v_cndmask_b32_e32 v37, v3, v37, vcc
	v_cndmask_b32_e32 v3, v7, v36, vcc
	s_add_i32 s5, s5, 28
	v_lshl_add_u64 v[14:15], v[54:55], 0, s[26:27]
	s_and_b32 s5, s5, 0x1e0
	v_and_b32_e32 v3, 0xffffff1f, v3
	v_or_b32_e32 v36, s5, v3
	v_readfirstlane_b32 s26, v14
	v_readfirstlane_b32 s27, v15
	s_nop 4
	global_store_dwordx4 v58, v[36:39], s[26:27]
	global_store_dwordx4 v58, v[40:43], s[26:27] offset:16
	global_store_dwordx4 v58, v[44:47], s[26:27] offset:32
	global_store_dwordx4 v58, v[48:51], s[26:27] offset:48
	s_and_saveexec_b64 s[26:27], s[2:3]
	s_cbranch_execz .LBB0_1158
; %bb.1151:                             ;   in Loop: Header=BB0_1087 Depth=2
	global_load_dwordx2 v[40:41], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[18:19], v9, s[16:17] offset:40
	v_mov_b32_e32 v38, s24
	v_mov_b32_e32 v39, s25
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s28, v18
	v_readfirstlane_b32 s29, v19
	s_and_b64 s[28:29], s[28:29], s[24:25]
	s_mul_i32 s5, s29, 24
	s_mul_hi_u32 s29, s28, 24
	s_mul_i32 s28, s28, 24
	s_add_i32 s29, s29, s5
	v_lshl_add_u64 v[18:19], v[52:53], 0, s[28:29]
	global_store_dwordx2 v[18:19], v[40:41], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v9, v[38:41], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[40:41]
	s_and_saveexec_b64 s[28:29], vcc
	s_cbranch_execz .LBB0_1154
; %bb.1152:                             ; %.preheader1.i.i16.i.1.i.preheader
                                        ;   in Loop: Header=BB0_1087 Depth=2
	s_mov_b64 s[30:31], 0
.LBB0_1153:                             ; %.preheader1.i.i16.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1087 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_sleep 1
	global_store_dwordx2 v[18:19], v[38:39], off
	v_mov_b32_e32 v36, s24
	v_mov_b32_e32 v37, s25
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[22:23], v9, v[36:39], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[22:23], v[38:39]
	s_or_b64 s[30:31], vcc, s[30:31]
	v_mov_b64_e32 v[38:39], v[22:23]
	s_andn2_b64 exec, exec, s[30:31]
	s_cbranch_execnz .LBB0_1153
.LBB0_1154:                             ; %Flow3836
                                        ;   in Loop: Header=BB0_1087 Depth=2
	s_or_b64 exec, exec, s[28:29]
	global_load_dwordx2 v[18:19], v9, s[16:17] offset:16
	s_mov_b64 s[30:31], exec
	v_mbcnt_lo_u32_b32 v3, s30, 0
	v_mbcnt_hi_u32_b32 v3, s31, v3
	v_cmp_eq_u32_e32 vcc, 0, v3
	s_and_saveexec_b64 s[28:29], vcc
	s_cbranch_execz .LBB0_1156
; %bb.1155:                             ;   in Loop: Header=BB0_1087 Depth=2
	s_bcnt1_i32_b64 s5, s[30:31]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[18:19], v[8:9], off offset:8 sc1
.LBB0_1156:                             ;   in Loop: Header=BB0_1087 Depth=2
	s_or_b64 exec, exec, s[28:29]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[22:23], v[18:19], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_cbranch_vccnz .LBB0_1158
; %bb.1157:                             ;   in Loop: Header=BB0_1087 Depth=2
	global_load_dword v8, v[18:19], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v3, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v3
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[22:23], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_1158:                             ; %Flow3837
                                        ;   in Loop: Header=BB0_1087 Depth=2
	s_or_b64 exec, exec, s[26:27]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[14:15], v[14:15], 0, v[58:59]
	s_branch .LBB0_1162
.LBB0_1159:                             ;   in Loop: Header=BB0_1162 Depth=3
	s_or_b64 exec, exec, s[26:27]
	v_readfirstlane_b32 s5, v3
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1161
; %bb.1160:                             ;   in Loop: Header=BB0_1162 Depth=3
	s_sleep 1
	s_cbranch_execnz .LBB0_1162
	s_branch .LBB0_1164
.LBB0_1161:                             ;   in Loop: Header=BB0_1087 Depth=2
	s_branch .LBB0_1164
.LBB0_1162:                             ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1087 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_mov_b32_e32 v3, 1
	s_and_saveexec_b64 s[26:27], s[2:3]
	s_cbranch_execz .LBB0_1159
; %bb.1163:                             ;   in Loop: Header=BB0_1162 Depth=3
	global_load_dword v3, v[10:11], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v3, 1, v3
	s_branch .LBB0_1159
.LBB0_1164:                             ;   in Loop: Header=BB0_1087 Depth=2
	global_load_dwordx4 v[36:39], v[14:15], off
	s_and_saveexec_b64 s[26:27], s[2:3]
	s_cbranch_execz .LBB0_1086
; %bb.1165:                             ;   in Loop: Header=BB0_1087 Depth=2
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[18:19], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[22:23], v[10:11], 0, 1
	v_lshl_add_u64 v[26:27], v[22:23], 0, s[24:25]
	v_cmp_eq_u64_e32 vcc, 0, v[26:27]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v40, v14
	v_mov_b32_e32 v41, v15
	v_cndmask_b32_e32 v39, v27, v23, vcc
	v_cndmask_b32_e32 v38, v26, v22, vcc
	v_and_b32_e32 v3, v39, v11
	v_and_b32_e32 v7, v38, v10
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v8, v7, 24
	v_mul_lo_u32 v10, v7, 24
	v_add_u32_e32 v11, v8, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[18:19], 0, v[10:11]
	global_store_dwordx2 v[10:11], v[14:15], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[40:41], v9, v[38:41], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[40:41], v[14:15]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_1086
; %bb.1166:                             ; %.preheader.i.i15.i.1.i.preheader
                                        ;   in Loop: Header=BB0_1087 Depth=2
	s_mov_b64 s[2:3], 0
.LBB0_1167:                             ; %.preheader.i.i15.i.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1087 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_sleep 1
	global_store_dwordx2 v[10:11], v[40:41], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[14:15], v9, v[38:41], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[14:15], v[40:41]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[40:41], v[14:15]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_1167
	s_branch .LBB0_1086
.LBB0_1168:                             ; %Flow3876
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_1197
.LBB0_1169:                             ;   in Loop: Header=BB0_273 Depth=1
                                        ; implicit-def: $vgpr36_vgpr37
	s_cbranch_execz .LBB0_1197
; %bb.1170:                             ;   in Loop: Header=BB0_273 Depth=1
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1176
; %bb.1171:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[38:39], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v38
	v_and_b32_e32 v3, v3, v39
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[36:37], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[36:39], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[38:39]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_1175
; %bb.1172:                             ; %.preheader3.i.i.i38.1.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_1173:                             ; %.preheader3.i.i.i38.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[38:39], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v38
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v39
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[36:37], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[36:39], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[38:39]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_1173
; %bb.1174:                             ; %Flow3889
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_1175:                             ; %Flow3891
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_1176:                             ; %.loopexit4.i.i.i33.1.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[38:41], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[38:39], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_1178
; %bb.1177:                             ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_1178:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[40:41], 0, s[6:7]
	v_and_or_b32 v34, v34, s33, 32
	v_mov_b32_e32 v36, v9
	v_mov_b32_e32 v37, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	s_nop 1
	global_store_dwordx4 v58, v[34:37], s[22:23]
	s_nop 1
	v_mov_b64_e32 v[36:37], s[6:7]
	v_mov_b64_e32 v[34:35], s[4:5]
	global_store_dwordx4 v58, v[34:37], s[22:23] offset:16
	global_store_dwordx4 v58, v[34:37], s[22:23] offset:32
	global_store_dwordx4 v58, v[34:37], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1186
; %bb.1179:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[42:43], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v40, s20
	v_mov_b32_e32 v41, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[38:39], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[42:43], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[40:43], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[42:43]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1182
; %bb.1180:                             ; %.preheader1.i.i.i36.1.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_1181:                             ; %.preheader1.i.i.i36.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[36:37], off
	v_mov_b32_e32 v34, s20
	v_mov_b32_e32 v35, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[34:37], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[36:37]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[36:37], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_1181
.LBB0_1182:                             ; %Flow3887
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1184
; %bb.1183:                             ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_1184:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_1186
; %bb.1185:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_1186:                             ; %Flow3888
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_1190
.LBB0_1187:                             ;   in Loop: Header=BB0_1190 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1189
; %bb.1188:                             ;   in Loop: Header=BB0_1190 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_1190
	s_branch .LBB0_1192
.LBB0_1189:                             ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_1192
.LBB0_1190:                             ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1187
; %bb.1191:                             ;   in Loop: Header=BB0_1190 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_1187
.LBB0_1192:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[36:37], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1196
; %bb.1193:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[18:19], v[2:3], 0, 1
	v_lshl_add_u64 v[22:23], v[18:19], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v40, v10
	v_mov_b32_e32 v41, v11
	v_cndmask_b32_e32 v39, v23, v19, vcc
	v_cndmask_b32_e32 v38, v22, v18, vcc
	v_and_b32_e32 v3, v39, v3
	v_and_b32_e32 v2, v38, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[40:41], v9, v[38:41], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[40:41], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_1196
; %bb.1194:                             ; %.preheader.i.i.i35.1.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_1195:                             ; %.preheader.i.i.i35.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[40:41], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[38:41], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[40:41]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[40:41], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_1195
.LBB0_1196:                             ; %Flow3880
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
.LBB0_1197:                             ; %__ockl_printf_append_string_n.exit.1.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1203
; %bb.1198:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[40:41], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v40
	v_and_b32_e32 v3, v3, v41
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[38:39], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[38:41], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[40:41]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_1202
; %bb.1199:                             ; %.preheader3.i.i.i45.1.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_1200:                             ; %.preheader3.i.i.i45.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[40:41], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v40
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v41
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[38:39], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[38:41], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[40:41]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_1200
; %bb.1201:                             ; %Flow3824
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_1202:                             ; %Flow3826
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_1203:                             ; %.loopexit4.i.i.i39.1.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[40:43], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[40:41], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_1205
; %bb.1204:                             ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_1205:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[42:43], 0, s[6:7]
	v_and_or_b32 v36, v36, s33, 32
	v_mov_b32_e32 v38, v0
	v_mov_b32_e32 v39, v1
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	s_nop 1
	global_store_dwordx4 v58, v[36:39], s[22:23]
	s_nop 1
	v_mov_b64_e32 v[36:37], s[6:7]
	v_mov_b64_e32 v[34:35], s[4:5]
	global_store_dwordx4 v58, v[34:37], s[22:23] offset:16
	global_store_dwordx4 v58, v[34:37], s[22:23] offset:32
	global_store_dwordx4 v58, v[34:37], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1213
; %bb.1206:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[38:39], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v36, s20
	v_mov_b32_e32 v37, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[40:41], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[38:39], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[36:39], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[38:39]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1209
; %bb.1207:                             ; %.preheader1.i.i.i43.1.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_1208:                             ; %.preheader1.i.i.i43.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[36:37], off
	v_mov_b32_e32 v34, s20
	v_mov_b32_e32 v35, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[34:37], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[36:37]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[36:37], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_1208
.LBB0_1209:                             ; %Flow3822
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1211
; %bb.1210:                             ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_1211:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_1213
; %bb.1212:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_1213:                             ; %Flow3823
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_1217
.LBB0_1214:                             ;   in Loop: Header=BB0_1217 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1216
; %bb.1215:                             ;   in Loop: Header=BB0_1217 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_1217
	s_branch .LBB0_1219
.LBB0_1216:                             ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_1219
.LBB0_1217:                             ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1214
; %bb.1218:                             ;   in Loop: Header=BB0_1217 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_1214
.LBB0_1219:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[26:27], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1223
; %bb.1220:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[18:19], v[2:3], 0, 1
	v_lshl_add_u64 v[22:23], v[18:19], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v36, v10
	v_mov_b32_e32 v37, v11
	v_cndmask_b32_e32 v35, v23, v19, vcc
	v_cndmask_b32_e32 v34, v22, v18, vcc
	v_and_b32_e32 v3, v35, v3
	v_and_b32_e32 v2, v34, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_1223
; %bb.1221:                             ; %.preheader.i.i.i42.1.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_1222:                             ; %.preheader.i.i.i42.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[36:37], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[36:37]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[36:37], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_1222
.LBB0_1223:                             ; %__ockl_printf_append_args.exit.1.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1229
; %bb.1224:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[36:37], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v36
	v_and_b32_e32 v3, v3, v37
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[34:35], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[36:37]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_1228
; %bb.1225:                             ; %.preheader3.i.i.i52.1.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_1226:                             ; %.preheader3.i.i.i52.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[36:37], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v36
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v37
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[34:35], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[36:37]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_1226
; %bb.1227:                             ; %Flow3810
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_1228:                             ; %Flow3812
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_1229:                             ; %.loopexit4.i.i.i46.1.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[34:37], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[34:35], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_1231
; %bb.1230:                             ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_1231:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[36:37], 0, s[6:7]
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	v_mov_b64_e32 v[38:39], s[6:7]
	v_and_or_b32 v26, v26, s33, 32
	v_mov_b32_e32 v29, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	v_mov_b64_e32 v[36:37], s[4:5]
	s_nop 3
	global_store_dwordx4 v58, v[26:29], s[22:23]
	global_store_dwordx4 v58, v[36:39], s[22:23] offset:16
	global_store_dwordx4 v58, v[36:39], s[22:23] offset:32
	global_store_dwordx4 v58, v[36:39], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1239
; %bb.1232:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[38:39], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v36, s20
	v_mov_b32_e32 v37, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[34:35], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[38:39], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[36:39], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[38:39]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1235
; %bb.1233:                             ; %.preheader1.i.i.i50.1.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_1234:                             ; %.preheader1.i.i.i50.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[36:37], off
	v_mov_b32_e32 v34, s20
	v_mov_b32_e32 v35, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[34:37], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[36:37]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[36:37], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_1234
.LBB0_1235:                             ; %Flow3808
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1237
; %bb.1236:                             ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_1237:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_1239
; %bb.1238:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_1239:                             ; %Flow3809
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_1243
.LBB0_1240:                             ;   in Loop: Header=BB0_1243 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1242
; %bb.1241:                             ;   in Loop: Header=BB0_1243 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_1243
	s_branch .LBB0_1245
.LBB0_1242:                             ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_1245
.LBB0_1243:                             ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1240
; %bb.1244:                             ;   in Loop: Header=BB0_1243 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_1240
.LBB0_1245:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[10:11], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1249
; %bb.1246:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[18:19], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[22:23], v[2:3], 0, 1
	v_lshl_add_u64 v[26:27], v[22:23], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[26:27]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v36, v14
	v_mov_b32_e32 v37, v15
	v_cndmask_b32_e32 v35, v27, v23, vcc
	v_cndmask_b32_e32 v34, v26, v22, vcc
	v_and_b32_e32 v3, v35, v3
	v_and_b32_e32 v2, v34, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[18:19], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[14:15], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[14:15]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_1249
; %bb.1247:                             ; %.preheader.i.i.i49.1.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_1248:                             ; %.preheader.i.i.i49.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[36:37], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[14:15], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[14:15], v[36:37]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[36:37], v[14:15]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_1248
.LBB0_1249:                             ; %__ockl_printf_append_args.exit53.1.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1255
; %bb.1250:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[36:37], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v36
	v_and_b32_e32 v3, v3, v37
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_load_dwordx2 v[34:35], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[36:37]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_1254
; %bb.1251:                             ; %.preheader3.i.i.i60.1.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_1252:                             ; %.preheader3.i.i.i60.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	global_load_dwordx2 v[18:19], v9, s[16:17]
	v_mov_b64_e32 v[36:37], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v14, v36
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[18:19]
	v_and_b32_e32 v7, v15, v37
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[14:15], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v14
	global_load_dwordx2 v[34:35], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[36:37]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_1252
; %bb.1253:                             ; %Flow3796
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_1254:                             ; %Flow3798
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_1255:                             ; %.loopexit4.i.i.i54.1.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	global_load_dwordx4 v[34:37], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[34:35], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_1257
; %bb.1256:                             ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_1257:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[14:15], v[36:37], 0, s[6:7]
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	v_mov_b64_e32 v[38:39], s[6:7]
	v_and_or_b32 v10, v10, s33, 32
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	v_mov_b64_e32 v[36:37], s[4:5]
	s_nop 3
	global_store_dwordx4 v58, v[10:13], s[22:23]
	global_store_dwordx4 v58, v[36:39], s[22:23] offset:16
	global_store_dwordx4 v58, v[36:39], s[22:23] offset:32
	global_store_dwordx4 v58, v[36:39], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1265
; %bb.1258:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[38:39], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	v_mov_b32_e32 v36, s20
	v_mov_b32_e32 v37, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[10:11], v[34:35], 0, s[22:23]
	global_store_dwordx2 v[10:11], v[38:39], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[36:39], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[38:39]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1261
; %bb.1259:                             ; %.preheader1.i.i.i58.1.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_1260:                             ; %.preheader1.i.i.i58.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[10:11], v[36:37], off
	v_mov_b32_e32 v34, s20
	v_mov_b32_e32 v35, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[34:37], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[36:37]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[36:37], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_1260
.LBB0_1261:                             ; %Flow3794
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1263
; %bb.1262:                             ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[10:11], v[8:9], off offset:8 sc1
.LBB0_1263:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[10:11], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_1265
; %bb.1264:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[10:11], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_1265:                             ; %Flow3795
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[14:15], 0, v[58:59]
	s_branch .LBB0_1269
.LBB0_1266:                             ;   in Loop: Header=BB0_1269 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1268
; %bb.1267:                             ;   in Loop: Header=BB0_1269 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_1269
	s_branch .LBB0_1271
.LBB0_1268:                             ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_1271
.LBB0_1269:                             ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1266
; %bb.1270:                             ;   in Loop: Header=BB0_1269 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_1266
.LBB0_1271:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[30:31], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1275
; %bb.1272:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[18:19], v[2:3], 0, 1
	v_lshl_add_u64 v[22:23], v[18:19], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v36, v10
	v_mov_b32_e32 v37, v11
	v_cndmask_b32_e32 v35, v23, v19, vcc
	v_cndmask_b32_e32 v34, v22, v18, vcc
	v_and_b32_e32 v3, v35, v3
	v_and_b32_e32 v2, v34, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_1275
; %bb.1273:                             ; %.preheader.i.i.i57.1.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_1274:                             ; %.preheader.i.i.i57.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[36:37], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[36:37]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[36:37], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_1274
.LBB0_1275:                             ; %__ockl_printf_append_args.exit61.1.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1281
; %bb.1276:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[36:37], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v36
	v_and_b32_e32 v3, v3, v37
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[34:35], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[36:37]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_1280
; %bb.1277:                             ; %.preheader3.i.i.i68.1.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_1278:                             ; %.preheader3.i.i.i68.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[36:37], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v36
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v37
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[34:35], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[36:37]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_1278
; %bb.1279:                             ; %Flow3782
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_1280:                             ; %Flow3784
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_1281:                             ; %.loopexit4.i.i.i62.1.i
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[34:37], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[34:35], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_1283
; %bb.1282:                             ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_1283:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[36:37], 0, s[6:7]
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	v_mov_b64_e32 v[38:39], s[6:7]
	v_and_or_b32 v30, v30, s34, 34
	v_mov_b32_e32 v33, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	v_mov_b64_e32 v[36:37], s[4:5]
	s_nop 3
	global_store_dwordx4 v58, v[30:33], s[22:23]
	global_store_dwordx4 v58, v[36:39], s[22:23] offset:16
	global_store_dwordx4 v58, v[36:39], s[22:23] offset:32
	global_store_dwordx4 v58, v[36:39], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1291
; %bb.1284:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[38:39], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	v_mov_b32_e32 v36, s20
	v_mov_b32_e32 v37, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[10:11], v[34:35], 0, s[22:23]
	global_store_dwordx2 v[10:11], v[38:39], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[36:39], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[38:39]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1287
; %bb.1285:                             ; %.preheader1.i.i.i66.1.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_1286:                             ; %.preheader1.i.i.i66.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[10:11], v[36:37], off
	v_mov_b32_e32 v34, s20
	v_mov_b32_e32 v35, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[14:15], v9, v[34:37], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[14:15], v[36:37]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[36:37], v[14:15]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_1286
.LBB0_1287:                             ; %Flow3780
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1289
; %bb.1288:                             ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[10:11], v[8:9], off offset:8 sc1
.LBB0_1289:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[14:15], v[10:11], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[14:15]
	s_cbranch_vccnz .LBB0_1291
; %bb.1290:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[10:11], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[14:15], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_1291:                             ; %Flow3781
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	s_branch .LBB0_1295
.LBB0_1292:                             ;   in Loop: Header=BB0_1295 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1294
; %bb.1293:                             ;   in Loop: Header=BB0_1295 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_1295
	s_branch .LBB0_1297
.LBB0_1294:                             ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_1297
.LBB0_1295:                             ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1292
; %bb.1296:                             ;   in Loop: Header=BB0_1295 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_1292
.LBB0_1297:                             ;   in Loop: Header=BB0_273 Depth=1
	s_and_b64 exec, exec, s[2:3]
	s_cbranch_execz .LBB0_1301
; %bb.1298:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[18:19], v[2:3], 0, 1
	v_lshl_add_u64 v[22:23], v[18:19], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v36, v10
	v_mov_b32_e32 v37, v11
	v_cndmask_b32_e32 v35, v23, v19, vcc
	v_cndmask_b32_e32 v34, v22, v18, vcc
	v_and_b32_e32 v3, v35, v3
	v_and_b32_e32 v2, v34, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_1301
; %bb.1299:                             ; %.preheader.i.i.i65.1.i.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_1300:                             ; %.preheader.i.i.i65.1.i
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[36:37], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[36:37]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[36:37], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_1300
.LBB0_1301:                             ; %Flow3908
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[18:19]
	v_add_u32_e32 v28, 16, v28
	v_lshl_add_u32 v64, v28, 7, s15
	v_add_u32_e32 v26, v64, v60
	;;#ASMSTART
	ds_read_b128 v[30:33], v32
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	scratch_store_dwordx4 v63, v[30:33], off offset:16
	s_and_saveexec_b64 s[18:19], s[0:1]
	s_cbranch_execz .LBB0_1571
; %bb.1302:                             ; %if.then.i.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1308
; %bb.1303:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[32:33], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v32
	v_and_b32_e32 v3, v3, v33
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[32:33]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_1307
; %bb.1304:                             ; %.preheader3.i.i.i.i.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_1305:                             ; %.preheader3.i.i.i.i.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[32:33], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v32
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v33
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[32:33]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_1305
; %bb.1306:                             ; %Flow3766
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_1307:                             ; %Flow3768
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_1308:                             ; %.loopexit4.i.i.i.i.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[30:33], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[30:31], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_1310
; %bb.1309:                             ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_1310:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[32:33], 0, s[6:7]
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	v_mov_b64_e32 v[34:35], s[6:7]
	v_mov_b32_e32 v7, v9
	v_mov_b32_e32 v8, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	v_mov_b64_e32 v[32:33], s[4:5]
	s_nop 3
	global_store_dwordx4 v58, v[6:9], s[22:23]
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:16
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:32
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1318
; %bb.1311:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[34:35], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v32, s20
	v_mov_b32_e32 v33, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[30:31], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[34:35], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[32:35], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[34:35]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1314
; %bb.1312:                             ; %.preheader1.i.i.i.i.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_1313:                             ; %.preheader1.i.i.i.i.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[32:33], off
	v_mov_b32_e32 v30, s20
	v_mov_b32_e32 v31, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[30:33], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[32:33]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[32:33], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_1313
.LBB0_1314:                             ; %Flow3764
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1316
; %bb.1315:                             ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_1316:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_1318
; %bb.1317:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_1318:                             ; %Flow3765
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_1322
.LBB0_1319:                             ;   in Loop: Header=BB0_1322 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1321
; %bb.1320:                             ;   in Loop: Header=BB0_1322 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_1322
	s_branch .LBB0_1324
.LBB0_1321:                             ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_1324
.LBB0_1322:                             ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1319
; %bb.1323:                             ;   in Loop: Header=BB0_1322 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_1319
.LBB0_1324:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[30:31], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1328
; %bb.1325:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[18:19], v[2:3], 0, 1
	v_lshl_add_u64 v[22:23], v[18:19], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v34, v10
	v_mov_b32_e32 v35, v11
	v_cndmask_b32_e32 v33, v23, v19, vcc
	v_cndmask_b32_e32 v32, v22, v18, vcc
	v_and_b32_e32 v3, v33, v3
	v_and_b32_e32 v2, v32, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[34:35], v9, v[32:35], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[34:35], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_1328
; %bb.1326:                             ; %.preheader.i.i.i.i.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_1327:                             ; %.preheader.i.i.i.i.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[34:35], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[32:35], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[34:35]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[34:35], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_1327
.LBB0_1328:                             ; %__ockl_printf_begin.exit.i.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	s_andn2_b64 vcc, exec, s[8:9]
	s_cbranch_vccnz .LBB0_1413
; %bb.1329:                             ;   in Loop: Header=BB0_273 Depth=1
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v2, 2, v30
	v_and_b32_e32 v32, -3, v30
	v_mov_b32_e32 v33, v31
	s_mov_b64 s[20:21], 0x43
	s_getpc_b64 s[6:7]
	s_add_u32 s6, s6, .str.3@rel32@lo+4
	s_addc_u32 s7, s7, .str.3@rel32@hi+12
	s_branch .LBB0_1331
.LBB0_1330:                             ; %__ockl_hostcall_preview.exit19.i.i.i.1
                                        ;   in Loop: Header=BB0_1331 Depth=2
	s_or_b64 exec, exec, s[26:27]
	s_sub_u32 s20, s20, s22
	s_subb_u32 s21, s21, s23
	s_add_u32 s6, s6, s22
	s_addc_u32 s7, s7, s23
	s_cmp_eq_u64 s[20:21], 0
	s_cbranch_scc1 .LBB0_1412
.LBB0_1331:                             ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_1334 Depth 3
                                        ;       Child Loop BB0_1341 Depth 3
                                        ;       Child Loop BB0_1349 Depth 3
                                        ;       Child Loop BB0_1357 Depth 3
                                        ;       Child Loop BB0_1365 Depth 3
                                        ;       Child Loop BB0_1373 Depth 3
                                        ;       Child Loop BB0_1381 Depth 3
                                        ;       Child Loop BB0_1389 Depth 3
                                        ;       Child Loop BB0_1397 Depth 3
                                        ;       Child Loop BB0_1406 Depth 3
                                        ;       Child Loop BB0_1411 Depth 3
	v_cmp_lt_u64_e64 s[2:3], s[20:21], 56
	s_and_b64 s[2:3], s[2:3], exec
	v_cmp_gt_u64_e64 s[2:3], s[20:21], 7
	s_cselect_b32 s23, s21, 0
	s_cselect_b32 s22, s20, 56
	s_and_b64 vcc, exec, s[2:3]
	s_cbranch_vccnz .LBB0_1336
; %bb.1332:                             ;   in Loop: Header=BB0_1331 Depth=2
	s_mov_b64 s[2:3], 0
	s_cmp_eq_u64 s[20:21], 0
	s_waitcnt vmcnt(0)
	v_mov_b64_e32 v[34:35], 0
	s_cbranch_scc1 .LBB0_1335
; %bb.1333:                             ; %.preheader30.i.i.i.1.preheader
                                        ;   in Loop: Header=BB0_1331 Depth=2
	s_lshl_b64 s[24:25], s[22:23], 3
	s_mov_b64 s[26:27], 0
	v_mov_b64_e32 v[34:35], 0
	s_mov_b64 s[28:29], s[6:7]
.LBB0_1334:                             ; %.preheader30.i.i.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1331 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	global_load_ubyte v3, v9, s[28:29]
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s26, v[8:9]
	s_add_u32 s26, s26, 8
	s_addc_u32 s27, s27, 0
	s_add_u32 s28, s28, 1
	s_addc_u32 s29, s29, 0
	v_or_b32_e32 v34, v10, v34
	s_cmp_eq_u32 s24, s26
	v_or_b32_e32 v35, v11, v35
	s_cbranch_scc0 .LBB0_1334
.LBB0_1335:                             ; %Flow3734
                                        ;   in Loop: Header=BB0_1331 Depth=2
	s_mov_b32 s5, 0
	s_andn2_b64 vcc, exec, s[2:3]
	s_mov_b64 s[2:3], s[6:7]
	s_cbranch_vccz .LBB0_1337
	s_branch .LBB0_1338
.LBB0_1336:                             ;   in Loop: Header=BB0_1331 Depth=2
                                        ; implicit-def: $vgpr34_vgpr35
                                        ; implicit-def: $sgpr5
	s_mov_b64 s[2:3], s[6:7]
.LBB0_1337:                             ;   in Loop: Header=BB0_1331 Depth=2
	global_load_dwordx2 v[34:35], v9, s[6:7]
	s_add_i32 s5, s22, -8
	s_add_u32 s2, s6, 8
	s_addc_u32 s3, s7, 0
.LBB0_1338:                             ; %.loopexit31.i.i.i.1
                                        ;   in Loop: Header=BB0_1331 Depth=2
	s_cmp_gt_u32 s5, 7
	s_cbranch_scc1 .LBB0_1342
; %bb.1339:                             ;   in Loop: Header=BB0_1331 Depth=2
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1343
; %bb.1340:                             ; %.preheader28.i.i.i.1.preheader
                                        ;   in Loop: Header=BB0_1331 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[36:37], 0
	s_mov_b64 s[26:27], 0
.LBB0_1341:                             ; %.preheader28.i.i.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1331 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s28, s2, s26
	s_addc_u32 s29, s3, s27
	global_load_ubyte v3, v9, s[28:29]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v36, v10, v36
	s_cmp_eq_u32 s5, s26
	v_or_b32_e32 v37, v11, v37
	s_cbranch_scc0 .LBB0_1341
	s_branch .LBB0_1344
.LBB0_1342:                             ;   in Loop: Header=BB0_1331 Depth=2
                                        ; implicit-def: $vgpr36_vgpr37
                                        ; implicit-def: $sgpr28
	s_branch .LBB0_1345
.LBB0_1343:                             ;   in Loop: Header=BB0_1331 Depth=2
	v_mov_b64_e32 v[36:37], 0
.LBB0_1344:                             ; %Flow3729
                                        ;   in Loop: Header=BB0_1331 Depth=2
	s_mov_b32 s28, 0
	s_cbranch_execnz .LBB0_1346
.LBB0_1345:                             ;   in Loop: Header=BB0_1331 Depth=2
	global_load_dwordx2 v[36:37], v9, s[2:3]
	s_add_i32 s28, s5, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_1346:                             ; %.loopexit29.i.i.i.1
                                        ;   in Loop: Header=BB0_1331 Depth=2
	s_cmp_gt_u32 s28, 7
	s_cbranch_scc1 .LBB0_1350
; %bb.1347:                             ;   in Loop: Header=BB0_1331 Depth=2
	s_cmp_eq_u32 s28, 0
	s_cbranch_scc1 .LBB0_1351
; %bb.1348:                             ; %.preheader26.i.i.i.1.preheader
                                        ;   in Loop: Header=BB0_1331 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[38:39], 0
	s_mov_b64 s[26:27], 0
.LBB0_1349:                             ; %.preheader26.i.i.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1331 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s30, s2, s26
	s_addc_u32 s31, s3, s27
	global_load_ubyte v3, v9, s[30:31]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v38, v10, v38
	s_cmp_eq_u32 s28, s26
	v_or_b32_e32 v39, v11, v39
	s_cbranch_scc0 .LBB0_1349
	s_branch .LBB0_1352
.LBB0_1350:                             ;   in Loop: Header=BB0_1331 Depth=2
                                        ; implicit-def: $sgpr5
	s_branch .LBB0_1353
.LBB0_1351:                             ;   in Loop: Header=BB0_1331 Depth=2
	v_mov_b64_e32 v[38:39], 0
.LBB0_1352:                             ; %Flow3724
                                        ;   in Loop: Header=BB0_1331 Depth=2
	s_mov_b32 s5, 0
	s_cbranch_execnz .LBB0_1354
.LBB0_1353:                             ;   in Loop: Header=BB0_1331 Depth=2
	global_load_dwordx2 v[38:39], v9, s[2:3]
	s_add_i32 s5, s28, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_1354:                             ; %.loopexit27.i.i.i.1
                                        ;   in Loop: Header=BB0_1331 Depth=2
	s_cmp_gt_u32 s5, 7
	s_cbranch_scc1 .LBB0_1358
; %bb.1355:                             ;   in Loop: Header=BB0_1331 Depth=2
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1359
; %bb.1356:                             ; %.preheader24.i.i.i.1.preheader
                                        ;   in Loop: Header=BB0_1331 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[40:41], 0
	s_mov_b64 s[26:27], 0
.LBB0_1357:                             ; %.preheader24.i.i.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1331 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s28, s2, s26
	s_addc_u32 s29, s3, s27
	global_load_ubyte v3, v9, s[28:29]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v40, v10, v40
	s_cmp_eq_u32 s5, s26
	v_or_b32_e32 v41, v11, v41
	s_cbranch_scc0 .LBB0_1357
	s_branch .LBB0_1360
.LBB0_1358:                             ;   in Loop: Header=BB0_1331 Depth=2
                                        ; implicit-def: $vgpr40_vgpr41
                                        ; implicit-def: $sgpr28
	s_branch .LBB0_1361
.LBB0_1359:                             ;   in Loop: Header=BB0_1331 Depth=2
	v_mov_b64_e32 v[40:41], 0
.LBB0_1360:                             ; %Flow3719
                                        ;   in Loop: Header=BB0_1331 Depth=2
	s_mov_b32 s28, 0
	s_cbranch_execnz .LBB0_1362
.LBB0_1361:                             ;   in Loop: Header=BB0_1331 Depth=2
	global_load_dwordx2 v[40:41], v9, s[2:3]
	s_add_i32 s28, s5, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_1362:                             ; %.loopexit25.i.i.i.1
                                        ;   in Loop: Header=BB0_1331 Depth=2
	s_cmp_gt_u32 s28, 7
	s_cbranch_scc1 .LBB0_1366
; %bb.1363:                             ;   in Loop: Header=BB0_1331 Depth=2
	s_cmp_eq_u32 s28, 0
	s_cbranch_scc1 .LBB0_1367
; %bb.1364:                             ; %.preheader22.i.i.i.1.preheader
                                        ;   in Loop: Header=BB0_1331 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[42:43], 0
	s_mov_b64 s[26:27], 0
.LBB0_1365:                             ; %.preheader22.i.i.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1331 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s30, s2, s26
	s_addc_u32 s31, s3, s27
	global_load_ubyte v3, v9, s[30:31]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v42, v10, v42
	s_cmp_eq_u32 s28, s26
	v_or_b32_e32 v43, v11, v43
	s_cbranch_scc0 .LBB0_1365
	s_branch .LBB0_1368
.LBB0_1366:                             ;   in Loop: Header=BB0_1331 Depth=2
                                        ; implicit-def: $sgpr5
	s_branch .LBB0_1369
.LBB0_1367:                             ;   in Loop: Header=BB0_1331 Depth=2
	v_mov_b64_e32 v[42:43], 0
.LBB0_1368:                             ; %Flow3714
                                        ;   in Loop: Header=BB0_1331 Depth=2
	s_mov_b32 s5, 0
	s_cbranch_execnz .LBB0_1370
.LBB0_1369:                             ;   in Loop: Header=BB0_1331 Depth=2
	global_load_dwordx2 v[42:43], v9, s[2:3]
	s_add_i32 s5, s28, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_1370:                             ; %.loopexit23.i.i.i.1
                                        ;   in Loop: Header=BB0_1331 Depth=2
	s_cmp_gt_u32 s5, 7
	s_cbranch_scc1 .LBB0_1374
; %bb.1371:                             ;   in Loop: Header=BB0_1331 Depth=2
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1375
; %bb.1372:                             ; %.preheader20.i.i.i.1.preheader
                                        ;   in Loop: Header=BB0_1331 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[44:45], 0
	s_mov_b64 s[26:27], 0
.LBB0_1373:                             ; %.preheader20.i.i.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1331 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s28, s2, s26
	s_addc_u32 s29, s3, s27
	global_load_ubyte v3, v9, s[28:29]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v44, v10, v44
	s_cmp_eq_u32 s5, s26
	v_or_b32_e32 v45, v11, v45
	s_cbranch_scc0 .LBB0_1373
	s_branch .LBB0_1376
.LBB0_1374:                             ;   in Loop: Header=BB0_1331 Depth=2
                                        ; implicit-def: $vgpr44_vgpr45
                                        ; implicit-def: $sgpr28
	s_branch .LBB0_1377
.LBB0_1375:                             ;   in Loop: Header=BB0_1331 Depth=2
	v_mov_b64_e32 v[44:45], 0
.LBB0_1376:                             ; %Flow3709
                                        ;   in Loop: Header=BB0_1331 Depth=2
	s_mov_b32 s28, 0
	s_cbranch_execnz .LBB0_1378
.LBB0_1377:                             ;   in Loop: Header=BB0_1331 Depth=2
	global_load_dwordx2 v[44:45], v9, s[2:3]
	s_add_i32 s28, s5, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_1378:                             ; %.loopexit21.i.i.i.1
                                        ;   in Loop: Header=BB0_1331 Depth=2
	s_cmp_gt_u32 s28, 7
	s_cbranch_scc1 .LBB0_1382
; %bb.1379:                             ;   in Loop: Header=BB0_1331 Depth=2
	s_cmp_eq_u32 s28, 0
	s_cbranch_scc1 .LBB0_1383
; %bb.1380:                             ; %.preheader.i.i.i.1.preheader
                                        ;   in Loop: Header=BB0_1331 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[46:47], 0
	s_mov_b64 s[26:27], s[2:3]
.LBB0_1381:                             ; %.preheader.i.i.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1331 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	global_load_ubyte v3, v9, s[26:27]
	s_add_i32 s28, s28, -1
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	v_or_b32_e32 v46, v10, v46
	s_cmp_eq_u32 s28, 0
	v_or_b32_e32 v47, v11, v47
	s_cbranch_scc0 .LBB0_1381
	s_branch .LBB0_1384
.LBB0_1382:                             ;   in Loop: Header=BB0_1331 Depth=2
	s_branch .LBB0_1385
.LBB0_1383:                             ;   in Loop: Header=BB0_1331 Depth=2
	v_mov_b64_e32 v[46:47], 0
.LBB0_1384:                             ; %Flow3704
                                        ;   in Loop: Header=BB0_1331 Depth=2
	s_cbranch_execnz .LBB0_1386
.LBB0_1385:                             ;   in Loop: Header=BB0_1331 Depth=2
	global_load_dwordx2 v[46:47], v9, s[2:3]
.LBB0_1386:                             ; %.loopexit.i.i.i.1
                                        ;   in Loop: Header=BB0_1331 Depth=2
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[10:11], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_1392
; %bb.1387:                             ;   in Loop: Header=BB0_1331 Depth=2
	global_load_dwordx2 v[50:51], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v3, v10, v50
	v_and_b32_e32 v7, v11, v51
	v_mul_lo_u32 v7, v7, 24
	v_mul_hi_u32 v8, v3, 24
	v_add_u32_e32 v11, v8, v7
	v_mul_lo_u32 v10, v3, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[14:15], 0, v[10:11]
	global_load_dwordx2 v[48:49], v[10:11], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[48:51], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[10:11], v[50:51]
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB0_1391
; %bb.1388:                             ; %.preheader3.i.i18.i.i.i.1.preheader
                                        ;   in Loop: Header=BB0_1331 Depth=2
	s_mov_b64 s[28:29], 0
.LBB0_1389:                             ; %.preheader3.i.i18.i.i.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1331 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_sleep 1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	global_load_dwordx2 v[18:19], v9, s[16:17]
	v_mov_b64_e32 v[50:51], v[10:11]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v7, v14, v50
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[10:11], s[30:31], v7, 24, v[18:19]
	v_and_b32_e32 v3, v15, v51
	v_mov_b32_e32 v8, v11
	v_mad_u64_u32 v[14:15], s[30:31], v3, 24, v[8:9]
	v_mov_b32_e32 v11, v14
	global_load_dwordx2 v[48:49], v[10:11], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[48:51], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[10:11], v[50:51]
	s_or_b64 s[28:29], vcc, s[28:29]
	s_andn2_b64 exec, exec, s[28:29]
	s_cbranch_execnz .LBB0_1389
; %bb.1390:                             ; %Flow3699
                                        ;   in Loop: Header=BB0_1331 Depth=2
	s_or_b64 exec, exec, s[28:29]
.LBB0_1391:                             ; %Flow3701
                                        ;   in Loop: Header=BB0_1331 Depth=2
	s_or_b64 exec, exec, s[26:27]
.LBB0_1392:                             ; %.loopexit4.i.i13.i.i.i.1
                                        ;   in Loop: Header=BB0_1331 Depth=2
	s_or_b64 exec, exec, s[24:25]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	global_load_dwordx4 v[48:51], v9, s[16:17]
	v_readfirstlane_b32 s24, v10
	v_readfirstlane_b32 s25, v11
	s_mov_b64 s[26:27], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s28, v14
	v_readfirstlane_b32 s29, v15
	s_and_b64 s[28:29], s[24:25], s[28:29]
	s_mul_i32 s5, s29, 24
	s_mul_hi_u32 s30, s28, 24
	s_add_i32 s31, s30, s5
	s_mul_i32 s30, s28, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[48:49], 0, s[30:31]
	s_and_saveexec_b64 s[30:31], s[2:3]
	s_cbranch_execz .LBB0_1394
; %bb.1393:                             ;   in Loop: Header=BB0_1331 Depth=2
	v_mov_b64_e32 v[22:23], s[26:27]
	global_store_dwordx4 v[10:11], v[22:25], off offset:8
.LBB0_1394:                             ;   in Loop: Header=BB0_1331 Depth=2
	s_or_b64 exec, exec, s[30:31]
	v_or_b32_e32 v3, 0, v33
	v_or_b32_e32 v7, v32, v2
	v_cmp_gt_u64_e64 vcc, s[20:21], 56
	s_lshl_b32 s5, s22, 2
	s_lshl_b64 s[26:27], s[28:29], 12
	v_cndmask_b32_e32 v33, v3, v33, vcc
	v_cndmask_b32_e32 v3, v7, v32, vcc
	s_add_i32 s5, s5, 28
	v_lshl_add_u64 v[14:15], v[50:51], 0, s[26:27]
	s_and_b32 s5, s5, 0x1e0
	v_and_b32_e32 v3, 0xffffff1f, v3
	v_or_b32_e32 v32, s5, v3
	v_readfirstlane_b32 s26, v14
	v_readfirstlane_b32 s27, v15
	s_nop 4
	global_store_dwordx4 v58, v[32:35], s[26:27]
	global_store_dwordx4 v58, v[36:39], s[26:27] offset:16
	global_store_dwordx4 v58, v[40:43], s[26:27] offset:32
	global_store_dwordx4 v58, v[44:47], s[26:27] offset:48
	s_and_saveexec_b64 s[26:27], s[2:3]
	s_cbranch_execz .LBB0_1402
; %bb.1395:                             ;   in Loop: Header=BB0_1331 Depth=2
	global_load_dwordx2 v[36:37], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[18:19], v9, s[16:17] offset:40
	v_mov_b32_e32 v34, s24
	v_mov_b32_e32 v35, s25
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s28, v18
	v_readfirstlane_b32 s29, v19
	s_and_b64 s[28:29], s[28:29], s[24:25]
	s_mul_i32 s5, s29, 24
	s_mul_hi_u32 s29, s28, 24
	s_mul_i32 s28, s28, 24
	s_add_i32 s29, s29, s5
	v_lshl_add_u64 v[18:19], v[48:49], 0, s[28:29]
	global_store_dwordx2 v[18:19], v[36:37], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[34:35], v9, v[34:37], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[34:35], v[36:37]
	s_and_saveexec_b64 s[28:29], vcc
	s_cbranch_execz .LBB0_1398
; %bb.1396:                             ; %.preheader1.i.i16.i.i.i.1.preheader
                                        ;   in Loop: Header=BB0_1331 Depth=2
	s_mov_b64 s[30:31], 0
.LBB0_1397:                             ; %.preheader1.i.i16.i.i.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1331 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_sleep 1
	global_store_dwordx2 v[18:19], v[34:35], off
	v_mov_b32_e32 v32, s24
	v_mov_b32_e32 v33, s25
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[22:23], v9, v[32:35], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[22:23], v[34:35]
	s_or_b64 s[30:31], vcc, s[30:31]
	v_mov_b64_e32 v[34:35], v[22:23]
	s_andn2_b64 exec, exec, s[30:31]
	s_cbranch_execnz .LBB0_1397
.LBB0_1398:                             ; %Flow3697
                                        ;   in Loop: Header=BB0_1331 Depth=2
	s_or_b64 exec, exec, s[28:29]
	global_load_dwordx2 v[18:19], v9, s[16:17] offset:16
	s_mov_b64 s[30:31], exec
	v_mbcnt_lo_u32_b32 v3, s30, 0
	v_mbcnt_hi_u32_b32 v3, s31, v3
	v_cmp_eq_u32_e32 vcc, 0, v3
	s_and_saveexec_b64 s[28:29], vcc
	s_cbranch_execz .LBB0_1400
; %bb.1399:                             ;   in Loop: Header=BB0_1331 Depth=2
	s_bcnt1_i32_b64 s5, s[30:31]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[18:19], v[8:9], off offset:8 sc1
.LBB0_1400:                             ;   in Loop: Header=BB0_1331 Depth=2
	s_or_b64 exec, exec, s[28:29]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[22:23], v[18:19], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_cbranch_vccnz .LBB0_1402
; %bb.1401:                             ;   in Loop: Header=BB0_1331 Depth=2
	global_load_dword v8, v[18:19], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v3, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v3
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[22:23], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_1402:                             ; %Flow3698
                                        ;   in Loop: Header=BB0_1331 Depth=2
	s_or_b64 exec, exec, s[26:27]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[14:15], v[14:15], 0, v[58:59]
	s_branch .LBB0_1406
.LBB0_1403:                             ;   in Loop: Header=BB0_1406 Depth=3
	s_or_b64 exec, exec, s[26:27]
	v_readfirstlane_b32 s5, v3
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1405
; %bb.1404:                             ;   in Loop: Header=BB0_1406 Depth=3
	s_sleep 1
	s_cbranch_execnz .LBB0_1406
	s_branch .LBB0_1408
.LBB0_1405:                             ;   in Loop: Header=BB0_1331 Depth=2
	s_branch .LBB0_1408
.LBB0_1406:                             ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1331 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_mov_b32_e32 v3, 1
	s_and_saveexec_b64 s[26:27], s[2:3]
	s_cbranch_execz .LBB0_1403
; %bb.1407:                             ;   in Loop: Header=BB0_1406 Depth=3
	global_load_dword v3, v[10:11], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v3, 1, v3
	s_branch .LBB0_1403
.LBB0_1408:                             ;   in Loop: Header=BB0_1331 Depth=2
	global_load_dwordx4 v[32:35], v[14:15], off
	s_and_saveexec_b64 s[26:27], s[2:3]
	s_cbranch_execz .LBB0_1330
; %bb.1409:                             ;   in Loop: Header=BB0_1331 Depth=2
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[18:19], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[22:23], v[10:11], 0, 1
	v_lshl_add_u64 v[34:35], v[22:23], 0, s[24:25]
	v_cmp_eq_u64_e32 vcc, 0, v[34:35]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v36, v14
	v_mov_b32_e32 v37, v15
	v_cndmask_b32_e32 v35, v35, v23, vcc
	v_cndmask_b32_e32 v34, v34, v22, vcc
	v_and_b32_e32 v3, v35, v11
	v_and_b32_e32 v7, v34, v10
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v8, v7, 24
	v_mul_lo_u32 v10, v7, 24
	v_add_u32_e32 v11, v8, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[18:19], 0, v[10:11]
	global_store_dwordx2 v[10:11], v[14:15], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[14:15]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_1330
; %bb.1410:                             ; %.preheader.i.i15.i.i.i.1.preheader
                                        ;   in Loop: Header=BB0_1331 Depth=2
	s_mov_b64 s[2:3], 0
.LBB0_1411:                             ; %.preheader.i.i15.i.i.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1331 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_sleep 1
	global_store_dwordx2 v[10:11], v[36:37], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[14:15], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[14:15], v[36:37]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[36:37], v[14:15]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_1411
	s_branch .LBB0_1330
.LBB0_1412:                             ; %Flow3737
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_1441
.LBB0_1413:                             ;   in Loop: Header=BB0_273 Depth=1
                                        ; implicit-def: $vgpr32_vgpr33
	s_cbranch_execz .LBB0_1441
; %bb.1414:                             ;   in Loop: Header=BB0_273 Depth=1
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1420
; %bb.1415:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[34:35], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v34
	v_and_b32_e32 v3, v3, v35
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[32:33], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[32:35], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[34:35]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_1419
; %bb.1416:                             ; %.preheader3.i.i.i9.i.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_1417:                             ; %.preheader3.i.i.i9.i.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[34:35], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v34
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v35
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[32:33], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[32:35], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[34:35]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_1417
; %bb.1418:                             ; %Flow3750
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_1419:                             ; %Flow3752
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_1420:                             ; %.loopexit4.i.i.i4.i.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[34:37], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[34:35], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_1422
; %bb.1421:                             ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_1422:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[36:37], 0, s[6:7]
	v_and_or_b32 v30, v30, s33, 32
	v_mov_b32_e32 v32, v9
	v_mov_b32_e32 v33, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	s_nop 1
	global_store_dwordx4 v58, v[30:33], s[22:23]
	s_nop 1
	v_mov_b64_e32 v[32:33], s[6:7]
	v_mov_b64_e32 v[30:31], s[4:5]
	global_store_dwordx4 v58, v[30:33], s[22:23] offset:16
	global_store_dwordx4 v58, v[30:33], s[22:23] offset:32
	global_store_dwordx4 v58, v[30:33], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1430
; %bb.1423:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[38:39], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v36, s20
	v_mov_b32_e32 v37, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[34:35], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[38:39], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[36:39], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[38:39]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1426
; %bb.1424:                             ; %.preheader1.i.i.i7.i.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_1425:                             ; %.preheader1.i.i.i7.i.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[32:33], off
	v_mov_b32_e32 v30, s20
	v_mov_b32_e32 v31, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[30:33], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[32:33]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[32:33], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_1425
.LBB0_1426:                             ; %Flow3748
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1428
; %bb.1427:                             ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_1428:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_1430
; %bb.1429:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_1430:                             ; %Flow3749
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_1434
.LBB0_1431:                             ;   in Loop: Header=BB0_1434 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1433
; %bb.1432:                             ;   in Loop: Header=BB0_1434 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_1434
	s_branch .LBB0_1436
.LBB0_1433:                             ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_1436
.LBB0_1434:                             ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1431
; %bb.1435:                             ;   in Loop: Header=BB0_1434 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_1431
.LBB0_1436:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[32:33], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1440
; %bb.1437:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[18:19], v[2:3], 0, 1
	v_lshl_add_u64 v[22:23], v[18:19], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v36, v10
	v_mov_b32_e32 v37, v11
	v_cndmask_b32_e32 v35, v23, v19, vcc
	v_cndmask_b32_e32 v34, v22, v18, vcc
	v_and_b32_e32 v3, v35, v3
	v_and_b32_e32 v2, v34, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_1440
; %bb.1438:                             ; %.preheader.i.i.i6.i.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_1439:                             ; %.preheader.i.i.i6.i.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[36:37], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[36:37]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[36:37], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_1439
.LBB0_1440:                             ; %Flow3741
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
.LBB0_1441:                             ; %__ockl_printf_append_string_n.exit.i.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1447
; %bb.1442:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[36:37], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v36
	v_and_b32_e32 v3, v3, v37
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[34:35], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[36:37]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_1446
; %bb.1443:                             ; %.preheader3.i.i.i16.i.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_1444:                             ; %.preheader3.i.i.i16.i.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[36:37], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v36
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v37
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[34:35], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[36:37]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_1444
; %bb.1445:                             ; %Flow3685
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_1446:                             ; %Flow3687
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_1447:                             ; %.loopexit4.i.i.i10.i.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[36:39], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[36:37], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_1449
; %bb.1448:                             ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_1449:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[38:39], 0, s[6:7]
	v_and_or_b32 v32, v32, s33, 32
	v_mov_b32_e32 v34, v26
	v_mov_b32_e32 v35, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	s_nop 1
	global_store_dwordx4 v58, v[32:35], s[22:23]
	s_nop 1
	v_mov_b64_e32 v[32:33], s[6:7]
	v_mov_b64_e32 v[30:31], s[4:5]
	global_store_dwordx4 v58, v[30:33], s[22:23] offset:16
	global_store_dwordx4 v58, v[30:33], s[22:23] offset:32
	global_store_dwordx4 v58, v[30:33], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1457
; %bb.1450:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[34:35], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v32, s20
	v_mov_b32_e32 v33, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[36:37], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[34:35], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[32:35], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[34:35]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1453
; %bb.1451:                             ; %.preheader1.i.i.i14.i.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_1452:                             ; %.preheader1.i.i.i14.i.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[32:33], off
	v_mov_b32_e32 v30, s20
	v_mov_b32_e32 v31, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[30:33], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[32:33]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[32:33], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_1452
.LBB0_1453:                             ; %Flow3683
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1455
; %bb.1454:                             ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_1455:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_1457
; %bb.1456:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_1457:                             ; %Flow3684
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_1461
.LBB0_1458:                             ;   in Loop: Header=BB0_1461 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1460
; %bb.1459:                             ;   in Loop: Header=BB0_1461 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_1461
	s_branch .LBB0_1463
.LBB0_1460:                             ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_1463
.LBB0_1461:                             ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1458
; %bb.1462:                             ;   in Loop: Header=BB0_1461 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_1458
.LBB0_1463:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[14:15], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1467
; %bb.1464:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[18:19], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[22:23], v[2:3], 0, 1
	v_lshl_add_u64 v[30:31], v[22:23], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[30:31]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v32, v10
	v_mov_b32_e32 v33, v11
	v_cndmask_b32_e32 v31, v31, v23, vcc
	v_cndmask_b32_e32 v30, v30, v22, vcc
	v_and_b32_e32 v3, v31, v3
	v_and_b32_e32 v2, v30, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[18:19], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_1467
; %bb.1465:                             ; %.preheader.i.i.i13.i.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_1466:                             ; %.preheader.i.i.i13.i.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[32:33], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[32:33]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[32:33], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_1466
.LBB0_1467:                             ; %__ockl_printf_append_args.exit.i.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1473
; %bb.1468:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[32:33], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v32
	v_and_b32_e32 v3, v3, v33
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[32:33]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_1472
; %bb.1469:                             ; %.preheader3.i.i.i23.i.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_1470:                             ; %.preheader3.i.i.i23.i.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[18:19], v9, s[16:17]
	v_mov_b64_e32 v[32:33], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v32
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[18:19]
	v_and_b32_e32 v7, v11, v33
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[32:33]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_1470
; %bb.1471:                             ; %Flow3671
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_1472:                             ; %Flow3673
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_1473:                             ; %.loopexit4.i.i.i17.i.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[30:33], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[30:31], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_1475
; %bb.1474:                             ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_1475:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[32:33], 0, s[6:7]
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	v_mov_b64_e32 v[34:35], s[6:7]
	v_and_or_b32 v14, v14, s33, 32
	v_mov_b32_e32 v17, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	v_mov_b64_e32 v[32:33], s[4:5]
	s_nop 3
	global_store_dwordx4 v58, v[14:17], s[22:23]
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:16
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:32
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1483
; %bb.1476:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[34:35], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v32, s20
	v_mov_b32_e32 v33, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[30:31], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[34:35], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[32:35], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[34:35]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1479
; %bb.1477:                             ; %.preheader1.i.i.i21.i.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_1478:                             ; %.preheader1.i.i.i21.i.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[32:33], off
	v_mov_b32_e32 v30, s20
	v_mov_b32_e32 v31, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[30:33], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[32:33]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[32:33], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_1478
.LBB0_1479:                             ; %Flow3669
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1481
; %bb.1480:                             ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_1481:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_1483
; %bb.1482:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_1483:                             ; %Flow3670
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_1487
.LBB0_1484:                             ;   in Loop: Header=BB0_1487 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1486
; %bb.1485:                             ;   in Loop: Header=BB0_1487 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_1487
	s_branch .LBB0_1489
.LBB0_1486:                             ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_1489
.LBB0_1487:                             ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1484
; %bb.1488:                             ;   in Loop: Header=BB0_1487 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_1484
.LBB0_1489:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[18:19], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1493
; %bb.1490:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[22:23], v[2:3], 0, 1
	v_lshl_add_u64 v[30:31], v[22:23], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[30:31]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v32, v10
	v_mov_b32_e32 v33, v11
	v_cndmask_b32_e32 v31, v31, v23, vcc
	v_cndmask_b32_e32 v30, v30, v22, vcc
	v_and_b32_e32 v3, v31, v3
	v_and_b32_e32 v2, v30, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_1493
; %bb.1491:                             ; %.preheader.i.i.i20.i.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_1492:                             ; %.preheader.i.i.i20.i.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[32:33], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[32:33]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[32:33], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_1492
.LBB0_1493:                             ; %__ockl_printf_append_args.exit24.i.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1499
; %bb.1494:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[32:33], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v32
	v_and_b32_e32 v3, v3, v33
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[32:33]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_1498
; %bb.1495:                             ; %.preheader3.i.i.i31.i.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_1496:                             ; %.preheader3.i.i.i31.i.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[32:33], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v32
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v33
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[32:33]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_1496
; %bb.1497:                             ; %Flow3657
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_1498:                             ; %Flow3659
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_1499:                             ; %.loopexit4.i.i.i25.i.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[30:33], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[30:31], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_1501
; %bb.1500:                             ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_1501:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[32:33], 0, s[6:7]
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	v_mov_b64_e32 v[34:35], s[6:7]
	v_and_or_b32 v18, v18, s33, 32
	v_mov_b32_e32 v21, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	v_mov_b64_e32 v[32:33], s[4:5]
	s_nop 3
	global_store_dwordx4 v58, v[18:21], s[22:23]
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:16
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:32
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1509
; %bb.1502:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[34:35], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v32, s20
	v_mov_b32_e32 v33, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[30:31], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[34:35], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[32:35], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[34:35]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1505
; %bb.1503:                             ; %.preheader1.i.i.i29.i.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_1504:                             ; %.preheader1.i.i.i29.i.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[32:33], off
	v_mov_b32_e32 v30, s20
	v_mov_b32_e32 v31, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[30:33], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[32:33]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[32:33], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_1504
.LBB0_1505:                             ; %Flow3655
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1507
; %bb.1506:                             ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_1507:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_1509
; %bb.1508:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_1509:                             ; %Flow3656
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_1513
.LBB0_1510:                             ;   in Loop: Header=BB0_1513 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1512
; %bb.1511:                             ;   in Loop: Header=BB0_1513 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_1513
	s_branch .LBB0_1515
.LBB0_1512:                             ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_1515
.LBB0_1513:                             ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1510
; %bb.1514:                             ;   in Loop: Header=BB0_1513 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_1510
.LBB0_1515:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[18:19], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1519
; %bb.1516:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[22:23], v[2:3], 0, 1
	v_lshl_add_u64 v[30:31], v[22:23], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[30:31]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v32, v10
	v_mov_b32_e32 v33, v11
	v_cndmask_b32_e32 v31, v31, v23, vcc
	v_cndmask_b32_e32 v30, v30, v22, vcc
	v_and_b32_e32 v3, v31, v3
	v_and_b32_e32 v2, v30, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_1519
; %bb.1517:                             ; %.preheader.i.i.i28.i.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_1518:                             ; %.preheader.i.i.i28.i.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[32:33], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[32:33]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[32:33], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_1518
.LBB0_1519:                             ; %__ockl_printf_append_args.exit32.i.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1525
; %bb.1520:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[32:33], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v32
	v_and_b32_e32 v3, v3, v33
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[32:33]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_1524
; %bb.1521:                             ; %.preheader3.i.i.i39.i.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_1522:                             ; %.preheader3.i.i.i39.i.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[32:33], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v32
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v33
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[32:33]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_1522
; %bb.1523:                             ; %Flow3643
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_1524:                             ; %Flow3645
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_1525:                             ; %.loopexit4.i.i.i33.i.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[30:33], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[30:31], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_1527
; %bb.1526:                             ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_1527:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[32:33], 0, s[6:7]
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	v_mov_b64_e32 v[34:35], s[6:7]
	v_and_or_b32 v18, v18, s33, 32
	v_mov_b32_e32 v21, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	v_mov_b64_e32 v[32:33], s[4:5]
	s_nop 3
	global_store_dwordx4 v58, v[18:21], s[22:23]
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:16
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:32
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1535
; %bb.1528:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[34:35], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v32, s20
	v_mov_b32_e32 v33, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[30:31], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[34:35], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[32:35], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[34:35]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1531
; %bb.1529:                             ; %.preheader1.i.i.i37.i.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_1530:                             ; %.preheader1.i.i.i37.i.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[32:33], off
	v_mov_b32_e32 v30, s20
	v_mov_b32_e32 v31, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[30:33], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[32:33]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[32:33], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_1530
.LBB0_1531:                             ; %Flow3641
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1533
; %bb.1532:                             ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_1533:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_1535
; %bb.1534:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_1535:                             ; %Flow3642
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_1539
.LBB0_1536:                             ;   in Loop: Header=BB0_1539 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1538
; %bb.1537:                             ;   in Loop: Header=BB0_1539 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_1539
	s_branch .LBB0_1541
.LBB0_1538:                             ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_1541
.LBB0_1539:                             ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1536
; %bb.1540:                             ;   in Loop: Header=BB0_1539 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_1536
.LBB0_1541:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[18:19], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1545
; %bb.1542:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[22:23], v[2:3], 0, 1
	v_lshl_add_u64 v[30:31], v[22:23], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[30:31]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v32, v10
	v_mov_b32_e32 v33, v11
	v_cndmask_b32_e32 v31, v31, v23, vcc
	v_cndmask_b32_e32 v30, v30, v22, vcc
	v_and_b32_e32 v3, v31, v3
	v_and_b32_e32 v2, v30, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_1545
; %bb.1543:                             ; %.preheader.i.i.i36.i.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_1544:                             ; %.preheader.i.i.i36.i.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[32:33], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[32:33]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[32:33], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_1544
.LBB0_1545:                             ; %__ockl_printf_append_args.exit40.i.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1551
; %bb.1546:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[32:33], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v32
	v_and_b32_e32 v3, v3, v33
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[32:33]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_1550
; %bb.1547:                             ; %.preheader3.i.i.i47.i.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_1548:                             ; %.preheader3.i.i.i47.i.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[32:33], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v32
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v33
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[32:33]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_1548
; %bb.1549:                             ; %Flow3629
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_1550:                             ; %Flow3631
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_1551:                             ; %.loopexit4.i.i.i41.i.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[30:33], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[30:31], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_1553
; %bb.1552:                             ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_1553:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[32:33], 0, s[6:7]
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	v_mov_b64_e32 v[34:35], s[6:7]
	v_and_or_b32 v18, v18, s34, 34
	v_mov_b32_e32 v21, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	v_mov_b64_e32 v[32:33], s[4:5]
	s_nop 3
	global_store_dwordx4 v58, v[18:21], s[22:23]
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:16
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:32
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1561
; %bb.1554:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[34:35], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	v_mov_b32_e32 v32, s20
	v_mov_b32_e32 v33, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[10:11], v[30:31], 0, s[22:23]
	global_store_dwordx2 v[10:11], v[34:35], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[32:35], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[34:35]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1557
; %bb.1555:                             ; %.preheader1.i.i.i45.i.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_1556:                             ; %.preheader1.i.i.i45.i.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[10:11], v[32:33], off
	v_mov_b32_e32 v30, s20
	v_mov_b32_e32 v31, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[14:15], v9, v[30:33], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[14:15], v[32:33]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[32:33], v[14:15]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_1556
.LBB0_1557:                             ; %Flow3627
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1559
; %bb.1558:                             ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[10:11], v[8:9], off offset:8 sc1
.LBB0_1559:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[14:15], v[10:11], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[14:15]
	s_cbranch_vccnz .LBB0_1561
; %bb.1560:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[10:11], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[14:15], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_1561:                             ; %Flow3628
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	s_branch .LBB0_1565
.LBB0_1562:                             ;   in Loop: Header=BB0_1565 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1564
; %bb.1563:                             ;   in Loop: Header=BB0_1565 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_1565
	s_branch .LBB0_1567
.LBB0_1564:                             ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_1567
.LBB0_1565:                             ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1562
; %bb.1566:                             ;   in Loop: Header=BB0_1565 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_1562
.LBB0_1567:                             ;   in Loop: Header=BB0_273 Depth=1
	s_and_b64 exec, exec, s[2:3]
	s_cbranch_execz .LBB0_1571
; %bb.1568:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[18:19], v[2:3], 0, 1
	v_lshl_add_u64 v[22:23], v[18:19], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v32, v10
	v_mov_b32_e32 v33, v11
	v_cndmask_b32_e32 v31, v23, v19, vcc
	v_cndmask_b32_e32 v30, v22, v18, vcc
	v_and_b32_e32 v3, v31, v3
	v_and_b32_e32 v2, v30, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_1571
; %bb.1569:                             ; %.preheader.i.i.i44.i.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_1570:                             ; %.preheader.i.i.i44.i.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[32:33], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[32:33]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[32:33], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_1570
.LBB0_1571:                             ; %Flow3769
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[18:19]
	v_lshrrev_b32_e32 v2, 3, v26
	v_bitop3_b32 v32, v2, v26, s35 bitop3:0x6c
	s_and_saveexec_b64 s[18:19], s[0:1]
	s_cbranch_execz .LBB0_1815
; %bb.1572:                             ; %if.then.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1578
; %bb.1573:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[36:37], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v36
	v_and_b32_e32 v3, v3, v37
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[34:35], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[36:37]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_1577
; %bb.1574:                             ; %.preheader3.i.i.i.i123.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_1575:                             ; %.preheader3.i.i.i.i123.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[36:37], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v36
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v37
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[34:35], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[36:37]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_1575
; %bb.1576:                             ; %Flow3613
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_1577:                             ; %Flow3615
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_1578:                             ; %.loopexit4.i.i.i.i87.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[34:37], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[34:35], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_1580
; %bb.1579:                             ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_1580:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[36:37], 0, s[6:7]
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	v_mov_b64_e32 v[38:39], s[6:7]
	v_mov_b32_e32 v7, v9
	v_mov_b32_e32 v8, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	v_mov_b64_e32 v[36:37], s[4:5]
	s_nop 3
	global_store_dwordx4 v58, v[6:9], s[22:23]
	global_store_dwordx4 v58, v[36:39], s[22:23] offset:16
	global_store_dwordx4 v58, v[36:39], s[22:23] offset:32
	global_store_dwordx4 v58, v[36:39], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1588
; %bb.1581:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[38:39], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v36, s20
	v_mov_b32_e32 v37, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[34:35], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[38:39], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[36:39], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[38:39]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1584
; %bb.1582:                             ; %.preheader1.i.i.i.i121.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_1583:                             ; %.preheader1.i.i.i.i121.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[36:37], off
	v_mov_b32_e32 v34, s20
	v_mov_b32_e32 v35, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[34:37], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[36:37]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[36:37], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_1583
.LBB0_1584:                             ; %Flow3611
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1586
; %bb.1585:                             ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_1586:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_1588
; %bb.1587:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_1588:                             ; %Flow3612
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_1592
.LBB0_1589:                             ;   in Loop: Header=BB0_1592 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1591
; %bb.1590:                             ;   in Loop: Header=BB0_1592 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_1592
	s_branch .LBB0_1594
.LBB0_1591:                             ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_1594
.LBB0_1592:                             ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1589
; %bb.1593:                             ;   in Loop: Header=BB0_1592 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_1589
.LBB0_1594:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[34:35], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1598
; %bb.1595:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[18:19], v[2:3], 0, 1
	v_lshl_add_u64 v[22:23], v[18:19], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v38, v10
	v_mov_b32_e32 v39, v11
	v_cndmask_b32_e32 v37, v23, v19, vcc
	v_cndmask_b32_e32 v36, v22, v18, vcc
	v_and_b32_e32 v3, v37, v3
	v_and_b32_e32 v2, v36, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v9, v[36:39], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_1598
; %bb.1596:                             ; %.preheader.i.i.i.i120.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_1597:                             ; %.preheader.i.i.i.i120.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[38:39], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[36:39], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[38:39]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[38:39], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_1597
.LBB0_1598:                             ; %__ockl_printf_begin.exit.i89.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	s_andn2_b64 vcc, exec, s[10:11]
	s_cbranch_vccnz .LBB0_1683
; %bb.1599:                             ;   in Loop: Header=BB0_273 Depth=1
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v2, 2, v34
	v_and_b32_e32 v36, -3, v34
	v_mov_b32_e32 v37, v35
	s_mov_b64 s[20:21], 49
	s_getpc_b64 s[6:7]
	s_add_u32 s6, s6, .str.2@rel32@lo+4
	s_addc_u32 s7, s7, .str.2@rel32@hi+12
	s_branch .LBB0_1601
.LBB0_1600:                             ; %__ockl_hostcall_preview.exit19.i.i106.1
                                        ;   in Loop: Header=BB0_1601 Depth=2
	s_or_b64 exec, exec, s[26:27]
	s_sub_u32 s20, s20, s22
	s_subb_u32 s21, s21, s23
	s_add_u32 s6, s6, s22
	s_addc_u32 s7, s7, s23
	s_cmp_eq_u64 s[20:21], 0
	s_cbranch_scc1 .LBB0_1682
.LBB0_1601:                             ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_1604 Depth 3
                                        ;       Child Loop BB0_1611 Depth 3
                                        ;       Child Loop BB0_1619 Depth 3
                                        ;       Child Loop BB0_1627 Depth 3
                                        ;       Child Loop BB0_1635 Depth 3
                                        ;       Child Loop BB0_1643 Depth 3
                                        ;       Child Loop BB0_1651 Depth 3
                                        ;       Child Loop BB0_1659 Depth 3
                                        ;       Child Loop BB0_1667 Depth 3
                                        ;       Child Loop BB0_1676 Depth 3
                                        ;       Child Loop BB0_1681 Depth 3
	v_cmp_lt_u64_e64 s[2:3], s[20:21], 56
	s_and_b64 s[2:3], s[2:3], exec
	v_cmp_gt_u64_e64 s[2:3], s[20:21], 7
	s_cselect_b32 s23, s21, 0
	s_cselect_b32 s22, s20, 56
	s_and_b64 vcc, exec, s[2:3]
	s_cbranch_vccnz .LBB0_1606
; %bb.1602:                             ;   in Loop: Header=BB0_1601 Depth=2
	s_mov_b64 s[2:3], 0
	s_cmp_eq_u64 s[20:21], 0
	s_waitcnt vmcnt(0)
	v_mov_b64_e32 v[38:39], 0
	s_cbranch_scc1 .LBB0_1605
; %bb.1603:                             ; %.preheader30.i.i90.1.preheader
                                        ;   in Loop: Header=BB0_1601 Depth=2
	s_lshl_b64 s[24:25], s[22:23], 3
	s_mov_b64 s[26:27], 0
	v_mov_b64_e32 v[38:39], 0
	s_mov_b64 s[28:29], s[6:7]
.LBB0_1604:                             ; %.preheader30.i.i90.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1601 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	global_load_ubyte v3, v9, s[28:29]
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s26, v[8:9]
	s_add_u32 s26, s26, 8
	s_addc_u32 s27, s27, 0
	s_add_u32 s28, s28, 1
	s_addc_u32 s29, s29, 0
	v_or_b32_e32 v38, v10, v38
	s_cmp_eq_u32 s24, s26
	v_or_b32_e32 v39, v11, v39
	s_cbranch_scc0 .LBB0_1604
.LBB0_1605:                             ; %Flow3581
                                        ;   in Loop: Header=BB0_1601 Depth=2
	s_mov_b32 s5, 0
	s_andn2_b64 vcc, exec, s[2:3]
	s_mov_b64 s[2:3], s[6:7]
	s_cbranch_vccz .LBB0_1607
	s_branch .LBB0_1608
.LBB0_1606:                             ;   in Loop: Header=BB0_1601 Depth=2
                                        ; implicit-def: $vgpr38_vgpr39
                                        ; implicit-def: $sgpr5
	s_mov_b64 s[2:3], s[6:7]
.LBB0_1607:                             ;   in Loop: Header=BB0_1601 Depth=2
	global_load_dwordx2 v[38:39], v9, s[6:7]
	s_add_i32 s5, s22, -8
	s_add_u32 s2, s6, 8
	s_addc_u32 s3, s7, 0
.LBB0_1608:                             ; %.loopexit31.i.i91.1
                                        ;   in Loop: Header=BB0_1601 Depth=2
	s_cmp_gt_u32 s5, 7
	s_cbranch_scc1 .LBB0_1612
; %bb.1609:                             ;   in Loop: Header=BB0_1601 Depth=2
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1613
; %bb.1610:                             ; %.preheader28.i.i92.1.preheader
                                        ;   in Loop: Header=BB0_1601 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[40:41], 0
	s_mov_b64 s[26:27], 0
.LBB0_1611:                             ; %.preheader28.i.i92.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1601 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s28, s2, s26
	s_addc_u32 s29, s3, s27
	global_load_ubyte v3, v9, s[28:29]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v40, v10, v40
	s_cmp_eq_u32 s5, s26
	v_or_b32_e32 v41, v11, v41
	s_cbranch_scc0 .LBB0_1611
	s_branch .LBB0_1614
.LBB0_1612:                             ;   in Loop: Header=BB0_1601 Depth=2
                                        ; implicit-def: $vgpr40_vgpr41
                                        ; implicit-def: $sgpr28
	s_branch .LBB0_1615
.LBB0_1613:                             ;   in Loop: Header=BB0_1601 Depth=2
	v_mov_b64_e32 v[40:41], 0
.LBB0_1614:                             ; %Flow3576
                                        ;   in Loop: Header=BB0_1601 Depth=2
	s_mov_b32 s28, 0
	s_cbranch_execnz .LBB0_1616
.LBB0_1615:                             ;   in Loop: Header=BB0_1601 Depth=2
	global_load_dwordx2 v[40:41], v9, s[2:3]
	s_add_i32 s28, s5, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_1616:                             ; %.loopexit29.i.i93.1
                                        ;   in Loop: Header=BB0_1601 Depth=2
	s_cmp_gt_u32 s28, 7
	s_cbranch_scc1 .LBB0_1620
; %bb.1617:                             ;   in Loop: Header=BB0_1601 Depth=2
	s_cmp_eq_u32 s28, 0
	s_cbranch_scc1 .LBB0_1621
; %bb.1618:                             ; %.preheader26.i.i94.1.preheader
                                        ;   in Loop: Header=BB0_1601 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[42:43], 0
	s_mov_b64 s[26:27], 0
.LBB0_1619:                             ; %.preheader26.i.i94.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1601 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s30, s2, s26
	s_addc_u32 s31, s3, s27
	global_load_ubyte v3, v9, s[30:31]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v42, v10, v42
	s_cmp_eq_u32 s28, s26
	v_or_b32_e32 v43, v11, v43
	s_cbranch_scc0 .LBB0_1619
	s_branch .LBB0_1622
.LBB0_1620:                             ;   in Loop: Header=BB0_1601 Depth=2
                                        ; implicit-def: $sgpr5
	s_branch .LBB0_1623
.LBB0_1621:                             ;   in Loop: Header=BB0_1601 Depth=2
	v_mov_b64_e32 v[42:43], 0
.LBB0_1622:                             ; %Flow3571
                                        ;   in Loop: Header=BB0_1601 Depth=2
	s_mov_b32 s5, 0
	s_cbranch_execnz .LBB0_1624
.LBB0_1623:                             ;   in Loop: Header=BB0_1601 Depth=2
	global_load_dwordx2 v[42:43], v9, s[2:3]
	s_add_i32 s5, s28, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_1624:                             ; %.loopexit27.i.i95.1
                                        ;   in Loop: Header=BB0_1601 Depth=2
	s_cmp_gt_u32 s5, 7
	s_cbranch_scc1 .LBB0_1628
; %bb.1625:                             ;   in Loop: Header=BB0_1601 Depth=2
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1629
; %bb.1626:                             ; %.preheader24.i.i96.1.preheader
                                        ;   in Loop: Header=BB0_1601 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[44:45], 0
	s_mov_b64 s[26:27], 0
.LBB0_1627:                             ; %.preheader24.i.i96.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1601 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s28, s2, s26
	s_addc_u32 s29, s3, s27
	global_load_ubyte v3, v9, s[28:29]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v44, v10, v44
	s_cmp_eq_u32 s5, s26
	v_or_b32_e32 v45, v11, v45
	s_cbranch_scc0 .LBB0_1627
	s_branch .LBB0_1630
.LBB0_1628:                             ;   in Loop: Header=BB0_1601 Depth=2
                                        ; implicit-def: $vgpr44_vgpr45
                                        ; implicit-def: $sgpr28
	s_branch .LBB0_1631
.LBB0_1629:                             ;   in Loop: Header=BB0_1601 Depth=2
	v_mov_b64_e32 v[44:45], 0
.LBB0_1630:                             ; %Flow3566
                                        ;   in Loop: Header=BB0_1601 Depth=2
	s_mov_b32 s28, 0
	s_cbranch_execnz .LBB0_1632
.LBB0_1631:                             ;   in Loop: Header=BB0_1601 Depth=2
	global_load_dwordx2 v[44:45], v9, s[2:3]
	s_add_i32 s28, s5, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_1632:                             ; %.loopexit25.i.i97.1
                                        ;   in Loop: Header=BB0_1601 Depth=2
	s_cmp_gt_u32 s28, 7
	s_cbranch_scc1 .LBB0_1636
; %bb.1633:                             ;   in Loop: Header=BB0_1601 Depth=2
	s_cmp_eq_u32 s28, 0
	s_cbranch_scc1 .LBB0_1637
; %bb.1634:                             ; %.preheader22.i.i98.1.preheader
                                        ;   in Loop: Header=BB0_1601 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[46:47], 0
	s_mov_b64 s[26:27], 0
.LBB0_1635:                             ; %.preheader22.i.i98.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1601 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s30, s2, s26
	s_addc_u32 s31, s3, s27
	global_load_ubyte v3, v9, s[30:31]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v46, v10, v46
	s_cmp_eq_u32 s28, s26
	v_or_b32_e32 v47, v11, v47
	s_cbranch_scc0 .LBB0_1635
	s_branch .LBB0_1638
.LBB0_1636:                             ;   in Loop: Header=BB0_1601 Depth=2
                                        ; implicit-def: $sgpr5
	s_branch .LBB0_1639
.LBB0_1637:                             ;   in Loop: Header=BB0_1601 Depth=2
	v_mov_b64_e32 v[46:47], 0
.LBB0_1638:                             ; %Flow3561
                                        ;   in Loop: Header=BB0_1601 Depth=2
	s_mov_b32 s5, 0
	s_cbranch_execnz .LBB0_1640
.LBB0_1639:                             ;   in Loop: Header=BB0_1601 Depth=2
	global_load_dwordx2 v[46:47], v9, s[2:3]
	s_add_i32 s5, s28, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_1640:                             ; %.loopexit23.i.i99.1
                                        ;   in Loop: Header=BB0_1601 Depth=2
	s_cmp_gt_u32 s5, 7
	s_cbranch_scc1 .LBB0_1644
; %bb.1641:                             ;   in Loop: Header=BB0_1601 Depth=2
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1645
; %bb.1642:                             ; %.preheader20.i.i100.1.preheader
                                        ;   in Loop: Header=BB0_1601 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[48:49], 0
	s_mov_b64 s[26:27], 0
.LBB0_1643:                             ; %.preheader20.i.i100.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1601 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s28, s2, s26
	s_addc_u32 s29, s3, s27
	global_load_ubyte v3, v9, s[28:29]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v48, v10, v48
	s_cmp_eq_u32 s5, s26
	v_or_b32_e32 v49, v11, v49
	s_cbranch_scc0 .LBB0_1643
	s_branch .LBB0_1646
.LBB0_1644:                             ;   in Loop: Header=BB0_1601 Depth=2
                                        ; implicit-def: $vgpr48_vgpr49
                                        ; implicit-def: $sgpr28
	s_branch .LBB0_1647
.LBB0_1645:                             ;   in Loop: Header=BB0_1601 Depth=2
	v_mov_b64_e32 v[48:49], 0
.LBB0_1646:                             ; %Flow3556
                                        ;   in Loop: Header=BB0_1601 Depth=2
	s_mov_b32 s28, 0
	s_cbranch_execnz .LBB0_1648
.LBB0_1647:                             ;   in Loop: Header=BB0_1601 Depth=2
	global_load_dwordx2 v[48:49], v9, s[2:3]
	s_add_i32 s28, s5, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_1648:                             ; %.loopexit21.i.i101.1
                                        ;   in Loop: Header=BB0_1601 Depth=2
	s_cmp_gt_u32 s28, 7
	s_cbranch_scc1 .LBB0_1652
; %bb.1649:                             ;   in Loop: Header=BB0_1601 Depth=2
	s_cmp_eq_u32 s28, 0
	s_cbranch_scc1 .LBB0_1653
; %bb.1650:                             ; %.preheader.i.i102.1.preheader
                                        ;   in Loop: Header=BB0_1601 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[50:51], 0
	s_mov_b64 s[26:27], s[2:3]
.LBB0_1651:                             ; %.preheader.i.i102.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1601 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	global_load_ubyte v3, v9, s[26:27]
	s_add_i32 s28, s28, -1
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	v_or_b32_e32 v50, v10, v50
	s_cmp_eq_u32 s28, 0
	v_or_b32_e32 v51, v11, v51
	s_cbranch_scc0 .LBB0_1651
	s_branch .LBB0_1654
.LBB0_1652:                             ;   in Loop: Header=BB0_1601 Depth=2
	s_branch .LBB0_1655
.LBB0_1653:                             ;   in Loop: Header=BB0_1601 Depth=2
	v_mov_b64_e32 v[50:51], 0
.LBB0_1654:                             ; %Flow3551
                                        ;   in Loop: Header=BB0_1601 Depth=2
	s_cbranch_execnz .LBB0_1656
.LBB0_1655:                             ;   in Loop: Header=BB0_1601 Depth=2
	global_load_dwordx2 v[50:51], v9, s[2:3]
.LBB0_1656:                             ; %.loopexit.i.i103.1
                                        ;   in Loop: Header=BB0_1601 Depth=2
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[10:11], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_1662
; %bb.1657:                             ;   in Loop: Header=BB0_1601 Depth=2
	global_load_dwordx2 v[54:55], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v3, v10, v54
	v_and_b32_e32 v7, v11, v55
	v_mul_lo_u32 v7, v7, 24
	v_mul_hi_u32 v8, v3, 24
	v_add_u32_e32 v11, v8, v7
	v_mul_lo_u32 v10, v3, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[14:15], 0, v[10:11]
	global_load_dwordx2 v[52:53], v[10:11], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[52:55], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[10:11], v[54:55]
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB0_1661
; %bb.1658:                             ; %.preheader3.i.i18.i.i118.1.preheader
                                        ;   in Loop: Header=BB0_1601 Depth=2
	s_mov_b64 s[28:29], 0
.LBB0_1659:                             ; %.preheader3.i.i18.i.i118.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1601 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_sleep 1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	global_load_dwordx2 v[18:19], v9, s[16:17]
	v_mov_b64_e32 v[54:55], v[10:11]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v7, v14, v54
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[10:11], s[30:31], v7, 24, v[18:19]
	v_and_b32_e32 v3, v15, v55
	v_mov_b32_e32 v8, v11
	v_mad_u64_u32 v[14:15], s[30:31], v3, 24, v[8:9]
	v_mov_b32_e32 v11, v14
	global_load_dwordx2 v[52:53], v[10:11], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[52:55], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[10:11], v[54:55]
	s_or_b64 s[28:29], vcc, s[28:29]
	s_andn2_b64 exec, exec, s[28:29]
	s_cbranch_execnz .LBB0_1659
; %bb.1660:                             ; %Flow3546
                                        ;   in Loop: Header=BB0_1601 Depth=2
	s_or_b64 exec, exec, s[28:29]
.LBB0_1661:                             ; %Flow3548
                                        ;   in Loop: Header=BB0_1601 Depth=2
	s_or_b64 exec, exec, s[26:27]
.LBB0_1662:                             ; %.loopexit4.i.i13.i.i104.1
                                        ;   in Loop: Header=BB0_1601 Depth=2
	s_or_b64 exec, exec, s[24:25]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	global_load_dwordx4 v[52:55], v9, s[16:17]
	v_readfirstlane_b32 s24, v10
	v_readfirstlane_b32 s25, v11
	s_mov_b64 s[26:27], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s28, v14
	v_readfirstlane_b32 s29, v15
	s_and_b64 s[28:29], s[24:25], s[28:29]
	s_mul_i32 s5, s29, 24
	s_mul_hi_u32 s30, s28, 24
	s_add_i32 s31, s30, s5
	s_mul_i32 s30, s28, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[52:53], 0, s[30:31]
	s_and_saveexec_b64 s[30:31], s[2:3]
	s_cbranch_execz .LBB0_1664
; %bb.1663:                             ;   in Loop: Header=BB0_1601 Depth=2
	v_mov_b64_e32 v[22:23], s[26:27]
	global_store_dwordx4 v[10:11], v[22:25], off offset:8
.LBB0_1664:                             ;   in Loop: Header=BB0_1601 Depth=2
	s_or_b64 exec, exec, s[30:31]
	v_or_b32_e32 v3, 0, v37
	v_or_b32_e32 v7, v36, v2
	v_cmp_gt_u64_e64 vcc, s[20:21], 56
	s_lshl_b32 s5, s22, 2
	s_lshl_b64 s[26:27], s[28:29], 12
	v_cndmask_b32_e32 v37, v3, v37, vcc
	v_cndmask_b32_e32 v3, v7, v36, vcc
	s_add_i32 s5, s5, 28
	v_lshl_add_u64 v[14:15], v[54:55], 0, s[26:27]
	s_and_b32 s5, s5, 0x1e0
	v_and_b32_e32 v3, 0xffffff1f, v3
	v_or_b32_e32 v36, s5, v3
	v_readfirstlane_b32 s26, v14
	v_readfirstlane_b32 s27, v15
	s_nop 4
	global_store_dwordx4 v58, v[36:39], s[26:27]
	global_store_dwordx4 v58, v[40:43], s[26:27] offset:16
	global_store_dwordx4 v58, v[44:47], s[26:27] offset:32
	global_store_dwordx4 v58, v[48:51], s[26:27] offset:48
	s_and_saveexec_b64 s[26:27], s[2:3]
	s_cbranch_execz .LBB0_1672
; %bb.1665:                             ;   in Loop: Header=BB0_1601 Depth=2
	global_load_dwordx2 v[40:41], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[18:19], v9, s[16:17] offset:40
	v_mov_b32_e32 v38, s24
	v_mov_b32_e32 v39, s25
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s28, v18
	v_readfirstlane_b32 s29, v19
	s_and_b64 s[28:29], s[28:29], s[24:25]
	s_mul_i32 s5, s29, 24
	s_mul_hi_u32 s29, s28, 24
	s_mul_i32 s28, s28, 24
	s_add_i32 s29, s29, s5
	v_lshl_add_u64 v[18:19], v[52:53], 0, s[28:29]
	global_store_dwordx2 v[18:19], v[40:41], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v9, v[38:41], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[40:41]
	s_and_saveexec_b64 s[28:29], vcc
	s_cbranch_execz .LBB0_1668
; %bb.1666:                             ; %.preheader1.i.i16.i.i116.1.preheader
                                        ;   in Loop: Header=BB0_1601 Depth=2
	s_mov_b64 s[30:31], 0
.LBB0_1667:                             ; %.preheader1.i.i16.i.i116.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1601 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_sleep 1
	global_store_dwordx2 v[18:19], v[38:39], off
	v_mov_b32_e32 v36, s24
	v_mov_b32_e32 v37, s25
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[22:23], v9, v[36:39], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[22:23], v[38:39]
	s_or_b64 s[30:31], vcc, s[30:31]
	v_mov_b64_e32 v[38:39], v[22:23]
	s_andn2_b64 exec, exec, s[30:31]
	s_cbranch_execnz .LBB0_1667
.LBB0_1668:                             ; %Flow3544
                                        ;   in Loop: Header=BB0_1601 Depth=2
	s_or_b64 exec, exec, s[28:29]
	global_load_dwordx2 v[18:19], v9, s[16:17] offset:16
	s_mov_b64 s[30:31], exec
	v_mbcnt_lo_u32_b32 v3, s30, 0
	v_mbcnt_hi_u32_b32 v3, s31, v3
	v_cmp_eq_u32_e32 vcc, 0, v3
	s_and_saveexec_b64 s[28:29], vcc
	s_cbranch_execz .LBB0_1670
; %bb.1669:                             ;   in Loop: Header=BB0_1601 Depth=2
	s_bcnt1_i32_b64 s5, s[30:31]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[18:19], v[8:9], off offset:8 sc1
.LBB0_1670:                             ;   in Loop: Header=BB0_1601 Depth=2
	s_or_b64 exec, exec, s[28:29]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[22:23], v[18:19], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_cbranch_vccnz .LBB0_1672
; %bb.1671:                             ;   in Loop: Header=BB0_1601 Depth=2
	global_load_dword v8, v[18:19], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v3, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v3
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[22:23], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_1672:                             ; %Flow3545
                                        ;   in Loop: Header=BB0_1601 Depth=2
	s_or_b64 exec, exec, s[26:27]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[14:15], v[14:15], 0, v[58:59]
	s_branch .LBB0_1676
.LBB0_1673:                             ;   in Loop: Header=BB0_1676 Depth=3
	s_or_b64 exec, exec, s[26:27]
	v_readfirstlane_b32 s5, v3
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1675
; %bb.1674:                             ;   in Loop: Header=BB0_1676 Depth=3
	s_sleep 1
	s_cbranch_execnz .LBB0_1676
	s_branch .LBB0_1678
.LBB0_1675:                             ;   in Loop: Header=BB0_1601 Depth=2
	s_branch .LBB0_1678
.LBB0_1676:                             ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1601 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_mov_b32_e32 v3, 1
	s_and_saveexec_b64 s[26:27], s[2:3]
	s_cbranch_execz .LBB0_1673
; %bb.1677:                             ;   in Loop: Header=BB0_1676 Depth=3
	global_load_dword v3, v[10:11], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v3, 1, v3
	s_branch .LBB0_1673
.LBB0_1678:                             ;   in Loop: Header=BB0_1601 Depth=2
	global_load_dwordx4 v[36:39], v[14:15], off
	s_and_saveexec_b64 s[26:27], s[2:3]
	s_cbranch_execz .LBB0_1600
; %bb.1679:                             ;   in Loop: Header=BB0_1601 Depth=2
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[18:19], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[22:23], v[10:11], 0, 1
	v_lshl_add_u64 v[26:27], v[22:23], 0, s[24:25]
	v_cmp_eq_u64_e32 vcc, 0, v[26:27]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v40, v14
	v_mov_b32_e32 v41, v15
	v_cndmask_b32_e32 v39, v27, v23, vcc
	v_cndmask_b32_e32 v38, v26, v22, vcc
	v_and_b32_e32 v3, v39, v11
	v_and_b32_e32 v7, v38, v10
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v8, v7, 24
	v_mul_lo_u32 v10, v7, 24
	v_add_u32_e32 v11, v8, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[18:19], 0, v[10:11]
	global_store_dwordx2 v[10:11], v[14:15], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[40:41], v9, v[38:41], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[40:41], v[14:15]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_1600
; %bb.1680:                             ; %.preheader.i.i15.i.i115.1.preheader
                                        ;   in Loop: Header=BB0_1601 Depth=2
	s_mov_b64 s[2:3], 0
.LBB0_1681:                             ; %.preheader.i.i15.i.i115.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1601 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_sleep 1
	global_store_dwordx2 v[10:11], v[40:41], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[14:15], v9, v[38:41], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[14:15], v[40:41]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[40:41], v[14:15]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_1681
	s_branch .LBB0_1600
.LBB0_1682:                             ; %Flow3584
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_1711
.LBB0_1683:                             ;   in Loop: Header=BB0_273 Depth=1
                                        ; implicit-def: $vgpr36_vgpr37
	s_cbranch_execz .LBB0_1711
; %bb.1684:                             ;   in Loop: Header=BB0_273 Depth=1
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1690
; %bb.1685:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[38:39], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v38
	v_and_b32_e32 v3, v3, v39
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[36:37], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[36:39], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[38:39]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_1689
; %bb.1686:                             ; %.preheader3.i.i.i38.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_1687:                             ; %.preheader3.i.i.i38.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[38:39], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v38
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v39
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[36:37], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[36:39], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[38:39]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_1687
; %bb.1688:                             ; %Flow3597
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_1689:                             ; %Flow3599
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_1690:                             ; %.loopexit4.i.i.i33.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[38:41], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[38:39], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_1692
; %bb.1691:                             ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_1692:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[40:41], 0, s[6:7]
	v_and_or_b32 v34, v34, s33, 32
	v_mov_b32_e32 v36, v9
	v_mov_b32_e32 v37, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	s_nop 1
	global_store_dwordx4 v58, v[34:37], s[22:23]
	s_nop 1
	v_mov_b64_e32 v[36:37], s[6:7]
	v_mov_b64_e32 v[34:35], s[4:5]
	global_store_dwordx4 v58, v[34:37], s[22:23] offset:16
	global_store_dwordx4 v58, v[34:37], s[22:23] offset:32
	global_store_dwordx4 v58, v[34:37], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1700
; %bb.1693:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[42:43], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v40, s20
	v_mov_b32_e32 v41, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[38:39], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[42:43], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[40:43], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[42:43]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1696
; %bb.1694:                             ; %.preheader1.i.i.i36.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_1695:                             ; %.preheader1.i.i.i36.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[36:37], off
	v_mov_b32_e32 v34, s20
	v_mov_b32_e32 v35, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[34:37], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[36:37]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[36:37], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_1695
.LBB0_1696:                             ; %Flow3595
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1698
; %bb.1697:                             ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_1698:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_1700
; %bb.1699:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_1700:                             ; %Flow3596
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_1704
.LBB0_1701:                             ;   in Loop: Header=BB0_1704 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1703
; %bb.1702:                             ;   in Loop: Header=BB0_1704 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_1704
	s_branch .LBB0_1706
.LBB0_1703:                             ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_1706
.LBB0_1704:                             ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1701
; %bb.1705:                             ;   in Loop: Header=BB0_1704 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_1701
.LBB0_1706:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[36:37], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1710
; %bb.1707:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[18:19], v[2:3], 0, 1
	v_lshl_add_u64 v[22:23], v[18:19], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v40, v10
	v_mov_b32_e32 v41, v11
	v_cndmask_b32_e32 v39, v23, v19, vcc
	v_cndmask_b32_e32 v38, v22, v18, vcc
	v_and_b32_e32 v3, v39, v3
	v_and_b32_e32 v2, v38, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[40:41], v9, v[38:41], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[40:41], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_1710
; %bb.1708:                             ; %.preheader.i.i.i35.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_1709:                             ; %.preheader.i.i.i35.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[40:41], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[38:41], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[40:41]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[40:41], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_1709
.LBB0_1710:                             ; %Flow3588
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
.LBB0_1711:                             ; %__ockl_printf_append_string_n.exit.i107.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1717
; %bb.1712:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[40:41], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v40
	v_and_b32_e32 v3, v3, v41
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[38:39], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[38:41], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[40:41]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_1716
; %bb.1713:                             ; %.preheader3.i.i.i45.i114.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_1714:                             ; %.preheader3.i.i.i45.i114.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[40:41], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v40
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v41
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[38:39], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[38:41], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[40:41]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_1714
; %bb.1715:                             ; %Flow3532
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_1716:                             ; %Flow3534
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_1717:                             ; %.loopexit4.i.i.i39.i108.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[40:43], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[40:41], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_1719
; %bb.1718:                             ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_1719:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[42:43], 0, s[6:7]
	v_and_or_b32 v36, v36, s33, 32
	v_mov_b32_e32 v38, v0
	v_mov_b32_e32 v39, v1
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	s_nop 1
	global_store_dwordx4 v58, v[36:39], s[22:23]
	s_nop 1
	v_mov_b64_e32 v[36:37], s[6:7]
	v_mov_b64_e32 v[34:35], s[4:5]
	global_store_dwordx4 v58, v[34:37], s[22:23] offset:16
	global_store_dwordx4 v58, v[34:37], s[22:23] offset:32
	global_store_dwordx4 v58, v[34:37], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1727
; %bb.1720:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[38:39], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v36, s20
	v_mov_b32_e32 v37, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[40:41], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[38:39], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[36:39], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[38:39]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1723
; %bb.1721:                             ; %.preheader1.i.i.i43.i112.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_1722:                             ; %.preheader1.i.i.i43.i112.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[36:37], off
	v_mov_b32_e32 v34, s20
	v_mov_b32_e32 v35, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[34:37], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[36:37]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[36:37], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_1722
.LBB0_1723:                             ; %Flow3530
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1725
; %bb.1724:                             ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_1725:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_1727
; %bb.1726:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_1727:                             ; %Flow3531
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_1731
.LBB0_1728:                             ;   in Loop: Header=BB0_1731 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1730
; %bb.1729:                             ;   in Loop: Header=BB0_1731 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_1731
	s_branch .LBB0_1733
.LBB0_1730:                             ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_1733
.LBB0_1731:                             ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1728
; %bb.1732:                             ;   in Loop: Header=BB0_1731 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_1728
.LBB0_1733:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[26:27], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1737
; %bb.1734:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[18:19], v[2:3], 0, 1
	v_lshl_add_u64 v[22:23], v[18:19], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v36, v10
	v_mov_b32_e32 v37, v11
	v_cndmask_b32_e32 v35, v23, v19, vcc
	v_cndmask_b32_e32 v34, v22, v18, vcc
	v_and_b32_e32 v3, v35, v3
	v_and_b32_e32 v2, v34, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_1737
; %bb.1735:                             ; %.preheader.i.i.i42.i111.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_1736:                             ; %.preheader.i.i.i42.i111.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[36:37], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[36:37]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[36:37], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_1736
.LBB0_1737:                             ; %__ockl_printf_append_args.exit.i110.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1743
; %bb.1738:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[36:37], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v36
	v_and_b32_e32 v3, v3, v37
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[34:35], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[36:37]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_1742
; %bb.1739:                             ; %.preheader3.i.i.i52.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_1740:                             ; %.preheader3.i.i.i52.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[36:37], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v36
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v37
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[34:35], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[36:37]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_1740
; %bb.1741:                             ; %Flow3518
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_1742:                             ; %Flow3520
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_1743:                             ; %.loopexit4.i.i.i46.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[34:37], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[34:35], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_1745
; %bb.1744:                             ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_1745:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[36:37], 0, s[6:7]
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	v_mov_b64_e32 v[38:39], s[6:7]
	v_and_or_b32 v26, v26, s33, 32
	v_mov_b32_e32 v29, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	v_mov_b64_e32 v[36:37], s[4:5]
	s_nop 3
	global_store_dwordx4 v58, v[26:29], s[22:23]
	global_store_dwordx4 v58, v[36:39], s[22:23] offset:16
	global_store_dwordx4 v58, v[36:39], s[22:23] offset:32
	global_store_dwordx4 v58, v[36:39], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1753
; %bb.1746:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[38:39], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v36, s20
	v_mov_b32_e32 v37, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[34:35], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[38:39], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[36:39], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[38:39]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1749
; %bb.1747:                             ; %.preheader1.i.i.i50.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_1748:                             ; %.preheader1.i.i.i50.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[36:37], off
	v_mov_b32_e32 v34, s20
	v_mov_b32_e32 v35, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[34:37], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[36:37]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[36:37], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_1748
.LBB0_1749:                             ; %Flow3516
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1751
; %bb.1750:                             ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_1751:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_1753
; %bb.1752:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_1753:                             ; %Flow3517
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_1757
.LBB0_1754:                             ;   in Loop: Header=BB0_1757 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1756
; %bb.1755:                             ;   in Loop: Header=BB0_1757 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_1757
	s_branch .LBB0_1759
.LBB0_1756:                             ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_1759
.LBB0_1757:                             ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1754
; %bb.1758:                             ;   in Loop: Header=BB0_1757 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_1754
.LBB0_1759:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1763
; %bb.1760:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[18:19], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[22:23], v[10:11], 0, 1
	v_lshl_add_u64 v[26:27], v[22:23], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[26:27]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v36, v14
	v_mov_b32_e32 v37, v15
	v_cndmask_b32_e32 v35, v27, v23, vcc
	v_cndmask_b32_e32 v34, v26, v22, vcc
	v_and_b32_e32 v7, v35, v11
	v_and_b32_e32 v8, v34, v10
	v_mul_lo_u32 v7, v7, 24
	v_mul_hi_u32 v11, v8, 24
	v_mul_lo_u32 v10, v8, 24
	v_add_u32_e32 v11, v11, v7
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[18:19], 0, v[10:11]
	global_store_dwordx2 v[10:11], v[14:15], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[14:15]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_1763
; %bb.1761:                             ; %.preheader.i.i.i49.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_1762:                             ; %.preheader.i.i.i49.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[10:11], v[36:37], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[14:15], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[14:15], v[36:37]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[36:37], v[14:15]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_1762
.LBB0_1763:                             ; %__ockl_printf_append_args.exit53.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[10:11], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1769
; %bb.1764:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[36:37], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v7, v10, v36
	v_and_b32_e32 v8, v11, v37
	v_mul_lo_u32 v8, v8, 24
	v_mul_hi_u32 v10, v7, 24
	v_add_u32_e32 v11, v10, v8
	v_mul_lo_u32 v10, v7, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[14:15], 0, v[10:11]
	global_load_dwordx2 v[34:35], v[10:11], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[10:11], v[36:37]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_1768
; %bb.1765:                             ; %.preheader3.i.i.i60.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_1766:                             ; %.preheader3.i.i.i60.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	global_load_dwordx2 v[18:19], v9, s[16:17]
	v_mov_b64_e32 v[36:37], v[10:11]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v8, v14, v36
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[10:11], s[24:25], v8, 24, v[18:19]
	v_and_b32_e32 v7, v15, v37
	v_mov_b32_e32 v8, v11
	v_mad_u64_u32 v[14:15], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v11, v14
	global_load_dwordx2 v[34:35], v[10:11], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[10:11], v[36:37]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_1766
; %bb.1767:                             ; %Flow3504
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_1768:                             ; %Flow3506
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_1769:                             ; %.loopexit4.i.i.i54.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	global_load_dwordx4 v[34:37], v9, s[16:17]
	v_readfirstlane_b32 s20, v10
	v_readfirstlane_b32 s21, v11
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[34:35], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_1771
; %bb.1770:                             ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[10:11], v[22:25], off offset:8
.LBB0_1771:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[14:15], v[36:37], 0, s[6:7]
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	v_mov_b64_e32 v[38:39], s[6:7]
	v_and_or_b32 v2, v2, s33, 32
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	v_mov_b64_e32 v[36:37], s[4:5]
	s_nop 3
	global_store_dwordx4 v58, v[2:5], s[22:23]
	global_store_dwordx4 v58, v[36:39], s[22:23] offset:16
	global_store_dwordx4 v58, v[36:39], s[22:23] offset:32
	global_store_dwordx4 v58, v[36:39], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1779
; %bb.1772:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[38:39], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	v_mov_b32_e32 v36, s20
	v_mov_b32_e32 v37, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v2
	v_readfirstlane_b32 s23, v3
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[2:3], v[34:35], 0, s[22:23]
	global_store_dwordx2 v[2:3], v[38:39], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[36:39], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[38:39]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1775
; %bb.1773:                             ; %.preheader1.i.i.i58.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_1774:                             ; %.preheader1.i.i.i58.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[36:37], off
	v_mov_b32_e32 v34, s20
	v_mov_b32_e32 v35, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[34:37], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[36:37]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[36:37], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_1774
.LBB0_1775:                             ; %Flow3502
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1777
; %bb.1776:                             ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[2:3], v[8:9], off offset:8 sc1
.LBB0_1777:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[2:3], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_1779
; %bb.1778:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[2:3], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v2, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v2
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_1779:                             ; %Flow3503
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[58:59]
	s_branch .LBB0_1783
.LBB0_1780:                             ;   in Loop: Header=BB0_1783 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1782
; %bb.1781:                             ;   in Loop: Header=BB0_1783 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_1783
	s_branch .LBB0_1785
.LBB0_1782:                             ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_1785
.LBB0_1783:                             ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1780
; %bb.1784:                             ;   in Loop: Header=BB0_1783 Depth=2
	global_load_dword v7, v[10:11], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_1780
.LBB0_1785:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[30:31], v[2:3], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1789
; %bb.1786:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[18:19], v[2:3], 0, 1
	v_lshl_add_u64 v[22:23], v[18:19], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v36, v10
	v_mov_b32_e32 v37, v11
	v_cndmask_b32_e32 v35, v23, v19, vcc
	v_cndmask_b32_e32 v34, v22, v18, vcc
	v_and_b32_e32 v3, v35, v3
	v_and_b32_e32 v2, v34, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_1789
; %bb.1787:                             ; %.preheader.i.i.i57.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_1788:                             ; %.preheader.i.i.i57.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[36:37], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[36:37]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[36:37], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_1788
.LBB0_1789:                             ; %__ockl_printf_append_args.exit61.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1795
; %bb.1790:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[36:37], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v36
	v_and_b32_e32 v3, v3, v37
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[34:35], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[36:37]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_1794
; %bb.1791:                             ; %.preheader3.i.i.i68.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_1792:                             ; %.preheader3.i.i.i68.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[36:37], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v36
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v37
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[34:35], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[36:37]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_1792
; %bb.1793:                             ; %Flow3490
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_1794:                             ; %Flow3492
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_1795:                             ; %.loopexit4.i.i.i62.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[34:37], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[34:35], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_1797
; %bb.1796:                             ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_1797:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[36:37], 0, s[6:7]
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	v_mov_b64_e32 v[38:39], s[6:7]
	v_and_or_b32 v30, v30, s34, 34
	v_mov_b32_e32 v33, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	v_mov_b64_e32 v[36:37], s[4:5]
	s_nop 3
	global_store_dwordx4 v58, v[30:33], s[22:23]
	global_store_dwordx4 v58, v[36:39], s[22:23] offset:16
	global_store_dwordx4 v58, v[36:39], s[22:23] offset:32
	global_store_dwordx4 v58, v[36:39], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1805
; %bb.1798:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[38:39], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	v_mov_b32_e32 v36, s20
	v_mov_b32_e32 v37, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[10:11], v[34:35], 0, s[22:23]
	global_store_dwordx2 v[10:11], v[38:39], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[36:39], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[38:39]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1801
; %bb.1799:                             ; %.preheader1.i.i.i66.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_1800:                             ; %.preheader1.i.i.i66.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[10:11], v[36:37], off
	v_mov_b32_e32 v34, s20
	v_mov_b32_e32 v35, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[14:15], v9, v[34:37], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[14:15], v[36:37]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[36:37], v[14:15]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_1800
.LBB0_1801:                             ; %Flow3488
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1803
; %bb.1802:                             ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[10:11], v[8:9], off offset:8 sc1
.LBB0_1803:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[14:15], v[10:11], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[14:15]
	s_cbranch_vccnz .LBB0_1805
; %bb.1804:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[10:11], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[14:15], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_1805:                             ; %Flow3489
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	s_branch .LBB0_1809
.LBB0_1806:                             ;   in Loop: Header=BB0_1809 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1808
; %bb.1807:                             ;   in Loop: Header=BB0_1809 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_1809
	s_branch .LBB0_1811
.LBB0_1808:                             ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_1811
.LBB0_1809:                             ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1806
; %bb.1810:                             ;   in Loop: Header=BB0_1809 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_1806
.LBB0_1811:                             ;   in Loop: Header=BB0_273 Depth=1
	s_and_b64 exec, exec, s[2:3]
	s_cbranch_execz .LBB0_1815
; %bb.1812:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[18:19], v[2:3], 0, 1
	v_lshl_add_u64 v[22:23], v[18:19], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v36, v10
	v_mov_b32_e32 v37, v11
	v_cndmask_b32_e32 v35, v23, v19, vcc
	v_cndmask_b32_e32 v34, v22, v18, vcc
	v_and_b32_e32 v3, v35, v3
	v_and_b32_e32 v2, v34, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_1815
; %bb.1813:                             ; %.preheader.i.i.i65.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_1814:                             ; %.preheader.i.i.i65.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[36:37], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[36:37]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[36:37], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_1814
.LBB0_1815:                             ; %Flow3616
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[18:19]
	v_add_u32_e32 v26, v64, v61
	;;#ASMSTART
	ds_read_b128 v[30:33], v32
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	scratch_store_dwordx4 v63, v[30:33], off offset:32
	s_and_saveexec_b64 s[18:19], s[0:1]
	s_cbranch_execz .LBB0_2085
; %bb.1816:                             ; %if.then.i.1.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1822
; %bb.1817:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[32:33], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v32
	v_and_b32_e32 v3, v3, v33
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[32:33]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_1821
; %bb.1818:                             ; %.preheader3.i.i.i.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_1819:                             ; %.preheader3.i.i.i.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[32:33], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v32
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v33
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[32:33]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_1819
; %bb.1820:                             ; %Flow3474
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_1821:                             ; %Flow3476
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_1822:                             ; %.loopexit4.i.i.i.i.1.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[30:33], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[30:31], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_1824
; %bb.1823:                             ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_1824:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[32:33], 0, s[6:7]
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	v_mov_b64_e32 v[34:35], s[6:7]
	v_mov_b32_e32 v7, v9
	v_mov_b32_e32 v8, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	v_mov_b64_e32 v[32:33], s[4:5]
	s_nop 3
	global_store_dwordx4 v58, v[6:9], s[22:23]
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:16
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:32
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1832
; %bb.1825:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[34:35], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v32, s20
	v_mov_b32_e32 v33, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[30:31], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[34:35], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[32:35], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[34:35]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1828
; %bb.1826:                             ; %.preheader1.i.i.i.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_1827:                             ; %.preheader1.i.i.i.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[32:33], off
	v_mov_b32_e32 v30, s20
	v_mov_b32_e32 v31, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[30:33], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[32:33]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[32:33], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_1827
.LBB0_1828:                             ; %Flow3472
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1830
; %bb.1829:                             ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_1830:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_1832
; %bb.1831:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_1832:                             ; %Flow3473
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_1836
.LBB0_1833:                             ;   in Loop: Header=BB0_1836 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1835
; %bb.1834:                             ;   in Loop: Header=BB0_1836 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_1836
	s_branch .LBB0_1838
.LBB0_1835:                             ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_1838
.LBB0_1836:                             ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1833
; %bb.1837:                             ;   in Loop: Header=BB0_1836 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_1833
.LBB0_1838:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[30:31], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1842
; %bb.1839:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[18:19], v[2:3], 0, 1
	v_lshl_add_u64 v[22:23], v[18:19], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v34, v10
	v_mov_b32_e32 v35, v11
	v_cndmask_b32_e32 v33, v23, v19, vcc
	v_cndmask_b32_e32 v32, v22, v18, vcc
	v_and_b32_e32 v3, v33, v3
	v_and_b32_e32 v2, v32, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[34:35], v9, v[32:35], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[34:35], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_1842
; %bb.1840:                             ; %.preheader.i.i.i.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_1841:                             ; %.preheader.i.i.i.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[34:35], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[32:35], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[34:35]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[34:35], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_1841
.LBB0_1842:                             ; %__ockl_printf_begin.exit.i.1.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	s_andn2_b64 vcc, exec, s[8:9]
	s_cbranch_vccnz .LBB0_1927
; %bb.1843:                             ;   in Loop: Header=BB0_273 Depth=1
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v2, 2, v30
	v_and_b32_e32 v32, -3, v30
	v_mov_b32_e32 v33, v31
	s_mov_b64 s[20:21], 0x43
	s_getpc_b64 s[6:7]
	s_add_u32 s6, s6, .str.3@rel32@lo+4
	s_addc_u32 s7, s7, .str.3@rel32@hi+12
	s_branch .LBB0_1845
.LBB0_1844:                             ; %__ockl_hostcall_preview.exit19.i.i.1.i.1
                                        ;   in Loop: Header=BB0_1845 Depth=2
	s_or_b64 exec, exec, s[26:27]
	s_sub_u32 s20, s20, s22
	s_subb_u32 s21, s21, s23
	s_add_u32 s6, s6, s22
	s_addc_u32 s7, s7, s23
	s_cmp_eq_u64 s[20:21], 0
	s_cbranch_scc1 .LBB0_1926
.LBB0_1845:                             ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_1848 Depth 3
                                        ;       Child Loop BB0_1855 Depth 3
                                        ;       Child Loop BB0_1863 Depth 3
                                        ;       Child Loop BB0_1871 Depth 3
                                        ;       Child Loop BB0_1879 Depth 3
                                        ;       Child Loop BB0_1887 Depth 3
                                        ;       Child Loop BB0_1895 Depth 3
                                        ;       Child Loop BB0_1903 Depth 3
                                        ;       Child Loop BB0_1911 Depth 3
                                        ;       Child Loop BB0_1920 Depth 3
                                        ;       Child Loop BB0_1925 Depth 3
	v_cmp_lt_u64_e64 s[2:3], s[20:21], 56
	s_and_b64 s[2:3], s[2:3], exec
	v_cmp_gt_u64_e64 s[2:3], s[20:21], 7
	s_cselect_b32 s23, s21, 0
	s_cselect_b32 s22, s20, 56
	s_and_b64 vcc, exec, s[2:3]
	s_cbranch_vccnz .LBB0_1850
; %bb.1846:                             ;   in Loop: Header=BB0_1845 Depth=2
	s_mov_b64 s[2:3], 0
	s_cmp_eq_u64 s[20:21], 0
	s_waitcnt vmcnt(0)
	v_mov_b64_e32 v[34:35], 0
	s_cbranch_scc1 .LBB0_1849
; %bb.1847:                             ; %.preheader30.i.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_1845 Depth=2
	s_lshl_b64 s[24:25], s[22:23], 3
	s_mov_b64 s[26:27], 0
	v_mov_b64_e32 v[34:35], 0
	s_mov_b64 s[28:29], s[6:7]
.LBB0_1848:                             ; %.preheader30.i.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1845 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	global_load_ubyte v3, v9, s[28:29]
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s26, v[8:9]
	s_add_u32 s26, s26, 8
	s_addc_u32 s27, s27, 0
	s_add_u32 s28, s28, 1
	s_addc_u32 s29, s29, 0
	v_or_b32_e32 v34, v10, v34
	s_cmp_eq_u32 s24, s26
	v_or_b32_e32 v35, v11, v35
	s_cbranch_scc0 .LBB0_1848
.LBB0_1849:                             ; %Flow3441
                                        ;   in Loop: Header=BB0_1845 Depth=2
	s_mov_b32 s5, 0
	s_andn2_b64 vcc, exec, s[2:3]
	s_mov_b64 s[2:3], s[6:7]
	s_cbranch_vccz .LBB0_1851
	s_branch .LBB0_1852
.LBB0_1850:                             ;   in Loop: Header=BB0_1845 Depth=2
                                        ; implicit-def: $vgpr34_vgpr35
                                        ; implicit-def: $sgpr5
	s_mov_b64 s[2:3], s[6:7]
.LBB0_1851:                             ;   in Loop: Header=BB0_1845 Depth=2
	global_load_dwordx2 v[34:35], v9, s[6:7]
	s_add_i32 s5, s22, -8
	s_add_u32 s2, s6, 8
	s_addc_u32 s3, s7, 0
.LBB0_1852:                             ; %.loopexit31.i.i.1.i.1
                                        ;   in Loop: Header=BB0_1845 Depth=2
	s_cmp_gt_u32 s5, 7
	s_cbranch_scc1 .LBB0_1856
; %bb.1853:                             ;   in Loop: Header=BB0_1845 Depth=2
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1857
; %bb.1854:                             ; %.preheader28.i.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_1845 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[36:37], 0
	s_mov_b64 s[26:27], 0
.LBB0_1855:                             ; %.preheader28.i.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1845 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s28, s2, s26
	s_addc_u32 s29, s3, s27
	global_load_ubyte v3, v9, s[28:29]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v36, v10, v36
	s_cmp_eq_u32 s5, s26
	v_or_b32_e32 v37, v11, v37
	s_cbranch_scc0 .LBB0_1855
	s_branch .LBB0_1858
.LBB0_1856:                             ;   in Loop: Header=BB0_1845 Depth=2
                                        ; implicit-def: $vgpr36_vgpr37
                                        ; implicit-def: $sgpr28
	s_branch .LBB0_1859
.LBB0_1857:                             ;   in Loop: Header=BB0_1845 Depth=2
	v_mov_b64_e32 v[36:37], 0
.LBB0_1858:                             ; %Flow3436
                                        ;   in Loop: Header=BB0_1845 Depth=2
	s_mov_b32 s28, 0
	s_cbranch_execnz .LBB0_1860
.LBB0_1859:                             ;   in Loop: Header=BB0_1845 Depth=2
	global_load_dwordx2 v[36:37], v9, s[2:3]
	s_add_i32 s28, s5, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_1860:                             ; %.loopexit29.i.i.1.i.1
                                        ;   in Loop: Header=BB0_1845 Depth=2
	s_cmp_gt_u32 s28, 7
	s_cbranch_scc1 .LBB0_1864
; %bb.1861:                             ;   in Loop: Header=BB0_1845 Depth=2
	s_cmp_eq_u32 s28, 0
	s_cbranch_scc1 .LBB0_1865
; %bb.1862:                             ; %.preheader26.i.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_1845 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[38:39], 0
	s_mov_b64 s[26:27], 0
.LBB0_1863:                             ; %.preheader26.i.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1845 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s30, s2, s26
	s_addc_u32 s31, s3, s27
	global_load_ubyte v3, v9, s[30:31]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v38, v10, v38
	s_cmp_eq_u32 s28, s26
	v_or_b32_e32 v39, v11, v39
	s_cbranch_scc0 .LBB0_1863
	s_branch .LBB0_1866
.LBB0_1864:                             ;   in Loop: Header=BB0_1845 Depth=2
                                        ; implicit-def: $sgpr5
	s_branch .LBB0_1867
.LBB0_1865:                             ;   in Loop: Header=BB0_1845 Depth=2
	v_mov_b64_e32 v[38:39], 0
.LBB0_1866:                             ; %Flow3431
                                        ;   in Loop: Header=BB0_1845 Depth=2
	s_mov_b32 s5, 0
	s_cbranch_execnz .LBB0_1868
.LBB0_1867:                             ;   in Loop: Header=BB0_1845 Depth=2
	global_load_dwordx2 v[38:39], v9, s[2:3]
	s_add_i32 s5, s28, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_1868:                             ; %.loopexit27.i.i.1.i.1
                                        ;   in Loop: Header=BB0_1845 Depth=2
	s_cmp_gt_u32 s5, 7
	s_cbranch_scc1 .LBB0_1872
; %bb.1869:                             ;   in Loop: Header=BB0_1845 Depth=2
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1873
; %bb.1870:                             ; %.preheader24.i.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_1845 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[40:41], 0
	s_mov_b64 s[26:27], 0
.LBB0_1871:                             ; %.preheader24.i.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1845 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s28, s2, s26
	s_addc_u32 s29, s3, s27
	global_load_ubyte v3, v9, s[28:29]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v40, v10, v40
	s_cmp_eq_u32 s5, s26
	v_or_b32_e32 v41, v11, v41
	s_cbranch_scc0 .LBB0_1871
	s_branch .LBB0_1874
.LBB0_1872:                             ;   in Loop: Header=BB0_1845 Depth=2
                                        ; implicit-def: $vgpr40_vgpr41
                                        ; implicit-def: $sgpr28
	s_branch .LBB0_1875
.LBB0_1873:                             ;   in Loop: Header=BB0_1845 Depth=2
	v_mov_b64_e32 v[40:41], 0
.LBB0_1874:                             ; %Flow3426
                                        ;   in Loop: Header=BB0_1845 Depth=2
	s_mov_b32 s28, 0
	s_cbranch_execnz .LBB0_1876
.LBB0_1875:                             ;   in Loop: Header=BB0_1845 Depth=2
	global_load_dwordx2 v[40:41], v9, s[2:3]
	s_add_i32 s28, s5, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_1876:                             ; %.loopexit25.i.i.1.i.1
                                        ;   in Loop: Header=BB0_1845 Depth=2
	s_cmp_gt_u32 s28, 7
	s_cbranch_scc1 .LBB0_1880
; %bb.1877:                             ;   in Loop: Header=BB0_1845 Depth=2
	s_cmp_eq_u32 s28, 0
	s_cbranch_scc1 .LBB0_1881
; %bb.1878:                             ; %.preheader22.i.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_1845 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[42:43], 0
	s_mov_b64 s[26:27], 0
.LBB0_1879:                             ; %.preheader22.i.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1845 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s30, s2, s26
	s_addc_u32 s31, s3, s27
	global_load_ubyte v3, v9, s[30:31]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v42, v10, v42
	s_cmp_eq_u32 s28, s26
	v_or_b32_e32 v43, v11, v43
	s_cbranch_scc0 .LBB0_1879
	s_branch .LBB0_1882
.LBB0_1880:                             ;   in Loop: Header=BB0_1845 Depth=2
                                        ; implicit-def: $sgpr5
	s_branch .LBB0_1883
.LBB0_1881:                             ;   in Loop: Header=BB0_1845 Depth=2
	v_mov_b64_e32 v[42:43], 0
.LBB0_1882:                             ; %Flow3421
                                        ;   in Loop: Header=BB0_1845 Depth=2
	s_mov_b32 s5, 0
	s_cbranch_execnz .LBB0_1884
.LBB0_1883:                             ;   in Loop: Header=BB0_1845 Depth=2
	global_load_dwordx2 v[42:43], v9, s[2:3]
	s_add_i32 s5, s28, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_1884:                             ; %.loopexit23.i.i.1.i.1
                                        ;   in Loop: Header=BB0_1845 Depth=2
	s_cmp_gt_u32 s5, 7
	s_cbranch_scc1 .LBB0_1888
; %bb.1885:                             ;   in Loop: Header=BB0_1845 Depth=2
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1889
; %bb.1886:                             ; %.preheader20.i.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_1845 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[44:45], 0
	s_mov_b64 s[26:27], 0
.LBB0_1887:                             ; %.preheader20.i.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1845 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s28, s2, s26
	s_addc_u32 s29, s3, s27
	global_load_ubyte v3, v9, s[28:29]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v44, v10, v44
	s_cmp_eq_u32 s5, s26
	v_or_b32_e32 v45, v11, v45
	s_cbranch_scc0 .LBB0_1887
	s_branch .LBB0_1890
.LBB0_1888:                             ;   in Loop: Header=BB0_1845 Depth=2
                                        ; implicit-def: $vgpr44_vgpr45
                                        ; implicit-def: $sgpr28
	s_branch .LBB0_1891
.LBB0_1889:                             ;   in Loop: Header=BB0_1845 Depth=2
	v_mov_b64_e32 v[44:45], 0
.LBB0_1890:                             ; %Flow3416
                                        ;   in Loop: Header=BB0_1845 Depth=2
	s_mov_b32 s28, 0
	s_cbranch_execnz .LBB0_1892
.LBB0_1891:                             ;   in Loop: Header=BB0_1845 Depth=2
	global_load_dwordx2 v[44:45], v9, s[2:3]
	s_add_i32 s28, s5, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_1892:                             ; %.loopexit21.i.i.1.i.1
                                        ;   in Loop: Header=BB0_1845 Depth=2
	s_cmp_gt_u32 s28, 7
	s_cbranch_scc1 .LBB0_1896
; %bb.1893:                             ;   in Loop: Header=BB0_1845 Depth=2
	s_cmp_eq_u32 s28, 0
	s_cbranch_scc1 .LBB0_1897
; %bb.1894:                             ; %.preheader.i.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_1845 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[46:47], 0
	s_mov_b64 s[26:27], s[2:3]
.LBB0_1895:                             ; %.preheader.i.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1845 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	global_load_ubyte v3, v9, s[26:27]
	s_add_i32 s28, s28, -1
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	v_or_b32_e32 v46, v10, v46
	s_cmp_eq_u32 s28, 0
	v_or_b32_e32 v47, v11, v47
	s_cbranch_scc0 .LBB0_1895
	s_branch .LBB0_1898
.LBB0_1896:                             ;   in Loop: Header=BB0_1845 Depth=2
	s_branch .LBB0_1899
.LBB0_1897:                             ;   in Loop: Header=BB0_1845 Depth=2
	v_mov_b64_e32 v[46:47], 0
.LBB0_1898:                             ; %Flow3411
                                        ;   in Loop: Header=BB0_1845 Depth=2
	s_cbranch_execnz .LBB0_1900
.LBB0_1899:                             ;   in Loop: Header=BB0_1845 Depth=2
	global_load_dwordx2 v[46:47], v9, s[2:3]
.LBB0_1900:                             ; %.loopexit.i.i.1.i.1
                                        ;   in Loop: Header=BB0_1845 Depth=2
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[10:11], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_1906
; %bb.1901:                             ;   in Loop: Header=BB0_1845 Depth=2
	global_load_dwordx2 v[50:51], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v3, v10, v50
	v_and_b32_e32 v7, v11, v51
	v_mul_lo_u32 v7, v7, 24
	v_mul_hi_u32 v8, v3, 24
	v_add_u32_e32 v11, v8, v7
	v_mul_lo_u32 v10, v3, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[14:15], 0, v[10:11]
	global_load_dwordx2 v[48:49], v[10:11], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[48:51], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[10:11], v[50:51]
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB0_1905
; %bb.1902:                             ; %.preheader3.i.i18.i.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_1845 Depth=2
	s_mov_b64 s[28:29], 0
.LBB0_1903:                             ; %.preheader3.i.i18.i.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1845 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_sleep 1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	global_load_dwordx2 v[18:19], v9, s[16:17]
	v_mov_b64_e32 v[50:51], v[10:11]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v7, v14, v50
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[10:11], s[30:31], v7, 24, v[18:19]
	v_and_b32_e32 v3, v15, v51
	v_mov_b32_e32 v8, v11
	v_mad_u64_u32 v[14:15], s[30:31], v3, 24, v[8:9]
	v_mov_b32_e32 v11, v14
	global_load_dwordx2 v[48:49], v[10:11], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[48:51], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[10:11], v[50:51]
	s_or_b64 s[28:29], vcc, s[28:29]
	s_andn2_b64 exec, exec, s[28:29]
	s_cbranch_execnz .LBB0_1903
; %bb.1904:                             ; %Flow3406
                                        ;   in Loop: Header=BB0_1845 Depth=2
	s_or_b64 exec, exec, s[28:29]
.LBB0_1905:                             ; %Flow3408
                                        ;   in Loop: Header=BB0_1845 Depth=2
	s_or_b64 exec, exec, s[26:27]
.LBB0_1906:                             ; %.loopexit4.i.i13.i.i.1.i.1
                                        ;   in Loop: Header=BB0_1845 Depth=2
	s_or_b64 exec, exec, s[24:25]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	global_load_dwordx4 v[48:51], v9, s[16:17]
	v_readfirstlane_b32 s24, v10
	v_readfirstlane_b32 s25, v11
	s_mov_b64 s[26:27], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s28, v14
	v_readfirstlane_b32 s29, v15
	s_and_b64 s[28:29], s[24:25], s[28:29]
	s_mul_i32 s5, s29, 24
	s_mul_hi_u32 s30, s28, 24
	s_add_i32 s31, s30, s5
	s_mul_i32 s30, s28, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[48:49], 0, s[30:31]
	s_and_saveexec_b64 s[30:31], s[2:3]
	s_cbranch_execz .LBB0_1908
; %bb.1907:                             ;   in Loop: Header=BB0_1845 Depth=2
	v_mov_b64_e32 v[22:23], s[26:27]
	global_store_dwordx4 v[10:11], v[22:25], off offset:8
.LBB0_1908:                             ;   in Loop: Header=BB0_1845 Depth=2
	s_or_b64 exec, exec, s[30:31]
	v_or_b32_e32 v3, 0, v33
	v_or_b32_e32 v7, v32, v2
	v_cmp_gt_u64_e64 vcc, s[20:21], 56
	s_lshl_b32 s5, s22, 2
	s_lshl_b64 s[26:27], s[28:29], 12
	v_cndmask_b32_e32 v33, v3, v33, vcc
	v_cndmask_b32_e32 v3, v7, v32, vcc
	s_add_i32 s5, s5, 28
	v_lshl_add_u64 v[14:15], v[50:51], 0, s[26:27]
	s_and_b32 s5, s5, 0x1e0
	v_and_b32_e32 v3, 0xffffff1f, v3
	v_or_b32_e32 v32, s5, v3
	v_readfirstlane_b32 s26, v14
	v_readfirstlane_b32 s27, v15
	s_nop 4
	global_store_dwordx4 v58, v[32:35], s[26:27]
	global_store_dwordx4 v58, v[36:39], s[26:27] offset:16
	global_store_dwordx4 v58, v[40:43], s[26:27] offset:32
	global_store_dwordx4 v58, v[44:47], s[26:27] offset:48
	s_and_saveexec_b64 s[26:27], s[2:3]
	s_cbranch_execz .LBB0_1916
; %bb.1909:                             ;   in Loop: Header=BB0_1845 Depth=2
	global_load_dwordx2 v[36:37], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[18:19], v9, s[16:17] offset:40
	v_mov_b32_e32 v34, s24
	v_mov_b32_e32 v35, s25
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s28, v18
	v_readfirstlane_b32 s29, v19
	s_and_b64 s[28:29], s[28:29], s[24:25]
	s_mul_i32 s5, s29, 24
	s_mul_hi_u32 s29, s28, 24
	s_mul_i32 s28, s28, 24
	s_add_i32 s29, s29, s5
	v_lshl_add_u64 v[18:19], v[48:49], 0, s[28:29]
	global_store_dwordx2 v[18:19], v[36:37], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[34:35], v9, v[34:37], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[34:35], v[36:37]
	s_and_saveexec_b64 s[28:29], vcc
	s_cbranch_execz .LBB0_1912
; %bb.1910:                             ; %.preheader1.i.i16.i.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_1845 Depth=2
	s_mov_b64 s[30:31], 0
.LBB0_1911:                             ; %.preheader1.i.i16.i.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1845 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_sleep 1
	global_store_dwordx2 v[18:19], v[34:35], off
	v_mov_b32_e32 v32, s24
	v_mov_b32_e32 v33, s25
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[22:23], v9, v[32:35], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[22:23], v[34:35]
	s_or_b64 s[30:31], vcc, s[30:31]
	v_mov_b64_e32 v[34:35], v[22:23]
	s_andn2_b64 exec, exec, s[30:31]
	s_cbranch_execnz .LBB0_1911
.LBB0_1912:                             ; %Flow3404
                                        ;   in Loop: Header=BB0_1845 Depth=2
	s_or_b64 exec, exec, s[28:29]
	global_load_dwordx2 v[18:19], v9, s[16:17] offset:16
	s_mov_b64 s[30:31], exec
	v_mbcnt_lo_u32_b32 v3, s30, 0
	v_mbcnt_hi_u32_b32 v3, s31, v3
	v_cmp_eq_u32_e32 vcc, 0, v3
	s_and_saveexec_b64 s[28:29], vcc
	s_cbranch_execz .LBB0_1914
; %bb.1913:                             ;   in Loop: Header=BB0_1845 Depth=2
	s_bcnt1_i32_b64 s5, s[30:31]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[18:19], v[8:9], off offset:8 sc1
.LBB0_1914:                             ;   in Loop: Header=BB0_1845 Depth=2
	s_or_b64 exec, exec, s[28:29]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[22:23], v[18:19], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_cbranch_vccnz .LBB0_1916
; %bb.1915:                             ;   in Loop: Header=BB0_1845 Depth=2
	global_load_dword v8, v[18:19], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v3, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v3
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[22:23], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_1916:                             ; %Flow3405
                                        ;   in Loop: Header=BB0_1845 Depth=2
	s_or_b64 exec, exec, s[26:27]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[14:15], v[14:15], 0, v[58:59]
	s_branch .LBB0_1920
.LBB0_1917:                             ;   in Loop: Header=BB0_1920 Depth=3
	s_or_b64 exec, exec, s[26:27]
	v_readfirstlane_b32 s5, v3
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1919
; %bb.1918:                             ;   in Loop: Header=BB0_1920 Depth=3
	s_sleep 1
	s_cbranch_execnz .LBB0_1920
	s_branch .LBB0_1922
.LBB0_1919:                             ;   in Loop: Header=BB0_1845 Depth=2
	s_branch .LBB0_1922
.LBB0_1920:                             ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1845 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_mov_b32_e32 v3, 1
	s_and_saveexec_b64 s[26:27], s[2:3]
	s_cbranch_execz .LBB0_1917
; %bb.1921:                             ;   in Loop: Header=BB0_1920 Depth=3
	global_load_dword v3, v[10:11], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v3, 1, v3
	s_branch .LBB0_1917
.LBB0_1922:                             ;   in Loop: Header=BB0_1845 Depth=2
	global_load_dwordx4 v[32:35], v[14:15], off
	s_and_saveexec_b64 s[26:27], s[2:3]
	s_cbranch_execz .LBB0_1844
; %bb.1923:                             ;   in Loop: Header=BB0_1845 Depth=2
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[18:19], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[22:23], v[10:11], 0, 1
	v_lshl_add_u64 v[34:35], v[22:23], 0, s[24:25]
	v_cmp_eq_u64_e32 vcc, 0, v[34:35]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v36, v14
	v_mov_b32_e32 v37, v15
	v_cndmask_b32_e32 v35, v35, v23, vcc
	v_cndmask_b32_e32 v34, v34, v22, vcc
	v_and_b32_e32 v3, v35, v11
	v_and_b32_e32 v7, v34, v10
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v8, v7, 24
	v_mul_lo_u32 v10, v7, 24
	v_add_u32_e32 v11, v8, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[18:19], 0, v[10:11]
	global_store_dwordx2 v[10:11], v[14:15], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[14:15]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_1844
; %bb.1924:                             ; %.preheader.i.i15.i.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_1845 Depth=2
	s_mov_b64 s[2:3], 0
.LBB0_1925:                             ; %.preheader.i.i15.i.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_1845 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_sleep 1
	global_store_dwordx2 v[10:11], v[36:37], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[14:15], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[14:15], v[36:37]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[36:37], v[14:15]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_1925
	s_branch .LBB0_1844
.LBB0_1926:                             ; %Flow3444
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_1955
.LBB0_1927:                             ;   in Loop: Header=BB0_273 Depth=1
                                        ; implicit-def: $vgpr32_vgpr33
	s_cbranch_execz .LBB0_1955
; %bb.1928:                             ;   in Loop: Header=BB0_273 Depth=1
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1934
; %bb.1929:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[34:35], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v34
	v_and_b32_e32 v3, v3, v35
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[32:33], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[32:35], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[34:35]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_1933
; %bb.1930:                             ; %.preheader3.i.i.i9.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_1931:                             ; %.preheader3.i.i.i9.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[34:35], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v34
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v35
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[32:33], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[32:35], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[34:35]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_1931
; %bb.1932:                             ; %Flow3457
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_1933:                             ; %Flow3459
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_1934:                             ; %.loopexit4.i.i.i4.i.1.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[34:37], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[34:35], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_1936
; %bb.1935:                             ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_1936:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[36:37], 0, s[6:7]
	v_and_or_b32 v30, v30, s33, 32
	v_mov_b32_e32 v32, v9
	v_mov_b32_e32 v33, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	s_nop 1
	global_store_dwordx4 v58, v[30:33], s[22:23]
	s_nop 1
	v_mov_b64_e32 v[32:33], s[6:7]
	v_mov_b64_e32 v[30:31], s[4:5]
	global_store_dwordx4 v58, v[30:33], s[22:23] offset:16
	global_store_dwordx4 v58, v[30:33], s[22:23] offset:32
	global_store_dwordx4 v58, v[30:33], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1944
; %bb.1937:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[38:39], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v36, s20
	v_mov_b32_e32 v37, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[34:35], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[38:39], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[36:39], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[38:39]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1940
; %bb.1938:                             ; %.preheader1.i.i.i7.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_1939:                             ; %.preheader1.i.i.i7.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[32:33], off
	v_mov_b32_e32 v30, s20
	v_mov_b32_e32 v31, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[30:33], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[32:33]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[32:33], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_1939
.LBB0_1940:                             ; %Flow3455
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1942
; %bb.1941:                             ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_1942:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_1944
; %bb.1943:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_1944:                             ; %Flow3456
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_1948
.LBB0_1945:                             ;   in Loop: Header=BB0_1948 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1947
; %bb.1946:                             ;   in Loop: Header=BB0_1948 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_1948
	s_branch .LBB0_1950
.LBB0_1947:                             ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_1950
.LBB0_1948:                             ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1945
; %bb.1949:                             ;   in Loop: Header=BB0_1948 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_1945
.LBB0_1950:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[32:33], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1954
; %bb.1951:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[18:19], v[2:3], 0, 1
	v_lshl_add_u64 v[22:23], v[18:19], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v36, v10
	v_mov_b32_e32 v37, v11
	v_cndmask_b32_e32 v35, v23, v19, vcc
	v_cndmask_b32_e32 v34, v22, v18, vcc
	v_and_b32_e32 v3, v35, v3
	v_and_b32_e32 v2, v34, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_1954
; %bb.1952:                             ; %.preheader.i.i.i6.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_1953:                             ; %.preheader.i.i.i6.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[36:37], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[36:37]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[36:37], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_1953
.LBB0_1954:                             ; %Flow3448
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
.LBB0_1955:                             ; %__ockl_printf_append_string_n.exit.i.1.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1961
; %bb.1956:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[36:37], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v36
	v_and_b32_e32 v3, v3, v37
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[34:35], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[36:37]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_1960
; %bb.1957:                             ; %.preheader3.i.i.i16.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_1958:                             ; %.preheader3.i.i.i16.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[36:37], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v36
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v37
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[34:35], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[36:37]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_1958
; %bb.1959:                             ; %Flow3392
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_1960:                             ; %Flow3394
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_1961:                             ; %.loopexit4.i.i.i10.i.1.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[36:39], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[36:37], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_1963
; %bb.1962:                             ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_1963:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[38:39], 0, s[6:7]
	v_and_or_b32 v32, v32, s33, 32
	v_mov_b32_e32 v34, v26
	v_mov_b32_e32 v35, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	s_nop 1
	global_store_dwordx4 v58, v[32:35], s[22:23]
	s_nop 1
	v_mov_b64_e32 v[32:33], s[6:7]
	v_mov_b64_e32 v[30:31], s[4:5]
	global_store_dwordx4 v58, v[30:33], s[22:23] offset:16
	global_store_dwordx4 v58, v[30:33], s[22:23] offset:32
	global_store_dwordx4 v58, v[30:33], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1971
; %bb.1964:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[34:35], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v32, s20
	v_mov_b32_e32 v33, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[36:37], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[34:35], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[32:35], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[34:35]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1967
; %bb.1965:                             ; %.preheader1.i.i.i14.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_1966:                             ; %.preheader1.i.i.i14.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[32:33], off
	v_mov_b32_e32 v30, s20
	v_mov_b32_e32 v31, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[30:33], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[32:33]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[32:33], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_1966
.LBB0_1967:                             ; %Flow3390
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1969
; %bb.1968:                             ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_1969:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_1971
; %bb.1970:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_1971:                             ; %Flow3391
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_1975
.LBB0_1972:                             ;   in Loop: Header=BB0_1975 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_1974
; %bb.1973:                             ;   in Loop: Header=BB0_1975 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_1975
	s_branch .LBB0_1977
.LBB0_1974:                             ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_1977
.LBB0_1975:                             ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1972
; %bb.1976:                             ;   in Loop: Header=BB0_1975 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_1972
.LBB0_1977:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[14:15], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1981
; %bb.1978:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[18:19], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[22:23], v[2:3], 0, 1
	v_lshl_add_u64 v[30:31], v[22:23], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[30:31]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v32, v10
	v_mov_b32_e32 v33, v11
	v_cndmask_b32_e32 v31, v31, v23, vcc
	v_cndmask_b32_e32 v30, v30, v22, vcc
	v_and_b32_e32 v3, v31, v3
	v_and_b32_e32 v2, v30, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[18:19], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_1981
; %bb.1979:                             ; %.preheader.i.i.i13.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_1980:                             ; %.preheader.i.i.i13.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[32:33], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[32:33]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[32:33], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_1980
.LBB0_1981:                             ; %__ockl_printf_append_args.exit.i.1.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1987
; %bb.1982:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[32:33], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v32
	v_and_b32_e32 v3, v3, v33
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[32:33]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_1986
; %bb.1983:                             ; %.preheader3.i.i.i23.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_1984:                             ; %.preheader3.i.i.i23.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[18:19], v9, s[16:17]
	v_mov_b64_e32 v[32:33], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v32
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[18:19]
	v_and_b32_e32 v7, v11, v33
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[32:33]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_1984
; %bb.1985:                             ; %Flow3378
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_1986:                             ; %Flow3380
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_1987:                             ; %.loopexit4.i.i.i17.i.1.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[30:33], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[30:31], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_1989
; %bb.1988:                             ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_1989:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[32:33], 0, s[6:7]
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	v_mov_b64_e32 v[34:35], s[6:7]
	v_and_or_b32 v14, v14, s33, 32
	v_mov_b32_e32 v17, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	v_mov_b64_e32 v[32:33], s[4:5]
	s_nop 3
	global_store_dwordx4 v58, v[14:17], s[22:23]
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:16
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:32
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1997
; %bb.1990:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[34:35], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v32, s20
	v_mov_b32_e32 v33, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[30:31], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[34:35], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[32:35], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[34:35]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1993
; %bb.1991:                             ; %.preheader1.i.i.i21.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_1992:                             ; %.preheader1.i.i.i21.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[32:33], off
	v_mov_b32_e32 v30, s20
	v_mov_b32_e32 v31, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[30:33], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[32:33]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[32:33], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_1992
.LBB0_1993:                             ; %Flow3376
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_1995
; %bb.1994:                             ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_1995:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_1997
; %bb.1996:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_1997:                             ; %Flow3377
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_2001
.LBB0_1998:                             ;   in Loop: Header=BB0_2001 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_2000
; %bb.1999:                             ;   in Loop: Header=BB0_2001 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_2001
	s_branch .LBB0_2003
.LBB0_2000:                             ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_2003
.LBB0_2001:                             ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_1998
; %bb.2002:                             ;   in Loop: Header=BB0_2001 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_1998
.LBB0_2003:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[18:19], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_2007
; %bb.2004:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[22:23], v[2:3], 0, 1
	v_lshl_add_u64 v[30:31], v[22:23], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[30:31]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v32, v10
	v_mov_b32_e32 v33, v11
	v_cndmask_b32_e32 v31, v31, v23, vcc
	v_cndmask_b32_e32 v30, v30, v22, vcc
	v_and_b32_e32 v3, v31, v3
	v_and_b32_e32 v2, v30, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_2007
; %bb.2005:                             ; %.preheader.i.i.i20.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_2006:                             ; %.preheader.i.i.i20.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[32:33], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[32:33]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[32:33], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_2006
.LBB0_2007:                             ; %__ockl_printf_append_args.exit24.i.1.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_2013
; %bb.2008:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[32:33], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v32
	v_and_b32_e32 v3, v3, v33
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[32:33]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_2012
; %bb.2009:                             ; %.preheader3.i.i.i31.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_2010:                             ; %.preheader3.i.i.i31.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[32:33], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v32
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v33
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[32:33]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_2010
; %bb.2011:                             ; %Flow3364
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_2012:                             ; %Flow3366
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_2013:                             ; %.loopexit4.i.i.i25.i.1.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[30:33], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[30:31], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_2015
; %bb.2014:                             ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_2015:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[32:33], 0, s[6:7]
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	v_mov_b64_e32 v[34:35], s[6:7]
	v_and_or_b32 v18, v18, s33, 32
	v_mov_b32_e32 v21, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	v_mov_b64_e32 v[32:33], s[4:5]
	s_nop 3
	global_store_dwordx4 v58, v[18:21], s[22:23]
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:16
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:32
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_2023
; %bb.2016:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[34:35], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v32, s20
	v_mov_b32_e32 v33, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[30:31], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[34:35], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[32:35], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[34:35]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_2019
; %bb.2017:                             ; %.preheader1.i.i.i29.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_2018:                             ; %.preheader1.i.i.i29.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[32:33], off
	v_mov_b32_e32 v30, s20
	v_mov_b32_e32 v31, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[30:33], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[32:33]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[32:33], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_2018
.LBB0_2019:                             ; %Flow3362
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_2021
; %bb.2020:                             ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_2021:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_2023
; %bb.2022:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_2023:                             ; %Flow3363
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_2027
.LBB0_2024:                             ;   in Loop: Header=BB0_2027 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_2026
; %bb.2025:                             ;   in Loop: Header=BB0_2027 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_2027
	s_branch .LBB0_2029
.LBB0_2026:                             ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_2029
.LBB0_2027:                             ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_2024
; %bb.2028:                             ;   in Loop: Header=BB0_2027 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_2024
.LBB0_2029:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[18:19], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_2033
; %bb.2030:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[22:23], v[2:3], 0, 1
	v_lshl_add_u64 v[30:31], v[22:23], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[30:31]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v32, v10
	v_mov_b32_e32 v33, v11
	v_cndmask_b32_e32 v31, v31, v23, vcc
	v_cndmask_b32_e32 v30, v30, v22, vcc
	v_and_b32_e32 v3, v31, v3
	v_and_b32_e32 v2, v30, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_2033
; %bb.2031:                             ; %.preheader.i.i.i28.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_2032:                             ; %.preheader.i.i.i28.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[32:33], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[32:33]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[32:33], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_2032
.LBB0_2033:                             ; %__ockl_printf_append_args.exit32.i.1.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_2039
; %bb.2034:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[32:33], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v32
	v_and_b32_e32 v3, v3, v33
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[32:33]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_2038
; %bb.2035:                             ; %.preheader3.i.i.i39.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_2036:                             ; %.preheader3.i.i.i39.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[32:33], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v32
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v33
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[32:33]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_2036
; %bb.2037:                             ; %Flow3350
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_2038:                             ; %Flow3352
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_2039:                             ; %.loopexit4.i.i.i33.i.1.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[30:33], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[30:31], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_2041
; %bb.2040:                             ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_2041:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[32:33], 0, s[6:7]
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	v_mov_b64_e32 v[34:35], s[6:7]
	v_and_or_b32 v18, v18, s33, 32
	v_mov_b32_e32 v21, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	v_mov_b64_e32 v[32:33], s[4:5]
	s_nop 3
	global_store_dwordx4 v58, v[18:21], s[22:23]
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:16
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:32
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_2049
; %bb.2042:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[34:35], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v32, s20
	v_mov_b32_e32 v33, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[30:31], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[34:35], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[32:35], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[34:35]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_2045
; %bb.2043:                             ; %.preheader1.i.i.i37.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_2044:                             ; %.preheader1.i.i.i37.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[32:33], off
	v_mov_b32_e32 v30, s20
	v_mov_b32_e32 v31, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[30:33], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[32:33]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[32:33], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_2044
.LBB0_2045:                             ; %Flow3348
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_2047
; %bb.2046:                             ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_2047:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_2049
; %bb.2048:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_2049:                             ; %Flow3349
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_2053
.LBB0_2050:                             ;   in Loop: Header=BB0_2053 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_2052
; %bb.2051:                             ;   in Loop: Header=BB0_2053 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_2053
	s_branch .LBB0_2055
.LBB0_2052:                             ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_2055
.LBB0_2053:                             ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_2050
; %bb.2054:                             ;   in Loop: Header=BB0_2053 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_2050
.LBB0_2055:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[18:19], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_2059
; %bb.2056:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[22:23], v[2:3], 0, 1
	v_lshl_add_u64 v[30:31], v[22:23], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[30:31]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v32, v10
	v_mov_b32_e32 v33, v11
	v_cndmask_b32_e32 v31, v31, v23, vcc
	v_cndmask_b32_e32 v30, v30, v22, vcc
	v_and_b32_e32 v3, v31, v3
	v_and_b32_e32 v2, v30, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_2059
; %bb.2057:                             ; %.preheader.i.i.i36.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_2058:                             ; %.preheader.i.i.i36.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[32:33], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[32:33]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[32:33], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_2058
.LBB0_2059:                             ; %__ockl_printf_append_args.exit40.i.1.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_2065
; %bb.2060:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[32:33], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v32
	v_and_b32_e32 v3, v3, v33
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[32:33]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_2064
; %bb.2061:                             ; %.preheader3.i.i.i47.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_2062:                             ; %.preheader3.i.i.i47.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[32:33], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v32
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v33
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[30:31], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[32:33]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_2062
; %bb.2063:                             ; %Flow3336
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_2064:                             ; %Flow3338
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_2065:                             ; %.loopexit4.i.i.i41.i.1.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[30:33], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[30:31], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_2067
; %bb.2066:                             ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_2067:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[32:33], 0, s[6:7]
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	v_mov_b64_e32 v[34:35], s[6:7]
	v_and_or_b32 v18, v18, s34, 34
	v_mov_b32_e32 v21, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	v_mov_b64_e32 v[32:33], s[4:5]
	s_nop 3
	global_store_dwordx4 v58, v[18:21], s[22:23]
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:16
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:32
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_2075
; %bb.2068:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[34:35], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	v_mov_b32_e32 v32, s20
	v_mov_b32_e32 v33, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[10:11], v[30:31], 0, s[22:23]
	global_store_dwordx2 v[10:11], v[34:35], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[32:35], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[34:35]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_2071
; %bb.2069:                             ; %.preheader1.i.i.i45.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_2070:                             ; %.preheader1.i.i.i45.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[10:11], v[32:33], off
	v_mov_b32_e32 v30, s20
	v_mov_b32_e32 v31, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[14:15], v9, v[30:33], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[14:15], v[32:33]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[32:33], v[14:15]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_2070
.LBB0_2071:                             ; %Flow3334
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_2073
; %bb.2072:                             ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[10:11], v[8:9], off offset:8 sc1
.LBB0_2073:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[14:15], v[10:11], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[14:15]
	s_cbranch_vccnz .LBB0_2075
; %bb.2074:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[10:11], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[14:15], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_2075:                             ; %Flow3335
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	s_branch .LBB0_2079
.LBB0_2076:                             ;   in Loop: Header=BB0_2079 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_2078
; %bb.2077:                             ;   in Loop: Header=BB0_2079 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_2079
	s_branch .LBB0_2081
.LBB0_2078:                             ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_2081
.LBB0_2079:                             ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_2076
; %bb.2080:                             ;   in Loop: Header=BB0_2079 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_2076
.LBB0_2081:                             ;   in Loop: Header=BB0_273 Depth=1
	s_and_b64 exec, exec, s[2:3]
	s_cbranch_execz .LBB0_2085
; %bb.2082:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[18:19], v[2:3], 0, 1
	v_lshl_add_u64 v[22:23], v[18:19], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v32, v10
	v_mov_b32_e32 v33, v11
	v_cndmask_b32_e32 v31, v23, v19, vcc
	v_cndmask_b32_e32 v30, v22, v18, vcc
	v_and_b32_e32 v3, v31, v3
	v_and_b32_e32 v2, v30, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[32:33], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_2085
; %bb.2083:                             ; %.preheader.i.i.i44.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_2084:                             ; %.preheader.i.i.i44.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[32:33], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[30:33], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[32:33]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[32:33], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_2084
.LBB0_2085:                             ; %Flow3477
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[18:19]
	v_lshrrev_b32_e32 v2, 3, v26
	v_bitop3_b32 v30, v2, v26, s35 bitop3:0x6c
	s_and_saveexec_b64 s[18:19], s[0:1]
	s_cbranch_execz .LBB0_272
; %bb.2086:                             ; %if.then.1.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_2092
; %bb.2087:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[34:35], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v34
	v_and_b32_e32 v3, v3, v35
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[32:33], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[32:35], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[34:35]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_2091
; %bb.2088:                             ; %.preheader3.i.i.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_2089:                             ; %.preheader3.i.i.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[34:35], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v34
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v35
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[32:33], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[32:35], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[34:35]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_2089
; %bb.2090:                             ; %Flow3320
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_2091:                             ; %Flow3322
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_2092:                             ; %.loopexit4.i.i.i.1.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[32:35], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[32:33], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_2094
; %bb.2093:                             ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_2094:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[34:35], 0, s[6:7]
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	v_mov_b64_e32 v[36:37], s[6:7]
	v_mov_b32_e32 v7, v9
	v_mov_b32_e32 v8, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	v_mov_b64_e32 v[34:35], s[4:5]
	s_nop 3
	global_store_dwordx4 v58, v[6:9], s[22:23]
	global_store_dwordx4 v58, v[34:37], s[22:23] offset:16
	global_store_dwordx4 v58, v[34:37], s[22:23] offset:32
	global_store_dwordx4 v58, v[34:37], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_2102
; %bb.2095:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[36:37], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v34, s20
	v_mov_b32_e32 v35, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[32:33], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[36:37], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[34:35], v9, v[34:37], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[34:35], v[36:37]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_2098
; %bb.2096:                             ; %.preheader1.i.i.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_2097:                             ; %.preheader1.i.i.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[34:35], off
	v_mov_b32_e32 v32, s20
	v_mov_b32_e32 v33, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[32:35], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[34:35]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[34:35], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_2097
.LBB0_2098:                             ; %Flow3318
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_2100
; %bb.2099:                             ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_2100:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_2102
; %bb.2101:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_2102:                             ; %Flow3319
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_2106
.LBB0_2103:                             ;   in Loop: Header=BB0_2106 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_2105
; %bb.2104:                             ;   in Loop: Header=BB0_2106 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_2106
	s_branch .LBB0_2108
.LBB0_2105:                             ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_2108
.LBB0_2106:                             ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_2103
; %bb.2107:                             ;   in Loop: Header=BB0_2106 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_2103
.LBB0_2108:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[32:33], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_2112
; %bb.2109:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[18:19], v[2:3], 0, 1
	v_lshl_add_u64 v[22:23], v[18:19], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v36, v10
	v_mov_b32_e32 v37, v11
	v_cndmask_b32_e32 v35, v23, v19, vcc
	v_cndmask_b32_e32 v34, v22, v18, vcc
	v_and_b32_e32 v3, v35, v3
	v_and_b32_e32 v2, v34, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_2112
; %bb.2110:                             ; %.preheader.i.i.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_2111:                             ; %.preheader.i.i.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[36:37], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[36:37]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[36:37], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_2111
.LBB0_2112:                             ; %__ockl_printf_begin.exit.1.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	s_andn2_b64 vcc, exec, s[10:11]
	s_cbranch_vccnz .LBB0_2197
; %bb.2113:                             ;   in Loop: Header=BB0_273 Depth=1
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v2, 2, v32
	v_and_b32_e32 v34, -3, v32
	v_mov_b32_e32 v35, v33
	s_mov_b64 s[20:21], 49
	s_getpc_b64 s[6:7]
	s_add_u32 s6, s6, .str.2@rel32@lo+4
	s_addc_u32 s7, s7, .str.2@rel32@hi+12
	s_branch .LBB0_2115
.LBB0_2114:                             ; %__ockl_hostcall_preview.exit19.i.1.i.1
                                        ;   in Loop: Header=BB0_2115 Depth=2
	s_or_b64 exec, exec, s[26:27]
	s_sub_u32 s20, s20, s22
	s_subb_u32 s21, s21, s23
	s_add_u32 s6, s6, s22
	s_addc_u32 s7, s7, s23
	s_cmp_eq_u64 s[20:21], 0
	s_cbranch_scc1 .LBB0_2196
.LBB0_2115:                             ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_2118 Depth 3
                                        ;       Child Loop BB0_2125 Depth 3
                                        ;       Child Loop BB0_2133 Depth 3
                                        ;       Child Loop BB0_2141 Depth 3
                                        ;       Child Loop BB0_2149 Depth 3
                                        ;       Child Loop BB0_2157 Depth 3
                                        ;       Child Loop BB0_2165 Depth 3
                                        ;       Child Loop BB0_2173 Depth 3
                                        ;       Child Loop BB0_2181 Depth 3
                                        ;       Child Loop BB0_2190 Depth 3
                                        ;       Child Loop BB0_2195 Depth 3
	v_cmp_lt_u64_e64 s[2:3], s[20:21], 56
	s_and_b64 s[2:3], s[2:3], exec
	v_cmp_gt_u64_e64 s[2:3], s[20:21], 7
	s_cselect_b32 s23, s21, 0
	s_cselect_b32 s22, s20, 56
	s_and_b64 vcc, exec, s[2:3]
	s_cbranch_vccnz .LBB0_2120
; %bb.2116:                             ;   in Loop: Header=BB0_2115 Depth=2
	s_mov_b64 s[2:3], 0
	s_cmp_eq_u64 s[20:21], 0
	s_waitcnt vmcnt(0)
	v_mov_b64_e32 v[36:37], 0
	s_cbranch_scc1 .LBB0_2119
; %bb.2117:                             ; %.preheader30.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_2115 Depth=2
	s_lshl_b64 s[24:25], s[22:23], 3
	s_mov_b64 s[26:27], 0
	v_mov_b64_e32 v[36:37], 0
	s_mov_b64 s[28:29], s[6:7]
.LBB0_2118:                             ; %.preheader30.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_2115 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	global_load_ubyte v3, v9, s[28:29]
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s26, v[8:9]
	s_add_u32 s26, s26, 8
	s_addc_u32 s27, s27, 0
	s_add_u32 s28, s28, 1
	s_addc_u32 s29, s29, 0
	v_or_b32_e32 v36, v10, v36
	s_cmp_eq_u32 s24, s26
	v_or_b32_e32 v37, v11, v37
	s_cbranch_scc0 .LBB0_2118
.LBB0_2119:                             ; %Flow3287
                                        ;   in Loop: Header=BB0_2115 Depth=2
	s_mov_b32 s5, 0
	s_andn2_b64 vcc, exec, s[2:3]
	s_mov_b64 s[2:3], s[6:7]
	s_cbranch_vccz .LBB0_2121
	s_branch .LBB0_2122
.LBB0_2120:                             ;   in Loop: Header=BB0_2115 Depth=2
                                        ; implicit-def: $vgpr36_vgpr37
                                        ; implicit-def: $sgpr5
	s_mov_b64 s[2:3], s[6:7]
.LBB0_2121:                             ;   in Loop: Header=BB0_2115 Depth=2
	global_load_dwordx2 v[36:37], v9, s[6:7]
	s_add_i32 s5, s22, -8
	s_add_u32 s2, s6, 8
	s_addc_u32 s3, s7, 0
.LBB0_2122:                             ; %.loopexit31.i.1.i.1
                                        ;   in Loop: Header=BB0_2115 Depth=2
	s_cmp_gt_u32 s5, 7
	s_cbranch_scc1 .LBB0_2126
; %bb.2123:                             ;   in Loop: Header=BB0_2115 Depth=2
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_2127
; %bb.2124:                             ; %.preheader28.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_2115 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[38:39], 0
	s_mov_b64 s[26:27], 0
.LBB0_2125:                             ; %.preheader28.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_2115 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s28, s2, s26
	s_addc_u32 s29, s3, s27
	global_load_ubyte v3, v9, s[28:29]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v38, v10, v38
	s_cmp_eq_u32 s5, s26
	v_or_b32_e32 v39, v11, v39
	s_cbranch_scc0 .LBB0_2125
	s_branch .LBB0_2128
.LBB0_2126:                             ;   in Loop: Header=BB0_2115 Depth=2
                                        ; implicit-def: $vgpr38_vgpr39
                                        ; implicit-def: $sgpr28
	s_branch .LBB0_2129
.LBB0_2127:                             ;   in Loop: Header=BB0_2115 Depth=2
	v_mov_b64_e32 v[38:39], 0
.LBB0_2128:                             ; %Flow3282
                                        ;   in Loop: Header=BB0_2115 Depth=2
	s_mov_b32 s28, 0
	s_cbranch_execnz .LBB0_2130
.LBB0_2129:                             ;   in Loop: Header=BB0_2115 Depth=2
	global_load_dwordx2 v[38:39], v9, s[2:3]
	s_add_i32 s28, s5, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_2130:                             ; %.loopexit29.i.1.i.1
                                        ;   in Loop: Header=BB0_2115 Depth=2
	s_cmp_gt_u32 s28, 7
	s_cbranch_scc1 .LBB0_2134
; %bb.2131:                             ;   in Loop: Header=BB0_2115 Depth=2
	s_cmp_eq_u32 s28, 0
	s_cbranch_scc1 .LBB0_2135
; %bb.2132:                             ; %.preheader26.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_2115 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[40:41], 0
	s_mov_b64 s[26:27], 0
.LBB0_2133:                             ; %.preheader26.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_2115 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s30, s2, s26
	s_addc_u32 s31, s3, s27
	global_load_ubyte v3, v9, s[30:31]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v40, v10, v40
	s_cmp_eq_u32 s28, s26
	v_or_b32_e32 v41, v11, v41
	s_cbranch_scc0 .LBB0_2133
	s_branch .LBB0_2136
.LBB0_2134:                             ;   in Loop: Header=BB0_2115 Depth=2
                                        ; implicit-def: $sgpr5
	s_branch .LBB0_2137
.LBB0_2135:                             ;   in Loop: Header=BB0_2115 Depth=2
	v_mov_b64_e32 v[40:41], 0
.LBB0_2136:                             ; %Flow3277
                                        ;   in Loop: Header=BB0_2115 Depth=2
	s_mov_b32 s5, 0
	s_cbranch_execnz .LBB0_2138
.LBB0_2137:                             ;   in Loop: Header=BB0_2115 Depth=2
	global_load_dwordx2 v[40:41], v9, s[2:3]
	s_add_i32 s5, s28, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_2138:                             ; %.loopexit27.i.1.i.1
                                        ;   in Loop: Header=BB0_2115 Depth=2
	s_cmp_gt_u32 s5, 7
	s_cbranch_scc1 .LBB0_2142
; %bb.2139:                             ;   in Loop: Header=BB0_2115 Depth=2
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_2143
; %bb.2140:                             ; %.preheader24.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_2115 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[42:43], 0
	s_mov_b64 s[26:27], 0
.LBB0_2141:                             ; %.preheader24.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_2115 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s28, s2, s26
	s_addc_u32 s29, s3, s27
	global_load_ubyte v3, v9, s[28:29]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v42, v10, v42
	s_cmp_eq_u32 s5, s26
	v_or_b32_e32 v43, v11, v43
	s_cbranch_scc0 .LBB0_2141
	s_branch .LBB0_2144
.LBB0_2142:                             ;   in Loop: Header=BB0_2115 Depth=2
                                        ; implicit-def: $vgpr42_vgpr43
                                        ; implicit-def: $sgpr28
	s_branch .LBB0_2145
.LBB0_2143:                             ;   in Loop: Header=BB0_2115 Depth=2
	v_mov_b64_e32 v[42:43], 0
.LBB0_2144:                             ; %Flow3272
                                        ;   in Loop: Header=BB0_2115 Depth=2
	s_mov_b32 s28, 0
	s_cbranch_execnz .LBB0_2146
.LBB0_2145:                             ;   in Loop: Header=BB0_2115 Depth=2
	global_load_dwordx2 v[42:43], v9, s[2:3]
	s_add_i32 s28, s5, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_2146:                             ; %.loopexit25.i.1.i.1
                                        ;   in Loop: Header=BB0_2115 Depth=2
	s_cmp_gt_u32 s28, 7
	s_cbranch_scc1 .LBB0_2150
; %bb.2147:                             ;   in Loop: Header=BB0_2115 Depth=2
	s_cmp_eq_u32 s28, 0
	s_cbranch_scc1 .LBB0_2151
; %bb.2148:                             ; %.preheader22.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_2115 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[44:45], 0
	s_mov_b64 s[26:27], 0
.LBB0_2149:                             ; %.preheader22.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_2115 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s30, s2, s26
	s_addc_u32 s31, s3, s27
	global_load_ubyte v3, v9, s[30:31]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v44, v10, v44
	s_cmp_eq_u32 s28, s26
	v_or_b32_e32 v45, v11, v45
	s_cbranch_scc0 .LBB0_2149
	s_branch .LBB0_2152
.LBB0_2150:                             ;   in Loop: Header=BB0_2115 Depth=2
                                        ; implicit-def: $sgpr5
	s_branch .LBB0_2153
.LBB0_2151:                             ;   in Loop: Header=BB0_2115 Depth=2
	v_mov_b64_e32 v[44:45], 0
.LBB0_2152:                             ; %Flow3267
                                        ;   in Loop: Header=BB0_2115 Depth=2
	s_mov_b32 s5, 0
	s_cbranch_execnz .LBB0_2154
.LBB0_2153:                             ;   in Loop: Header=BB0_2115 Depth=2
	global_load_dwordx2 v[44:45], v9, s[2:3]
	s_add_i32 s5, s28, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_2154:                             ; %.loopexit23.i.1.i.1
                                        ;   in Loop: Header=BB0_2115 Depth=2
	s_cmp_gt_u32 s5, 7
	s_cbranch_scc1 .LBB0_2158
; %bb.2155:                             ;   in Loop: Header=BB0_2115 Depth=2
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_2159
; %bb.2156:                             ; %.preheader20.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_2115 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[46:47], 0
	s_mov_b64 s[26:27], 0
.LBB0_2157:                             ; %.preheader20.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_2115 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s28, s2, s26
	s_addc_u32 s29, s3, s27
	global_load_ubyte v3, v9, s[28:29]
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	v_or_b32_e32 v46, v10, v46
	s_cmp_eq_u32 s5, s26
	v_or_b32_e32 v47, v11, v47
	s_cbranch_scc0 .LBB0_2157
	s_branch .LBB0_2160
.LBB0_2158:                             ;   in Loop: Header=BB0_2115 Depth=2
                                        ; implicit-def: $vgpr46_vgpr47
                                        ; implicit-def: $sgpr28
	s_branch .LBB0_2161
.LBB0_2159:                             ;   in Loop: Header=BB0_2115 Depth=2
	v_mov_b64_e32 v[46:47], 0
.LBB0_2160:                             ; %Flow3262
                                        ;   in Loop: Header=BB0_2115 Depth=2
	s_mov_b32 s28, 0
	s_cbranch_execnz .LBB0_2162
.LBB0_2161:                             ;   in Loop: Header=BB0_2115 Depth=2
	global_load_dwordx2 v[46:47], v9, s[2:3]
	s_add_i32 s28, s5, -8
	s_add_u32 s2, s2, 8
	s_addc_u32 s3, s3, 0
.LBB0_2162:                             ; %.loopexit21.i.1.i.1
                                        ;   in Loop: Header=BB0_2115 Depth=2
	s_cmp_gt_u32 s28, 7
	s_cbranch_scc1 .LBB0_2166
; %bb.2163:                             ;   in Loop: Header=BB0_2115 Depth=2
	s_cmp_eq_u32 s28, 0
	s_cbranch_scc1 .LBB0_2167
; %bb.2164:                             ; %.preheader.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_2115 Depth=2
	s_mov_b64 s[24:25], 0
	v_mov_b64_e32 v[48:49], 0
	s_mov_b64 s[26:27], s[2:3]
.LBB0_2165:                             ; %.preheader.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_2115 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	global_load_ubyte v3, v9, s[26:27]
	s_add_i32 s28, s28, -1
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffff, v3
	v_lshlrev_b64 v[10:11], s24, v[8:9]
	s_add_u32 s24, s24, 8
	s_addc_u32 s25, s25, 0
	s_add_u32 s26, s26, 1
	s_addc_u32 s27, s27, 0
	v_or_b32_e32 v48, v10, v48
	s_cmp_eq_u32 s28, 0
	v_or_b32_e32 v49, v11, v49
	s_cbranch_scc0 .LBB0_2165
	s_branch .LBB0_2168
.LBB0_2166:                             ;   in Loop: Header=BB0_2115 Depth=2
	s_branch .LBB0_2169
.LBB0_2167:                             ;   in Loop: Header=BB0_2115 Depth=2
	v_mov_b64_e32 v[48:49], 0
.LBB0_2168:                             ; %Flow3257
                                        ;   in Loop: Header=BB0_2115 Depth=2
	s_cbranch_execnz .LBB0_2170
.LBB0_2169:                             ;   in Loop: Header=BB0_2115 Depth=2
	global_load_dwordx2 v[48:49], v9, s[2:3]
.LBB0_2170:                             ; %.loopexit.i.1.i.1
                                        ;   in Loop: Header=BB0_2115 Depth=2
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[10:11], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_2176
; %bb.2171:                             ;   in Loop: Header=BB0_2115 Depth=2
	global_load_dwordx2 v[52:53], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v3, v10, v52
	v_and_b32_e32 v7, v11, v53
	v_mul_lo_u32 v7, v7, 24
	v_mul_hi_u32 v8, v3, 24
	v_add_u32_e32 v11, v8, v7
	v_mul_lo_u32 v10, v3, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[14:15], 0, v[10:11]
	global_load_dwordx2 v[50:51], v[10:11], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[50:53], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[10:11], v[52:53]
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB0_2175
; %bb.2172:                             ; %.preheader3.i.i18.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_2115 Depth=2
	s_mov_b64 s[28:29], 0
.LBB0_2173:                             ; %.preheader3.i.i18.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_2115 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_sleep 1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	global_load_dwordx2 v[18:19], v9, s[16:17]
	v_mov_b64_e32 v[52:53], v[10:11]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v7, v14, v52
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[10:11], s[30:31], v7, 24, v[18:19]
	v_and_b32_e32 v3, v15, v53
	v_mov_b32_e32 v8, v11
	v_mad_u64_u32 v[14:15], s[30:31], v3, 24, v[8:9]
	v_mov_b32_e32 v11, v14
	global_load_dwordx2 v[50:51], v[10:11], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[50:53], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[10:11], v[52:53]
	s_or_b64 s[28:29], vcc, s[28:29]
	s_andn2_b64 exec, exec, s[28:29]
	s_cbranch_execnz .LBB0_2173
; %bb.2174:                             ; %Flow3252
                                        ;   in Loop: Header=BB0_2115 Depth=2
	s_or_b64 exec, exec, s[28:29]
.LBB0_2175:                             ; %Flow3254
                                        ;   in Loop: Header=BB0_2115 Depth=2
	s_or_b64 exec, exec, s[26:27]
.LBB0_2176:                             ; %.loopexit4.i.i13.i.1.i.1
                                        ;   in Loop: Header=BB0_2115 Depth=2
	s_or_b64 exec, exec, s[24:25]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	global_load_dwordx4 v[50:53], v9, s[16:17]
	v_readfirstlane_b32 s24, v10
	v_readfirstlane_b32 s25, v11
	s_mov_b64 s[26:27], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s28, v14
	v_readfirstlane_b32 s29, v15
	s_and_b64 s[28:29], s[24:25], s[28:29]
	s_mul_i32 s5, s29, 24
	s_mul_hi_u32 s30, s28, 24
	s_add_i32 s31, s30, s5
	s_mul_i32 s30, s28, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[50:51], 0, s[30:31]
	s_and_saveexec_b64 s[30:31], s[2:3]
	s_cbranch_execz .LBB0_2178
; %bb.2177:                             ;   in Loop: Header=BB0_2115 Depth=2
	v_mov_b64_e32 v[22:23], s[26:27]
	global_store_dwordx4 v[10:11], v[22:25], off offset:8
.LBB0_2178:                             ;   in Loop: Header=BB0_2115 Depth=2
	s_or_b64 exec, exec, s[30:31]
	v_or_b32_e32 v3, 0, v35
	v_or_b32_e32 v7, v34, v2
	v_cmp_gt_u64_e64 vcc, s[20:21], 56
	s_lshl_b32 s5, s22, 2
	s_lshl_b64 s[26:27], s[28:29], 12
	v_cndmask_b32_e32 v35, v3, v35, vcc
	v_cndmask_b32_e32 v3, v7, v34, vcc
	s_add_i32 s5, s5, 28
	v_lshl_add_u64 v[14:15], v[52:53], 0, s[26:27]
	s_and_b32 s5, s5, 0x1e0
	v_and_b32_e32 v3, 0xffffff1f, v3
	v_or_b32_e32 v34, s5, v3
	v_readfirstlane_b32 s26, v14
	v_readfirstlane_b32 s27, v15
	s_nop 4
	global_store_dwordx4 v58, v[34:37], s[26:27]
	global_store_dwordx4 v58, v[38:41], s[26:27] offset:16
	global_store_dwordx4 v58, v[42:45], s[26:27] offset:32
	global_store_dwordx4 v58, v[46:49], s[26:27] offset:48
	s_and_saveexec_b64 s[26:27], s[2:3]
	s_cbranch_execz .LBB0_2186
; %bb.2179:                             ;   in Loop: Header=BB0_2115 Depth=2
	global_load_dwordx2 v[38:39], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[18:19], v9, s[16:17] offset:40
	v_mov_b32_e32 v36, s24
	v_mov_b32_e32 v37, s25
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s28, v18
	v_readfirstlane_b32 s29, v19
	s_and_b64 s[28:29], s[28:29], s[24:25]
	s_mul_i32 s5, s29, 24
	s_mul_hi_u32 s29, s28, 24
	s_mul_i32 s28, s28, 24
	s_add_i32 s29, s29, s5
	v_lshl_add_u64 v[18:19], v[50:51], 0, s[28:29]
	global_store_dwordx2 v[18:19], v[38:39], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v9, v[36:39], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[38:39]
	s_and_saveexec_b64 s[28:29], vcc
	s_cbranch_execz .LBB0_2182
; %bb.2180:                             ; %.preheader1.i.i16.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_2115 Depth=2
	s_mov_b64 s[30:31], 0
.LBB0_2181:                             ; %.preheader1.i.i16.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_2115 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_sleep 1
	global_store_dwordx2 v[18:19], v[36:37], off
	v_mov_b32_e32 v34, s24
	v_mov_b32_e32 v35, s25
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[22:23], v9, v[34:37], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[22:23], v[36:37]
	s_or_b64 s[30:31], vcc, s[30:31]
	v_mov_b64_e32 v[36:37], v[22:23]
	s_andn2_b64 exec, exec, s[30:31]
	s_cbranch_execnz .LBB0_2181
.LBB0_2182:                             ; %Flow3250
                                        ;   in Loop: Header=BB0_2115 Depth=2
	s_or_b64 exec, exec, s[28:29]
	global_load_dwordx2 v[18:19], v9, s[16:17] offset:16
	s_mov_b64 s[30:31], exec
	v_mbcnt_lo_u32_b32 v3, s30, 0
	v_mbcnt_hi_u32_b32 v3, s31, v3
	v_cmp_eq_u32_e32 vcc, 0, v3
	s_and_saveexec_b64 s[28:29], vcc
	s_cbranch_execz .LBB0_2184
; %bb.2183:                             ;   in Loop: Header=BB0_2115 Depth=2
	s_bcnt1_i32_b64 s5, s[30:31]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[18:19], v[8:9], off offset:8 sc1
.LBB0_2184:                             ;   in Loop: Header=BB0_2115 Depth=2
	s_or_b64 exec, exec, s[28:29]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[22:23], v[18:19], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_cbranch_vccnz .LBB0_2186
; %bb.2185:                             ;   in Loop: Header=BB0_2115 Depth=2
	global_load_dword v8, v[18:19], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v3, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v3
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[22:23], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_2186:                             ; %Flow3251
                                        ;   in Loop: Header=BB0_2115 Depth=2
	s_or_b64 exec, exec, s[26:27]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[14:15], v[14:15], 0, v[58:59]
	s_branch .LBB0_2190
.LBB0_2187:                             ;   in Loop: Header=BB0_2190 Depth=3
	s_or_b64 exec, exec, s[26:27]
	v_readfirstlane_b32 s5, v3
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_2189
; %bb.2188:                             ;   in Loop: Header=BB0_2190 Depth=3
	s_sleep 1
	s_cbranch_execnz .LBB0_2190
	s_branch .LBB0_2192
.LBB0_2189:                             ;   in Loop: Header=BB0_2115 Depth=2
	s_branch .LBB0_2192
.LBB0_2190:                             ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_2115 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_mov_b32_e32 v3, 1
	s_and_saveexec_b64 s[26:27], s[2:3]
	s_cbranch_execz .LBB0_2187
; %bb.2191:                             ;   in Loop: Header=BB0_2190 Depth=3
	global_load_dword v3, v[10:11], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v3, 1, v3
	s_branch .LBB0_2187
.LBB0_2192:                             ;   in Loop: Header=BB0_2115 Depth=2
	global_load_dwordx4 v[34:37], v[14:15], off
	s_and_saveexec_b64 s[26:27], s[2:3]
	s_cbranch_execz .LBB0_2114
; %bb.2193:                             ;   in Loop: Header=BB0_2115 Depth=2
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[18:19], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[22:23], v[10:11], 0, 1
	v_lshl_add_u64 v[26:27], v[22:23], 0, s[24:25]
	v_cmp_eq_u64_e32 vcc, 0, v[26:27]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v38, v14
	v_mov_b32_e32 v39, v15
	v_cndmask_b32_e32 v37, v27, v23, vcc
	v_cndmask_b32_e32 v36, v26, v22, vcc
	v_and_b32_e32 v3, v37, v11
	v_and_b32_e32 v7, v36, v10
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v8, v7, 24
	v_mul_lo_u32 v10, v7, 24
	v_add_u32_e32 v11, v8, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[18:19], 0, v[10:11]
	global_store_dwordx2 v[10:11], v[14:15], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v9, v[36:39], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[14:15]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_2114
; %bb.2194:                             ; %.preheader.i.i15.i.1.i.1.preheader
                                        ;   in Loop: Header=BB0_2115 Depth=2
	s_mov_b64 s[2:3], 0
.LBB0_2195:                             ; %.preheader.i.i15.i.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ;     Parent Loop BB0_2115 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_sleep 1
	global_store_dwordx2 v[10:11], v[38:39], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[14:15], v9, v[36:39], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[14:15], v[38:39]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[38:39], v[14:15]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_2195
	s_branch .LBB0_2114
.LBB0_2196:                             ; %Flow3290
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_2225
.LBB0_2197:                             ;   in Loop: Header=BB0_273 Depth=1
                                        ; implicit-def: $vgpr34_vgpr35
	s_cbranch_execz .LBB0_2225
; %bb.2198:                             ;   in Loop: Header=BB0_273 Depth=1
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_2204
; %bb.2199:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[36:37], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v36
	v_and_b32_e32 v3, v3, v37
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[34:35], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[36:37]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_2203
; %bb.2200:                             ; %.preheader3.i.i.i38.1.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_2201:                             ; %.preheader3.i.i.i38.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[36:37], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v36
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v37
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[34:35], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[34:37], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[36:37]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_2201
; %bb.2202:                             ; %Flow3303
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_2203:                             ; %Flow3305
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_2204:                             ; %.loopexit4.i.i.i33.1.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[36:39], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[36:37], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_2206
; %bb.2205:                             ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_2206:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[38:39], 0, s[6:7]
	v_and_or_b32 v32, v32, s33, 32
	v_mov_b32_e32 v34, v9
	v_mov_b32_e32 v35, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	s_nop 1
	global_store_dwordx4 v58, v[32:35], s[22:23]
	s_nop 1
	v_mov_b64_e32 v[34:35], s[6:7]
	v_mov_b64_e32 v[32:33], s[4:5]
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:16
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:32
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_2214
; %bb.2207:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[40:41], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v38, s20
	v_mov_b32_e32 v39, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[36:37], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[40:41], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[34:35], v9, v[38:41], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[34:35], v[40:41]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_2210
; %bb.2208:                             ; %.preheader1.i.i.i36.1.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_2209:                             ; %.preheader1.i.i.i36.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[34:35], off
	v_mov_b32_e32 v32, s20
	v_mov_b32_e32 v33, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[32:35], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[34:35]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[34:35], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_2209
.LBB0_2210:                             ; %Flow3301
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_2212
; %bb.2211:                             ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_2212:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_2214
; %bb.2213:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_2214:                             ; %Flow3302
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_2218
.LBB0_2215:                             ;   in Loop: Header=BB0_2218 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_2217
; %bb.2216:                             ;   in Loop: Header=BB0_2218 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_2218
	s_branch .LBB0_2220
.LBB0_2217:                             ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_2220
.LBB0_2218:                             ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_2215
; %bb.2219:                             ;   in Loop: Header=BB0_2218 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_2215
.LBB0_2220:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[34:35], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_2224
; %bb.2221:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[18:19], v[2:3], 0, 1
	v_lshl_add_u64 v[22:23], v[18:19], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v38, v10
	v_mov_b32_e32 v39, v11
	v_cndmask_b32_e32 v37, v23, v19, vcc
	v_cndmask_b32_e32 v36, v22, v18, vcc
	v_and_b32_e32 v3, v37, v3
	v_and_b32_e32 v2, v36, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v9, v[36:39], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_2224
; %bb.2222:                             ; %.preheader.i.i.i35.1.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_2223:                             ; %.preheader.i.i.i35.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[38:39], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[36:39], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[38:39]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[38:39], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_2223
.LBB0_2224:                             ; %Flow3294
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
.LBB0_2225:                             ; %__ockl_printf_append_string_n.exit.1.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_2231
; %bb.2226:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[38:39], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v38
	v_and_b32_e32 v3, v3, v39
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[36:37], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[36:39], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[38:39]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_2230
; %bb.2227:                             ; %.preheader3.i.i.i45.1.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_2228:                             ; %.preheader3.i.i.i45.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[38:39], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v38
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v39
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[36:37], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[36:39], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[38:39]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_2228
; %bb.2229:                             ; %Flow3238
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_2230:                             ; %Flow3240
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_2231:                             ; %.loopexit4.i.i.i39.1.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[38:41], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[38:39], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_2233
; %bb.2232:                             ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_2233:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[40:41], 0, s[6:7]
	v_and_or_b32 v34, v34, s33, 32
	v_mov_b32_e32 v36, v0
	v_mov_b32_e32 v37, v1
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	s_nop 1
	global_store_dwordx4 v58, v[34:37], s[22:23]
	s_nop 1
	v_mov_b64_e32 v[34:35], s[6:7]
	v_mov_b64_e32 v[32:33], s[4:5]
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:16
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:32
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_2241
; %bb.2234:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[36:37], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v34, s20
	v_mov_b32_e32 v35, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[38:39], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[36:37], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[34:35], v9, v[34:37], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[34:35], v[36:37]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_2237
; %bb.2235:                             ; %.preheader1.i.i.i43.1.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_2236:                             ; %.preheader1.i.i.i43.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[34:35], off
	v_mov_b32_e32 v32, s20
	v_mov_b32_e32 v33, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[32:35], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[34:35]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[34:35], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_2236
.LBB0_2237:                             ; %Flow3236
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_2239
; %bb.2238:                             ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_2239:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_2241
; %bb.2240:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_2241:                             ; %Flow3237
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_2245
.LBB0_2242:                             ;   in Loop: Header=BB0_2245 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_2244
; %bb.2243:                             ;   in Loop: Header=BB0_2245 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_2245
	s_branch .LBB0_2247
.LBB0_2244:                             ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_2247
.LBB0_2245:                             ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_2242
; %bb.2246:                             ;   in Loop: Header=BB0_2245 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_2242
.LBB0_2247:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[26:27], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_2251
; %bb.2248:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[18:19], v[2:3], 0, 1
	v_lshl_add_u64 v[22:23], v[18:19], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v34, v10
	v_mov_b32_e32 v35, v11
	v_cndmask_b32_e32 v33, v23, v19, vcc
	v_cndmask_b32_e32 v32, v22, v18, vcc
	v_and_b32_e32 v3, v33, v3
	v_and_b32_e32 v2, v32, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[34:35], v9, v[32:35], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[34:35], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_2251
; %bb.2249:                             ; %.preheader.i.i.i42.1.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_2250:                             ; %.preheader.i.i.i42.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[34:35], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[32:35], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[34:35]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[34:35], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_2250
.LBB0_2251:                             ; %__ockl_printf_append_args.exit.1.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_2257
; %bb.2252:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[34:35], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v34
	v_and_b32_e32 v3, v3, v35
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[32:33], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[32:35], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[34:35]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_2256
; %bb.2253:                             ; %.preheader3.i.i.i52.1.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_2254:                             ; %.preheader3.i.i.i52.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[34:35], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v34
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v35
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[32:33], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[32:35], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[34:35]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_2254
; %bb.2255:                             ; %Flow3224
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_2256:                             ; %Flow3226
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_2257:                             ; %.loopexit4.i.i.i46.1.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[32:35], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[32:33], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_2259
; %bb.2258:                             ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_2259:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[34:35], 0, s[6:7]
	v_and_or_b32 v26, v26, s33, 32
	v_mov_b32_e32 v29, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	s_nop 1
	global_store_dwordx4 v58, v[26:29], s[22:23]
	s_nop 1
	v_mov_b64_e32 v[28:29], s[6:7]
	v_mov_b64_e32 v[26:27], s[4:5]
	global_store_dwordx4 v58, v[26:29], s[22:23] offset:16
	global_store_dwordx4 v58, v[26:29], s[22:23] offset:32
	global_store_dwordx4 v58, v[26:29], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_2267
; %bb.2260:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[36:37], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	v_mov_b32_e32 v34, s20
	v_mov_b32_e32 v35, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[14:15], v[32:33], 0, s[22:23]
	global_store_dwordx2 v[14:15], v[36:37], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[28:29], v9, v[34:37], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[28:29], v[36:37]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_2263
; %bb.2261:                             ; %.preheader1.i.i.i50.1.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_2262:                             ; %.preheader1.i.i.i50.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[28:29], off
	v_mov_b32_e32 v26, s20
	v_mov_b32_e32 v27, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[26:29], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[28:29]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[28:29], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_2262
.LBB0_2263:                             ; %Flow3222
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_2265
; %bb.2264:                             ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[14:15], v[8:9], off offset:8 sc1
.LBB0_2265:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[14:15], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_2267
; %bb.2266:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[14:15], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_2267:                             ; %Flow3223
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[58:59]
	s_branch .LBB0_2271
.LBB0_2268:                             ;   in Loop: Header=BB0_2271 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_2270
; %bb.2269:                             ;   in Loop: Header=BB0_2271 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_2271
	s_branch .LBB0_2273
.LBB0_2270:                             ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_2273
.LBB0_2271:                             ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_2268
; %bb.2272:                             ;   in Loop: Header=BB0_2271 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_2268
.LBB0_2273:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[10:11], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_2277
; %bb.2274:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[18:19], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[22:23], v[2:3], 0, 1
	v_lshl_add_u64 v[26:27], v[22:23], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[26:27]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v28, v14
	v_mov_b32_e32 v29, v15
	v_cndmask_b32_e32 v27, v27, v23, vcc
	v_cndmask_b32_e32 v26, v26, v22, vcc
	v_and_b32_e32 v3, v27, v3
	v_and_b32_e32 v2, v26, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[18:19], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[14:15], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[28:29], v9, v[26:29], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[28:29], v[14:15]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_2277
; %bb.2275:                             ; %.preheader.i.i.i49.1.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_2276:                             ; %.preheader.i.i.i49.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[28:29], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[14:15], v9, v[26:29], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[14:15], v[28:29]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[28:29], v[14:15]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_2276
.LBB0_2277:                             ; %__ockl_printf_append_args.exit53.1.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_2283
; %bb.2278:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[28:29], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v28
	v_and_b32_e32 v3, v3, v29
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_load_dwordx2 v[26:27], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[26:29], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[28:29]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_2282
; %bb.2279:                             ; %.preheader3.i.i.i60.1.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_2280:                             ; %.preheader3.i.i.i60.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	global_load_dwordx2 v[18:19], v9, s[16:17]
	v_mov_b64_e32 v[28:29], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v14, v28
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[18:19]
	v_and_b32_e32 v7, v15, v29
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[14:15], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v14
	global_load_dwordx2 v[26:27], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[26:29], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[28:29]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_2280
; %bb.2281:                             ; %Flow3210
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_2282:                             ; %Flow3212
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_2283:                             ; %.loopexit4.i.i.i54.1.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[14:15], v9, s[16:17] offset:40
	global_load_dwordx4 v[26:29], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[26:27], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_2285
; %bb.2284:                             ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_2285:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[14:15], v[28:29], 0, s[6:7]
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	v_mov_b64_e32 v[34:35], s[6:7]
	v_and_or_b32 v10, v10, s33, 32
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	v_mov_b64_e32 v[32:33], s[4:5]
	s_nop 3
	global_store_dwordx4 v58, v[10:13], s[22:23]
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:16
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:32
	global_store_dwordx4 v58, v[32:35], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_2293
; %bb.2286:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[34:35], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	v_mov_b32_e32 v32, s20
	v_mov_b32_e32 v33, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[10:11], v[26:27], 0, s[22:23]
	global_store_dwordx2 v[10:11], v[34:35], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[28:29], v9, v[32:35], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[28:29], v[34:35]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_2289
; %bb.2287:                             ; %.preheader1.i.i.i58.1.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_2288:                             ; %.preheader1.i.i.i58.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[10:11], v[28:29], off
	v_mov_b32_e32 v26, s20
	v_mov_b32_e32 v27, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v9, v[26:29], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[28:29]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[28:29], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_2288
.LBB0_2289:                             ; %Flow3208
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_2291
; %bb.2290:                             ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[10:11], v[8:9], off offset:8 sc1
.LBB0_2291:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[10:11], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_2293
; %bb.2292:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[10:11], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_2293:                             ; %Flow3209
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v59, v9
	v_lshl_add_u64 v[10:11], v[14:15], 0, v[58:59]
	s_branch .LBB0_2297
.LBB0_2294:                             ;   in Loop: Header=BB0_2297 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_2296
; %bb.2295:                             ;   in Loop: Header=BB0_2297 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_2297
	s_branch .LBB0_2299
.LBB0_2296:                             ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_2299
.LBB0_2297:                             ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_2294
; %bb.2298:                             ;   in Loop: Header=BB0_2297 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_2294
.LBB0_2299:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[28:29], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_2303
; %bb.2300:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[18:19], v[2:3], 0, 1
	v_lshl_add_u64 v[22:23], v[18:19], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v34, v10
	v_mov_b32_e32 v35, v11
	v_cndmask_b32_e32 v33, v23, v19, vcc
	v_cndmask_b32_e32 v32, v22, v18, vcc
	v_and_b32_e32 v3, v33, v3
	v_and_b32_e32 v2, v32, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[34:35], v9, v[32:35], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[34:35], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_2303
; %bb.2301:                             ; %.preheader.i.i.i57.1.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_2302:                             ; %.preheader.i.i.i57.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[34:35], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[32:35], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[34:35]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[34:35], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_2302
.LBB0_2303:                             ; %__ockl_printf_append_args.exit61.1.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s2, v56
	v_mov_b64_e32 v[2:3], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v56
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_2309
; %bb.2304:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[34:35], v9, s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v34
	v_and_b32_e32 v3, v3, v35
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_add_u32_e32 v3, v7, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[2:3]
	global_load_dwordx2 v[32:33], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[32:35], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[34:35]
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_2308
; %bb.2305:                             ; %.preheader3.i.i.i68.1.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[22:23], 0
.LBB0_2306:                             ; %.preheader3.i.i.i68.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx2 v[14:15], v9, s[16:17]
	v_mov_b64_e32 v[34:35], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v34
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[24:25], v2, 24, v[14:15]
	v_and_b32_e32 v7, v11, v35
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[10:11], s[24:25], v7, 24, v[8:9]
	v_mov_b32_e32 v3, v10
	global_load_dwordx2 v[32:33], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[32:35], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[34:35]
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_2306
; %bb.2307:                             ; %Flow3196
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB0_2308:                             ; %Flow3198
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[20:21]
.LBB0_2309:                             ; %.loopexit4.i.i.i62.1.i.1
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	global_load_dwordx4 v[32:35], v9, s[16:17]
	v_readfirstlane_b32 s20, v2
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[20:21], s[22:23]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_add_i32 s25, s24, s5
	s_mul_i32 s24, s22, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[32:33], 0, s[24:25]
	s_and_saveexec_b64 s[24:25], s[2:3]
	s_cbranch_execz .LBB0_2311
; %bb.2310:                             ;   in Loop: Header=BB0_273 Depth=1
	v_mov_b64_e32 v[22:23], s[6:7]
	global_store_dwordx4 v[2:3], v[22:25], off offset:8
.LBB0_2311:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_lshl_b64 s[6:7], s[22:23], 12
	v_lshl_add_u64 v[10:11], v[34:35], 0, s[6:7]
	v_and_or_b32 v28, v28, s34, 34
	v_mov_b32_e32 v31, v9
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	s_nop 1
	global_store_dwordx4 v58, v[28:31], s[22:23]
	s_nop 1
	v_mov_b64_e32 v[28:29], s[6:7]
	v_mov_b64_e32 v[26:27], s[4:5]
	global_store_dwordx4 v58, v[26:29], s[22:23] offset:16
	global_store_dwordx4 v58, v[26:29], s[22:23] offset:32
	global_store_dwordx4 v58, v[26:29], s[22:23] offset:48
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_2319
; %bb.2312:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[36:37], v9, s[16:17] offset:32 sc0 sc1
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:40
	v_mov_b32_e32 v34, s20
	v_mov_b32_e32 v35, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v10
	v_readfirstlane_b32 s23, v11
	s_and_b64 s[22:23], s[22:23], s[20:21]
	s_mul_i32 s5, s23, 24
	s_mul_hi_u32 s23, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s23, s5
	v_lshl_add_u64 v[10:11], v[32:33], 0, s[22:23]
	global_store_dwordx2 v[10:11], v[36:37], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[28:29], v9, v[34:37], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[28:29], v[36:37]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_2315
; %bb.2313:                             ; %.preheader1.i.i.i66.1.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[24:25], 0
.LBB0_2314:                             ; %.preheader1.i.i.i66.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[10:11], v[28:29], off
	v_mov_b32_e32 v26, s20
	v_mov_b32_e32 v27, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[14:15], v9, v[26:29], s[16:17] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[14:15], v[28:29]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[28:29], v[14:15]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_2314
.LBB0_2315:                             ; %Flow3194
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v7, s24, 0
	v_mbcnt_hi_u32_b32 v7, s25, v7
	v_cmp_eq_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_2317
; %bb.2316:                             ;   in Loop: Header=BB0_273 Depth=1
	s_bcnt1_i32_b64 s5, s[24:25]
	v_mov_b32_e32 v8, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[10:11], v[8:9], off offset:8 sc1
.LBB0_2317:                             ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[14:15], v[10:11], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[14:15]
	s_cbranch_vccnz .LBB0_2319
; %bb.2318:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dword v8, v[10:11], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0xffffff, v8
	s_nop 0
	v_readfirstlane_b32 s5, v7
	s_mov_b32 m0, s5
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[14:15], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_2319:                             ; %Flow3195
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_or_b64 exec, exec, s[6:7]
	s_branch .LBB0_2323
.LBB0_2320:                             ;   in Loop: Header=BB0_2323 Depth=2
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s5, v7
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc1 .LBB0_2322
; %bb.2321:                             ;   in Loop: Header=BB0_2323 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_2323
	s_branch .LBB0_2325
.LBB0_2322:                             ;   in Loop: Header=BB0_273 Depth=1
	s_branch .LBB0_2325
.LBB0_2323:                             ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v7, 1
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_2320
; %bb.2324:                             ;   in Loop: Header=BB0_2323 Depth=2
	global_load_dword v7, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v7, 1, v7
	s_branch .LBB0_2320
.LBB0_2325:                             ;   in Loop: Header=BB0_273 Depth=1
	s_and_b64 exec, exec, s[2:3]
	s_cbranch_execz .LBB0_272
; %bb.2326:                             ;   in Loop: Header=BB0_273 Depth=1
	global_load_dwordx2 v[2:3], v9, s[16:17] offset:40
	global_load_dwordx2 v[10:11], v9, s[16:17] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v9, s[16:17]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[18:19], v[2:3], 0, 1
	v_lshl_add_u64 v[22:23], v[18:19], 0, s[20:21]
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v28, v10
	v_mov_b32_e32 v29, v11
	v_cndmask_b32_e32 v27, v23, v19, vcc
	v_cndmask_b32_e32 v26, v22, v18, vcc
	v_and_b32_e32 v3, v27, v3
	v_and_b32_e32 v2, v26, v2
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v7, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	global_store_dwordx2 v[2:3], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[28:29], v9, v[26:29], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[28:29], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_272
; %bb.2327:                             ; %.preheader.i.i.i65.1.i.1.preheader
                                        ;   in Loop: Header=BB0_273 Depth=1
	s_mov_b64 s[2:3], 0
.LBB0_2328:                             ; %.preheader.i.i.i65.1.i.1
                                        ;   Parent Loop BB0_273 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[2:3], v[28:29], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v9, v[26:29], s[16:17] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[28:29]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[28:29], v[10:11]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_2328
	s_branch .LBB0_272
.LBB0_2329:                             ; %_ZL12load_lds_regITkN7kittens5ducks2rt3allENS0_2rtI14__hip_bfloat16Li64ELi64ENS1_9rt_layout3rowEEETkNS1_2st3allENS0_2stIS4_Li64ELi64EEEEvRT_RKT0_.exit
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	; wave barrier
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(0)
	; wave barrier
	s_waitcnt lgkmcnt(0)
	scratch_load_dwordx4 v[0:3], off, off
	scratch_load_dwordx4 v[6:9], off, off offset:16
	scratch_load_dwordx4 v[10:13], off, off offset:32
	scratch_load_dwordx4 v[14:17], off, off offset:48
	scratch_load_dwordx4 v[18:21], off, off offset:64
	scratch_load_dwordx4 v[22:25], off, off offset:80
	scratch_load_dwordx4 v[26:29], off, off offset:96
	scratch_load_dwordx4 v[30:33], off, off offset:112
	v_mad_u64_u32 v[4:5], s[0:1], v57, s14, v[4:5]
	s_lshl_b32 s0, s14, 4
	s_nop 0
	v_add_u32_e32 v34, s0, v4
	v_add_u32_e32 v36, s0, v34
	v_ashrrev_i32_e32 v5, 31, v4
	v_add_u32_e32 v38, s0, v36
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[12:13]
	v_ashrrev_i32_e32 v35, 31, v34
	v_ashrrev_i32_e32 v37, 31, v36
	v_ashrrev_i32_e32 v39, 31, v38
	v_lshl_add_u64 v[34:35], v[34:35], 1, s[12:13]
	v_lshl_add_u64 v[36:37], v[36:37], 1, s[12:13]
	v_lshl_add_u64 v[38:39], v[38:39], 1, s[12:13]
	s_waitcnt vmcnt(7)
	global_store_dwordx4 v[4:5], v[0:3], off
	s_waitcnt vmcnt(7)
	global_store_dwordx4 v[4:5], v[6:9], off offset:64
	s_waitcnt vmcnt(7)
	global_store_dwordx4 v[34:35], v[10:13], off
	s_waitcnt vmcnt(7)
	global_store_dwordx4 v[34:35], v[14:17], off offset:64
	s_waitcnt vmcnt(7)
	global_store_dwordx4 v[36:37], v[18:21], off
	s_waitcnt vmcnt(7)
	global_store_dwordx4 v[36:37], v[22:25], off offset:64
	s_waitcnt vmcnt(7)
	global_store_dwordx4 v[38:39], v[26:29], off
	s_waitcnt vmcnt(7)
	global_store_dwordx4 v[38:39], v[30:33], off offset:64
	s_waitcnt lgkmcnt(0)
	; wave barrier
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z8micro_tk13micro_globals
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 132
		.amdhsa_kernarg_size 400
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 65
		.amdhsa_next_free_sgpr 39
		.amdhsa_accum_offset 68
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	_Z8micro_tk13micro_globals, .Lfunc_end0-_Z8micro_tk13micro_globals
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 88244
; NumSgprs: 45
; NumVgprs: 65
; NumAgprs: 0
; TotalNumVgprs: 65
; ScratchSize: 132
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 8
; NumSGPRsForWavesPerEU: 45
; NumVGPRsForWavesPerEU: 65
; AccumOffset: 68
; Occupancy: 7
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 16
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.type	.str.1,@object                  ; @.str.1
	.section	.rodata.str1.1,"aMS",@progbits,1
.str.1:
	.asciz	"threadIdx.x: %d, row: %d, col: %d, global_elem_offset: %d, final_global_elem_offset: %d\n"
	.size	.str.1, 89

	.type	.str.2,@object                  ; @.str.2
.str.2:
	.asciz	"threadIdx.x: %d, row: %d, col: %d, out_addr: %x\n"
	.size	.str.2, 49

	.type	.str.3,@object                  ; @.str.3
.str.3:
	.asciz	"addr: %x, swizzle_bytes: %d, subtile_cols: %d, rows: %d, cols: %d\n"
	.size	.str.3, 67

	.type	__hip_cuid_6b3e5d0c69c0ea5b,@object ; @__hip_cuid_6b3e5d0c69c0ea5b
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_6b3e5d0c69c0ea5b
__hip_cuid_6b3e5d0c69c0ea5b:
	.byte	0                               ; 0x0
	.size	__hip_cuid_6b3e5d0c69c0ea5b, 1

	.ident	"AMD clang version 19.0.0git (ssh://github-emu/AMD-Lightning-Internal/llvm-project  25152 3450a62cc77a5a3e97438da52b7bb6c5243520b9)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __shm
	.addrsig_sym __hip_cuid_6b3e5d0c69c0ea5b
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           144
        .value_kind:     by_value
      - .offset:         144
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         148
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         152
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         156
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         158
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         160
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         162
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         164
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         166
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         184
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         192
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         200
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         208
        .size:           2
        .value_kind:     hidden_grid_dims
      - .offset:         224
        .size:           8
        .value_kind:     hidden_hostcall_buffer
      - .offset:         264
        .size:           4
        .value_kind:     hidden_dynamic_lds_size
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 400
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 64
    .name:           _Z8micro_tk13micro_globals
    .private_segment_fixed_size: 132
    .sgpr_count:     45
    .sgpr_spill_count: 0
    .symbol:         _Z8micro_tk13micro_globals.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     65
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
