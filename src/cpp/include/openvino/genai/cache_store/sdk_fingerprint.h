/* Copyright (C) 2026 Intel Corporation */
/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

/*
 * Build fingerprint of the SDK a KV cache store plugin compiles against (provider_abi.hpp,
 * OvKVCacheStoreModuleInfo::sdk_build_fingerprint). The host rejects a module whose fingerprint differs from this
 * string in any byte, so a plugin must be rebuilt against the GenAI release it is deployed with.
 *
 * The fingerprint names the GenAI release and the store ABI version; it changes with every GenAI release.
 * Pure C, so plugins written in C can include it.
 */
#define OV_KVCACHE_STORE_SDK_BUILD_FINGERPRINT "openvino.genai 2026.4.0.0; kvcache store abi 1.0"
