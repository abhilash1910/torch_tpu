/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TORCH_TPU_CSRC_COMMON_UNBOUNDED_DYNAMIC_KERNEL_REGISTRY_H_
#define TORCH_TPU_CSRC_COMMON_UNBOUNDED_DYNAMIC_KERNEL_REGISTRY_H_

#include <memory>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/shape.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"

namespace torch_tpu {

// An unbounded-dynamic kernel that the compilation cache can load instead of
// compiling a traced computation.
struct UnboundedDynamicKernel {
  using BuildFn =
      absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> (*)(
          xla::PjRtClient& client, UniqueCompileOptions compile_options);

  BuildFn absl_nonnull build;
  UnboundedDynamicExecutionMetadata metadata;
  CompilationCacheKey key;
};

// Process-wide registry of unbounded dynamic kernels, grouped by shapeless key.
// Several kernels may share a shapeless key, e.g. with different compatibility
// predicates; a lookup returns the first compatible one in registration order.
class UnboundedDynamicKernelRegistry {
 public:
  // Owns the kernels, indexed by their exact cache key.
  using KernelMap = absl::flat_hash_map<CompilationCacheKey,
                                        std::unique_ptr<UnboundedDynamicKernel>,
                                        CompilationCacheKey::Hash>;
  // Non-owning index from shapeless key to the kernels registered under it, in
  // registration order. Pointers point into `KernelMap`'s heap-allocated
  // values, which are never freed or moved (the registry is append-only).
  using KeysMap =
      absl::flat_hash_map<ShapelessKey,
                          std::vector<const UnboundedDynamicKernel*>,
                          ShapelessKey::Hash>;

  static UnboundedDynamicKernelRegistry& GetInstance();

  UnboundedDynamicKernelRegistry(const UnboundedDynamicKernelRegistry&) =
      delete;
  UnboundedDynamicKernelRegistry& operator=(
      const UnboundedDynamicKernelRegistry&) = delete;

  // Registers `kernel` under `kernel.key` and its shapeless key. Must only be
  // called during static initialization; it is not synchronized with lookups.
  // Returns kAlreadyExists if a kernel is already registered under that key.
  absl::Status RegisterKernel(UnboundedDynamicKernel kernel);

  // Returns the first kernel registered under `key`, in registration order,
  // whose compatibility predicate accepts the given logical input and output
  // shapes, or null if there is none.
  [[nodiscard]] const UnboundedDynamicKernel* absl_nullable
  FindUnboundedDynamicKernel(const ShapelessKey& key,
                             absl::Span<const Shape> input_shapes,
                             absl::Span<const Shape> output_shapes) const;

  // Returns the kernel registered under exactly `key`, or null if there is
  // none.
  [[nodiscard]] const UnboundedDynamicKernel* absl_nullable
  GetUnboundedDynamicKernel(const CompilationCacheKey& key) const;

 private:
  friend class absl::NoDestructor<UnboundedDynamicKernelRegistry>;
  UnboundedDynamicKernelRegistry() = default;

  KernelMap kernels_;
  KeysMap keys_;
};

// Registers a kernel with `UnboundedDynamicKernelRegistry` when constructed.
// Prefer `TT_REGISTER_DYNAMIC_KERNEL` over instantiating this class directly.
class UnboundedDynamicKernelRegistrar {
 public:
  UnboundedDynamicKernelRegistrar(CompilationCacheKey key,
                                  UnboundedDynamicKernel::BuildFn
                                  absl_nonnull build_fn,
                                  UnboundedDynamicExecutionMetadata metadata) {
    ABSL_CHECK_OK(  // CRASH_OK=static-init registration bug
        UnboundedDynamicKernelRegistry::GetInstance().RegisterKernel(
            UnboundedDynamicKernel{.build = build_fn,
                                   .metadata = std::move(metadata),
                                   .key = std::move(key)}));
  }
};

#define TT_REGISTER_DYNAMIC_KERNEL_CONCAT_INNER_(x, y) x##y
#define TT_REGISTER_DYNAMIC_KERNEL_CONCAT_(x, y) \
  TT_REGISTER_DYNAMIC_KERNEL_CONCAT_INNER_(x, y)

// Registers an unbounded dynamic kernel with `UnboundedDynamicKernelRegistry`
// at static initialization time. Invoke at namespace scope in a library built
// with `alwayslink = True`:
//
//   TT_REGISTER_DYNAMIC_KERNEL(NonZeroCacheKey(),
//                              BuildNonzeroIndicesExecutable,
//                              MakeNonzeroDynamicMetadata());
//
// The arguments are evaluated during static initialization, so they must not
// depend on dynamically initialized globals from other translation units
// (flags, environment-derived settings, other registries, ...).
//
// Registering the same compilation cache key twice is a programming error and
// crashes at startup.
#define TT_REGISTER_DYNAMIC_KERNEL(key, build_fn, metadata)             \
  const ::torch_tpu::UnboundedDynamicKernelRegistrar                    \
  TT_REGISTER_DYNAMIC_KERNEL_CONCAT_(kUnboundedDynamicKernelRegistrar_, \
                                     __COUNTER__)((key), (build_fn),    \
                                                  (metadata))

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_UNBOUNDED_DYNAMIC_KERNEL_REGISTRY_H_
