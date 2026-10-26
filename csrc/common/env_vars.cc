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

#include "csrc/common/env_vars.h"

#include <string_view>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/synchronization/mutex.h"

namespace torch_tpu::env_vars_internal {
namespace {

ABSL_CONST_INIT absl::Mutex memoized_env_vars_mutex(absl::kConstInit);

// The names of the env vars whose values have been memoized by GetEnvOnce().
// The names are keyed by content rather than by address, as the kFooEnvVar
// arrays may have different addresses in different shared libraries. They
// point to the static kFooEnvVar arrays, so they never dangle.
absl::flat_hash_set<std::string_view>& MemoizedEnvVars()
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(memoized_env_vars_mutex) {
  static absl::NoDestructor<absl::flat_hash_set<std::string_view>>
      memoized_env_vars;
  return *memoized_env_vars;
}

}  // namespace

void MarkEnvVarMemoized(std::string_view name) {
  absl::MutexLock lock(memoized_env_vars_mutex);
  MemoizedEnvVars().insert(name);
}

void CheckEnvVarNotMemoized(std::string_view name) {
  absl::MutexLock lock(memoized_env_vars_mutex);
  ABSL_CHECK(!MemoizedEnvVars().contains(name))  // CRASH_OK
      << "SetEnv() or UnsetEnv() called on " << name
      << " after its value was memoized by GetEnvOnce(). The new value would "
         "be ignored by GetEnvOnce(). If TorchTPU needs to change the env var "
         "at runtime, read it with std::getenv instead of GetEnvOnce().";
}

}  // namespace torch_tpu::env_vars_internal
