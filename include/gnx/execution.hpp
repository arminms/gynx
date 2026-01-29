// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Armin Sobhani
//
#pragma once

#include <type_traits>

namespace gnx::execution {

/// @brief Execution policy for sequential execution (no parallelism)
struct sequenced_policy
{};
/// @brief Execution policy for parallel execution with OpenMP
struct parallel_policy
{};
/// @brief Execution policy for parallel execution with unsequenced iterations
struct parallel_unsequenced_policy
{};
/// @brief Execution policy for unsequenced execution (vectorization)
struct unsequenced_policy
{};
/// @brief Execution policy for data-parallel execution on CUDA-enabled GPUs
struct data_parallel_cuda_policy
{};
/// @brief Execution policy for data-parallel execution on ROCm-enabled GPUs
struct data_parallel_rocm_policy
{};
/// @brief Execution policy for data-parallel execution on oneAPI-enabled devices
struct data_parallel_oneapi_policy
{};
/// @brief Sequential execution policy instance
inline constexpr sequenced_policy seq;

/// @brief Parallel execution policy instance
inline constexpr parallel_policy par;

/// @brief Parallel unsequenced execution policy instance
inline constexpr parallel_unsequenced_policy par_unseq;

/// @brief Unsequenced execution policy instance
inline constexpr unsequenced_policy unseq;

/// @brief Data-parallel CUDA execution policy instance
inline constexpr data_parallel_cuda_policy cuda;

/// @brief Data-parallel ROCm execution policy instance
inline constexpr data_parallel_rocm_policy rocm;

/// @brief Data-parallel oneAPI execution policy instance   
inline constexpr data_parallel_oneapi_policy oneapi;

} // namespace gnx::execution

namespace gnx {

/// @brief Type trait to check if a type is an execution policy
template<typename T>
struct is_execution_policy
:   std::false_type
{};
/// @brief Specialization for sequenced_policy
template<>
struct is_execution_policy<gnx::execution::sequenced_policy>
:   std::true_type
{};
/// @brief Specialization for parallel_policy
template<>
struct is_execution_policy<gnx::execution::parallel_policy>
:   std::true_type
{};
/// @brief Specialization for parallel_unsequenced_policy
template<>
struct is_execution_policy<gnx::execution::parallel_unsequenced_policy>
:   std::true_type
{};
/// @brief Specialization for unsequenced_policy
template<>
struct is_execution_policy<gnx::execution::unsequenced_policy>
:   std::true_type
{};
/// @brief Specialization for data_parallel_cuda_policy
template<>
struct is_execution_policy<gnx::execution::data_parallel_cuda_policy>
:   std::true_type
{};
/// @brief Specialization for data_parallel_rocm_policy
template<>
struct is_execution_policy<gnx::execution::data_parallel_rocm_policy>
:   std::true_type
{};
/// @brief Specialization for data_parallel_oneapi_policy
template<>
struct is_execution_policy<gnx::execution::data_parallel_oneapi_policy>
:   std::true_type
{};
/// @brief Helper variable template for is_execution_policy
template<typename T>
inline constexpr bool is_execution_policy_v = is_execution_policy<T>::value;

} // namespace gnx
