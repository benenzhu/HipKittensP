/**
 * @file
 * @brief Layouts and their manipulations for register tiles.
 */

#pragma once

#include <concepts>

namespace kittens {
namespace ducks {
/**
 * @namespace st_layout
 * 
 * @brief A namespace for template metaprogramming with register tile layouts.
 */
namespace st_layout {

/**
 * @brief A dummy type used to identify a row-major layout for a register tile.
 */
struct classical {}; // for most matrices

/**
 * @brief A dummy type used to identify an accumulator col-major layout for a register tile.
 */
struct accumulator {};

/**
 * @brief A concept to check if a type is a register tile layout.
 */

template<typename T>
concept all = std::is_same_v<T, classical> || std::is_same_v<T, accumulator>;

} // namespace st_layout
} // namespace ducks
} // namespace kittens