#pragma once

#include <cmath>
#include <cstdlib>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace hymson3d {
namespace utility {

/// hash_tuple defines a general hash function for std::tuple
/// See this post for details:
///   http://stackoverflow.com/questions/7110301
/// The hash_combine code is from boost
/// Reciprocal of the golden ratio helps spread entropy and handles duplicates.
/// See Mike Seymour in magic-numbers-in-boosthash-combine:
///   http://stackoverflow.com/questions/4948780

template <typename TT>
struct hash_tuple {
    size_t operator()(TT const& tt) const { return std::hash<TT>()(tt); }
};

namespace {

template <class T>
inline void hash_combine(std::size_t& hash_seed, T const& v) {
    hash_seed ^= std::hash<T>()(v) + 0x9e3779b9 + (hash_seed << 6) +
                 (hash_seed >> 2);
}

template <class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
struct HashValueImpl {
    static void apply(size_t& hash_seed, Tuple const& tuple) {
        HashValueImpl<Tuple, Index - 1>::apply(hash_seed, tuple);
        hash_combine(hash_seed, std::get<Index>(tuple));
    }
};

template <class Tuple>
struct HashValueImpl<Tuple, 0> {
    static void apply(size_t& hash_seed, Tuple const& tuple) {
        hash_combine(hash_seed, std::get<0>(tuple));
    }
};

}  // unnamed namespace

template <typename... TT>
struct hash_tuple<std::tuple<TT...>> {
    size_t operator()(std::tuple<TT...> const& tt) const {
        size_t hash_seed = 0;
        HashValueImpl<std::tuple<TT...>>::apply(hash_seed, tt);
        return hash_seed;
    }
};

template <typename T>
struct hash_eigen {
    std::size_t operator()(T const& matrix) const {
        size_t hash_seed = 0;
        for (int i = 0; i < (int)matrix.size(); i++) {
            auto elem = *(matrix.data() + i);
            hash_seed ^= std::hash<typename T::Scalar>()(elem) + 0x9e3779b9 +
                         (hash_seed << 6) + (hash_seed >> 2);
        }
        return hash_seed;
    }
};

// Hash function for enum class for C++ standard less than C++14
// https://stackoverflow.com/a/24847480/1255535
struct hash_enum_class {
    template <typename T>
    std::size_t operator()(T t) const {
        return static_cast<std::size_t>(t);
    }
};

}  // namespace utility
}  // namespace hymson3d