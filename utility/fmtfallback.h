#pragma once

#include <fmt/format.h>

#include <type_traits>

namespace fmt {

// 只对 enum 生效
template <typename T, typename Char>
struct formatter<T, Char, std::enable_if_t<std::is_enum_v<T>, void>> {
    // 用 fmt 自带的 basic_format_parse_context
    using parse_context = basic_format_parse_context<Char>;

    // C++17 支持 auto 返回类型
    constexpr auto parse(parse_context& ctx) -> decltype(ctx.begin()) {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(T value, FormatContext& ctx) const {
        return format_to(ctx.out(), "{}",
                         static_cast<std::underlying_type_t<T>>(value));
    }
};

}  // namespace fmt

//first version
//#pragma once
//
//#include <fmt/format.h>
//#include <fmt/core.h>
//#include <type_traits>
//
//// 只为 enum 类型注册一个默认的 formatter<T>
//namespace fmt {
//
//template <typename T, typename Char>
//struct formatter<T,
//                 Char,
//                 // 只有当 T 是枚举类型时，这个特化才生效
//                 std::enable_if_t<std::is_enum_v<T>, void>> {
//    // 不消费任何格式选项
//    constexpr auto parse(format_parse_context<Char>& ctx) {
//        return ctx.begin();
//    }
//
//    template <typename FormatContext>
//    auto format(T value, FormatContext& ctx) {
//        // 将枚举转换为它的底层整型再输出
//        return format_to(ctx.out(), "{}",
//                         static_cast<std::underlying_type_t<T>>(value));
//    }
//};
//
//}  // namespace fmt

//second version
//#pragma once
//
//#include <fmt/core.h>
//
//#include <string>
//#include <type_traits>
//
//namespace fmt_fallback {
//
//template <typename T>
//inline std::string fallback_to_string(const T& value) {
//    if constexpr (std::is_enum_v<T>) {
//        return std::to_string(static_cast<std::underlying_type_t<T>>(value));
//    } else {
//        return "[unsupported type]";
//    }
//}
//
//}  // namespace fmt_fallback
//
//// 统一注册 fallback formatter
//namespace fmt {
//
//template <typename T>
//struct formatter<T,
//                 char,
//                 std::enable_if_t<!std::is_arithmetic_v<T> &&
//                                  !std::is_same_v<T, std::string>>> {
//    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
//
//    template <typename FormatContext>
//    auto format(const T& value, FormatContext& ctx) {
//        return format_to(ctx.out(), "{}",
//                         fmt_fallback::fallback_to_string(value));
//    }
//};
//
//}  // namespace fmt
