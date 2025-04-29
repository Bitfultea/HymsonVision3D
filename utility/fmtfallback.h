#pragma once

#include <fmt/format.h>

#include <type_traits>

namespace fmt {

// ֻ�� enum ��Ч
template <typename T, typename Char>
struct formatter<T, Char, std::enable_if_t<std::is_enum_v<T>, void>> {
    // �� fmt �Դ��� basic_format_parse_context
    using parse_context = basic_format_parse_context<Char>;

    // C++17 ֧�� auto ��������
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
//// ֻΪ enum ����ע��һ��Ĭ�ϵ� formatter<T>
//namespace fmt {
//
//template <typename T, typename Char>
//struct formatter<T,
//                 Char,
//                 // ֻ�е� T ��ö������ʱ������ػ�����Ч
//                 std::enable_if_t<std::is_enum_v<T>, void>> {
//    // �������κθ�ʽѡ��
//    constexpr auto parse(format_parse_context<Char>& ctx) {
//        return ctx.begin();
//    }
//
//    template <typename FormatContext>
//    auto format(T value, FormatContext& ctx) {
//        // ��ö��ת��Ϊ���ĵײ����������
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
//// ͳһע�� fallback formatter
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
