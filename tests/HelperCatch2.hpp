#pragma once

#include <catch2/matchers/catch_matchers.hpp>

#include <algorithm>
#include <span>
#include <string>

template<typename T>
class SpanMatcher : public Catch::Matchers::MatcherBase<std::span<const T>> {
    std::span<const T> m_expected;
public:
    SpanMatcher(std::span<const T> expected) : m_expected(expected) {}

    bool match(std::span<const T> const& actual) const override {
        return std::equal(actual.begin(), actual.end(), m_expected.begin(), m_expected.end());
    }

    std::string describe() const override {
        std::ostringstream ss;
        ss << "Expected span: { ";
        for (const auto& val : m_expected) ss << val << " ";
        ss << "}";
        return ss.str();
    }
};

template<typename T>
inline SpanMatcher<T> EqualsSpan(std::span<const T> expected) {
    return SpanMatcher(expected);
}

