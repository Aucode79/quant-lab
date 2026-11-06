#pragma once
/**
 * 日期工具函数
 *
 * A股交易相关日期处理:
 *   - 日期字符串 ↔ 整数转换
 *   - 日期加减
 *   - 自然日 ↔ 交易日判断（需配合交易日历）
 */

#include <string>

namespace quant::util {

/// 日期字符串转整数 "2024-01-15" → 20240115
inline int date_to_int(const std::string& date_str) {
    // "YYYY-MM-DD" → YYYYMMDD
    int year = std::stoi(date_str.substr(0, 4));
    int month = std::stoi(date_str.substr(5, 2));
    int day = std::stoi(date_str.substr(8, 2));
    return year * 10000 + month * 100 + day;
}

/// 整数转日期字符串 20240115 → "2024-01-15"
inline std::string int_to_date(int date_int) {
    int year = date_int / 10000;
    int month = (date_int % 10000) / 100;
    int day = date_int % 100;
    char buf[16];
    snprintf(buf, sizeof(buf), "%04d-%02d-%02d", year, month, day);
    return std::string(buf);
}

/// 日期比较
inline bool date_lt(const std::string& a, const std::string& b) {
    return date_to_int(a) < date_to_int(b);
}

inline bool date_le(const std::string& a, const std::string& b) {
    return date_to_int(a) <= date_to_int(b);
}

inline bool date_gt(const std::string& a, const std::string& b) {
    return date_to_int(a) > date_to_int(b);
}

inline bool date_ge(const std::string& a, const std::string& b) {
    return date_to_int(a) >= date_to_int(b);
}

}  // namespace quant::util
