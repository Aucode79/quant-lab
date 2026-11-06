#pragma once
/**
 * 数据提供者抽象接口 (Data Provider Interface)
 *
 * 所有数据源实现必须继承此接口，提供统一的数据访问方式。
 * 支持三种实现:
 *   - DuckDBProvider: 生产环境，从 DuckDB/Parquet 读取
 *   - MemoryProvider: 测试/基准测试，纯内存数据
 *   - CSVProvider: 快速调试，从 CSV 文件读取
 */

#include <memory>
#include <string>
#include <vector>

#include <arrow/api.h>  // NOLINT

namespace quant {

// ── 数据结构 ───────────────────────────────────────────────

/// 股票基本信息
struct StockInfo {
    std::string ts_code;       // 股票代码 (000001.SZ)
    std::string name;          // 股票名称
    std::string industry;      // 申万一级行业
    std::string list_date;     // 上市日期
    std::string delist_date;   // 退市日期 (空字符串表示正常)
    std::string exchange;      // SSE / SZSE / BSE
};

/// 单日截面快照（用于选股）
struct DailySnapshot {
    std::string ts_code;
    std::string trade_date;
    double open = 0.0;
    double high = 0.0;
    double low = 0.0;
    double close = 0.0;
    double pre_close = 0.0;
    double volume = 0.0;
    double amount = 0.0;
    double adj_factor = 1.0;
    bool is_st = false;
    bool is_suspend = false;
};

// ── 接口 ───────────────────────────────────────────────────

class IDataProvider {
public:
    virtual ~IDataProvider() = default;

    // ── 日K线数据 ─────────────────────────────────────────

    /// 获取单只股票的日K线 (Arrow Table 格式)
    /// @return schema: [trade_date, open, high, low, close,
    ///                   pre_close, volume, amount, adj_factor,
    ///                   is_st, is_suspend]
    virtual std::shared_ptr<arrow::Table> get_daily_kline(
        const std::string& ts_code,
        const std::string& start_date,
        const std::string& end_date) = 0;

    /// 获取多只股票的日K线（宽表格式，用于因子计算）
    /// @return schema: [trade_date, ts_code, open, ..., close, ...]
    virtual std::shared_ptr<arrow::Table> get_multi_daily_kline(
        const std::vector<std::string>& ts_codes,
        const std::string& start_date,
        const std::string& end_date) = 0;

    // ── 横截面数据 ─────────────────────────────────────────

    /// 获取全市场某一天的截面快照
    /// @param date 交易日 (YYYY-MM-DD)
    /// @param min_volume 最低成交量过滤（排除流动性差的股票）
    virtual std::shared_ptr<arrow::Table> get_universe_snapshot(
        const std::string& date,
        double min_volume = 0.0) = 0;

    // ── 参考数据 ───────────────────────────────────────────

    /// 获取交易日列表
    virtual std::vector<std::string> get_trade_dates(
        const std::string& start_date,
        const std::string& end_date) = 0;

    /// 获取全市场股票列表
    virtual std::vector<StockInfo> get_stock_list() = 0;

    /// 获取指定日期的可交易股票（排除停牌、ST）
    virtual std::vector<std::string> get_tradable_stocks(
        const std::string& date) = 0;
};

}  // namespace quant
