/**
 * DuckDB 数据提供者实现
 */

#include "core/duckdb_provider.h"

#include <algorithm>
#include <sstream>

#include <fmt/format.h>

namespace quant {

DuckDBProvider::DuckDBProvider(const std::string& db_path,
                               const std::string& data_dir)
    : data_dir_(data_dir) {
    duckdb::DBConfig config;
    // 增大内存限制，允许 DuckDB 充分利用系统内存
    config.options.maximum_memory = 8ULL * 1024 * 1024 * 1024;  // 8 GB
    // 启用多线程查询
    config.options.maximum_threads = std::thread::hardware_concurrency();

    db_ = std::make_unique<duckdb::DuckDB>(db_path, &config);
    conn_ = std::make_unique<duckdb::Connection>(*db_);
}

DuckDBProvider::~DuckDBProvider() = default;

void DuckDBProvider::init_database() {
    ensure_views();
}

void DuckDBProvider::ensure_views() {
    if (views_initialized_) return;

    // 注册 Parquet 文件为视图，方便 SQL 查询
    // 使用 glob 模式匹配分区 Parquet 文件

    try {
        // 日K线视图
        auto kline_path = data_dir_ + "/market/kline_daily/**/*.parquet";
        auto kline_sql = fmt::format(
            "CREATE OR REPLACE VIEW kline_daily AS "
            "SELECT * FROM '{}'",
            kline_path
        );
        conn_->Query(kline_sql);

        // 股票列表视图
        auto stock_path = data_dir_ + "/reference/stock_basic.parquet";
        auto stock_sql = fmt::format(
            "CREATE OR REPLACE VIEW IF NOT EXISTS stock_basic AS "
            "SELECT * FROM '{}'",
            stock_path
        );
        conn_->Query(stock_sql);

        views_initialized_ = true;
    } catch (const std::exception& e) {
        // 视图注册失败可能是文件不存在，后续查询时会报错
        // 这里不阻止程序启动
    }
}

// ── IDataProvider 实现 ─────────────────────────────────────

std::shared_ptr<arrow::Table> DuckDBProvider::get_daily_kline(
    const std::string& ts_code,
    const std::string& start_date,
    const std::string& end_date) {

    ensure_views();

    auto sql = fmt::format(
        "SELECT trade_date, open, high, low, close, pre_close, "
        "       volume, amount, adj_factor, is_st, is_suspend "
        "FROM kline_daily "
        "WHERE ts_code = '{}' "
        "  AND trade_date >= '{}' "
        "  AND trade_date <= '{}' "
        "ORDER BY trade_date",
        ts_code, start_date, end_date
    );

    auto result = conn_->Query(sql);
    if (result->HasError()) {
        throw std::runtime_error(
            fmt::format("DuckDB query failed: {}", result->GetError()));
    }

    // 将 DuckDB 结果转换为 Arrow Table
    return result->Fetch();
}

std::shared_ptr<arrow::Table> DuckDBProvider::get_multi_daily_kline(
    const std::vector<std::string>& ts_codes,
    const std::string& start_date,
    const std::string& end_date) {

    ensure_views();

    // 构建 IN 子句
    std::string code_list;
    for (size_t i = 0; i < ts_codes.size(); ++i) {
        if (i > 0) code_list += ", ";
        code_list += fmt::format("'{}'", ts_codes[i]);
    }

    auto sql = fmt::format(
        "SELECT ts_code, trade_date, open, high, low, close, pre_close, "
        "       volume, amount, adj_factor, is_st, is_suspend "
        "FROM kline_daily "
        "WHERE ts_code IN ({}) "
        "  AND trade_date >= '{}' "
        "  AND trade_date <= '{}' "
        "ORDER BY ts_code, trade_date",
        code_list, start_date, end_date
    );

    auto result = conn_->Query(sql);
    if (result->HasError()) {
        throw std::runtime_error(
            fmt::format("DuckDB query failed: {}", result->GetError()));
    }

    return result->Fetch();
}

std::shared_ptr<arrow::Table> DuckDBProvider::get_universe_snapshot(
    const std::string& date,
    double min_volume) {

    ensure_views();

    auto sql = fmt::format(
        "SELECT ts_code, trade_date, open, high, low, close, "
        "       pre_close, volume, amount, adj_factor, is_st, is_suspend "
        "FROM kline_daily "
        "WHERE trade_date = '{}' "
        "  AND volume >= {} "
        "  AND is_suspend = false "
        "ORDER BY ts_code",
        date, min_volume
    );

    auto result = conn_->Query(sql);
    if (result->HasError()) {
        throw std::runtime_error(
            fmt::format("DuckDB query failed: {}", result->GetError()));
    }

    return result->Fetch();
}

std::vector<std::string> DuckDBProvider::get_trade_dates(
    const std::string& start_date,
    const std::string& end_date) {

    ensure_views();

    auto sql = fmt::format(
        "SELECT DISTINCT trade_date "
        "FROM kline_daily "
        "WHERE trade_date >= '{}' AND trade_date <= '{}' "
        "ORDER BY trade_date",
        start_date, end_date
    );

    auto result = conn_->Query(sql);
    if (result->HasError()) {
        throw std::runtime_error(
            fmt::format("DuckDB query failed: {}", result->GetError()));
    }

    // DuckDB 的 MaterializedQueryResult -> Arrow Table -> vector<string>
    auto table = result->Fetch();
    if (!table || table->num_rows() == 0) return {};

    auto date_col = std::static_pointer_cast<arrow::Date32Array>(
        table->column(0)->chunk(0));

    std::vector<std::string> dates;
    dates.reserve(date_col->length());
    for (int64_t i = 0; i < date_col->length(); ++i) {
        if (!date_col->IsNull(i)) {
            // Date32 是 days since epoch
            int32_t days = date_col->Value(i);
            // 转换为 YYYY-MM-DD 格式
            auto millis = days * 86400000LL;
            // 简化处理：直接从 Arrow 的 Date32 转字符串
            // 实际使用中建议用 Arrow compute 函数
            dates.push_back(date_col->GetString(i));
        }
    }

    return dates;
}

std::vector<StockInfo> DuckDBProvider::get_stock_list() {
    ensure_views();

    auto sql = std::string(
        "SELECT ts_code, name, industry, list_date, delist_date, exchange "
        "FROM stock_basic ORDER BY ts_code"
    );

    auto result = conn_->Query(sql);
    if (result->HasError()) {
        // 视图可能不存在，返回空列表
        return {};
    }

    auto table = result->Fetch();
    if (!table || table->num_rows() == 0) return {};

    std::vector<StockInfo> stocks;
    stocks.reserve(table->num_rows());

    // 遍历 Arrow Table 的各列
    for (int64_t i = 0; i < table->num_rows(); ++i) {
        StockInfo info;
        // 使用 chunked_array 获取值
        auto code_arr = std::static_pointer_cast<arrow::StringArray>(
            table->column(0)->chunk(0));
        auto name_arr = std::static_pointer_cast<arrow::StringArray>(
            table->column(1)->chunk(0));
        auto ind_arr = std::static_pointer_cast<arrow::StringArray>(
            table->column(2)->chunk(0));
        auto exch_arr = std::static_pointer_cast<arrow::StringArray>(
            table->column(5)->chunk(0));

        info.ts_code = code_arr->GetString(i);
        info.name = name_arr->GetString(i);
        info.industry = ind_arr->GetString(i);
        info.exchange = exch_arr->GetString(i);

        // 简化日期处理
        info.list_date = "";
        info.delist_date = "";

        stocks.push_back(std::move(info));
    }

    return stocks;
}

std::vector<std::string> DuckDBProvider::get_tradable_stocks(
    const std::string& date) {

    auto snapshot = get_universe_snapshot(date, 0.0);
    if (!snapshot || snapshot->num_rows() == 0) return {};

    // 过滤 ST 和停牌股票
    auto code_arr = std::static_pointer_cast<arrow::StringArray>(
        snapshot->column(0)->chunk(0));
    auto st_arr = std::static_pointer_cast<arrow::BooleanArray>(
        snapshot->column(9)->chunk(0));
    auto sus_arr = std::static_pointer_cast<arrow::BooleanArray>(
        snapshot->column(10)->chunk(0));

    std::vector<std::string> tradable;
    for (int64_t i = 0; i < snapshot->num_rows(); ++i) {
        if (!st_arr->Value(i) && !sus_arr->Value(i)) {
            tradable.push_back(code_arr->GetString(i));
        }
    }

    return tradable;
}

}  // namespace quant
