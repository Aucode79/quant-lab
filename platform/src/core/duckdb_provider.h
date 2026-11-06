#pragma once
/**
 * DuckDB 数据提供者实现
 *
 * 从 DuckDB 管理的 Parquet 文件中读取市场数据。
 * 支持:
 *   - 直接查询分区 Parquet 文件
 *   - Arrow 格式零拷贝导出
 *   - SQL 窗口函数预计算指标
 */

#include <memory>
#include <string>

#include <duckdb.hpp>  // NOLINT

#include "core/data_provider.h"

namespace quant {

class DuckDBProvider : public IDataProvider {
public:
    /// 构造函数
    /// @param db_path DuckDB 数据库文件路径（可指向空目录，数据从Parquet加载）
    /// @param data_dir 数据根目录（包含 market/, reference/ 子目录）
    explicit DuckDBProvider(const std::string& db_path,
                            const std::string& data_dir);

    ~DuckDBProvider() override;

    // 禁用拷贝
    DuckDBProvider(const DuckDBProvider&) = delete;
    DuckDBProvider& operator=(const DuckDBProvider&) = delete;

    // ── IDataProvider 接口实现 ─────────────────────────────

    std::shared_ptr<arrow::Table> get_daily_kline(
        const std::string& ts_code,
        const std::string& start_date,
        const std::string& end_date) override;

    std::shared_ptr<arrow::Table> get_multi_daily_kline(
        const std::vector<std::string>& ts_codes,
        const std::string& start_date,
        const std::string& end_date) override;

    std::shared_ptr<arrow::Table> get_universe_snapshot(
        const std::string& date,
        double min_volume = 0.0) override;

    std::vector<std::string> get_trade_dates(
        const std::string& start_date,
        const std::string& end_date) override;

    std::vector<StockInfo> get_stock_list() override;

    std::vector<std::string> get_tradable_stocks(
        const std::string& date) override;

private:
    /// 初始化 DuckDB 连接并注册 Parquet 视图
    void init_database();

    /// 确保 Parquet 视图已注册
    void ensure_views();

    std::unique_ptr<duckdb::DuckDB> db_;
    std::unique_ptr<duckdb::Connection> conn_;
    std::string data_dir_;
    bool views_initialized_ = false;
};

}  // namespace quant
