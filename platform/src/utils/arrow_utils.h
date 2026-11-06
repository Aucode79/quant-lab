#pragma once
/**
 * Arrow ↔ Eigen 数据转换工具
 *
 * 核心转换:
 *   Arrow Table Column → Eigen::VectorXd (单列)
 *   Arrow Table → Eigen::MatrixXd (多列)
 *
 * 零拷贝策略:
 *   Arrow 的列式内存布局与 Eigen 的列优先布局天然匹配，
 *   在满足条件时可使用 Map 实现零拷贝。
 */

#include <Eigen/Core>
#include <arrow/api.h>  // NOLINT
#include <arrow/array/array_primitive.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace quant::util {

/// 将 Arrow ChunkedArray (Double) 提取为 Eigen::VectorXd
/// 注意：这是拷贝操作（Arrow 内部可能跨 chunk）
inline Eigen::VectorXd arrow_to_eigen_vector(
    const std::shared_ptr<arrow::ChunkedArray>& column) {

    if (!column) {
        return {};
    }

    Eigen::VectorXd result(column->length());

    // 逐 chunk 填充
    int64_t offset = 0;
    for (int i = 0; i < column->num_chunks(); ++i) {
        auto chunk = std::static_pointer_cast<arrow::DoubleArray>(
            column->chunk(i));
        int64_t chunk_len = chunk->length();

        const double* raw = chunk->raw_values();
        for (int64_t j = 0; j < chunk_len; ++j) {
            result(offset + j) = raw[j];
        }
        offset += chunk_len;
    }

    return result;
}

/// 将 Arrow Table 中某列提取为 Eigen::VectorXd
inline Eigen::VectorXd table_column_to_eigen(
    const arrow::Table& table,
    const std::string& col_name) {

    int idx = table.schema()->GetFieldIndex(col_name);
    if (idx < 0) {
        throw std::runtime_error("Column not found: " + col_name);
    }

    return arrow_to_eigen_vector(table.column(idx));
}

/// 将 Arrow Table 多列提取为 Eigen::MatrixXd
/// @param table Arrow 表
/// @param col_names 列名列表
/// @return [rows × cols] 矩阵，每列对应一个 col_name
inline Eigen::MatrixXd table_to_eigen_matrix(
    const arrow::Table& table,
    const std::vector<std::string>& col_names) {

    Eigen::MatrixXd result(table.num_rows(), col_names.size());

    for (size_t c = 0; c < col_names.size(); ++c) {
        int idx = table.schema()->GetFieldIndex(col_names[c]);
        if (idx < 0) {
            throw std::runtime_error("Column not found: " + col_names[c]);
        }

        auto vec = arrow_to_eigen_vector(table.column(idx));
        result.col(c) = vec;
    }

    return result;
}

/// Eigen::VectorXd → Arrow Array (拷贝)
inline std::shared_ptr<arrow::Array> eigen_to_arrow(
    const Eigen::VectorXd& vec) {

    arrow::DoubleBuilder builder;
    // 或许可以在某些场景下映射优化—但目前先做安全拷贝
    for (int i = 0; i < vec.size(); ++i) {
        if (std::isnan(vec(i))) {
            builder.AppendNull();
        } else {
            builder.Append(vec(i));
        }
    }

    auto result = builder.Finish();
    if (!result.ok()) {
        throw std::runtime_error(result.status().ToString());
    }
    return result.ValueOrDie();
}

}  // namespace quant::util
