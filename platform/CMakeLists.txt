cmake_minimum_required(VERSION 3.20)
project(quant-system
    VERSION 0.1.0
    DESCRIPTION "Local C++ quantitative trading system"
    LANGUAGES CXX
)

# C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Dependencies via vcpkg ─────────────────────────────────
find_package(DuckDB CONFIG REQUIRED)
find_package(Arrow CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(CLI11 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

# ── OpenMP ─────────────────────────────────────────────────
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP enabled")
else()
    message(WARNING "OpenMP not found - parallel optimization disabled")
endif()

# ── SQLite3 (DuckDB dependency, but we might use it directly) ─
find_package(SQLite3 QUIET)

# ── Library: quant-core ────────────────────────────────────
set(CORE_SOURCES
    src/core/duckdb_provider.cpp
    src/core/backtest_engine.cpp
    src/core/portfolio.cpp
    src/core/performance_eval.cpp
    src/strategy/ma_cross_strategy.cpp
    src/strategy/mean_reversion_strategy.cpp
    src/optimizer/grid_search.cpp
)

set(CORE_HEADERS
    src/core/data_provider.h
    src/core/duckdb_provider.h
    src/core/backtest_engine.h
    src/core/portfolio.h
    src/core/broker_model.h
    src/core/performance_eval.h
    src/strategy/strategy_base.h
    src/strategy/ma_cross_strategy.h
    src/strategy/mean_reversion_strategy.h
    src/optimizer/param_optimizer.h
    src/utils/date_utils.h
    src/utils/logger.h
    src/utils/arrow_utils.h
    src/indicators/ma.h
    src/indicators/rsi.h
    src/indicators/bollinger.h
    src/indicators/cross_sectional.h
)

add_library(quant-core STATIC ${CORE_SOURCES} ${CORE_HEADERS})
target_include_directories(quant-core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(quant-core PUBLIC
    DuckDB::duckdb
    Arrow::arrow
    Eigen3::Eigen
    fmt::fmt
    nlohmann_json::nlohmann_json
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(quant-core PUBLIC OpenMP::OpenMP_CXX)
endif()

# ── Library: quant-indicators (header-only style) ──────────
add_library(quant-indicators INTERFACE)
target_include_directories(quant-indicators INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(quant-indicators INTERFACE Eigen3::Eigen)

# ── Apps ───────────────────────────────────────────────────
add_executable(backtest_main apps/backtest_main.cpp)
target_link_libraries(backtest_main PRIVATE quant-core quant-indicators CLI11::CLI11)

add_executable(grid_search_main apps/grid_search_main.cpp)
target_link_libraries(grid_search_main PRIVATE quant-core quant-indicators CLI11::CLI11)

# ── Tests ──────────────────────────────────────────────────
enable_testing()
add_executable(test_indicators tests/test_indicators.cpp)
target_link_libraries(test_indicators PRIVATE quant-core quant-indicators)
add_test(NAME Indicators COMMAND test_indicators)

# ── Benchmark (optional) ───────────────────────────────────
find_package(benchmark QUIET)
if(benchmark_FOUND)
    add_executable(bench_backtest tests/bench_backtest.cpp)
    target_link_libraries(bench_backtest PRIVATE quant-core quant-indicators benchmark::benchmark)
endif()
