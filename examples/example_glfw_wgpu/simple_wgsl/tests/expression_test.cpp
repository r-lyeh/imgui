#include <gtest/gtest.h>
#include <algorithm>
#include <filesystem>
#include <sstream>
#include <vector>
#include <string>
#include <cstdio>
#include "test_utils.h"

namespace fs = std::filesystem;

// Discover all .wgsl files in the expressions directory
std::vector<std::string> DiscoverExpressionTests() {
    std::vector<std::string> tests;

    // Find expressions directory - use compile-time path from CMake
    fs::path expressions_dir;

#ifdef WGSL_SOURCE_DIR
    expressions_dir = fs::path(WGSL_SOURCE_DIR) / "expressions";
#else
    // Fall back to relative paths from common build locations
    for (const auto& candidate : {
        "../expressions",
        "../../expressions",
        "../../../expressions",
        "expressions"
    }) {
        if (fs::exists(candidate)) {
            expressions_dir = candidate;
            break;
        }
    }
#endif

    if (!fs::exists(expressions_dir)) {
        return tests;
    }

    for (const auto& entry : fs::recursive_directory_iterator(expressions_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".wgsl") {
            // Check if expected file exists
            auto expected_path = entry.path().string() + ".expected.spvasm";
            if (fs::exists(expected_path)) {
                tests.push_back(entry.path().string());
            }
        }
    }

    std::sort(tests.begin(), tests.end());
    return tests;
}

// Disassemble SPIR-V binary to text using spirv-dis
std::string DisassembleSpirv(const uint32_t* words, size_t word_count, std::string* out_error = nullptr) {
    std::string spv_path = wgsl_test::MakeTempSpvPath("wgsl_dis");
    if (!wgsl_test::WriteSpirvFile(spv_path, words, word_count)) {
        if (out_error) *out_error = "Failed to write temp SPIR-V file";
        return "";
    }

    std::string output;
    int ret = wgsl_test::RunCommand("spirv-dis " + spv_path + " 2>&1", &output);
    std::remove(spv_path.c_str());

    if (ret != 0) {
        if (out_error) *out_error = "spirv-dis failed: " + output;
        return "";
    }

    return output;
}

// Normalize SPIR-V assembly for comparison
// Removes generator-specific details and normalizes whitespace
std::string NormalizeSpvasm(const std::string& spvasm) {
    std::istringstream iss(spvasm);
    std::ostringstream oss;
    std::string line;

    while (std::getline(iss, line)) {
        // Skip generator line as it will differ between compilers
        if (line.find("; Generator:") != std::string::npos) {
            continue;
        }
        // Skip bound line as IDs may differ
        if (line.find("; Bound:") != std::string::npos) {
            continue;
        }
        // Normalize whitespace: trim trailing, collapse internal
        // Keep the line as-is otherwise for now
        while (!line.empty() && (line.back() == ' ' || line.back() == '\t')) {
            line.pop_back();
        }
        if (!line.empty()) {
            oss << line << "\n";
        }
    }

    return oss.str();
}

class ExpressionTest : public testing::TestWithParam<std::string> {
protected:
    void SetUp() override {
        wgsl_path_ = GetParam();
        expected_path_ = wgsl_path_ + ".expected.spvasm";
    }

    std::string wgsl_path_;
    std::string expected_path_;
};

TEST_P(ExpressionTest, CompilesAndMatchesExpected) {
    // Read WGSL source
    std::string wgsl_source = wgsl_test::ReadFile(wgsl_path_);
    ASSERT_FALSE(wgsl_source.empty()) << "Failed to read: " << wgsl_path_;

    // Compile WGSL to SPIR-V
    auto result = wgsl_test::CompileWgsl(wgsl_source.c_str());
    ASSERT_TRUE(result.success) << "Compile failed for " << wgsl_path_ << ": " << result.error;

    // Disassemble to text
    std::string dis_error;
    std::string actual_spvasm = DisassembleSpirv(result.spirv.data(), result.spirv.size(), &dis_error);
    ASSERT_FALSE(actual_spvasm.empty()) << "Disassembly failed for " << wgsl_path_ << ": " << dis_error;

    // Read expected output
    std::string expected_spvasm = wgsl_test::ReadFile(expected_path_);
    ASSERT_FALSE(expected_spvasm.empty()) << "Failed to read expected: " << expected_path_;

    // Normalize both for comparison
    std::string actual_normalized = NormalizeSpvasm(actual_spvasm);
    std::string expected_normalized = NormalizeSpvasm(expected_spvasm);

    EXPECT_EQ(actual_normalized, expected_normalized)
        << "Output mismatch for " << wgsl_path_ << "\n"
        << "=== Expected ===\n" << expected_normalized << "\n"
        << "=== Actual ===\n" << actual_normalized;
}

// Generate human-readable test names from file paths
std::string TestNameGenerator(const testing::TestParamInfo<std::string>& info) {
    std::string name = info.param;

    // Extract relative path from expressions/
    auto pos = name.find("expressions/");
    if (pos != std::string::npos) {
        name = name.substr(pos + 12);  // Skip "expressions/"
    }

    // Remove .wgsl extension
    if (name.size() > 5 && name.substr(name.size() - 5) == ".wgsl") {
        name = name.substr(0, name.size() - 5);
    }

    // Replace invalid characters with underscores
    for (char& c : name) {
        if (!std::isalnum(c)) {
            c = '_';
        }
    }

    // Remove consecutive underscores
    std::string result;
    for (char c : name) {
        if (c != '_' || result.empty() || result.back() != '_') {
            result += c;
        }
    }

    // Remove leading/trailing underscores
    while (!result.empty() && result.front() == '_') result.erase(0, 1);
    while (!result.empty() && result.back() == '_') result.pop_back();

    return result.empty() ? "test" : result;
}

INSTANTIATE_TEST_SUITE_P(
    Expressions,
    ExpressionTest,
    testing::ValuesIn(DiscoverExpressionTests()),
    TestNameGenerator
);
