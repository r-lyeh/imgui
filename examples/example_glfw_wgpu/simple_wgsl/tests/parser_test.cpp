#include <gtest/gtest.h>
extern "C" {
#include "simple_wgsl.h"
}

class ParserTest : public ::testing::Test {
protected:
    WgslAstNode* ast = nullptr;

    void TearDown() override {
        if (ast) { wgsl_free_ast(ast); ast = nullptr; }
    }

    WgslAstNode* Parse(const char* source) {
        ast = wgsl_parse(source);
        return ast;
    }
};

TEST_F(ParserTest, ParseStructWithLargeArray) {
    auto* node = Parse(R"(
        struct Struct {
            Dummy : array<vec2<u32>, 32768u>,
        };

        @group(0) @binding(0) var<storage, read_write> binding : Struct;")
        }
        )"
    );
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->type, WGSL_NODE_PROGRAM);
    ASSERT_EQ(node->program.decl_count, 2);
    EXPECT_EQ(node->program.decls[0]->type, WGSL_NODE_STRUCT);
    EXPECT_STREQ(node->program.decls[0]->struct_decl.name, "Struct");
}

TEST_F(ParserTest, ParseEmptyStruct) {
    auto* node = Parse("struct Empty {};");
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->type, WGSL_NODE_PROGRAM);
    ASSERT_EQ(node->program.decl_count, 1);
    EXPECT_EQ(node->program.decls[0]->type, WGSL_NODE_STRUCT);
    EXPECT_STREQ(node->program.decls[0]->struct_decl.name, "Empty");
}

TEST_F(ParserTest, ParseSimpleFunction) {
    auto* node = Parse("fn foo() {}");
    ASSERT_NE(node, nullptr);
    auto* fn = node->program.decls[0];
    ASSERT_EQ(fn->type, WGSL_NODE_FUNCTION);
    EXPECT_STREQ(fn->function.name, "foo");
}

TEST_F(ParserTest, ParseBinaryExpressions) {
    auto* node = Parse("fn f() { var x = 1 + 2 * 3; }");
    ASSERT_NE(node, nullptr);
    auto* fn = node->program.decls[0];
    auto* block = fn->function.body;
    ASSERT_EQ(block->type, WGSL_NODE_BLOCK);
    auto* var_decl = block->block.stmts[0];
    ASSERT_EQ(var_decl->type, WGSL_NODE_VAR_DECL);
    auto* init = var_decl->var_decl.init;
    ASSERT_EQ(init->type, WGSL_NODE_BINARY);
    EXPECT_STREQ(init->binary.op, "+");
}
