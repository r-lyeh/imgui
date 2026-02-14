#include <gtest/gtest.h>
extern "C" {
#include "simple_wgsl.h"
}

class ResolverTest : public ::testing::Test {
protected:
    WgslAstNode* ast = nullptr;
    WgslResolver* resolver = nullptr;

    void TearDown() override {
        if (resolver) { wgsl_resolver_free(resolver); resolver = nullptr; }
        if (ast) { wgsl_free_ast(ast); ast = nullptr; }
    }

    void ParseAndResolve(const char* source) {
        ast = wgsl_parse(source);
        ASSERT_NE(ast, nullptr);
        resolver = wgsl_resolver_build(ast);
        ASSERT_NE(resolver, nullptr);
    }
};

TEST_F(ResolverTest, ResolveBindingInfo) {
    ParseAndResolve("@group(0) @binding(5) var tex: texture_2d<f32>;");
    int count = 0;
    const WgslSymbolInfo* bindings = wgsl_resolver_binding_vars(resolver, &count);
    ASSERT_NE(bindings, nullptr);
    EXPECT_EQ(count, 1);
    EXPECT_TRUE(bindings[0].has_group);
    EXPECT_EQ(bindings[0].group_index, 0);
    EXPECT_TRUE(bindings[0].has_binding);
    EXPECT_EQ(bindings[0].binding_index, 5);
    wgsl_resolve_free((void*)bindings);
}

TEST_F(ResolverTest, ResolveEntrypoints) {
    ParseAndResolve(R"(
        @vertex fn vs() -> @builtin(position) vec4f { return vec4f(0.0); }
        @fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }
    )");
    int count = 0;
    const WgslResolverEntrypoint* eps = wgsl_resolver_entrypoints(resolver, &count);
    ASSERT_NE(eps, nullptr);
    EXPECT_EQ(count, 2);
    wgsl_resolve_free((void*)eps);
}
