#include <gtest/gtest.h>
#include "test_utils.h"

TEST(IntegrationTest, BuildLoweringEntrypoints) {
    const char* source = R"(
        @vertex fn vs() -> @builtin(position) vec4f { return vec4f(0.0); }
        @fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }
    )";

    wgsl_test::AstGuard ast(wgsl_parse(source));
    ASSERT_NE(ast.get(), nullptr);

    wgsl_test::ResolverGuard resolver(wgsl_resolver_build(ast.get()));
    ASSERT_NE(resolver.get(), nullptr);

    WgslLowerOptions opts = {};
    opts.env = WGSL_LOWER_ENV_VULKAN_1_3;

    WgslLower* lower = wgsl_lower_create(ast.get(), resolver.get(), &opts);
    ASSERT_NE(lower, nullptr);

    int count = 0;
    const WgslLowerEntrypointInfo* eps = wgsl_lower_entrypoints(lower, &count);
    ASSERT_NE(eps, nullptr);
    EXPECT_EQ(count, 2);
    EXPECT_NE(eps[0].function_id, 0u);
    EXPECT_NE(eps[1].function_id, 0u);

    wgsl_lower_destroy(lower);
}
