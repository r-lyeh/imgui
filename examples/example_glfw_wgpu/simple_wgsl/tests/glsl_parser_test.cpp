#include <gtest/gtest.h>
#include <cstring>
extern "C" {
#include "simple_wgsl.h"
}

class GlslParserTest : public ::testing::Test {
protected:
    WgslAstNode* ast = nullptr;

    void TearDown() override {
        if (ast) { wgsl_free_ast(ast); ast = nullptr; }
    }

    WgslAstNode* Parse(const char* source, WgslStage stage = WGSL_STAGE_UNKNOWN) {
        ast = glsl_parse(source, stage);
        return ast;
    }
};

/* ============================================================================
 * Lexer Tests (via parser - verify tokenization works correctly)
 * ============================================================================ */

TEST_F(GlslParserTest, EmptySource) {
    auto* node = Parse("");
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->type, WGSL_NODE_PROGRAM);
    EXPECT_EQ(node->program.decl_count, 0);
}

TEST_F(GlslParserTest, VersionDirectiveSkipped) {
    auto* node = Parse(R"(
        #version 450
        void main() {}
    )");
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->program.decl_count, 1);
    EXPECT_EQ(node->program.decls[0]->type, WGSL_NODE_FUNCTION);
}

TEST_F(GlslParserTest, ExtensionDirectiveSkipped) {
    auto* node = Parse(R"(
        #version 450
        #extension GL_ARB_separate_shader_objects : enable
        void main() {}
    )");
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->program.decl_count, 1);
}

TEST_F(GlslParserTest, MultiplePreprocessorDirectives) {
    auto* node = Parse(R"(
        #version 450
        #extension GL_EXT_scalar_block_layout : require
        #extension GL_KHR_shader_subgroup_basic : enable
        void main() {}
    )");
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->program.decl_count, 1);
}

/* ============================================================================
 * Struct Tests
 * ============================================================================ */

TEST_F(GlslParserTest, SimpleStruct) {
    auto* node = Parse(R"(
        struct Material {
            vec4 color;
            float roughness;
        };
    )");
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->program.decl_count, 1);
    auto* s = node->program.decls[0];
    EXPECT_EQ(s->type, WGSL_NODE_STRUCT);
    EXPECT_STREQ(s->struct_decl.name, "Material");
    EXPECT_EQ(s->struct_decl.field_count, 2);
    EXPECT_STREQ(s->struct_decl.fields[0]->struct_field.name, "color");
    EXPECT_STREQ(s->struct_decl.fields[0]->struct_field.type->type_node.name, "vec4f");
    EXPECT_STREQ(s->struct_decl.fields[1]->struct_field.name, "roughness");
    EXPECT_STREQ(s->struct_decl.fields[1]->struct_field.type->type_node.name, "f32");
}

TEST_F(GlslParserTest, EmptyStruct) {
    auto* node = Parse("struct Empty {};");
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->program.decl_count, 1);
    EXPECT_EQ(node->program.decls[0]->type, WGSL_NODE_STRUCT);
    EXPECT_STREQ(node->program.decls[0]->struct_decl.name, "Empty");
    EXPECT_EQ(node->program.decls[0]->struct_decl.field_count, 0);
}

TEST_F(GlslParserTest, StructWithArrayField) {
    auto* node = Parse(R"(
        struct Data {
            float values[4];
            int indices[16];
        };
    )");
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->program.decl_count, 1);
    auto* s = node->program.decls[0];
    EXPECT_EQ(s->struct_decl.field_count, 2);
    /* values[4] should be wrapped as array<float, 4> */
    auto* f0type = s->struct_decl.fields[0]->struct_field.type;
    EXPECT_STREQ(f0type->type_node.name, "array");
    EXPECT_EQ(f0type->type_node.type_arg_count, 1);
    EXPECT_STREQ(f0type->type_node.type_args[0]->type_node.name, "f32");
    EXPECT_EQ(f0type->type_node.expr_arg_count, 1);
}

TEST_F(GlslParserTest, StructWithMatrixFields) {
    auto* node = Parse(R"(
        struct Transform {
            mat4 model;
            mat4 view;
            mat4 projection;
        };
    )");
    ASSERT_NE(node, nullptr);
    auto* s = node->program.decls[0];
    EXPECT_EQ(s->struct_decl.field_count, 3);
    EXPECT_STREQ(s->struct_decl.fields[0]->struct_field.type->type_node.name, "mat4x4f");
}

/* ============================================================================
 * Global Variable Tests
 * ============================================================================ */

TEST_F(GlslParserTest, InputVariable) {
    auto* node = Parse("in vec3 position;");
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->program.decl_count, 1);
    auto* g = node->program.decls[0];
    EXPECT_EQ(g->type, WGSL_NODE_GLOBAL_VAR);
    EXPECT_STREQ(g->global_var.name, "position");
    EXPECT_STREQ(g->global_var.address_space, "in");
    EXPECT_STREQ(g->global_var.type->type_node.name, "vec3f");
}

TEST_F(GlslParserTest, OutputVariable) {
    auto* node = Parse("out vec4 fragColor;");
    ASSERT_NE(node, nullptr);
    auto* g = node->program.decls[0];
    EXPECT_EQ(g->type, WGSL_NODE_GLOBAL_VAR);
    EXPECT_STREQ(g->global_var.name, "fragColor");
    EXPECT_STREQ(g->global_var.address_space, "out");
}

TEST_F(GlslParserTest, InputWithLayoutLocation) {
    auto* node = Parse("layout(location = 0) in vec3 pos;");
    ASSERT_NE(node, nullptr);
    auto* g = node->program.decls[0];
    EXPECT_EQ(g->type, WGSL_NODE_GLOBAL_VAR);
    EXPECT_STREQ(g->global_var.name, "pos");
    EXPECT_STREQ(g->global_var.address_space, "in");
    ASSERT_GE(g->global_var.attr_count, 1);
    EXPECT_STREQ(g->global_var.attrs[0]->attribute.name, "location");
    EXPECT_EQ(g->global_var.attrs[0]->attribute.arg_count, 1);
}

TEST_F(GlslParserTest, OutputWithLayoutLocation) {
    auto* node = Parse("layout(location = 0) out vec4 outColor;");
    ASSERT_NE(node, nullptr);
    auto* g = node->program.decls[0];
    EXPECT_STREQ(g->global_var.name, "outColor");
    EXPECT_STREQ(g->global_var.address_space, "out");
    EXPECT_STREQ(g->global_var.attrs[0]->attribute.name, "location");
}

TEST_F(GlslParserTest, UniformVariable) {
    auto* node = Parse("uniform float time;");
    ASSERT_NE(node, nullptr);
    auto* g = node->program.decls[0];
    EXPECT_EQ(g->type, WGSL_NODE_GLOBAL_VAR);
    EXPECT_STREQ(g->global_var.name, "time");
    EXPECT_STREQ(g->global_var.address_space, "uniform");
}

TEST_F(GlslParserTest, LayoutSetBinding) {
    auto* node = Parse("layout(set = 0, binding = 1) uniform sampler2D tex;");
    ASSERT_NE(node, nullptr);
    auto* g = node->program.decls[0];
    EXPECT_EQ(g->type, WGSL_NODE_GLOBAL_VAR);
    /* Should have group and binding attributes */
    int found_group = 0, found_binding = 0;
    for (int i = 0; i < g->global_var.attr_count; i++) {
        if (strcmp(g->global_var.attrs[i]->attribute.name, "group") == 0)
            found_group = 1;
        if (strcmp(g->global_var.attrs[i]->attribute.name, "binding") == 0)
            found_binding = 1;
    }
    EXPECT_TRUE(found_group);
    EXPECT_TRUE(found_binding);
}

TEST_F(GlslParserTest, FlatInterpolation) {
    auto* node = Parse("flat in int instanceId;");
    ASSERT_NE(node, nullptr);
    auto* g = node->program.decls[0];
    EXPECT_EQ(g->type, WGSL_NODE_GLOBAL_VAR);
    EXPECT_STREQ(g->global_var.name, "instanceId");
    EXPECT_STREQ(g->global_var.address_space, "in");
    /* Should have flat interpolation attribute */
    int found_flat = 0;
    for (int i = 0; i < g->global_var.attr_count; i++) {
        if (strcmp(g->global_var.attrs[i]->attribute.name, "flat") == 0)
            found_flat = 1;
    }
    EXPECT_TRUE(found_flat);
}

TEST_F(GlslParserTest, SharedVariable) {
    auto* node = Parse("shared uint counter;");
    ASSERT_NE(node, nullptr);
    auto* g = node->program.decls[0];
    EXPECT_STREQ(g->global_var.address_space, "workgroup");
}

TEST_F(GlslParserTest, BufferVariable) {
    auto* node = Parse(R"(
        struct Data { float values[4]; };
        layout(set = 0, binding = 0) buffer DataBuf {
            float data[1024];
        } buf;
    )");
    ASSERT_NE(node, nullptr);
    /* Should produce: struct Data, struct DataBuf, global var buf */
    ASSERT_GE(node->program.decl_count, 2);
}

TEST_F(GlslParserTest, ConstGlobal) {
    auto* node = Parse("const float PI = 3.14159;");
    ASSERT_NE(node, nullptr);
    auto* v = node->program.decls[0];
    EXPECT_EQ(v->type, WGSL_NODE_VAR_DECL);
    EXPECT_STREQ(v->var_decl.name, "PI");
    ASSERT_NE(v->var_decl.init, nullptr);
}

TEST_F(GlslParserTest, MultipleInputOutputs) {
    auto* node = Parse(R"(
        layout(location = 0) in vec3 inPosition;
        layout(location = 1) in vec2 inTexCoord;
        layout(location = 2) in vec3 inNormal;
        layout(location = 0) out vec4 outColor;
    )");
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->program.decl_count, 4);
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(node->program.decls[i]->type, WGSL_NODE_GLOBAL_VAR);
    }
}

/* ============================================================================
 * Interface Block Tests
 * ============================================================================ */

TEST_F(GlslParserTest, UniformBlock) {
    auto* node = Parse(R"(
        layout(set = 0, binding = 0) uniform UBO {
            mat4 model;
            mat4 view;
            mat4 proj;
        } ubo;
    )");
    ASSERT_NE(node, nullptr);
    /* Should decompose into struct + global var */
    ASSERT_GE(node->program.decl_count, 2);

    /* First should be the struct */
    auto* s = node->program.decls[0];
    EXPECT_EQ(s->type, WGSL_NODE_STRUCT);
    EXPECT_STREQ(s->struct_decl.name, "UBO");
    EXPECT_EQ(s->struct_decl.field_count, 3);

    /* Second should be the global var */
    auto* g = node->program.decls[1];
    EXPECT_EQ(g->type, WGSL_NODE_GLOBAL_VAR);
    EXPECT_STREQ(g->global_var.name, "ubo");
    EXPECT_STREQ(g->global_var.type->type_node.name, "UBO");
    EXPECT_STREQ(g->global_var.address_space, "uniform");
}

TEST_F(GlslParserTest, StorageBlock) {
    auto* node = Parse(R"(
        layout(set = 0, binding = 0) buffer SSBO {
            float data[];
        } ssbo;
    )");
    ASSERT_NE(node, nullptr);
    ASSERT_GE(node->program.decl_count, 2);
    auto* g = node->program.decls[1];
    EXPECT_STREQ(g->global_var.address_space, "storage");
}

TEST_F(GlslParserTest, InterfaceBlockNoInstanceName) {
    auto* node = Parse(R"(
        layout(set = 0, binding = 0) uniform GlobalUBO {
            mat4 mvp;
        };
    )");
    ASSERT_NE(node, nullptr);
    ASSERT_GE(node->program.decl_count, 2);
    /* Instance name should default to block name */
    auto* g = node->program.decls[1];
    EXPECT_EQ(g->type, WGSL_NODE_GLOBAL_VAR);
    EXPECT_STREQ(g->global_var.name, "GlobalUBO");
}

TEST_F(GlslParserTest, PushConstantBlock) {
    auto* node = Parse(R"(
        layout(push_constant) uniform PushConstants {
            mat4 model;
            vec4 color;
        } pc;
    )");
    ASSERT_NE(node, nullptr);
    ASSERT_GE(node->program.decl_count, 2);
    auto* g = node->program.decls[1];
    /* Should have push_constant attribute */
    int found_pc = 0;
    for (int i = 0; i < g->global_var.attr_count; i++) {
        if (strcmp(g->global_var.attrs[i]->attribute.name, "push_constant") == 0)
            found_pc = 1;
    }
    EXPECT_TRUE(found_pc);
}

/* ============================================================================
 * Function Tests
 * ============================================================================ */

TEST_F(GlslParserTest, VoidMainFunction) {
    auto* node = Parse("void main() {}");
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->program.decl_count, 1);
    auto* fn = node->program.decls[0];
    EXPECT_EQ(fn->type, WGSL_NODE_FUNCTION);
    EXPECT_STREQ(fn->function.name, "main");
    ASSERT_NE(fn->function.return_type, nullptr);
    EXPECT_STREQ(fn->function.return_type->type_node.name, "void");
    EXPECT_EQ(fn->function.param_count, 0);
}

TEST_F(GlslParserTest, FunctionWithReturnType) {
    auto* node = Parse("float square(float x) { return x * x; }");
    ASSERT_NE(node, nullptr);
    auto* fn = node->program.decls[0];
    EXPECT_STREQ(fn->function.name, "square");
    EXPECT_STREQ(fn->function.return_type->type_node.name, "f32");
    EXPECT_EQ(fn->function.param_count, 1);
    EXPECT_STREQ(fn->function.params[0]->param.name, "x");
    EXPECT_STREQ(fn->function.params[0]->param.type->type_node.name, "f32");
}

TEST_F(GlslParserTest, FunctionMultipleParams) {
    auto* node = Parse("vec3 lerp(vec3 a, vec3 b, float t) { return a + (b - a) * t; }");
    ASSERT_NE(node, nullptr);
    auto* fn = node->program.decls[0];
    EXPECT_STREQ(fn->function.return_type->type_node.name, "vec3f");
    EXPECT_EQ(fn->function.param_count, 3);
    EXPECT_STREQ(fn->function.params[0]->param.name, "a");
    EXPECT_STREQ(fn->function.params[1]->param.name, "b");
    EXPECT_STREQ(fn->function.params[2]->param.name, "t");
}

TEST_F(GlslParserTest, FunctionOutParam) {
    auto* node = Parse("void getValues(in float x, out float y, inout float z) {}");
    ASSERT_NE(node, nullptr);
    auto* fn = node->program.decls[0];
    EXPECT_EQ(fn->function.param_count, 3);
    /* "out" and "inout" params should have attributes */
    EXPECT_EQ(fn->function.params[0]->param.attr_count, 0); /* in is default */
    EXPECT_GE(fn->function.params[1]->param.attr_count, 1); /* out */
    EXPECT_STREQ(fn->function.params[1]->param.attrs[0]->attribute.name, "out");
    EXPECT_GE(fn->function.params[2]->param.attr_count, 1); /* inout */
    EXPECT_STREQ(fn->function.params[2]->param.attrs[0]->attribute.name, "inout");
}

TEST_F(GlslParserTest, FunctionVoidParamList) {
    auto* node = Parse("void foo(void) {}");
    ASSERT_NE(node, nullptr);
    auto* fn = node->program.decls[0];
    EXPECT_EQ(fn->function.param_count, 0);
}

TEST_F(GlslParserTest, FunctionWithVectorReturn) {
    auto* node = Parse("vec4 getColor() { return vec4(1.0, 0.0, 0.0, 1.0); }");
    ASSERT_NE(node, nullptr);
    auto* fn = node->program.decls[0];
    EXPECT_STREQ(fn->function.return_type->type_node.name, "vec4f");
}

/* ============================================================================
 * Statement Tests
 * ============================================================================ */

TEST_F(GlslParserTest, ReturnStatement) {
    auto* node = Parse("float f() { return 1.0; }");
    ASSERT_NE(node, nullptr);
    auto* body = node->program.decls[0]->function.body;
    ASSERT_EQ(body->block.stmt_count, 1);
    EXPECT_EQ(body->block.stmts[0]->type, WGSL_NODE_RETURN);
    ASSERT_NE(body->block.stmts[0]->return_stmt.expr, nullptr);
}

TEST_F(GlslParserTest, ReturnVoid) {
    auto* node = Parse("void f() { return; }");
    ASSERT_NE(node, nullptr);
    auto* body = node->program.decls[0]->function.body;
    ASSERT_EQ(body->block.stmt_count, 1);
    EXPECT_EQ(body->block.stmts[0]->type, WGSL_NODE_RETURN);
    EXPECT_EQ(body->block.stmts[0]->return_stmt.expr, nullptr);
}

TEST_F(GlslParserTest, LocalVariableDeclaration) {
    auto* node = Parse("void f() { float x = 1.0; }");
    ASSERT_NE(node, nullptr);
    auto* body = node->program.decls[0]->function.body;
    ASSERT_EQ(body->block.stmt_count, 1);
    auto* vd = body->block.stmts[0];
    EXPECT_EQ(vd->type, WGSL_NODE_VAR_DECL);
    EXPECT_STREQ(vd->var_decl.name, "x");
    EXPECT_STREQ(vd->var_decl.type->type_node.name, "f32");
    ASSERT_NE(vd->var_decl.init, nullptr);
}

TEST_F(GlslParserTest, LocalVarNoInit) {
    auto* node = Parse("void f() { int x; }");
    ASSERT_NE(node, nullptr);
    auto* vd = node->program.decls[0]->function.body->block.stmts[0];
    EXPECT_EQ(vd->type, WGSL_NODE_VAR_DECL);
    EXPECT_STREQ(vd->var_decl.name, "x");
    EXPECT_EQ(vd->var_decl.init, nullptr);
}

TEST_F(GlslParserTest, LocalVarArray) {
    auto* node = Parse("void f() { float arr[3]; }");
    ASSERT_NE(node, nullptr);
    auto* vd = node->program.decls[0]->function.body->block.stmts[0];
    EXPECT_EQ(vd->type, WGSL_NODE_VAR_DECL);
    EXPECT_STREQ(vd->var_decl.name, "arr");
    /* Type should be array<float, 3> */
    EXPECT_STREQ(vd->var_decl.type->type_node.name, "array");
}

TEST_F(GlslParserTest, IfStatement) {
    auto* node = Parse("void f() { if (x > 0) { y = 1; } }");
    ASSERT_NE(node, nullptr);
    auto* body = node->program.decls[0]->function.body;
    ASSERT_GE(body->block.stmt_count, 1);
    auto* ifs = body->block.stmts[0];
    EXPECT_EQ(ifs->type, WGSL_NODE_IF);
    ASSERT_NE(ifs->if_stmt.cond, nullptr);
    ASSERT_NE(ifs->if_stmt.then_branch, nullptr);
    EXPECT_EQ(ifs->if_stmt.else_branch, nullptr);
}

TEST_F(GlslParserTest, IfElseStatement) {
    auto* node = Parse("void f() { if (x > 0) { y = 1; } else { y = 0; } }");
    ASSERT_NE(node, nullptr);
    auto* ifs = node->program.decls[0]->function.body->block.stmts[0];
    EXPECT_EQ(ifs->type, WGSL_NODE_IF);
    ASSERT_NE(ifs->if_stmt.else_branch, nullptr);
}

TEST_F(GlslParserTest, IfElseIfStatement) {
    auto* node = Parse(R"(
        void f() {
            if (x > 0) { y = 1; }
            else if (x < 0) { y = -1; }
            else { y = 0; }
        }
    )");
    ASSERT_NE(node, nullptr);
    auto* ifs = node->program.decls[0]->function.body->block.stmts[0];
    EXPECT_EQ(ifs->type, WGSL_NODE_IF);
    ASSERT_NE(ifs->if_stmt.else_branch, nullptr);
    EXPECT_EQ(ifs->if_stmt.else_branch->type, WGSL_NODE_IF);
}

TEST_F(GlslParserTest, IfWithoutBraces) {
    auto* node = Parse("void f() { if (x > 0) y = 1; }");
    ASSERT_NE(node, nullptr);
    auto* ifs = node->program.decls[0]->function.body->block.stmts[0];
    EXPECT_EQ(ifs->type, WGSL_NODE_IF);
    /* then_branch should be wrapped in a block */
    EXPECT_EQ(ifs->if_stmt.then_branch->type, WGSL_NODE_BLOCK);
}

TEST_F(GlslParserTest, WhileLoop) {
    auto* node = Parse("void f() { while (i < 10) { i = i + 1; } }");
    ASSERT_NE(node, nullptr);
    auto* w = node->program.decls[0]->function.body->block.stmts[0];
    EXPECT_EQ(w->type, WGSL_NODE_WHILE);
    ASSERT_NE(w->while_stmt.cond, nullptr);
    ASSERT_NE(w->while_stmt.body, nullptr);
}

TEST_F(GlslParserTest, DoWhileLoop) {
    auto* node = Parse("void f() { do { i = i + 1; } while (i < 10); }");
    ASSERT_NE(node, nullptr);
    auto* dw = node->program.decls[0]->function.body->block.stmts[0];
    EXPECT_EQ(dw->type, WGSL_NODE_DO_WHILE);
    ASSERT_NE(dw->do_while_stmt.body, nullptr);
    ASSERT_NE(dw->do_while_stmt.cond, nullptr);
}

TEST_F(GlslParserTest, ForLoop) {
    auto* node = Parse("void f() { for (int i = 0; i < 10; i++) { x = x + 1; } }");
    ASSERT_NE(node, nullptr);
    auto* fs = node->program.decls[0]->function.body->block.stmts[0];
    EXPECT_EQ(fs->type, WGSL_NODE_FOR);
    ASSERT_NE(fs->for_stmt.init, nullptr);
    ASSERT_NE(fs->for_stmt.cond, nullptr);
    ASSERT_NE(fs->for_stmt.cont, nullptr);
    ASSERT_NE(fs->for_stmt.body, nullptr);
}

TEST_F(GlslParserTest, ForLoopEmpty) {
    auto* node = Parse("void f() { for (;;) { break; } }");
    ASSERT_NE(node, nullptr);
    auto* fs = node->program.decls[0]->function.body->block.stmts[0];
    EXPECT_EQ(fs->type, WGSL_NODE_FOR);
    EXPECT_EQ(fs->for_stmt.init, nullptr);
    EXPECT_EQ(fs->for_stmt.cond, nullptr);
    EXPECT_EQ(fs->for_stmt.cont, nullptr);
}

TEST_F(GlslParserTest, SwitchStatement) {
    auto* node = Parse(R"(
        void f() {
            switch (x) {
                case 0:
                    y = 1;
                    break;
                case 1:
                    y = 2;
                    break;
                default:
                    y = 0;
                    break;
            }
        }
    )");
    ASSERT_NE(node, nullptr);
    auto* sw = node->program.decls[0]->function.body->block.stmts[0];
    EXPECT_EQ(sw->type, WGSL_NODE_SWITCH);
    ASSERT_NE(sw->switch_stmt.expr, nullptr);
    ASSERT_EQ(sw->switch_stmt.case_count, 3);
    /* First case: value 0 */
    EXPECT_NE(sw->switch_stmt.cases[0]->case_clause.expr, nullptr);
    /* Default case: no value */
    EXPECT_EQ(sw->switch_stmt.cases[2]->case_clause.expr, nullptr);
}

TEST_F(GlslParserTest, BreakStatement) {
    auto* node = Parse("void f() { while (true) { break; } }");
    ASSERT_NE(node, nullptr);
    auto* body = node->program.decls[0]->function.body->block.stmts[0];
    EXPECT_EQ(body->type, WGSL_NODE_WHILE);
    auto* inner = body->while_stmt.body;
    ASSERT_NE(inner, nullptr);
    /* Find break in the body */
    bool found_break = false;
    if (inner->type == WGSL_NODE_BLOCK) {
        for (int i = 0; i < inner->block.stmt_count; i++) {
            if (inner->block.stmts[i]->type == WGSL_NODE_BREAK)
                found_break = true;
        }
    } else if (inner->type == WGSL_NODE_BREAK) {
        found_break = true;
    }
    EXPECT_TRUE(found_break);
}

TEST_F(GlslParserTest, ContinueStatement) {
    auto* node = Parse("void f() { for (int i = 0; i < 10; i++) { continue; } }");
    ASSERT_NE(node, nullptr);
    auto* body = node->program.decls[0]->function.body->block.stmts[0]->for_stmt.body;
    ASSERT_NE(body, nullptr);
    bool found_continue = false;
    if (body->type == WGSL_NODE_BLOCK) {
        for (int i = 0; i < body->block.stmt_count; i++) {
            if (body->block.stmts[i]->type == WGSL_NODE_CONTINUE)
                found_continue = true;
        }
    }
    EXPECT_TRUE(found_continue);
}

TEST_F(GlslParserTest, DiscardStatement) {
    auto* node = Parse("void f() { discard; }");
    ASSERT_NE(node, nullptr);
    auto* body = node->program.decls[0]->function.body;
    ASSERT_GE(body->block.stmt_count, 1);
    EXPECT_EQ(body->block.stmts[0]->type, WGSL_NODE_DISCARD);
}

TEST_F(GlslParserTest, NestedBlocks) {
    auto* node = Parse("void f() { { { int x = 1; } } }");
    ASSERT_NE(node, nullptr);
    auto* outer = node->program.decls[0]->function.body;
    EXPECT_EQ(outer->type, WGSL_NODE_BLOCK);
    EXPECT_GE(outer->block.stmt_count, 1);
}

/* ============================================================================
 * Expression Tests
 * ============================================================================ */

TEST_F(GlslParserTest, BinaryAdd) {
    auto* node = Parse("void f() { float x = 1.0 + 2.0; }");
    ASSERT_NE(node, nullptr);
    auto* init = node->program.decls[0]->function.body->block.stmts[0]->var_decl.init;
    ASSERT_NE(init, nullptr);
    EXPECT_EQ(init->type, WGSL_NODE_BINARY);
    EXPECT_STREQ(init->binary.op, "+");
}

TEST_F(GlslParserTest, BinaryModulo) {
    auto* node = Parse("void f() { int x = 10 % 3; }");
    ASSERT_NE(node, nullptr);
    auto* init = node->program.decls[0]->function.body->block.stmts[0]->var_decl.init;
    EXPECT_EQ(init->type, WGSL_NODE_BINARY);
    EXPECT_STREQ(init->binary.op, "%");
}

TEST_F(GlslParserTest, BitwiseOperators) {
    auto* node = Parse("void f() { int x = a & b | c ^ d; }");
    ASSERT_NE(node, nullptr);
    auto* init = node->program.decls[0]->function.body->block.stmts[0]->var_decl.init;
    /* Precedence: & before ^ before | */
    EXPECT_EQ(init->type, WGSL_NODE_BINARY);
    EXPECT_STREQ(init->binary.op, "|");
}

TEST_F(GlslParserTest, PrecedenceChain) {
    auto* node = Parse("void f() { int x = 1 + 2 * 3; }");
    ASSERT_NE(node, nullptr);
    auto* init = node->program.decls[0]->function.body->block.stmts[0]->var_decl.init;
    /* Should parse as 1 + (2 * 3) */
    EXPECT_EQ(init->type, WGSL_NODE_BINARY);
    EXPECT_STREQ(init->binary.op, "+");
    EXPECT_EQ(init->binary.right->type, WGSL_NODE_BINARY);
    EXPECT_STREQ(init->binary.right->binary.op, "*");
}

TEST_F(GlslParserTest, CompoundAssignment) {
    auto* node = Parse("void f() { x += 1; }");
    ASSERT_NE(node, nullptr);
    auto* stmt = node->program.decls[0]->function.body->block.stmts[0];
    EXPECT_EQ(stmt->type, WGSL_NODE_EXPR_STMT);
    auto* assign = stmt->expr_stmt.expr;
    EXPECT_EQ(assign->type, WGSL_NODE_ASSIGN);
    EXPECT_STREQ(assign->assign.op, "+=");
}

TEST_F(GlslParserTest, CompoundAssignmentVariants) {
    auto* node = Parse(R"(
        void f() {
            x -= 1;
            y *= 2;
            z /= 3;
            w %= 4;
            a &= 0xFF;
            b |= 0x01;
            c ^= 0x0F;
        }
    )");
    ASSERT_NE(node, nullptr);
    auto* body = node->program.decls[0]->function.body;
    ASSERT_GE(body->block.stmt_count, 7);

    const char* expected_ops[] = {"-=", "*=", "/=", "%=", "&=", "|=", "^="};
    for (int i = 0; i < 7; i++) {
        auto* assign = body->block.stmts[i]->expr_stmt.expr;
        EXPECT_EQ(assign->type, WGSL_NODE_ASSIGN) << "stmt " << i;
        EXPECT_STREQ(assign->assign.op, expected_ops[i]) << "stmt " << i;
    }
}

TEST_F(GlslParserTest, TernaryExpression) {
    auto* node = Parse("void f() { float x = (a > b) ? a : b; }");
    ASSERT_NE(node, nullptr);
    auto* init = node->program.decls[0]->function.body->block.stmts[0]->var_decl.init;
    EXPECT_EQ(init->type, WGSL_NODE_TERNARY);
}

TEST_F(GlslParserTest, FunctionCall) {
    auto* node = Parse("void f() { float x = sin(1.0); }");
    ASSERT_NE(node, nullptr);
    auto* init = node->program.decls[0]->function.body->block.stmts[0]->var_decl.init;
    EXPECT_EQ(init->type, WGSL_NODE_CALL);
    EXPECT_STREQ(init->call.callee->ident.name, "sin");
    EXPECT_EQ(init->call.arg_count, 1);
}

TEST_F(GlslParserTest, TypeConstructor) {
    auto* node = Parse("void f() { vec3 v = vec3(1.0, 2.0, 3.0); }");
    ASSERT_NE(node, nullptr);
    auto* init = node->program.decls[0]->function.body->block.stmts[0]->var_decl.init;
    EXPECT_EQ(init->type, WGSL_NODE_CALL);
    EXPECT_EQ(init->call.callee->type, WGSL_NODE_TYPE);
    EXPECT_STREQ(init->call.callee->type_node.name, "vec3f");
    EXPECT_EQ(init->call.arg_count, 3);
}

TEST_F(GlslParserTest, MatrixConstructor) {
    auto* node = Parse("void f() { mat4 m = mat4(1.0); }");
    ASSERT_NE(node, nullptr);
    auto* init = node->program.decls[0]->function.body->block.stmts[0]->var_decl.init;
    EXPECT_EQ(init->type, WGSL_NODE_CALL);
    EXPECT_EQ(init->call.callee->type, WGSL_NODE_TYPE);
    EXPECT_STREQ(init->call.callee->type_node.name, "mat4x4f");
}

TEST_F(GlslParserTest, MemberAccess) {
    auto* node = Parse("void f() { float x = v.x; }");
    ASSERT_NE(node, nullptr);
    auto* init = node->program.decls[0]->function.body->block.stmts[0]->var_decl.init;
    EXPECT_EQ(init->type, WGSL_NODE_MEMBER);
    EXPECT_STREQ(init->member.member, "x");
}

TEST_F(GlslParserTest, Swizzle) {
    auto* node = Parse("void f() { vec3 n = v.xyz; }");
    ASSERT_NE(node, nullptr);
    auto* init = node->program.decls[0]->function.body->block.stmts[0]->var_decl.init;
    EXPECT_EQ(init->type, WGSL_NODE_MEMBER);
    EXPECT_STREQ(init->member.member, "xyz");
}

TEST_F(GlslParserTest, ArrayIndexing) {
    auto* node = Parse("void f() { float x = arr[0]; }");
    ASSERT_NE(node, nullptr);
    auto* init = node->program.decls[0]->function.body->block.stmts[0]->var_decl.init;
    EXPECT_EQ(init->type, WGSL_NODE_INDEX);
}

TEST_F(GlslParserTest, ChainedMemberIndex) {
    auto* node = Parse("void f() { float x = ubo.data[0].x; }");
    ASSERT_NE(node, nullptr);
    auto* init = node->program.decls[0]->function.body->block.stmts[0]->var_decl.init;
    EXPECT_EQ(init->type, WGSL_NODE_MEMBER);
    EXPECT_STREQ(init->member.member, "x");
}

TEST_F(GlslParserTest, UnaryNeg) {
    auto* node = Parse("void f() { float x = -1.0; }");
    ASSERT_NE(node, nullptr);
    auto* init = node->program.decls[0]->function.body->block.stmts[0]->var_decl.init;
    EXPECT_EQ(init->type, WGSL_NODE_UNARY);
    EXPECT_STREQ(init->unary.op, "-");
    EXPECT_EQ(init->unary.is_postfix, 0);
}

TEST_F(GlslParserTest, UnaryLogicalNot) {
    auto* node = Parse("void f() { bool b = !flag; }");
    ASSERT_NE(node, nullptr);
    auto* init = node->program.decls[0]->function.body->block.stmts[0]->var_decl.init;
    EXPECT_EQ(init->type, WGSL_NODE_UNARY);
    EXPECT_STREQ(init->unary.op, "!");
}

TEST_F(GlslParserTest, UnaryBitwiseNot) {
    auto* node = Parse("void f() { uint x = ~mask; }");
    ASSERT_NE(node, nullptr);
    auto* init = node->program.decls[0]->function.body->block.stmts[0]->var_decl.init;
    EXPECT_EQ(init->type, WGSL_NODE_UNARY);
    EXPECT_STREQ(init->unary.op, "~");
}

TEST_F(GlslParserTest, PostfixIncrement) {
    auto* node = Parse("void f() { i++; }");
    ASSERT_NE(node, nullptr);
    auto* expr = node->program.decls[0]->function.body->block.stmts[0]->expr_stmt.expr;
    EXPECT_EQ(expr->type, WGSL_NODE_UNARY);
    EXPECT_STREQ(expr->unary.op, "++");
    EXPECT_EQ(expr->unary.is_postfix, 1);
}

TEST_F(GlslParserTest, PrefixDecrement) {
    auto* node = Parse("void f() { --i; }");
    ASSERT_NE(node, nullptr);
    auto* expr = node->program.decls[0]->function.body->block.stmts[0]->expr_stmt.expr;
    EXPECT_EQ(expr->type, WGSL_NODE_UNARY);
    EXPECT_STREQ(expr->unary.op, "--");
    EXPECT_EQ(expr->unary.is_postfix, 0);
}

TEST_F(GlslParserTest, ParenthesizedExpression) {
    auto* node = Parse("void f() { float x = (1.0 + 2.0) * 3.0; }");
    ASSERT_NE(node, nullptr);
    auto* init = node->program.decls[0]->function.body->block.stmts[0]->var_decl.init;
    EXPECT_EQ(init->type, WGSL_NODE_BINARY);
    EXPECT_STREQ(init->binary.op, "*");
    EXPECT_EQ(init->binary.left->type, WGSL_NODE_BINARY);
    EXPECT_STREQ(init->binary.left->binary.op, "+");
}

TEST_F(GlslParserTest, LogicalOperators) {
    auto* node = Parse("void f() { bool b = a && b || c; }");
    ASSERT_NE(node, nullptr);
    auto* init = node->program.decls[0]->function.body->block.stmts[0]->var_decl.init;
    /* || has lower precedence than && */
    EXPECT_EQ(init->type, WGSL_NODE_BINARY);
    EXPECT_STREQ(init->binary.op, "||");
    EXPECT_EQ(init->binary.left->type, WGSL_NODE_BINARY);
    EXPECT_STREQ(init->binary.left->binary.op, "&&");
}

TEST_F(GlslParserTest, ShiftOperators) {
    auto* node = Parse("void f() { uint x = a << 2; }");
    ASSERT_NE(node, nullptr);
    auto* init = node->program.decls[0]->function.body->block.stmts[0]->var_decl.init;
    EXPECT_EQ(init->type, WGSL_NODE_BINARY);
    EXPECT_STREQ(init->binary.op, "<<");
}

TEST_F(GlslParserTest, ComparisonOperators) {
    auto* node = Parse("void f() { bool b = x <= y; }");
    ASSERT_NE(node, nullptr);
    auto* init = node->program.decls[0]->function.body->block.stmts[0]->var_decl.init;
    EXPECT_EQ(init->type, WGSL_NODE_BINARY);
    EXPECT_STREQ(init->binary.op, "<=");
}

TEST_F(GlslParserTest, NestedFunctionCalls) {
    auto* node = Parse("void f() { float x = max(sin(a), cos(b)); }");
    ASSERT_NE(node, nullptr);
    auto* init = node->program.decls[0]->function.body->block.stmts[0]->var_decl.init;
    EXPECT_EQ(init->type, WGSL_NODE_CALL);
    EXPECT_EQ(init->call.arg_count, 2);
    EXPECT_EQ(init->call.args[0]->type, WGSL_NODE_CALL);
    EXPECT_EQ(init->call.args[1]->type, WGSL_NODE_CALL);
}

/* ============================================================================
 * Precision Declaration Tests
 * ============================================================================ */

TEST_F(GlslParserTest, PrecisionDeclaration) {
    auto* node = Parse(R"(
        precision highp float;
        void main() {}
    )");
    ASSERT_NE(node, nullptr);
    /* precision declaration should be skipped */
    EXPECT_EQ(node->program.decl_count, 1);
    EXPECT_EQ(node->program.decls[0]->type, WGSL_NODE_FUNCTION);
}

/* ============================================================================
 * Complete Shader Tests
 * ============================================================================ */

TEST_F(GlslParserTest, SimpleVertexShader) {
    auto* node = Parse(R"(
        #version 450

        layout(location = 0) in vec3 inPosition;
        layout(location = 1) in vec2 inTexCoord;

        layout(location = 0) out vec2 fragTexCoord;

        layout(set = 0, binding = 0) uniform UBO {
            mat4 model;
            mat4 view;
            mat4 proj;
        } ubo;

        void main() {
            gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
            fragTexCoord = inTexCoord;
        }
    )");
    ASSERT_NE(node, nullptr);
    /* inPosition, inTexCoord, fragTexCoord, struct UBO, ubo global, main */
    ASSERT_GE(node->program.decl_count, 5);
}

TEST_F(GlslParserTest, SimpleFragmentShader) {
    auto* node = Parse(R"(
        #version 450

        layout(location = 0) in vec2 fragTexCoord;
        layout(location = 0) out vec4 outColor;

        layout(set = 0, binding = 1) uniform sampler2D texSampler;

        void main() {
            outColor = texture(texSampler, fragTexCoord);
        }
    )");
    ASSERT_NE(node, nullptr);
    ASSERT_GE(node->program.decl_count, 3);
}

TEST_F(GlslParserTest, ComputeShader) {
    auto* node = Parse(R"(
        #version 450

        layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

        layout(set = 0, binding = 0) buffer InputBuf {
            float data[];
        } inputBuf;

        layout(set = 0, binding = 1) buffer OutputBuf {
            float result[];
        } outputBuf;

        void main() {
            uint idx = gl_GlobalInvocationID.x;
            outputBuf.result[idx] = inputBuf.data[idx] * 2.0;
        }
    )");
    ASSERT_NE(node, nullptr);
    /* Verify we got struct + global declarations */
    ASSERT_GE(node->program.decl_count, 4);
}

TEST_F(GlslParserTest, ShaderWithMultipleFunctions) {
    auto* node = Parse(R"(
        #version 450

        layout(location = 0) out vec4 outColor;

        float square(float x) {
            return x * x;
        }

        vec3 computeColor(float t) {
            return vec3(square(t), 0.5, 1.0 - t);
        }

        void main() {
            outColor = vec4(computeColor(0.5), 1.0);
        }
    )");
    ASSERT_NE(node, nullptr);
    /* outColor, square, computeColor, main */
    ASSERT_GE(node->program.decl_count, 4);

    /* Find the square function */
    WgslAstNode* square_fn = nullptr;
    for (int i = 0; i < node->program.decl_count; i++) {
        if (node->program.decls[i]->type == WGSL_NODE_FUNCTION &&
            strcmp(node->program.decls[i]->function.name, "square") == 0) {
            square_fn = node->program.decls[i];
            break;
        }
    }
    ASSERT_NE(square_fn, nullptr);
    EXPECT_STREQ(square_fn->function.return_type->type_node.name, "f32");
    EXPECT_EQ(square_fn->function.param_count, 1);
}

TEST_F(GlslParserTest, ShaderWithControlFlow) {
    auto* node = Parse(R"(
        #version 450

        layout(location = 0) in float value;
        layout(location = 0) out vec4 outColor;

        void main() {
            vec4 color;
            if (value > 0.5) {
                color = vec4(1.0, 0.0, 0.0, 1.0);
            } else if (value > 0.25) {
                color = vec4(0.0, 1.0, 0.0, 1.0);
            } else {
                color = vec4(0.0, 0.0, 1.0, 1.0);
            }

            for (int i = 0; i < 4; i++) {
                color = color * 0.9;
            }

            outColor = color;
        }
    )");
    ASSERT_NE(node, nullptr);
    ASSERT_GE(node->program.decl_count, 3);
}

TEST_F(GlslParserTest, ShaderWithStructUsage) {
    auto* node = Parse(R"(
        #version 450

        struct Light {
            vec3 position;
            vec3 color;
            float intensity;
        };

        layout(set = 0, binding = 0) uniform LightData {
            Light lights[4];
            int numLights;
        } lightData;

        void main() {
            vec3 total = vec3(0.0);
            for (int i = 0; i < lightData.numLights; i++) {
                total = total + lightData.lights[i].color * lightData.lights[i].intensity;
            }
        }
    )");
    ASSERT_NE(node, nullptr);
    /* Light struct, LightData struct, lightData global, main */
    ASSERT_GE(node->program.decl_count, 4);
}

TEST_F(GlslParserTest, WorkgroupSizeAttribute) {
    auto* node = Parse(R"(
        #version 450
        layout(local_size_x = 16, local_size_y = 16) in;
        void main() {}
    )");
    ASSERT_NE(node, nullptr);
    /* The "in" with local_size becomes a global var with workgroup_size attribute */
    /* Find the global var with workgroup_size */
    bool found_ws = false;
    for (int i = 0; i < node->program.decl_count; i++) {
        auto* d = node->program.decls[i];
        if (d->type == WGSL_NODE_GLOBAL_VAR) {
            for (int j = 0; j < d->global_var.attr_count; j++) {
                if (strcmp(d->global_var.attrs[j]->attribute.name, "workgroup_size") == 0) {
                    found_ws = true;
                    EXPECT_EQ(d->global_var.attrs[j]->attribute.arg_count, 3);
                }
            }
        }
    }
    EXPECT_TRUE(found_ws);
}

TEST_F(GlslParserTest, MultipleRenderTargets) {
    auto* node = Parse(R"(
        #version 450

        layout(location = 0) out vec4 outColor;
        layout(location = 1) out vec4 outNormal;
        layout(location = 2) out vec4 outPosition;

        void main() {
            outColor = vec4(1.0);
            outNormal = vec4(0.0, 0.0, 1.0, 0.0);
            outPosition = vec4(0.0);
        }
    )");
    ASSERT_NE(node, nullptr);
    /* 3 outputs + main */
    ASSERT_EQ(node->program.decl_count, 4);
}

TEST_F(GlslParserTest, CommentsPreserved) {
    auto* node = Parse(R"(
        // Single line comment
        /* Multi-line
           comment */
        void main() {
            // inside function
            float x = 1.0; /* inline comment */
        }
    )");
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->program.decl_count, 1);
}

TEST_F(GlslParserTest, HexLiterals) {
    auto* node = Parse("void f() { uint x = 0xFF; }");
    ASSERT_NE(node, nullptr);
    auto* init = node->program.decls[0]->function.body->block.stmts[0]->var_decl.init;
    EXPECT_EQ(init->type, WGSL_NODE_LITERAL);
    EXPECT_STREQ(init->literal.lexeme, "0xFF");
}

TEST_F(GlslParserTest, FloatLiteralSuffix) {
    auto* node = Parse("void f() { float x = 1.0f; }");
    ASSERT_NE(node, nullptr);
    auto* init = node->program.decls[0]->function.body->block.stmts[0]->var_decl.init;
    EXPECT_EQ(init->type, WGSL_NODE_LITERAL);
    EXPECT_TRUE(init->literal.kind == WGSL_LIT_FLOAT);
}

TEST_F(GlslParserTest, BooleanLiterals) {
    auto* node = Parse("void f() { bool a = true; bool b = false; }");
    ASSERT_NE(node, nullptr);
    auto* body = node->program.decls[0]->function.body;
    ASSERT_GE(body->block.stmt_count, 2);
    auto* init1 = body->block.stmts[0]->var_decl.init;
    EXPECT_EQ(init1->type, WGSL_NODE_LITERAL);
    EXPECT_STREQ(init1->literal.lexeme, "true");
    auto* init2 = body->block.stmts[1]->var_decl.init;
    EXPECT_STREQ(init2->literal.lexeme, "false");
}

TEST_F(GlslParserTest, SimpleAssignment) {
    auto* node = Parse("void f() { x = 1; }");
    ASSERT_NE(node, nullptr);
    auto* expr = node->program.decls[0]->function.body->block.stmts[0]->expr_stmt.expr;
    EXPECT_EQ(expr->type, WGSL_NODE_ASSIGN);
    EXPECT_STREQ(expr->assign.op, "=");
}

TEST_F(GlslParserTest, StructVariableDecl) {
    auto* node = Parse(R"(
        struct Vertex { vec3 pos; vec2 uv; };
        void f() { Vertex v; }
    )");
    ASSERT_NE(node, nullptr);
    /* struct Vertex + function */
    ASSERT_GE(node->program.decl_count, 2);
    auto* body = node->program.decls[1]->function.body;
    ASSERT_GE(body->block.stmt_count, 1);
    auto* vd = body->block.stmts[0];
    EXPECT_EQ(vd->type, WGSL_NODE_VAR_DECL);
    EXPECT_STREQ(vd->var_decl.name, "v");
    EXPECT_STREQ(vd->var_decl.type->type_node.name, "Vertex");
}

TEST_F(GlslParserTest, TextureSampling) {
    auto* node = Parse(R"(
        layout(set = 0, binding = 0) uniform sampler2D tex;
        void f() {
            vec4 color = texture(tex, vec2(0.5, 0.5));
        }
    )");
    ASSERT_NE(node, nullptr);
    auto* init = node->program.decls[1]->function.body->block.stmts[0]->var_decl.init;
    EXPECT_EQ(init->type, WGSL_NODE_CALL);
    EXPECT_STREQ(init->call.callee->ident.name, "texture");
    EXPECT_EQ(init->call.arg_count, 2);
    /* Second arg is vec2 constructor */
    EXPECT_EQ(init->call.args[1]->type, WGSL_NODE_CALL);
    EXPECT_EQ(init->call.args[1]->call.callee->type, WGSL_NODE_TYPE);
}

TEST_F(GlslParserTest, ComplexExpression) {
    auto* node = Parse(R"(
        void f() {
            vec3 result = normalize(cross(a - b, c - d)) * length(e);
        }
    )");
    ASSERT_NE(node, nullptr);
    auto* init = node->program.decls[0]->function.body->block.stmts[0]->var_decl.init;
    EXPECT_EQ(init->type, WGSL_NODE_BINARY);
    EXPECT_STREQ(init->binary.op, "*");
}

TEST_F(GlslParserTest, ConstLocalDecl) {
    auto* node = Parse("void f() { const float PI = 3.14159; }");
    ASSERT_NE(node, nullptr);
    auto* vd = node->program.decls[0]->function.body->block.stmts[0];
    EXPECT_EQ(vd->type, WGSL_NODE_VAR_DECL);
    EXPECT_STREQ(vd->var_decl.name, "PI");
    ASSERT_NE(vd->var_decl.init, nullptr);
}

TEST_F(GlslParserTest, EmptyStatement) {
    auto* node = Parse("void f() { ; ; ; }");
    ASSERT_NE(node, nullptr);
    /* Empty statements should be skipped */
    auto* body = node->program.decls[0]->function.body;
    EXPECT_EQ(body->block.stmt_count, 0);
}

TEST_F(GlslParserTest, AllGLSLTypeKeywords) {
    /* Verify various type keywords parse correctly in declarations */
    auto* node = Parse(R"(
        void f() {
            int a;
            uint b;
            float c;
            bool d;
            vec2 e;
            vec3 f1;
            vec4 g;
            ivec2 h;
            ivec3 i1;
            ivec4 j;
            uvec2 k;
            uvec3 l;
            uvec4 m;
            mat2 n;
            mat3 o;
            mat4 p;
            bvec2 q;
            bvec3 r;
            bvec4 s;
        }
    )");
    ASSERT_NE(node, nullptr);
    auto* body = node->program.decls[0]->function.body;
    ASSERT_EQ(body->block.stmt_count, 19);

    const char* expected_types[] = {
        "i32", "u32", "f32", "bool",
        "vec2f", "vec3f", "vec4f",
        "vec2i", "vec3i", "vec4i",
        "vec2u", "vec3u", "vec4u",
        "mat2x2f", "mat3x3f", "mat4x4f",
        "vec2<bool>", "vec3<bool>", "vec4<bool>"
    };
    for (int i = 0; i < 19; i++) {
        EXPECT_STREQ(body->block.stmts[i]->var_decl.type->type_node.name,
                     expected_types[i]) << "type " << i;
    }
}

TEST_F(GlslParserTest, DoWhileNested) {
    auto* node = Parse(R"(
        void f() {
            do {
                if (x > 0) {
                    x = x - 1;
                }
            } while (x > 0);
        }
    )");
    ASSERT_NE(node, nullptr);
    auto* dw = node->program.decls[0]->function.body->block.stmts[0];
    EXPECT_EQ(dw->type, WGSL_NODE_DO_WHILE);
    auto* body = dw->do_while_stmt.body;
    EXPECT_EQ(body->type, WGSL_NODE_BLOCK);
    EXPECT_GE(body->block.stmt_count, 1);
    EXPECT_EQ(body->block.stmts[0]->type, WGSL_NODE_IF);
}

TEST_F(GlslParserTest, SwitchFallthrough) {
    auto* node = Parse(R"(
        void f() {
            switch (mode) {
                case 0:
                case 1:
                    x = 1;
                    break;
                case 2:
                    x = 2;
                    break;
                default:
                    x = 0;
            }
        }
    )");
    ASSERT_NE(node, nullptr);
    auto* sw = node->program.decls[0]->function.body->block.stmts[0];
    EXPECT_EQ(sw->type, WGSL_NODE_SWITCH);
    ASSERT_GE(sw->switch_stmt.case_count, 3);
    /* case 0 should have no statements (fallthrough) */
    EXPECT_EQ(sw->switch_stmt.cases[0]->case_clause.stmt_count, 0);
}
