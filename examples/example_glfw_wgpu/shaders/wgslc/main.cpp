#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include "src/tint/lang/wgsl/reader/reader.h"
#include "src/tint/lang/wgsl/inspector/inspector.h"
#include "src/tint/lang/spirv/writer/writer.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: wgslc <input.wgsl> -o <output.spv> [-e <entry_point>]\n");
        return 1;
    }

    const char* input_path = argv[1];
    const char* output_path = nullptr;
    const char* entry_point = nullptr;
    for (int i = 2; i < argc; i++) {
        if (std::string(argv[i]) == "-o" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (std::string(argv[i]) == "-e" && i + 1 < argc) {
            entry_point = argv[++i];
        }
    }
    if (!output_path) {
        fprintf(stderr, "Error: missing -o <output.spv>\n");
        return 1;
    }

    std::ifstream in(input_path);
    if (!in) {
        fprintf(stderr, "Error: cannot open '%s'\n", input_path);
        return 1;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    std::string source = ss.str();

    tint::Source::File file(input_path, source);
    auto program = tint::wgsl::reader::Parse(&file);
    if (!program.IsValid()) {
        fprintf(stderr, "%s\n", program.Diagnostics().Str().c_str());
        return 1;
    }

    tint::inspector::Inspector inspector(program);
    auto entry_points = inspector.GetEntryPoints();
    if (entry_points.empty()) {
        fprintf(stderr, "Error: no entry points found\n");
        return 1;
    }

    std::string ep_name;
    if (entry_point) {
        ep_name = entry_point;
    } else {
        ep_name = entry_points[0].name;
        if (entry_points.size() > 1) {
            fprintf(stderr, "Warning: multiple entry points found, using '%s'. Use -e to select.\n",
                    ep_name.c_str());
        }
    }

    auto ir = tint::wgsl::reader::ProgramToLoweredIR(program);
    if (ir != tint::Success) {
        fprintf(stderr, "%s\n", ir.Failure().reason.c_str());
        return 1;
    }

    tint::spirv::writer::Options spv_options;
    spv_options.entry_point_name = ep_name;
    auto result = tint::spirv::writer::Generate(ir.Get(), spv_options);
    if (result != tint::Success) {
        fprintf(stderr, "%s\n", result.Failure().reason.c_str());
        return 1;
    }

    auto& spirv = result->spirv;
    FILE* out = fopen(output_path, "wb");
    if (!out) {
        fprintf(stderr, "Error: cannot open '%s' for writing\n", output_path);
        return 1;
    }
    fwrite(spirv.data(), sizeof(uint32_t), spirv.size(), out);
    fclose(out);
    return 0;
}
