#include "splg/npy.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace splg {
namespace {

void read_exact(std::ifstream &in, char *dst, std::streamsize n, const std::string &path) {
    in.read(dst, n);
    if (in.gcount() != n) {
        throw std::runtime_error("short read: " + path);
    }
}

std::vector<int64_t> parse_shape(const std::string &header) {
    std::smatch m;
    std::regex re(R"('shape':\s*\(([^)]*)\))");
    if (!std::regex_search(header, m, re)) {
        throw std::runtime_error("npy header missing shape");
    }
    std::vector<int64_t> shape;
    std::stringstream ss(m[1].str());
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        tok.erase(0, tok.find_first_not_of(" \t"));
        tok.erase(tok.find_last_not_of(" \t") + 1);
        if (tok.empty()) {
            continue;
        }
        shape.push_back(std::stoll(tok));
    }
    if (shape.empty()) {
        shape.push_back(1);
    }
    return shape;
}

std::string parse_descr(const std::string &header) {
    std::smatch m;
    std::regex re(R"('descr':\s*'([^']+)')");
    if (!std::regex_search(header, m, re)) {
        throw std::runtime_error("npy header missing descr");
    }
    return m[1].str();
}

bool parse_fortran(const std::string &header) {
    return header.find("'fortran_order': True") != std::string::npos ||
           header.find("'fortran_order': true") != std::string::npos;
}

}  // namespace

NpyArray load_npy(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }
    char magic[6];
    read_exact(in, magic, 6, path);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("not an npy file: " + path);
    }
    uint8_t ver[2];
    read_exact(in, reinterpret_cast<char *>(ver), 2, path);
    uint32_t header_len = 0;
    if (ver[0] == 1) {
        uint16_t hl = 0;
        read_exact(in, reinterpret_cast<char *>(&hl), 2, path);
        header_len = hl;
    } else {
        read_exact(in, reinterpret_cast<char *>(&header_len), 4, path);
    }
    std::string header(header_len, '\0');
    read_exact(in, header.data(), static_cast<std::streamsize>(header_len), path);
    if (parse_fortran(header)) {
        throw std::runtime_error("fortran-order npy not supported: " + path);
    }

    NpyArray arr;
    arr.shape = parse_shape(header);
    const std::string descr = parse_descr(header);
    const int64_t n = arr.numel();
    if (descr == "<f4" || descr == "|f4") {
        arr.f32.resize(static_cast<size_t>(n));
        read_exact(in, reinterpret_cast<char *>(arr.f32.data()),
                   static_cast<std::streamsize>(n * 4), path);
    } else if (descr == "<i8" || descr == "|i8") {
        arr.is_int64 = true;
        arr.i64.resize(static_cast<size_t>(n));
        read_exact(in, reinterpret_cast<char *>(arr.i64.data()),
                   static_cast<std::streamsize>(n * 8), path);
    } else if (descr == "<i4" || descr == "|i4") {
        arr.is_int64 = true;
        std::vector<int32_t> tmp(static_cast<size_t>(n));
        read_exact(in, reinterpret_cast<char *>(tmp.data()),
                   static_cast<std::streamsize>(n * 4), path);
        arr.i64.assign(tmp.begin(), tmp.end());
    } else {
        throw std::runtime_error("unsupported npy dtype " + descr + " in " + path);
    }
    return arr;
}

namespace {

void write_npy_header(std::ofstream &out, const std::string &descr,
                      const std::vector<int64_t> &shape) {
    std::ostringstream dict;
    dict << "{'descr': '" << descr << "', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) {
        dict << shape[i];
        if (i + 1 < shape.size() || shape.size() == 1) {
            dict << ", ";
        }
    }
    dict << "), }";
    std::string d = dict.str();
    const size_t preamble = 10;  // magic(6)+ver(2)+hlen(2)
    size_t pad = 16 - ((preamble + d.size() + 1) % 16);
    if (pad == 16) {
        pad = 0;
    }
    d.append(pad, ' ');
    d.push_back('\n');
    const uint16_t hlen = static_cast<uint16_t>(d.size());
    out.write("\x93NUMPY", 6);
    const char ver[2] = {1, 0};
    out.write(ver, 2);
    out.write(reinterpret_cast<const char *>(&hlen), 2);
    out.write(d.data(), static_cast<std::streamsize>(d.size()));
}

}  // namespace

void save_npy_f32(const std::string &path, const std::vector<int64_t> &shape, const float *data) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("cannot write " + path);
    }
    write_npy_header(out, "<f4", shape);
    int64_t n = 1;
    for (int64_t d : shape) {
        n *= d;
    }
    if (n > 0 && data != nullptr) {
        out.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(n * 4));
    }
}

void save_npy_i64(const std::string &path, const std::vector<int64_t> &shape, const int64_t *data) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("cannot write " + path);
    }
    write_npy_header(out, "<i8", shape);
    int64_t n = 1;
    for (int64_t d : shape) {
        n *= d;
    }
    if (n > 0 && data != nullptr) {
        out.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(n * 8));
    }
}

}  // namespace splg
