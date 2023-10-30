#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GANGMUK_PARSER_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GANGMUK_PARSER_H_

#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <filesystem>

namespace tensorflow {

class GM_Parser {
  public:
    const char* file_path_ptr = std::getenv("ZICO_CONFIGURATION_PATH");
    assert(file_path_ptr && "ZICO_CONFIGURATION_PATH must be exported");
    std::string file_path(file_path_ptr);
    std::string delimiter = "=";
    int parse_target_config(std::string target);
    float parse_float_config(std::string target);
    void file_write(std::string text);
    std::string parse_string_config(std::string target);

    // std::vector<int> parse_list(std::string target);

};

}

#endif