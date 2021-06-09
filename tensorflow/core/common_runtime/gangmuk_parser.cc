#include "tensorflow/core/common_runtime/gangmuk_parser.h"
// #include "gangmuk_parser.h"

namespace tensorflow {

int GM_Parser::parse_target_config(std::string target) {
  std::ifstream openFile(file_path.data());
  if (openFile.is_open()) {
    std::string line;
    while (getline(openFile, line)) {
      std::size_t index = line.find(delimiter);
      std::string key = "";
      int value = -1;
      if (index != std::string::npos) {
        key = line.substr(0, index);
        if (key == target) {
          value = std::stoi(line.substr(index + 1));
          std::cout << "parse_target_config target key: " << key << ", value: " << value << std::endl;
          openFile.close();
          return value;
        }
      }
    }
    openFile.close();
  }
  std::cout << "!!! No target(" << target << ") exists. !!!" << std::endl;
  exit(-1);
}


float GM_Parser::parse_float_config(std::string target) {
  std::ifstream openFile(file_path.data());
  if (openFile.is_open()) {
    std::string line;
    while (getline(openFile, line)) {
      std::size_t index = line.find(delimiter);
      std::string key = "";
      float value = -1;
      if (index != std::string::npos) {
        key = line.substr(0, index);
        if (key == target) {
          value = std::stof(line.substr(index + 1));
          std::cout << "parse_float_config target key: " << key << ", value: " << value << std::endl;
          openFile.close();
          return value;
        }
      }
    }
    openFile.close();
  }
  std::cout << "!!! No target(" << target << ") exists. !!!" << std::endl;
  exit(-1);
}


std::string GM_Parser::parse_string_config(std::string target) {
  std::ifstream openFile(file_path.data());
  if (openFile.is_open()) {
    std::string line;
    while (getline(openFile, line)) {
      std::size_t index = line.find(delimiter);
      std::string key = "";
      std::string value = "";
      if (index != std::string::npos) {
        key = line.substr(0, index);
        if (key == target) {
          value = line.substr(index + 1);
          std::cout << "parse_string_config target key: " << key << ", value: " << value << std::endl;
          openFile.close();
          return value;
        }
      }
    }
    openFile.close();
  }
  std::cout << "!!! No target(" << target << ") exists. !!!" << std::endl;
  exit(-1);
}

void GM_Parser::file_write(std::string text) {
	std::string filePath = "/home/gnagmuk/mnt/ssd2/V100/TF_Output_Cache/doremi.txt";
	// write File
	std::ofstream writeFile(filePath.data());
	if( writeFile.is_open() ){
		writeFile << text;
		writeFile.close();
	}
}

// std::vector<int> GM_Parser::parse_list(std::string target) {
//   std::ifstream openFile(file_path.data());
//   if (openFile.is_open()) {
//     std::string line;
//     std::vector<int> ret_list_;
//     while (getline(openFile, line)) {
//       std::size_t index = line.find(delimiter);
//       if (index != std::string::npos) {
//         std::string key = line.substr(0, index);
//         std::string value = line.substr(index + 1);
//         if (key == target) {
//           int num_elements = 0;
//           int prev_idx = 0;
//           for (int i = 0; value.size(); i++) {
//             if (value[i] == ',') {
//               int elem = std::stoi(value.substr(prev_idx, i));
//               num_elements++;
//               prev_idx = (i + 1);
//               ret_list_.push_back(elem);
//             }
//             else if (num_elements > 0) {
//               int elem = std::stoi(value.substr(prev_idx));
//               num_elements++;
//               ret_list_.push_back(elem);
//             }
//           }
//           if (num_elements == 0) {
//             printf(
//                 "Error: %s is not list. Call parse_target_config instead of "
//                 "parse_list\n",
//                 key.c_str());
//             exit(-1);
//           }
//           std::cout << "parse target key: " << key << ", value: ";
//           for (int i = 0; i < ret_list_.size(); i++) {
//             printf("%d, ", ret_list_[i]);
//           }
//           printf("\n");
//           return ret_list_;
//         }
//       }
//     }
//     openFile.close();
//   }
//   std::cout << "!!! No target(" << target << ") exists. !!!" << std::endl;
//   exit(-1);
// }

} // namespace tensorflow
