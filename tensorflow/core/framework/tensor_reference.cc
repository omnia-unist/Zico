/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/tensor_reference.h"

namespace tensorflow {

// Original Copy Constructor
TensorReference::TensorReference(const Tensor& tensor)
    : buf_(tensor.buf_ ? tensor.buf_->root_buffer() : nullptr) {
  if (buf_) buf_->Ref();
}

}  // namespace tensorflow
