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

#include "tensorflow/core/framework/unique_tensor_references.h"

namespace tensorflow {

UniqueTensorReferences::~UniqueTensorReferences() {
  if (!frozen_) {
    // The references were not retrieved so discard them to avoid
    // leaking memory.
    TensorReferenceVector refs;
    FreezeAndReturnReferences(&refs);
    for (auto& tensor : refs) {
      tensor.Unref();
    }
  }
  delete referenced_tensors_set_;
}

void UniqueTensorReferences::Add(const Tensor& tensor) {
  DCHECK(!frozen_);
  // Do nothing if the tensor has a null buffer.
  if (tensor.IsInitialized() && tensor.NumElements() > 0) {
    // referenced_tensors_set_에 insert 하든, referenced_tensors_vector_에
    // push_back 하든 상관없다. FreezeAndReturnReferences 함수에서 두개 변수
    // 모두 확인을 하고 다 꺼내서 out_vector에 넣어주기 때문에.

    if (referenced_tensors_set_ != nullptr) {
      // 여기는 도달 안한다. 최소한 Resnet50는 도달 안한다.

      // Original Code
      // const TensorReference tensor_ref(tensor); // 이 컨스트럭터에서 Ref() 가 한번 된다.
      // There are enough tensors that we are using a hash set to de-duplicate.

      // Gangmuk: TensorReference ctor without Ref()
      const TensorReference tensor_ref(tensor);

      // 여기서 insert 할 때 tensor가 또 만들어지나? => No
      if (!referenced_tensors_set_->insert(tensor_ref).second) {
        // The tensor was a duplicate, so discard the reference.
        // 만약에 위에 insert를 했을 때, 뭔가 잘 안되서 false가 return이 되면,
        // 위 컨스트럭터에서 Ref()된 걸 Unref() 해준다.
        // printf("!!!!!!!!! referenced_tensors_set_->insert(tensor_ref) return false!\n");
        // 여기도 도달을 안할 줄 알았는데 최소한 Resnet50에선 매 스텝마다 일정 횟수 도달한다.
        tensor_ref.Unref(); // Original Code
      }
    }
    else {
      // printf("$$$$$$$$$$$$ referenced_tensors_set_ == nullptr\n");
      // 여기는 도달한다.

      for (size_t i = 0; i < referenced_tensors_vector_.size(); ++i) {
        if (referenced_tensors_vector_[i].SharesBufferWith(tensor)) {
          // printf("!! UniqueTensorReferences::Add(), tensor is a duplicate, so nothing to do.\n");
          // tensor is a duplicate, so nothing to do.
          return;
        }
      }

      // Original Code
      referenced_tensors_vector_.push_back(TensorReference(tensor));

      // Gangmuk: TensorReference ctor without Ref()
      // 여기서 push_back 할 때 tensor가 또 만들어지나? => No
      if (kInVector == referenced_tensors_vector_.size()) {
        // printf("$$$$$$$$$$$$$$$$$$$$$$$$ Too many tensors!\n");
        // 여기는 도달 할 때도 있고 안할 때도 있고 하다.

        // There are too many tensors to keep using the N^2 algorithm
        // so start de-duplicating using a set.
        // Transfer the refs from the vector to the set.
        DCHECK(referenced_tensors_set_ == nullptr);
        referenced_tensors_set_ = new ReferencedTensorsSet;
        referenced_tensors_set_->reserve(kInVector);
        referenced_tensors_set_->insert(referenced_tensors_vector_.begin(),
                                        referenced_tensors_vector_.end());
        DCHECK_EQ(kInVector, referenced_tensors_set_->size());
        referenced_tensors_vector_.clear();
      }
    }
  }
}

void UniqueTensorReferences::FreezeAndReturnReferences(
    TensorReferenceVector* out_vector) {
  // Prevent any further additions.
  frozen_ = true;
  if (referenced_tensors_set_ != nullptr) {
    DCHECK(referenced_tensors_vector_.empty());
    out_vector->reserve(referenced_tensors_set_->size());
    for (const auto& ref : *referenced_tensors_set_) {
      out_vector->push_back(ref); // 혹시 여기서 copy가 일어나면서...? ㅠㅠ
    }
    referenced_tensors_set_->clear();
    delete referenced_tensors_set_;
    referenced_tensors_set_ = nullptr;
  } else {
    out_vector->reserve(referenced_tensors_vector_.size());
    for (const auto& ref : referenced_tensors_vector_) {
      out_vector->push_back(ref);
    }
    referenced_tensors_vector_.clear();
  }
}

}  // namespace tensorflow
