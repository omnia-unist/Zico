/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/op_kernel.h"

#include <mutex>  // NOLINT
#include <unordered_map>
#include <utility>
#include <vector>

// Gangmuk
#include <map>

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/graph.pb_text.h"
#include "tensorflow/core/framework/kernel_def.pb_text.h"
#include "tensorflow/core/framework/kernel_def_util.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/memory_types.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/platform_strings.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {

// Gangmuk
std::unordered_set<std::string> OpKernelContext::failed_RefCountIsOne_nodes_set;

namespace {

Status MatchSignatureHelper(const DataTypeSlice expected_inputs,
                            const DataTypeSlice expected_outputs,
                            const DataTypeSlice inputs,
                            const DataTypeSlice outputs) {
  bool signature_mismatch = false;

  if (inputs.size() != expected_inputs.size()) signature_mismatch = true;
  for (size_t i = 0; !signature_mismatch && i < inputs.size(); ++i) {
    if (!TypesCompatible(expected_inputs[i], inputs[i])) {
      signature_mismatch = true;
    }
  }

  if (outputs.size() != expected_outputs.size()) signature_mismatch = true;
  for (size_t i = 0; !signature_mismatch && i < outputs.size(); ++i) {
    if (!TypesCompatible(expected_outputs[i], outputs[i])) {
      signature_mismatch = true;
    }
  }

  if (signature_mismatch) {
    return errors::InvalidArgument(
        "Signature mismatch, have: ", DataTypeSliceString(inputs), "->",
        DataTypeSliceString(outputs),
        " expected: ", DataTypeSliceString(expected_inputs), "->",
        DataTypeSliceString(expected_outputs));
  }
  return Status::OK();
}

}  // namespace

// OpKernel ------------------------------------------------------------------

OpKernel::OpKernel(OpKernelConstruction* context)
    : OpKernel(context, MakeUnique<const NodeDef>(context->def())) {}

OpKernel::OpKernel(OpKernelConstruction* context,
                   std::unique_ptr<const NodeDef> node_def)
    : def_(std::move(node_def)),
      input_types_(context->input_types().begin(),
                   context->input_types().end()),
      input_memory_types_(context->input_memory_types().begin(),
                          context->input_memory_types().end()),
      output_types_(context->output_types().begin(),
                    context->output_types().end()),
      output_memory_types_(context->output_memory_types().begin(),
                           context->output_memory_types().end()),
      graph_def_version_(context->graph_def_version()),
      is_internal_(str_util::StartsWith(type_string(), "_")),
      input_name_map_(context->num_inputs()),
      output_name_map_(context->num_outputs()) {
  OP_REQUIRES_OK(context,
                 NameRangesForNode(*def_, *context->op_def_, &input_name_map_,
                                   &output_name_map_));
  OP_REQUIRES_OK(context, CheckOpDeprecation(*context->op_def_,
                                             context->graph_def_version()));

  // Kernels executing on GPU/SYCL tie very few resources on the CPU where the
  // scheduler runs: we consider them as inexpensive.
  expensive_ = context->device_type() != DeviceType(DEVICE_GPU) &&
               context->device_type() != DeviceType(DEVICE_SYCL);
}

OpKernel::~OpKernel() {}

const string& OpKernel::name() const { return def_->name(); }
const string& OpKernel::type_string() const { return def_->op(); }
const string& OpKernel::requested_device() const { return def_->device(); }
const string& OpKernel::requested_input(int i) const { return def_->input(i); }

Status OpKernel::InputRange(StringPiece input_name, int* start,
                            int* stop) const {
  const auto result = input_name_map_.find(input_name);
  if (result == input_name_map_.end()) {
    return errors::InvalidArgument("Unknown input name: ", input_name);
  } else {
    *start = result->second.first;
    *stop = result->second.second;
    return Status::OK();
  }
}

Status OpKernel::OutputRange(StringPiece output_name, int* start,
                             int* stop) const {
  const auto result = output_name_map_.find(output_name);
  if (result == output_name_map_.end()) {
    return errors::InvalidArgument("Unknown output name: ", output_name);
  } else {
    *start = result->second.first;
    *stop = result->second.second;
    return Status::OK();
  }
}

Status OpKernel::MakeShape(const Tensor& shape, TensorShape* out) const {
  if (!IsLegacyVector(shape.shape())) {
    return errors::InvalidArgument(
        "shape must be a vector of {int32,int64}, got shape ",
        shape.shape().DebugString());
  }
  if (shape.dtype() == DataType::DT_INT32) {
    auto vec = shape.flat<int32>();
    return TensorShapeUtils::MakeShape(vec.data(), vec.size(), out);
  } else if (shape.dtype() == DataType::DT_INT64) {
    auto vec = shape.flat<int64>();
    return TensorShapeUtils::MakeShape(vec.data(), vec.size(), out);
  } else {
    return errors::InvalidArgument("shape must be a vector of {int32,int64}.");
  }
}

void AsyncOpKernel::Compute(OpKernelContext* context) {
  Notification n;
  ComputeAsync(context, [&n]() { n.Notify(); });
  n.WaitForNotification();
}

// PersistentTensor ----------------------------------------------------------

Tensor* PersistentTensor::AccessTensor(OpKernelConstruction* context) {
  // the caller has to have a valid context
  CHECK(context);
  return &tensor_;
}

Tensor* PersistentTensor::AccessTensor(OpKernelContext* context) {
  context->NotifyUseOfPersistentTensor(tensor_);
  return &tensor_;
}

// OpKernelConstruction ------------------------------------------------------

OpKernelConstruction::OpKernelConstruction(
    DeviceType device_type, DeviceBase* device, Allocator* allocator,
    const NodeDef* node_def, const OpDef* op_def, FunctionLibraryRuntime* flib,
    const DataTypeSlice& input_types, const MemoryTypeSlice& input_memory_types,
    const DataTypeSlice& output_types,
    const MemoryTypeSlice& output_memory_types, int graph_def_version,
    Status* status)
    : device_type_(std::move(device_type)),
      device_(device),
      allocator_(allocator),
      def_(node_def),
      op_def_(op_def),
      flib_(flib),
      input_types_(input_types),
      input_memory_types_(input_memory_types),
      output_types_(output_types),
      output_memory_types_(output_memory_types),
      graph_def_version_(graph_def_version),
      status_(status) {}

bool OpKernelConstruction::HasAttr(StringPiece attr_name) const {
  return HasNodeAttr(def(), attr_name);
}

void OpKernelConstruction::SetStatus(const Status& status) {
  status_->Update(status);
}

Status OpKernelConstruction::MatchSignature(
    const DataTypeSlice expected_inputs, const DataTypeSlice expected_outputs) {
  return MatchSignatureHelper(expected_inputs, expected_outputs, input_types_,
                              output_types_);
}

Status OpKernelConstruction::allocate_temp(DataType type,
                                           const TensorShape& shape,
                                           Tensor* out_temp) {
  AllocationAttributes attr;
  attr.allocation_will_be_logged = true;
  ///////////////////////////////////////////////////////
  //////////////Where a NEW TENSOR is created!!!/////////
  ///////////////////////////////////////////////////////
	if(false) std::cout<<"@@ allocate_temp() is creating new tensor"<<std::endl;
  // it doesn't seem to be called in resnet50
  Tensor new_temp(allocator_, type, shape, attr);

  if (!new_temp.IsInitialized()) {
    return errors::ResourceExhausted(
        "OOM when allocating temporary tensor with shape", shape.DebugString());
  }
  if (LogMemory::IsEnabled()) {
    LogMemory::RecordTensorAllocation(
        def_->name(), LogMemory::OP_KERNEL_CONSTRUCTION_STEP_ID, new_temp);
  }
  *out_temp = new_temp;
  return Status::OK();
}

Status OpKernelConstruction::allocate_persistent(
    DataType type, const TensorShape& shape, PersistentTensor* out_persistent,
    Tensor** out_tensor) {

  // for now just do the same thing as allocate_temp
  // TODO(misard) add specific memory tracking for persistent tensors
  Tensor persistent;
  // std::cout<<"@@ allocate_persistent() - 1 is calling allocate_temp()"<<std::endl;
  // it didn't appear in resnet50

  Status s = allocate_temp(type, shape, &persistent);
  if (!s.ok()) {
    return s;
  }
  *out_persistent = PersistentTensor(persistent);
  Tensor* allocated = out_persistent->AccessTensor(this);
  if (out_tensor) {
    *out_tensor = allocated;
  }
  return s;
}

// OpKernelContext -----------------------------------------------------------

const int OpKernelContext::Params::kNeverForward;
const int OpKernelContext::Params::kNoReservation;

OpKernelContext::OpKernelContext(Params* params)
    : OpKernelContext(
          params, static_cast<int>(params->op_kernel->output_types().size())) {}

OpKernelContext::OpKernelContext(Params* params, int num_outputs)
    : params_(params),
      outputs_(num_outputs),
      temp_memory_allocated_(0),
      persistent_memory_allocated_(0) {
  params_->ensure_eigen_gpu_device();
  if (params_->eigen_gpu_device != nullptr) {
    Allocator* eigen_gpu_allocator = get_allocator(AllocatorAttributes());
    Status s = params_->device->ReinitializeGpuDevice(
        this, params_->eigen_gpu_device, params_->op_device_context,
        eigen_gpu_allocator);
    if (!s.ok()) {
      SetStatus(s);
    }
  }
  if (params_->record_tensor_accesses) {
    referenced_tensors_.Init();
    // Gangmuk
    // referenced_tensors_->tensor_recording_original_version =
    //     params_->device->TensorRecordingOriginalVersion();
  }

  // Gangmuk
  // skip_all_test = params_->device->SkipAllTest();
  // skip_ref_count_is_one = params_->device->SkipRefCountIsOne();
  record_output_tensor_only = params_->device->RecordOutputTensorOnly();
  forward_input_profiling_mode = params_->device->ForwardInputProfilingMode();
  // print_forward_input_or_allocate_output = params_->device->PrintForwardInputOrAllocateOutput();
  inplace_profiling_step = params_->device->InplaceProfilingStep();
  force_no_inplace = params_->device->ForceNoInplace();
  print_forward_input = params_->device->PrintForwardInput();
}

OpKernelContext::~OpKernelContext() {
  for (TensorValue& value : outputs_) {
    if (!value.is_ref()) {
      delete value.tensor;
    }
  }
  if (params_->record_tensor_accesses) referenced_tensors_.Destroy();
  if (params_->track_allocations && !wrapped_allocators_.empty()) {
    LOG(WARNING) << "OpKernelContext is tracking allocations but they are not "
                 << "being consumed by the StepStatsCollector.";
    for (auto& wrapped_alloator : wrapped_allocators_) {
      wrapped_alloator.second->GetRecordsAndUnRef();
    }
  }
}

Allocator* OpKernelContext::get_allocator(AllocatorAttributes attr) {
  Allocator* allocator = nullptr;
  if (TF_PREDICT_FALSE(attr.scope_id > 0)) {
    allocator = params_->device->GetScopedAllocator(attr, step_id());
    CHECK(allocator);
  } else {
    allocator = params_->device->GetAllocator(attr);
  }
  if (TF_PREDICT_FALSE(track_allocations())) {
    mutex_lock lock(mu_);
    for (const auto& wrapped : wrapped_allocators_) {
      if (wrapped.first == allocator) {
        return wrapped.second;
      }
    }
    TrackingAllocator* wrapped_allocator =
        new TrackingAllocator(allocator, params_->track_allocations);
    wrapped_allocators_.push_back(std::make_pair(allocator, wrapped_allocator));
    return wrapped_allocator;
  } else {
    return allocator;
  }
}

void OpKernelContext::SetStatus(const Status& status) {
  status_.Update(status);
}

void OpKernelContext::really_record_tensor_reference(const Tensor& tensor) {
  mutex_lock l(mu_);
  // Keep a reference to the underlying memory around.
  referenced_tensors_->Add(tensor);
}

Status OpKernelContext::input(StringPiece name, const Tensor** tensor) {
  int start, stop;
  TF_RETURN_IF_ERROR(params_->op_kernel->InputRange(name, &start, &stop));
  if (stop != start + 1) {
    return errors::InvalidArgument("OpKernel used list-valued input name '",
                                   name,
                                   "' when single-valued input was "
                                   "expected");
  }
  if (input_is_ref(start)) {
    return errors::InvalidArgument("OpKernel used ref input name '", name,
                                   "' when non-ref input was expected");
  }
  *tensor = (*params_->inputs)[start].tensor;
  // record_tensor_reference(**tensor);
  // Gangmuk
  record_tensor_reference(**tensor, false);
  return Status::OK();
}

Status OpKernelContext::input_dtype(StringPiece name, DataType* dtype) const {
  int start, stop;
  TF_RETURN_IF_ERROR(params_->op_kernel->InputRange(name, &start, &stop));
  if (stop != start + 1) {
    return errors::InvalidArgument("OpKernel used list-valued input name '",
                                   name,
                                   "' when single-valued input was "
                                   "expected");
  }
  const TensorValue& value((*params_->inputs)[start]);
  if (value.is_ref()) {
    *dtype = MakeRefType(value->dtype());
  } else {
    *dtype = value->dtype();
  }
  return Status::OK();
}

Status OpKernelContext::input_ref_mutex(StringPiece name, mutex** out_mutex) {
  int start, stop;
  TF_RETURN_IF_ERROR(params_->op_kernel->InputRange(name, &start, &stop));
  if (stop != start + 1) {
    return errors::InvalidArgument("OpKernel used list-valued input name '",
                                   name,
                                   "' when single-valued input was expected");
  }
  *out_mutex = input_ref_mutex(start);
  return Status::OK();
}

const Tensor& OpKernelContext::input(int index) {
  DCHECK_GE(index, 0);
  DCHECK_LT(index, num_inputs()) << " name: " << op_kernel().name();
  DCHECK(!input_is_ref(index));
  const Tensor& tensor = *((*params_->inputs)[index].tensor);
  record_tensor_reference(tensor, false);
  return tensor;
}

Tensor OpKernelContext::mutable_input(int index, bool lock_held) {
  DCHECK_GE(index, 0);
  DCHECK_LT(index, num_inputs());
  DCHECK(input_is_ref(index));
  // return a copy of the Ref acquired while holding the mutex
  if (lock_held) {
    Tensor& tensor = *((*params_->inputs)[index].tensor);
    record_tensor_reference(tensor, false);
    return tensor;
  } else {
    mutex_lock l(*input_ref_mutex(index));
    Tensor& tensor = *((*params_->inputs)[index].tensor);
    record_tensor_reference(tensor, false);
    return tensor;
  }
}

void OpKernelContext::replace_ref_input(int index, const Tensor& tensor,
                                        bool lock_held) {
  DCHECK_GE(index, 0);
  DCHECK_LT(index, num_inputs());
  DCHECK(input_is_ref(index));
  // should only modify the tensor while holding the mutex
  if (lock_held) {
    *(*params_->inputs)[index].tensor = tensor;
  } else {
    mutex_lock l(*input_ref_mutex(index));
    *(*params_->inputs)[index].tensor = tensor;
  }
  record_tensor_reference(tensor, false);
}

void OpKernelContext::forward_ref_input_to_ref_output(int input_index,
                                                      int output_index) {
  DCHECK_GE(input_index, 0);
  DCHECK_LT(input_index, num_inputs());
  DCHECK(input_is_ref(input_index));
  set_output_ref(output_index, (*params_->inputs)[input_index].mutex_if_ref,
                 (*params_->inputs)[input_index].tensor);
}

//
bool OpKernelContext::forward_input_to_output_with_shape(
    int input_index, int output_index, const TensorShape& output_shape,
    Tensor** output) {
  const auto output_attr = params_->output_attr_array == nullptr
                               ? AllocatorAttributes()
                               : output_alloc_attr(output_index);
  std::unique_ptr<Tensor> new_tensor = forward_input(
      input_index, output_index, expected_output_dtype(output_index),
      output_shape, output_memory_type(output_index), output_attr);
  if (new_tensor != nullptr) {
    // Transfer ownership to the output slot in OpKernelContext.
    outputs_[output_index] = TensorValue(new_tensor.release());
    *output = outputs_[output_index].tensor;
    return true;
  } else {
    return false;
  }
}

// Gangmuk: overloading with reason_id
bool OpKernelContext::forward_input_to_output_with_shape(
    int input_index, int output_index, const TensorShape& output_shape,
    Tensor** output, int* reason_id) {
  const auto output_attr = params_->output_attr_array == nullptr
                               ? AllocatorAttributes()
                               : output_alloc_attr(output_index);
  std::unique_ptr<Tensor> new_tensor = forward_input(
      input_index, output_index, expected_output_dtype(output_index),
      output_shape, output_memory_type(output_index), output_attr, reason_id);
  if (new_tensor != nullptr) {
    // Transfer ownership to the output slot in OpKernelContext.
    outputs_[output_index] = TensorValue(new_tensor.release());
    *output = outputs_[output_index].tensor;
    // Gangmuk
    // printf(
    //     "**,OpKernelContext::forward_input_to_output_with_shape using "
    //     "index,new_tensor != "
    //     "nullptr,%s,%s\n",
    //     op_kernel().name().c_str(), op_kernel().type_string().c_str());
    return true;
  } else {
    // Gangmuk
    // printf(
    //     "**,OpKernelContext::forward_input_to_output_with_shape using "
    //     "index,new_tensor == "
    //     "nullptr,%s,%s\n",
    //     op_kernel().name().c_str(), op_kernel().type_string().c_str());
    return false;
  }
}

Status OpKernelContext::forward_input_to_output_with_shape(
    StringPiece input_name, StringPiece output_name,
    const TensorShape& output_shape, Tensor** output) {
  int input_index, output_index, stop;
  TF_RETURN_IF_ERROR(
      params_->op_kernel->InputRange(input_name, &input_index, &stop));
  if (stop != input_index + 1) {
    return errors::InvalidArgument("OpKernel used list-valued input name '",
                                   input_name,
                                   "' when single-valued input was "
                                   "expected");
  }
  TF_RETURN_IF_ERROR(
      params_->op_kernel->OutputRange(output_name, &output_index, &stop));
  if (stop != output_index + 1) {
    return errors::InvalidArgument("OpKernel used list-valued output name '",
                                   output_name,
                                   "' when single-valued output was "
                                   "expected");
  }
  if (!forward_input_to_output_with_shape(input_index, output_index,
                                          output_shape, output)) {
    return errors::FailedPrecondition("OpKernel could not forward input '",
                                      input_name, "' to output '", output_name);
  }
  return Status::OK();
}

//
std::unique_ptr<Tensor> OpKernelContext::forward_input(
    int input_index, int output_index, DataType output_dtype,
    const TensorShape& output_shape, MemoryType output_memory_type,
    const AllocatorAttributes& output_attr) {
  DCHECK_GE(input_index, 0);
  DCHECK_LT(input_index, num_inputs());

  // 강제로 forward_input 끄기 //
  // 다른데도 하나 더 있음 !!!!!!!!!!!!!!!!!!!!!!!!!!!!
  if (force_no_inplace) return nullptr;

  const TensorValue& input = (*params_->inputs)[input_index];
  // Check whether at graph construction time this output was marked
  // either for no forwarding or with a reservation for this input.
  // If it's reserved for this input we'll skip the refcount and
  // AllocatorAttribute checks.
  // TODO(tucker): Maybe we should skip all of the checks?
  // Gangmuk: 이 테스트를 전부 패스하면 crash 난다.

  /*
    Gangmuk: It doesn't work, since there are forward_input that return false from
    one of these coditions even when TensorRecording is off.
  */
  bool print_forward_input = false;

  // if (!skip_all_test) {
  bool never_forward =
      (params_->forward_from_array != nullptr && output_index >= 0 &&
      params_->forward_from_array[output_index] == Params::kNeverForward);
  if (never_forward) {
    if (print_forward_input)
      printf(
          "**,OpKernelContext::forward_input,%d,%d,never_forward,1,%s,%s\n",
          input_index, output_index, op_kernel().name().c_str(), op_kernel().type_string().c_str());
    return nullptr;
  }
  bool forward_expected =
      (params_->forward_from_array != nullptr && output_index >= 0 &&
      params_->forward_from_array[output_index] == input_index);
  if (!forward_expected && params_->forward_from_array != nullptr) {
    // Check for possibly conflicting forward.
    for (int i = 0; i < num_outputs(); ++i) {
      if (params_->forward_from_array[i] == input_index) {
        // This input is reserved for output i.
        if (print_forward_input)
          printf(
              "**,OpKernelContext::forward_input,%d,%d,inputIsReserved,2,%s,%"
              "s\n",
              input_index, output_index, op_kernel().name().c_str(),
              op_kernel().type_string().c_str());
        return nullptr;
      }
    }
  }
  // Check that input tensor exists and is not a ref.
  if (input.tensor == nullptr || input.is_ref()) {
    CHECK(!forward_expected);
    if (print_forward_input)
      printf(
          "**,OpKernelContext::forward_input,%d,%d,input.tensor == nullptr "
          "|| "
          "input.is_ref(),3,%s,%s\n",
          input_index, output_index, op_kernel().name().c_str(),
          op_kernel().type_string().c_str());
    return nullptr;
  }
  // Check that input type matches.
  if (input_dtype(input_index) != output_dtype) {
    CHECK(!forward_expected);
    if (print_forward_input)
      printf(
          "**,OpKernelContext::forward_input,%d,%d,input_dtype(input_index) "
          "!= "
          "output_dtype,4,%s,%s\n",
          input_index, output_index, op_kernel().name().c_str(),
          op_kernel().type_string().c_str());
    return nullptr;
  }
  // Check that the input and output sizes are compatible.
  if (input.tensor->shape().num_elements() != output_shape.num_elements()) {
    CHECK(!forward_expected);
    if (print_forward_input)
      printf(
          "**,OpKernelContext::forward_input,%d,%d,input.tensor->shape().num_"
          "elements() != output_shape.num_elements(),5,%s,%s\n",
          input_index, output_index, op_kernel().name().c_str(),
          op_kernel().type_string().c_str());
    return nullptr;
  }
  // Check that input and output memory types match, i.e.
  // that they either both live in host or both live in device memory.
  if (input_memory_type(input_index) != output_memory_type) {
    CHECK(!forward_expected);
    if (print_forward_input)
      printf(
          "**,OpKernelContext::forward_input,%d,%d,input_memory_type(input_"
          "index) != "
          "output_memory_type,6,%s,%s\n",
          input_index, output_index, op_kernel().name().c_str(),
          op_kernel().type_string().c_str());
    return nullptr;
  }
  if (!forward_expected) {
    // Gangmuk
    if (!input->RefCountIsOne()) {
      if (print_forward_input)
        printf(
            "**,OpKernelContext::forward_input,%d,%d,!input->RefCountIsOne(),"
            "7,%s,%s\n", input_index, output_index, op_kernel().name().c_str(),
            op_kernel().type_string().c_str());
        return nullptr;
    }
    // Check that output allocator attributes are not more restrictive than
    // input allocator attributes.
    const auto input_attr = params_->input_alloc_attrs == nullptr
                                ? AllocatorAttributes()
                                : input_alloc_attr(input_index);
    if (!output_attr.IsEqualOrLessRestrictiveThan(input_attr)) {
      if (print_forward_input)
        printf(
            "**,OpKernelContext::forward_input,%d,%d,!output_attr."
            "IsEqualOrLessRestrictiveThan(input_attr),8,%s,%s\n",
            input_index, output_index, op_kernel().name().c_str(),
            op_kernel().type_string().c_str());
      return nullptr;
    }
  }
  // }  // if(!skip_all_test)

  auto output_tensor = MakeUnique<Tensor>();
  CHECK(output_tensor->CopyFrom(*input.tensor, output_shape));
  if (print_forward_input) {
    printf("**,OpKernelContext::forward_input,%d,%d,Success,0,%s,%s\n",
           input_index, output_index, op_kernel().name().c_str(),
           op_kernel().type_string().c_str());
  }
  return output_tensor;
}

// Gangmuk: overloading with reason_id
std::unique_ptr<Tensor> OpKernelContext::forward_input(
    int input_index, int output_index, DataType output_dtype,
    const TensorShape& output_shape, MemoryType output_memory_type,
    const AllocatorAttributes& output_attr, int* reason_id) {
  DCHECK_GE(input_index, 0);
  DCHECK_LT(input_index, num_inputs());
  
  // 강제로 forward_input 끄기 //
  // 다른데도 하나 더 있음 !!!!!!!!!!!!!!!!!!!!!!!!!!!!
  if (force_no_inplace) return nullptr;

  const TensorValue& input = (*params_->inputs)[input_index];
  // Check whether at graph construction time this output was marked
  // either for no forwarding or with a reservation for this input.
  // If it's reserved for this input we'll skip the refcount and
  // AllocatorAttribute checks.
  // TODO(tucker): Maybe we should skip all of the checks?
  // Gangmuk: 이 테스트를 전부 패스하면 crash 난다.

  /*
    Gangmuk: It doesn't work, since there are forward_input that return false
    from one of these coditions even when TensorRecording is off.
  */

  bool never_forward =
      (params_->forward_from_array != nullptr && output_index >= 0 &&
        params_->forward_from_array[output_index] == Params::kNeverForward);
  if (never_forward) {
    *reason_id = 1;
    return nullptr;
  }
  bool forward_expected =
      (params_->forward_from_array != nullptr && output_index >= 0 &&
        params_->forward_from_array[output_index] == input_index);
  if (!forward_expected && params_->forward_from_array != nullptr) {
    // Check for possibly conflicting forward.
    for (int i = 0; i < num_outputs(); ++i) {
      if (params_->forward_from_array[i] == input_index) {
        // This input is reserved for output i.
        *reason_id = 2;
        return nullptr;
      }
    }
  }
  // Check that input tensor exists and is not a ref.
  if (input.tensor == nullptr || input.is_ref()) {
    CHECK(!forward_expected);
    *reason_id = 3;
    return nullptr;
  }
  // Check that input type matches.
  if (input_dtype(input_index) != output_dtype) {
    CHECK(!forward_expected);
    *reason_id = 4;
    return nullptr;
  }
  // Check that the input and output sizes are compatible.
  if (input.tensor->shape().num_elements() != output_shape.num_elements()) {
    CHECK(!forward_expected);
    *reason_id = 5;
    return nullptr;
  }
  // Check that input and output memory types match, i.e.
  // that they either both live in host or both live in device memory.
  if (input_memory_type(input_index) != output_memory_type) {
    CHECK(!forward_expected);
    *reason_id = 6;
    return nullptr;
  }
  if (!forward_expected) {
    // Gangmuk
    if (!input->RefCountIsOne()) {
      if (forward_input_profiling_mode) {
        if (step_id() > inplace_profiling_step) {
          std::string node_name_with_index = op_kernel().name() + "-" +
                                             std::to_string(input_index) + "-" +
                                             std::to_string(output_index);
          auto iter = failed_RefCountIsOne_nodes_set.find(node_name_with_index);
          if (iter == failed_RefCountIsOne_nodes_set.end()) {
            /*
              failed_RefCountIsOne_nodes_set 에서 찾을 수 없었음.
              (NoRecord mode에서 RefCountIsOne에서 """실패 한적이 없는"""
              노드라는 뜻이다.)

              원래는 forward_input이 성공해야 하는데, TensorRecording 기능으로
              인해 Referencing이 꼬이면서 Fail처럼 보이는 것이므로, 해당 if문에
              걸리면 return nullptr 하지 않고, success 하는 쪽으로 가게 한다.
              (맨 마지막 부분에 있는 return output_tensor로 간다.)

               Do Nothing !!! // go to "return output_tensor"
            */
            printf(
                "**,forward_input,FalseFail(RefCountIsOne),step_"
                "id,%d,%s,%s,Skip this fail\n",
                step_id(), node_name_with_index.c_str(),
                op_kernel().type_string().c_str());
          }
          else {
            // Profiling Mode도 맞고 프로파일링 스텝도 지났음.
            // failed_RefCountIsOne_nodes_set 기록에 있음.
            // 따라서, TrueFail! 
            printf(
                "**,forward_input,TrueFail(RefCountIsOne),step_id,%d,%s,%s,\n",
                step_id(), node_name_with_index.c_str(),op_kernel().type_string().c_str());
            *reason_id = 7;
            return nullptr;
          }
        }
        else {
          // Profiling Mode는 맞는데, 아직 프로파일링 스텝을 지나지 않았다.
          // 그 말은 아직 TensorRecording이 켜지지 않았고, 여기서의 fail은 진짜
          // fail이다.
          *reason_id = 7;
          return nullptr;
        }
      }
      else {
        // Profiling mode가 아니므로 다 진짜 Fail이다.
        *reason_id = 7;
        return nullptr;
      }
    }
    // Check that output allocator attributes are not more restrictive than
    // input allocator attributes.
    const auto input_attr = params_->input_alloc_attrs == nullptr
                                ? AllocatorAttributes()
                                : input_alloc_attr(input_index);
    if (!output_attr.IsEqualOrLessRestrictiveThan(input_attr)) {
      *reason_id = 8;
      return nullptr;
    }
  }

  // Success (pass all tests)
  auto output_tensor = MakeUnique<Tensor>();
  CHECK(output_tensor->CopyFrom(*input.tensor, output_shape));
  *reason_id = 0;
  return output_tensor;
}

Status OpKernelContext::forward_input_or_allocate_temp(
    gtl::ArraySlice<int> candidate_input_indices, DataType type,
    const TensorShape& shape, const AllocatorAttributes& allocator_attr,
    Tensor* out_temp) {
  printf("@@ forward_input_or_allocate_temp()\n");
  // it didn't appear in resnet50

  for (int input_index : candidate_input_indices) {
    std::unique_ptr<Tensor> new_tensor =
        forward_input(input_index, Params::kNoReservation /*output_index*/,
                      type, shape, DEVICE_MEMORY, allocator_attr);
    if (new_tensor != nullptr) {
      *out_temp = std::move(*new_tensor);
      return Status::OK();
    }
  }
  printf("@@ forward_intput_or_allocate_temp() is calling allocate_temp()\n");
  return allocate_temp(type, shape, out_temp, allocator_attr);
}

void OpKernelContext::delete_ref_input(int index, bool lock_held) {
  DCHECK_GE(index, 0);
  DCHECK_LT(index, num_inputs());
  DCHECK(input_is_ref(index));
  // should only modify the tensor while holding the mutex
  if (lock_held) {
    delete (*params_->inputs)[index].tensor;
  } else {
    mutex_lock l(*input_ref_mutex(index));
    delete (*params_->inputs)[index].tensor;
  }
}

Status OpKernelContext::mutable_input(StringPiece name, Tensor* tensor,
                                      bool lock_held) {
  int start, stop;
  TF_RETURN_IF_ERROR(params_->op_kernel->InputRange(name, &start, &stop));
  if (stop != start + 1) {
    return errors::InvalidArgument("OpKernel used list-valued input name '",
                                   name,
                                   "' when single-valued input was expected");
  }
  if (!input_is_ref(start)) {
    return errors::InvalidArgument("OpKernel used non-ref input name '", name,
                                   "' when ref input was expected");
  }
  // return a copy of the Ref acquired while holding the mutex
  if (lock_held) {
    *tensor = *(*params_->inputs)[start].tensor;
  } else {
    mutex_lock l(*input_ref_mutex(start));
    *tensor = *(*params_->inputs)[start].tensor;
  }
  record_tensor_reference(*tensor, false);
  return Status::OK();
}

Status OpKernelContext::replace_ref_input(StringPiece name,
                                          const Tensor& tensor,
                                          bool lock_held) {
  int start, stop;
  TF_RETURN_IF_ERROR(params_->op_kernel->InputRange(name, &start, &stop));
  if (stop != start + 1) {
    return errors::InvalidArgument("OpKernel used list-valued input name '",
                                   name,
                                   "' when single-valued input was expected");
  }
  if (!input_is_ref(start)) {
    return errors::InvalidArgument("OpKernel used immutable input name '", name,
                                   "' when ref input was expected");
  }
  replace_ref_input(start, tensor, lock_held);
  return Status::OK();
}

Status OpKernelContext::input_list(StringPiece name, OpInputList* list) {
  int start, stop;
  TF_RETURN_IF_ERROR(params_->op_kernel->InputRange(name, &start, &stop));
  *list = OpInputList(this, start, stop);
  return Status::OK();
}

Status OpKernelContext::mutable_input_list(StringPiece name,
                                           OpMutableInputList* list) {
  int start, stop;
  TF_RETURN_IF_ERROR(params_->op_kernel->InputRange(name, &start, &stop));
  *list = OpMutableInputList(this, start, stop);
  return Status::OK();
}

Status OpKernelContext::output_list(StringPiece name, OpOutputList* list) {
  int start, stop;
  TF_RETURN_IF_ERROR(params_->op_kernel->OutputRange(name, &start, &stop));
  *list = OpOutputList(this, start, stop);
  return Status::OK();
}

Status OpKernelContext::allocate_output(int index, const TensorShape& shape,
                                        Tensor** output) {
  // std::cout<<"@@ allocate_output() - 1"<<std::endl;
  // it calls allocate_output() - 4 after adding allocator attribute parameter

  DCHECK_GE(index, 0);
  DCHECK_LT(index, num_outputs());
  bool forward_expected =
      (params_->forward_from_array != nullptr && index >= 0 &&
       params_->forward_from_array[index] >= 0);
  if (forward_expected) {
    return errors::Internal(
        "Explicit allocate_output call where input forwarding required.  Try "
        "turning off the ScopedAllocator optimizer.");
  }
  AllocatorAttributes attr = output_alloc_attr(index);
  return allocate_output(index, shape, output, attr);
}

Status OpKernelContext::allocate_output(StringPiece name,
                                        const TensorShape& shape,
                                        Tensor** tensor) {
  if(false) std::cout<<"@@ allocate_output() - 2"<<std::endl;
  // it didn't appear in resnet50
  int start, stop;
  TF_RETURN_IF_ERROR(params_->op_kernel->OutputRange(name, &start, &stop));
  if (stop != start + 1) {
    return errors::InvalidArgument("OpKernel used list-valued output name '",
                                   name,
                                   "' when single-valued output was "
                                   "expected");
  }
  return allocate_output(start, shape, tensor);
}

Status OpKernelContext::allocate_output(StringPiece name,
                                        const TensorShape& shape,
                                        Tensor** tensor,
                                        AllocatorAttributes attr) {
  if(false) std::cout<<"@@ allocate_output() - 3"<<std::endl;
  // it didn't appear in resnet50

  int start, stop;
  TF_RETURN_IF_ERROR(params_->op_kernel->OutputRange(name, &start, &stop));
  if (stop != start + 1) {
    return errors::InvalidArgument("OpKernel used list-valued output name '",
                                   name,
                                   "' when single-valued output was "
                                   "expected");
  }
  return allocate_output(start, shape, tensor, attr);
}

// Gangmuk: third modified with additional parameter, struct NodeInfo
Status OpKernelContext::allocate_tensor(DataType type, const TensorShape& shape, Tensor* out_tensor,
                                        AllocatorAttributes attr, const AllocationAttributes& allocation_attr, 
                                        std::map< std::string, std::string > node_info) {
  Allocator* a = get_allocator(attr);
  AllocationAttributes logged_attr(allocation_attr);
  logged_attr.allocation_will_be_logged = true;

  // This is original new_tensor sentence
  // Tensor new_tensor(a, type, shape, logged_attr);

  // Gangmuk
  // This is new one with additional parameter NodeInfo
  // node_info.insert(std::pair< std::string, std::string >("executorstate_id", std::to_string(params_->executorstate_id)));
  node_info.insert(std::pair< std::string, std::string >("executorimpl_id", std::to_string(params_->executorimpl_id)));
  Tensor new_tensor(a, type, shape, logged_attr, node_info);

  if (!new_tensor.IsInitialized()) {
    return errors::ResourceExhausted(
        "OOM when allocating tensor with shape", shape.DebugString(),
        " and type ", DataTypeString(type), " on ", params_->device->name(),
        " by allocator ", a->Name());
  }
  if (params_->log_memory) {
    LogMemory::RecordTensorAllocation(params_->op_kernel->name(),
                                      params_->step_id, new_tensor);
  }
  //record tensor reference
  // Gangmuk
  std::string caller = node_info["caller"];
  if (caller == "allocate_output") {
    record_tensor_reference(new_tensor, true);
  }
  else {
    record_tensor_reference(new_tensor, false);
  }
  *out_tensor = std::move(new_tensor);
  return Status::OK();
}

// Gangmuk
// This "allocate_output" is the one calling "allocate_tensor()"
Status OpKernelContext::allocate_output(int index, const TensorShape& shape,
                                        Tensor** output,
                                        AllocatorAttributes attr) {
  DCHECK_GE(index, 0);
  DCHECK_LT(index, outputs_.size());
  const DataType type = params_->op_kernel->output_type(index);
  DCHECK(!IsRefType(type));
  DCHECK(mutable_output(index) == nullptr);
  auto output_tensor = MakeUnique<Tensor>();

  // Gangmuk: Dictionary data structure having tensor related information
  std::map< std::string, std::string > node_info;

  // Key
  std::string key1 = "kernel_name";
  std::string key2 = "op_type";
  std::string key3 = "caller";
  std::string key4 = "node_type";
  std::string key5 = "step_id";
  std::string key6 = "node_id";
  std::string key7 = "is_gradient";
  std::string key8 = "sharable";

  /* 
    Simpler Method to Classify the Activation 
    All tensors created through 'allocate_output' are activation.
  */
  // Value
  const NodeDef& ndef = op_kernel().def();
  std::string op_kernel_name = &ndef.name()[0];                  // value 1
  std::string op_kernel_type_string = &ndef.op()[0];             // value 2
  // std::string node_type = "bridge";                              // value 3
  /* Original Method to Classify the Activation */
  std::string node_type = this->params_->from_which_node;
  std::string caller = "allocate_output";                        // value 4
  std::string step_id = std::to_string(this->params_->step_id);  // value 5
  std::string node_id = std::to_string(params_->node_id);        // value 6
  std::string is_gradient = std::to_string(params_->is_gradient);  // value 7
  std::string sharable = "1";
  if (params_->is_gradient == 0) {
    sharable = "1";
  }

  // Insert
  node_info.insert(std::pair< std::string, std::string >(key1, op_kernel_name));
  node_info.insert(std::pair< std::string, std::string >(key2, op_kernel_type_string));
  node_info.insert(std::pair< std::string, std::string >(key3, caller));
  node_info.insert(std::pair< std::string, std::string >(key4, node_type));
  node_info.insert(std::pair< std::string, std::string >(key5, step_id));
  node_info.insert(std::pair< std::string, std::string >(key6, node_id));
  node_info.insert(std::pair<std::string, std::string>(key7, is_gradient));
  node_info.insert(std::pair<std::string, std::string>(key8, sharable));
  // Gangmuk
  //Status s = allocate_tensor(type, shape, output_tensor.get(), attr, "allocate_output");

  //Status s = allocate_tensor(type, shape, output_tensor.get(), attr, "allocate_output", op_kernel_name, op_kernel_type_string);

  Status s = allocate_tensor(type, shape, output_tensor.get(), attr, node_info);

  if (s.ok()) {
//    printf("!!!! allocate_output() !!!!\n");
    outputs_[index] = TensorValue(output_tensor.release());
    *output = outputs_[index].tensor;
  }
  return s;
}

Status OpKernelContext::allocate_temp(
    DataType type, const TensorShape& shape, Tensor* out_temp,
    AllocatorAttributes allocator_attr,
    const AllocationAttributes& allocation_attr) {
  // std::cout<<"@@ allocate_temp() - 1 is calling allocate_tensor()"<<std::endl;

  // Gangmuk: Dictionary data structure having tensor related information
  std::map< std::string, std::string > node_info;

  // Key
  std::string key1 = "kernel_name";
  std::string key2 = "op_type";
  std::string key3 = "caller";
  std::string key4 = "node_type";
  std::string key5 = "step_id";
  std::string key6 = "node_id";
  std::string key7 = "is_gradient";
  std::string key8 = "sharable";

  // Value
  const NodeDef& ndef = op_kernel().def();
  std::string op_kernel_name = &ndef.name()[0];       // value 1
  std::string op_kernel_type_string = &ndef.op()[0];  // value 2
  std::string caller = "allocate_temp";               // value 3
  std::string node_type = this->params_->from_which_node;     // value 4
  std::string step_id = std::to_string(this->params_->step_id);  // value 5
  std::string node_id = std::to_string(params_->node_id);        // value 6
  std::string is_gradient = std::to_string(params_->is_gradient);  // value 7
  std::string sharable = "0";

  // Insert
  node_info.insert(std::pair< std::string, std::string >(key1, op_kernel_name));
  node_info.insert(std::pair< std::string, std::string >(key2, op_kernel_type_string));
  node_info.insert(std::pair< std::string, std::string >(key3, caller));
  node_info.insert(std::pair< std::string, std::string >(key4, node_type));
  node_info.insert(std::pair< std::string, std::string >(key5, step_id));
  node_info.insert(std::pair< std::string, std::string >(key6, node_id));
  node_info.insert(std::pair<std::string, std::string>(key7, is_gradient));
  node_info.insert(std::pair<std::string, std::string>(key8, sharable));

  Status s = allocate_tensor(type, shape, out_temp, allocator_attr,
                             allocation_attr, node_info);
  if (track_allocations() && s.ok() && out_temp->TotalBytes() > 0) {
    Allocator* a = get_allocator(allocator_attr);
    if (a->TracksAllocationSizes()) {
      int64 alloc_size = a->AllocatedSize(out_temp->tensor_data().data());
      record_temp_memory_allocation(alloc_size, *out_temp);
    }
  }
  return s;
}

Status OpKernelContext::allocate_persistent(DataType type,
                                            const TensorShape& shape,
                                            PersistentTensor* out_persistent,
                                            Tensor** out_tensor,
                                            AllocatorAttributes attr) {
  Tensor persistent;
  // std::cout<<"@@ allocate_persistent() - 2 is calling allocate_tensor"<<std::endl;

  // Gangmuk: Dictionary data structure having tensor related information
  std::map< std::string, std::string > node_info;

  // Key
  std::string key1 = "kernel_name";
  std::string key2 = "op_type";
  std::string key3 = "caller";
  std::string key4 = "node_type";
  std::string key5 = "step_id";
  std::string key6 = "node_id";
  std::string key7 = "is_gradient";
  std::string key8 = "sharable";

  // Value
  const NodeDef& ndef = op_kernel().def();
  std::string op_kernel_name = &ndef.name()[0];       // value 1
  std::string op_kernel_type_string = &ndef.op()[0];  // value 2
  std::string caller = "allocate_persistent";         // value 3
  std::string node_type = this->params_->from_which_node;  // value 4
  std::string step_id = std::to_string(this->params_->step_id);     // value 5
  std::string node_id = std::to_string(params_->node_id);           // value 6
  std::string is_gradient = std::to_string(params_->is_gradient);   // value 7
  std::string sharable = "0";
  
  // Insert
  node_info.insert(std::pair< std::string, std::string >(key1, op_kernel_name));
  node_info.insert(std::pair< std::string, std::string >(key2, op_kernel_type_string));
  node_info.insert(std::pair< std::string, std::string >(key3, caller));
  node_info.insert(std::pair< std::string, std::string >(key4, node_type));
  node_info.insert(std::pair< std::string, std::string >(key5, step_id));
  node_info.insert(std::pair< std::string, std::string >(key6, node_id));
  node_info.insert(std::pair<std::string, std::string>(key7, is_gradient));
  node_info.insert(std::pair<std::string, std::string>(key8, sharable));

  Status s = allocate_tensor(type, shape, &persistent, attr, node_info);
  if (s.ok()) {
    *out_persistent = PersistentTensor(persistent);
    if (out_tensor) {
      *out_tensor = out_persistent->AccessTensor(this);
    }
    if (track_allocations()) {
      Tensor* t = out_persistent->AccessTensor(this);
      Allocator* a = get_allocator(attr);
      if (a->TracksAllocationSizes()) {
        int64 alloc_size = a->AllocatedSize(t->tensor_data().data());
        int64 alloc_id = a->AllocationId(t->tensor_data().data());
        record_persistent_memory_allocation(alloc_size, alloc_id);
      }
    }
  }
  return s;
}

Status OpKernelContext::set_output(StringPiece name, const Tensor& tensor) {
  int start, stop;
  TF_RETURN_IF_ERROR(params_->op_kernel->OutputRange(name, &start, &stop));
  if (stop != start + 1) {
    return errors::InvalidArgument("OpKernel used list-valued output name '",
                                   name,
                                   "' when single-valued output was "
                                   "expected");
  }
  set_output(start, tensor);
  return Status::OK();
}

void OpKernelContext::set_output(int index, const Tensor& tensor) {
  // if (params_->step_id == 40) {
  //   printf("@@ set_output\n");
  // }
  DCHECK_GE(index, 0);
  DCHECK_LT(index, outputs_.size());
  DCHECK(!IsRefType(params_->op_kernel->output_type(index)));
  DCHECK_EQ(mutable_output(index), nullptr);
  
  // Gangmuk
  // record_tensor_reference(tensor);
  // record_tensor_reference(tensor, true);
  record_tensor_reference(tensor, false);

  // Original code
  // outputs_[index] = TensorValue(new Tensor(tensor));
  // Ganmguk
  // printf("set_output debug 1\n");
  // fflush(stdout);
  Tensor *t = new Tensor(tensor);
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // printf("set_output debug 2\n");
  // fflush(stdout);
  // AllocatorAttributes attr = output_alloc_attr(index);
  // Allocator* allocator_ = get_allocator(attr);
  // // if (allocator_->Name().find("BFC") != std::string::npos) { // GPU_0_BFC가 아니라 GPU_0_bfc
  // //   printf("set_output debug 3 (Allocator is BFCAllocator.)\n"); // 이거 if문 하나도 안들어간다. 한번도 True가 안됐다. 뭔가 잘못된 if 문인건가.
  // // if (allocator_->Name().find("bfc") != std::string::npos) { // cuda_host_bfc도 있다. GPU bfc만 걸러내야한다.
  // if (allocator_->Name().find("GPU") != std::string::npos) { // 이렇게 해야 seg fault가 안난다.
  //   // printf("set_output debug 3 (Allocator is %s.)\n", allocator_->Name().c_str());
  //   // fflush(stdout);
  //   // if (tensor.buf_ != nullptr) {
  //   if (t->buf_ != nullptr) {
  //     // printf("set_output debug 4 (Checking tensor.buf_ != nullptr)\n");
  //     // fflush(stdout);
  //     void* data = t->buf_->data();
  //    //  void* data = tensor.buf_->data();
  //     // printf("set_output debug 5 (Get data pointer to chunk.)\n");
  //     // fflush(stdout);
  //     if (data != nullptr) {
  //       allocator_->TagAsOutput(data);
  //       // printf("set_output debug 6 (Get valid data pointer to chunk TagAsOutput)\n");
  //       // fflush(stdout);
  //     }
  //     // else {
  //     //   printf("set_output debug 7 (data pointer is nullptr...)\n");
  //     //   fflush(stdout);
  //     // }
  //   }
  //   // else {
  //   //   printf("set_output debug 8 (t->buf_ == nullptr)\n");
  //   //   fflush(stdout);
  //   // }
  // }
  // // else {
  // //   printf("set_output debug 9 (Another Allocator: %s)\n",allocator_->Name().c_str());
  // //   fflush(stdout);
  // // }
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  outputs_[index] = TensorValue(t);
  // printf("set_output debug 10 (set_output done)\n");
  // fflush(stdout);
  ///////////////////////////////////////////////////////////

  if (track_allocations() && tensor.TotalBytes() > 0) {
    mutex_lock l(stats_mu_);
    if (!temp_tensor_buffer_and_size_) {
      return;
    }
    auto it = std::find_if(temp_tensor_buffer_and_size_->begin(),
                           temp_tensor_buffer_and_size_->end(),
                           [&tensor](const std::pair<const void*, int64>& e) {
                             return e.first == static_cast<const void*>(
                                                   tensor.tensor_data().data());
                           });
    if (it != temp_tensor_buffer_and_size_->end()) {
      temp_memory_allocated_ -= it->second;
      temp_tensor_buffer_and_size_->erase(it);
    }
  }
}

void OpKernelContext::set_output_ref(int index, mutex* mu,
                                     Tensor* tensor_for_ref) {
  // printf("@@ set_output_ref\n");
  DCHECK_GE(index, 0);
  DCHECK_LT(index, outputs_.size());
  DCHECK(IsRefType(params_->op_kernel->output_type(index)));

  // Gangmuk
  // record_tensor_reference(*tensor_for_ref);
  // record_tensor_reference(*tensor_for_ref, true);
  record_tensor_reference(*tensor_for_ref, false);

  outputs_[index] = TensorValue(mu, tensor_for_ref);
}

Status OpKernelContext::set_output_ref(StringPiece name, mutex* mu,
                                       Tensor* tensor_for_ref) {
  int start, stop;
  TF_RETURN_IF_ERROR(params_->op_kernel->OutputRange(name, &start, &stop));
  if (stop != start + 1) {
    return errors::InvalidArgument("OpKernel used list-valued output name '",
                                   name,
                                   "' when single-valued output was "
                                   "expected");
  }
  set_output_ref(start, mu, tensor_for_ref);
  return Status::OK();
}

Status OpKernelContext::mutable_output(StringPiece name, Tensor** tensor) {
  int start, stop;
  TF_RETURN_IF_ERROR(params_->op_kernel->OutputRange(name, &start, &stop));
  if (stop != start + 1) {
    return errors::InvalidArgument("OpKernel used list-valued output name '",
                                   name,
                                   "' when single-valued output was "
                                   "expected");
  }
  *tensor = mutable_output(start);
  return Status::OK();
}

bool OpKernelContext::ValidateInputsAreSameShape(OpKernel* op) {
  const auto& inputs = *params_->inputs;
  for (size_t i = 1; i < inputs.size(); ++i) {
    if (!inputs[0]->IsSameSize(*(inputs[i].tensor))) {
      SetStatus(errors::InvalidArgument(
          "Inputs to operation ", op->name(), " of type ", op->type_string(),
          " must have the same size and shape.  Input 0: ",
          inputs[0]->shape().DebugString(), " != input ", i, ": ",
          inputs[i]->shape().DebugString()));
      return false;
    }
  }
  return true;
}

Status OpKernelContext::MatchSignature(const DataTypeSlice expected_inputs,
                                       const DataTypeSlice expected_outputs) {
  DataTypeVector inputs;
  for (const TensorValue& t : *params_->inputs) {
    inputs.push_back(t.is_ref() ? MakeRefType(t->dtype()) : t->dtype());
  }
  DataTypeVector outputs = params_->op_kernel->output_types();
  return MatchSignatureHelper(expected_inputs, expected_outputs, inputs,
                              outputs);
}

void OpKernelContext::record_temp_memory_allocation(int64 size,
                                                    const Tensor& t) {
  mutex_lock l(stats_mu_);
  temp_memory_allocated_ += size;
  if (!temp_tensor_buffer_and_size_) {
    temp_tensor_buffer_and_size_.reset(
        new gtl::InlinedVector<std::pair<const void*, int64>, 2>());
  }
  temp_tensor_buffer_and_size_->emplace_back(
      static_cast<const void*>(t.tensor_data().data()), size);
}

int64 OpKernelContext::temp_memory_allocated() const {
  mutex_lock l(stats_mu_);
  return temp_memory_allocated_;
}

void OpKernelContext::record_persistent_memory_allocation(int64 size,
                                                          int64 alloc_id) {
  mutex_lock l(stats_mu_);
  persistent_memory_allocated_ += size;
  if (alloc_id >= 0) {
    if (!persistent_alloc_ids_) {
      persistent_alloc_ids_.reset(new gtl::InlinedVector<int64, 2>());
    }
    persistent_alloc_ids_->push_back(alloc_id);
  }
}

int64 OpKernelContext::persistent_memory_allocated() const {
  mutex_lock l(stats_mu_);
  return persistent_memory_allocated_;
}

std::vector<int64> OpKernelContext::persistent_alloc_ids() const {
  mutex_lock l(stats_mu_);
  if (persistent_alloc_ids_) {
    return std::vector<int64>(persistent_alloc_ids_->begin(),
                              persistent_alloc_ids_->end());
  } else {
    return std::vector<int64>();
  }
}

void OpKernelContext::clear_recorded_memory() {
  mutex_lock l(stats_mu_);
  temp_memory_allocated_ = 0;
  persistent_memory_allocated_ = 0;
  if (temp_tensor_buffer_and_size_) {
    temp_tensor_buffer_and_size_->clear();
  }
  if (persistent_alloc_ids_) {
    persistent_alloc_ids_->clear();
  }
}

// OpKernel registration ------------------------------------------------------

struct KernelRegistration {
  KernelRegistration(const KernelDef& d, StringPiece c,
                     std::unique_ptr<kernel_factory::OpKernelFactory> f)
      : def(d), kernel_class_name(c), factory(std::move(f)) {}

  const KernelDef def;
  const string kernel_class_name;
  std::unique_ptr<kernel_factory::OpKernelFactory> factory;
};

// This maps from 'op_type' + DeviceType to the set of KernelDefs and
// factory functions for instantiating the OpKernel that matches the
// KernelDef.
typedef std::unordered_multimap<string, KernelRegistration> KernelRegistry;

#if defined(_WIN32)
static const char kKernelLibPattern[] = "libtfkernel*.dll";
#elif defined(__APPLE__)
static const char kKernelLibPattern[] = "libtfkernel*.dylib";
#else
static const char kKernelLibPattern[] = "libtfkernel*.so";
#endif

#define FEATURE(x) \
  { x, #x }

// Returns Status::OK if the dynamic library at the given path is safe to
// load with some level of confidence.
static Status IsProbablySafeToLoad(const string& path) {
  // A map of platform string to required CPU feature.
  using port::CPUFeature;
  static const auto* feature_map =
      new std::map<string, std::pair<CPUFeature, string>>{
          {"__AVX512VL__=1", FEATURE(CPUFeature::AVX512VL)},
      };

  std::vector<std::string> platform_strings;
  int result = GetPlatformStrings(path, &platform_strings);
  if (result) {
    return Status(error::Code::UNKNOWN, strerror(result));
  }
  if (platform_strings.empty()) {
    return Status(error::Code::FAILED_PRECONDITION,
                  "Didn't find any platform strings");
  }
  std::vector<std::string> missing_features;
  for (const auto& platform_string : platform_strings) {
    const auto& entry = feature_map->find(platform_string);
    if (entry != feature_map->end() &&
        !port::TestCPUFeature(entry->second.first)) {
      missing_features.emplace_back(entry->second.second);
    }
  }
  if (!missing_features.empty()) {
    string errmsg = "Missing CPU features: ";
    errmsg.append(str_util::Join(missing_features, ", "));
    return Status(errors::Code::FAILED_PRECONDITION, errmsg);
  }
  return Status::OK();
}

void LoadDynamicKernelsInternal() {
  Env* env = Env::Default();
  string bazel_kernel_dir = io::JoinPath(env->GetRunfilesDir(),
                                         "tensorflow",
                                         "core",
                                         "kernels");
  std::vector<string> files;
  Status s_kernel_dir = env->GetChildren(bazel_kernel_dir, &files);
  if (s_kernel_dir.ok()) {
    string dll_spec = io::JoinPath(bazel_kernel_dir, kKernelLibPattern);
    for (const auto& file : files) {
      string fullpath = io::JoinPath(bazel_kernel_dir, file);
      if (env->MatchPath(fullpath, dll_spec)) {
        Status s = IsProbablySafeToLoad(fullpath);
        if (s.ok()) {
          // TODO(gunan): Store the handles to the opened files.
          void* unused_filehandle;
          TF_CHECK_OK(env->LoadLibrary(fullpath.c_str(), &unused_filehandle));
        } else {
          LOG(WARNING) << "Not loading plugin library " << fullpath << ": "
                       << s.error_message();
        }
      }
    }
  }
}

// Mechanism for loading existing kernel libraries.
void LoadDynamicKernels() {
  // TODO(gunan): As more features are available, add intelligent kernel
  // selection, and dropping unsuitable kernel logic here.
  static std::once_flag dll_loader_flag;
  std::call_once(dll_loader_flag, LoadDynamicKernelsInternal);
}

void* GlobalKernelRegistry() {
  static KernelRegistry* global_kernel_registry = new KernelRegistry;
  return global_kernel_registry;
}

static KernelRegistry* GlobalKernelRegistryTyped() {
#ifdef AUTOLOAD_DYNAMIC_KERNELS
  LoadDynamicKernels();
#endif  // AUTOLOAD_DYNAMIC_KERNELS
  return reinterpret_cast<KernelRegistry*>(GlobalKernelRegistry());
}

static string Key(StringPiece op_type, const DeviceType& device_type,
                  StringPiece label) {
  return strings::StrCat(op_type, ":", DeviceTypeString(device_type), ":",
                         label);
}

namespace kernel_factory {

void OpKernelRegistrar::InitInternal(const KernelDef* kernel_def,
                                     StringPiece kernel_class_name,
                                     std::unique_ptr<OpKernelFactory> factory) {
  // See comments in register_kernel::Name in header for info on _no_register.
  if (kernel_def->op() != "_no_register") {
    const string key =
        Key(kernel_def->op(), DeviceType(kernel_def->device_type()),
            kernel_def->label());

    // To avoid calling LoadDynamicKernels DO NOT CALL GlobalKernelRegistryTyped
    // here.
    // InitInternal gets called by static initializers, so it ends up executing
    // before main. This causes LoadKernelLibraries function to get called
    // before some file libraries can initialize, which in turn crashes the
    // program flakily. Until we get rid of static initializers in kernel
    // registration mechanism, we have this workaround here.
    reinterpret_cast<KernelRegistry*>(GlobalKernelRegistry())
        ->emplace(key, KernelRegistration(*kernel_def, kernel_class_name,
                                          std::move(factory)));
  }
  delete kernel_def;
}

}  // namespace kernel_factory

namespace {

static const StringPiece kKernelAttr("_kernel");

// TODO(irving): Replace with const Node& version below.
Status FindKernelRegistration(const DeviceType& device_type,
                              const NodeDef& node_def,
                              const KernelRegistration** reg,
                              bool* was_attr_mismatch) {
  *reg = nullptr;
  *was_attr_mismatch = false;
  // Label defaults to empty if not found in NodeDef.
  const string& label = GetNodeAttrString(node_def, kKernelAttr);

  const string key = Key(node_def.op(), device_type, label);
  auto regs = GlobalKernelRegistryTyped()->equal_range(key);
  for (auto iter = regs.first; iter != regs.second; ++iter) {
    // If there is a kernel registered for the op and device_type,
    // check that the attrs match.
    bool match;
    TF_RETURN_IF_ERROR(KernelAttrsMatch(iter->second.def, node_def, &match));
    if (match) {
      if (*reg != nullptr) {
        return errors::InvalidArgument(
            "Multiple OpKernel registrations match NodeDef '",
            FormatNodeDefForError(node_def), "': '",
            ProtoShortDebugString((*reg)->def), "' and '",
            ProtoShortDebugString(iter->second.def), "'");
      }
      *reg = &iter->second;
    } else {
      *was_attr_mismatch = true;
    }
  }
  return Status::OK();
}

}  // namespace

bool KernelDefAvailable(const DeviceType& device_type,
                        const NodeDef& node_def) {
  const KernelRegistration* reg = nullptr;
  bool was_attr_mismatch;
  Status result =
      FindKernelRegistration(device_type, node_def, &reg, &was_attr_mismatch);
  return result.ok() && reg != nullptr;
}

// TODO(irving): Change const NodeDef& to const Node&
Status FindKernelDef(const DeviceType& device_type, const NodeDef& node_def,
                     const KernelDef** def, string* kernel_class_name) {
  const KernelRegistration* reg = nullptr;
  bool was_attr_mismatch;
  TF_RETURN_IF_ERROR(
      FindKernelRegistration(device_type, node_def, &reg, &was_attr_mismatch));
  if (reg == nullptr) {
    Status s = errors::NotFound(
        "No registered '", node_def.op(), "' OpKernel for ",
        DeviceTypeString(device_type), " devices compatible with node ",
        FormatNodeDefForError(node_def));
    if (was_attr_mismatch) {
      errors::AppendToMessage(
          &s, " (OpKernel was found, but attributes didn't match) ",
          "Requested Attributes: ", SummarizeAttrs(node_def));
    }
    errors::AppendToMessage(
        &s, ".  Registered:", KernelsRegisteredForOp(node_def.op()));
    return s;
  }
  if (def != nullptr) *def = &reg->def;
  if (kernel_class_name != nullptr) *kernel_class_name = reg->kernel_class_name;
  return Status::OK();
}

Status SupportedDeviceTypesForNode(
    const std::vector<DeviceType>& prioritized_types, const NodeDef& def,
    PrioritizedDeviceTypeVector* prioritized_device_types) {
  // TODO(zhifengc): Changes the callers (SimplePlacer and
  // DynamicPlacer) to consider the possibility that 'def' is call to
  // a user-defined function and only calls this
  // SupportedDeviceTypesForNode for primitive ops.
  const OpRegistrationData* op_reg_data;
  const Status s = OpRegistry::Global()->LookUp(def.op(), &op_reg_data);
  if (s.ok()) {
    for (const DeviceType& device_type : prioritized_types) {
      const KernelRegistration* reg = nullptr;
      bool was_attr_mismatch;
      TF_RETURN_IF_ERROR(
          FindKernelRegistration(device_type, def, &reg, &was_attr_mismatch));
      if (reg != nullptr) {
        int32 priority = reg->def.priority();
        prioritized_device_types->emplace_back(device_type, priority);
      }
    }
    std::sort(prioritized_device_types->begin(),
              prioritized_device_types->end(),
              [](const std::pair<DeviceType, int32>& a,
                 const std::pair<DeviceType, int32>& b) {
                return a.second > b.second;
              });
  } else {
    // Assumes that all device types support this node.
    for (const DeviceType& device_type : prioritized_types) {
      prioritized_device_types->push_back(std::make_pair(device_type, 0));
    }
  }
  return Status::OK();
}

void LogAllRegisteredKernels() {
  KernelList kernel_list = GetAllRegisteredKernels();
  for (const auto& kernel_def : kernel_list.kernel()) {
    LOG(INFO) << "OpKernel ('" << ProtoShortDebugString(kernel_def) << "')";
  }
}

KernelList GetAllRegisteredKernels() {
  return GetFilteredRegisteredKernels([](const KernelDef& k) { return true; });
}

KernelList GetFilteredRegisteredKernels(
    const std::function<bool(const KernelDef&)>& predicate) {
  const KernelRegistry* const typed_registry = GlobalKernelRegistryTyped();
  KernelList kernel_list;
  kernel_list.mutable_kernel()->Reserve(typed_registry->size());
  for (const auto& p : *typed_registry) {
    const KernelDef& kernel_def = p.second.def;
    if (predicate(kernel_def)) {
      *kernel_list.add_kernel() = kernel_def;
    }
  }
  return kernel_list;
}

KernelList GetRegisteredKernelsForOp(StringPiece op_name) {
  auto op_pred = [op_name](const KernelDef& k) { return k.op() == op_name; };
  return GetFilteredRegisteredKernels(op_pred);
}

string KernelsRegisteredForOp(StringPiece op_name) {
  KernelList kernel_list = GetRegisteredKernelsForOp(op_name);
  if (kernel_list.kernel_size() == 0) return "  <no registered kernels>\n";
  string ret;
  for (const auto& kernel_def : kernel_list.kernel()) {
    strings::StrAppend(&ret, "  device='", kernel_def.device_type(), "'");
    if (!kernel_def.label().empty()) {
      strings::StrAppend(&ret, "; label='", kernel_def.label(), "'");
    }
    for (int i = 0; i < kernel_def.constraint_size(); ++i) {
      strings::StrAppend(
          &ret, "; ", kernel_def.constraint(i).name(), " in ",
          SummarizeAttrValue(kernel_def.constraint(i).allowed_values()));
    }
    strings::StrAppend(&ret, "\n");
  }
  return ret;
}

std::unique_ptr<OpKernel> CreateOpKernel(
    DeviceType device_type, DeviceBase* device, Allocator* allocator,
    const NodeDef& node_def, int graph_def_version, Status* status) {
  OpKernel* kernel = nullptr;
  *status = CreateOpKernel(std::move(device_type), device, allocator, nullptr,
                           node_def, graph_def_version, &kernel);
  return std::unique_ptr<OpKernel>(kernel);
}

Status CreateOpKernel(DeviceType device_type, DeviceBase* device,
                      Allocator* allocator, FunctionLibraryRuntime* flib,
                      const NodeDef& node_def, int graph_def_version,
                      OpKernel** kernel) {
  VLOG(1) << "Instantiating kernel for node: " << SummarizeNodeDef(node_def);

  // Look up the Op registered for this op name.
  const OpDef* op_def = nullptr;
  Status s = OpRegistry::Global()->LookUpOpDef(node_def.op(), &op_def);
  if (!s.ok()) return s;

  // Validate node_def against OpDef.
  s = ValidateNodeDef(node_def, *op_def);
  if (!s.ok()) return s;

  // Look up kernel registration.
  const KernelRegistration* registration;
  bool was_attr_mismatch;
  s = FindKernelRegistration(device_type, node_def, &registration,
                             &was_attr_mismatch);
  if (!s.ok()) {
    errors::AppendToMessage(&s, " when instantiating ", node_def.op());
    return s;
  }
  if (registration == nullptr) {
    s.Update(errors::NotFound("No registered '", node_def.op(),
                              "' OpKernel for ", DeviceTypeString(device_type),
                              " devices compatible with node ",
                              FormatNodeDefForError(node_def)));
    if (was_attr_mismatch) {
      errors::AppendToMessage(
          &s, " (OpKernel was found, but attributes didn't match) ",
          "Requested Attributes: ", SummarizeAttrs(node_def));
    }
    errors::AppendToMessage(
        &s, ".  Registered:", KernelsRegisteredForOp(node_def.op()));
    return s;
  }

  // Get signature from the OpDef & NodeDef
  DataTypeVector inputs;
  DataTypeVector outputs;
  s.Update(InOutTypesForNode(node_def, *op_def, &inputs, &outputs));
  if (!s.ok()) {
    errors::AppendToMessage(&s, " for node: ", FormatNodeDefForError(node_def));
    return s;
  }

  // We are creating a kernel for an op registered in
  // OpRegistry::Global(), we consult the kernel registry to decide
  // the kernel's input and output memory types.
  MemoryTypeVector input_memory_types;
  MemoryTypeVector output_memory_types;
  TF_RETURN_IF_ERROR(MemoryTypesForNode(OpRegistry::Global(), device_type,
                                        node_def, &input_memory_types,
                                        &output_memory_types));

  // Everything needed for OpKernel construction.
  OpKernelConstruction context(
      device_type, device, allocator, &node_def, op_def, flib, inputs,
      input_memory_types, outputs, output_memory_types, graph_def_version, &s);
  *kernel = registration->factory->Create(&context);
  if (!s.ok()) {
    delete *kernel;
    *kernel = nullptr;
  }
  return s;
}

namespace {

bool FindArgInOp(StringPiece arg_name,
                 const protobuf::RepeatedPtrField<OpDef::ArgDef>& args) {
  for (const auto& arg : args) {
    if (arg_name == arg.name()) {
      return true;
    }
  }
  return false;
}

}  // namespace

Status ValidateKernelRegistrations(const OpRegistryInterface& op_registry) {
  for (const auto& key_registration : *GlobalKernelRegistryTyped()) {
    const KernelDef& kernel_def(key_registration.second.def);
    const OpRegistrationData* op_reg_data;
    const Status status = op_registry.LookUp(kernel_def.op(), &op_reg_data);
    if (!status.ok()) {
      // TODO(josh11b): Make this a hard error.
      LOG(ERROR) << "OpKernel ('" << ProtoShortDebugString(kernel_def)
                 << "') for unknown op: " << kernel_def.op();
      continue;
    }
    const OpDef& op_def = op_reg_data->op_def;
    for (const auto& host_memory_arg : kernel_def.host_memory_arg()) {
      if (!FindArgInOp(host_memory_arg, op_def.input_arg()) &&
          !FindArgInOp(host_memory_arg, op_def.output_arg())) {
        return errors::InvalidArgument(
            "HostMemory arg '", host_memory_arg,
            "' not found in OpDef: ", SummarizeOpDef(op_def));
      }
    }
  }
  return Status::OK();
}

template <>
const Eigen::ThreadPoolDevice& OpKernelContext::eigen_device() const {
  return eigen_cpu_device();
}

template <>
const Eigen::GpuDevice& OpKernelContext::eigen_device() const {
  return eigen_gpu_device();
}

#ifdef TENSORFLOW_USE_SYCL
template <>
const Eigen::SyclDevice& OpKernelContext::eigen_device() const {
  return eigen_sycl_device();
}
#endif

void OpKernelConstruction::CtxFailure(const Status& s) {
  VLOG(1) << s;
  SetStatus(s);
}

void OpKernelConstruction::CtxFailureWithWarning(const Status& s) {
  LOG(WARNING) << s;
  SetStatus(s);
}

void OpKernelConstruction::CtxFailure(const char* file, int line,
                                      const Status& s) {
  VLOG(1) << "OP_REQUIRES failed at " << io::Basename(file) << ":" << line
          << " : " << s;
  SetStatus(s);
}

void OpKernelConstruction::CtxFailureWithWarning(const char* file, int line,
                                                 const Status& s) {
  LOG(WARNING) << "OP_REQUIRES failed at " << io::Basename(file) << ":" << line
               << " : " << s;
  SetStatus(s);
}

void OpKernelContext::CtxFailure(const Status& s) {
  VLOG(1) << s;
  SetStatus(s);
}

void OpKernelContext::CtxFailureWithWarning(const Status& s) {
  LOG(WARNING) << s;
  SetStatus(s);
}

void OpKernelContext::CtxFailure(const char* file, int line, const Status& s) {
  VLOG(1) << "OP_REQUIRES failed at " << io::Basename(file) << ":" << line
          << " : " << s;
  SetStatus(s);
}

void OpKernelContext::CtxFailureWithWarning(const char* file, int line,
                                            const Status& s) {
  LOG(WARNING) << "OP_REQUIRES failed at " << io::Basename(file) << ":" << line
               << " : " << s;
  SetStatus(s);
}

void CheckNotInComputeAsync(OpKernelContext* ctx,
                            const char* correct_macro_name) {
  CHECK_EQ(nullptr, ctx->op_kernel().AsAsync())
      << "Use " << correct_macro_name << " in AsyncOpKernel implementations.";
}

}  // namespace tensorflow
