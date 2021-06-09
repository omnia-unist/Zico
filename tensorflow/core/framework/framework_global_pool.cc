#include "tensorflow/core/framework/framework_global_pool.h"

namespace tensorflow {

// oh no...
// This was the problem...
// Declaration but no definition anywhere.
// That's why it keeps telling defined reference for GlobalPool() and ~GlobalPool()
//
// GlobalPool();
//
// ~GlobalPool();

GlobalPool::GlobalPool() {
//	if (print_globalpool) {
//		printf("*************** GlobalPool() Constructor. Press any button to continue.\n"); 
////		char tmp = 'a';
////		std::cin >> tmp;
//	}
}

GlobalPool::~GlobalPool() {
//	if (print_globalpool) { 
//		printf("--------------- ~GlobalPool() Destructor. Press any button to continue.\n"); 
//	}
}

/*
	Local BFC ====<AllocationRegion>====> Global Pool
*/

// Wrapper function for BFC Allocator
// This function will be called from local BFC Allocator's GlobalPool instance
// when ACT AR is freed from BFC Allocator.

void GlobalPool::FreeARFromBFC(void* memory_ptr, size_t size_in_b, size_t region_id, int job_id) {
	return GlobalPool::AddFreeARToGlobalPool(memory_ptr, size_in_b, region_id, job_id);
}

void GlobalPool::AddFreeARToGlobalPool(void* memory_ptr, size_t size_in_b, size_t region_id, int job_id) {
	if (print_globalpool) {
		printf("BFC,GlobalPool,%d,id:%zu,%zuMB,%lu\n", job_id, region_id, size_in_b/(1024*1024), memory_ptr);
	}
	// Creating new FreeAR object with given information
	FreeAR new_free_ar;
	new_free_ar.job_id = job_id;
	new_free_ar.size_in_b = size_in_b;
	new_free_ar.memory_ptr = memory_ptr;
	new_free_ar.region_id = region_id;

	// Sorting해서 가지고 있어야하나. 가장 작은 거 중에 fit하는 걸 먼저 return 하기 위해
	// first fit
	for (std::vector<FreeAR>::iterator it = free_ARs_.begin() ; it != free_ARs_.end(); ++it) {
		if (it->size_in_b > size_in_b){
			free_ARs_.insert(it, new_free_ar);
			return;
		}
	}
	free_ARs_.push_back(new_free_ar);
}

// overloading for const AllocationRegion* parameter
//void GlobalPool::FreeARFromBFC(const AllocationRegion* ar_ptr, void* memory_ptr, size_t size_in_b, size_t region_id, int job_id) {
//	return GlobalPool::AddFreeARToGlobalPool(ar_ptr, memory_ptr, size_in_b, region_id, job_id);
//}
//
//void GlobalPool::AddFreeARToGlobalPool(const AllocationRegion* ar_ptr, void* memory_ptr, size_t size_in_b, size_t region_id, int job_id) {
//	if (print_globalpool) {
//		printf("BFC,GlobalPool,%d,id:%zu,%zuMB,%lu\n", job_id, region_id, size_in_b/(1024*1024), memory_ptr);
//	}
//	// Creating new FreeAR object with given information
//	FreeAR new_free_ar;
//	new_free_ar.job_id = job_id;
//	new_free_ar.size_in_b = size_in_b;
//	new_free_ar.memory_ptr = memory_ptr;
//	new_free_ar.region_id = region_id;
//	new_free_ar.ar_ptr = ar_ptr;
//
//	// Sorting해서 가지고 있어야하나. 가장 작은 거 중에 fit하는 걸 먼저 return 하기 위해
//	// first fit
//	for (std::vector<FreeAR>::iterator it = free_ARs_.begin() ; it != free_ARs_.end(); ++it) {
//		if (it->size_in_b > size_in_b){
//			free_ARs_.insert(it, new_free_ar);
//			return;
//		}
//	}
//	free_ARs_.push_back(new_free_ar);
//}

// Overloading for unique_ptr parameter
//void GlobalPool::FreeARFromBFC(std::unique_ptr<ChunkHandle[]> handles_, void* memory_ptr, size_t size_in_b, size_t region_id, int job_id) {
//	return GlobalPool::AddFreeARToGlobalPool(std::move(handles_), memory_ptr, size_in_b, region_id, job_id);
//}
//
//void GlobalPool::AddFreeARToGlobalPool(std::unique_ptr<ChunkHandle[]> handles_, void* memory_ptr, size_t size_in_b, size_t region_id, int job_id) {
//	if (print_globalpool) {
//		printf("BFC,GlobalPool,%d,id:%zu,%zuMB,%lu\n", job_id, region_id, size_in_b/(1024*1024), memory_ptr);
//	}
//	// Creating new FreeAR object with given information
//	FreeAR new_free_ar;
//	new_free_ar.job_id = job_id;
//	new_free_ar.size_in_b = size_in_b;
//	new_free_ar.memory_ptr = memory_ptr;
//	new_free_ar.region_id = region_id;
////	new_free_ar.ar_ptr = ar_ptr;
////	new_free_ar.handles_ = std::move(handles_);
//
//	// Sorting해서 가지고 있어야하나. 가장 작은 거 중에 fit하는 걸 먼저 return 하기 위해
//	// first fit
//	for (std::vector<FreeAR>::iterator it = free_ARs_.begin() ; it != free_ARs_.end(); ++it) {
//		if (it->size_in_b > size_in_b){
//			free_ARs_.insert(it, new_free_ar);
//			return;
//		}
//	}
//	free_ARs_.push_back(new_free_ar);
//}


/*
	Global Pool ====<AllocationRegion>====> Local BFC
*/

// Wrapper function for BFC Allocator
// This function will be called from local BFC Allocator's GlobalPool instance
// when ACT AR is allocated to BFC Allocator.
FreeAR GlobalPool::AllocARToBFC(size_t rounded_bytes) {
	return GlobalPool::EraseFreeARFromGlobalPool(rounded_bytes);
}

FreeAR GlobalPool::EraseFreeARFromGlobalPool(size_t rounded_bytes) {
	if (print_globalpool) {
		printf("$$++ Trying to find Free AR which is bigger than requested bytes: %zu\n", rounded_bytes);
	}
	for (std::vector<FreeAR>::iterator it = free_ARs_.begin() ; it != free_ARs_.end(); ++it) {
		if (it->size_in_b >= rounded_bytes) {
			FreeAR ret_free_ar = *it;
			num_bytes -= it->size_in_b;
			if (print_globalpool) {
				printf("GlobalPool,BFC,%d,id:%zu,%zuMB,%lu\n", it->job_id, it->region_id, it->size_in_b/(1024*1024), it->memory_ptr);
			}
			free_ARs_.erase(it);
			return ret_free_ar;
		}
	}
	// sentinel object for invalid return
	FreeAR null_;
	null_.job_id = -2;
	null_.size_in_b = -2;
	null_.region_id = -2;
	null_.memory_ptr = nullptr;
	if (print_globalpool) {
		printf("$$xx Fail to find FreeAR in GlobalPool. Returning sentinel object...\n");
	}
	return null_;
}

int GlobalPool::num_AR() {
	return free_ARs_.size();
}

} // namespace tensorflow
