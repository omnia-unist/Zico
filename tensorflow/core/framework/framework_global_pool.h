#ifndef TENSORFLOW_CORE_FRAMEWORK_FRAMEWORK_GLOBAL_POOL_H_
#define TENSORFLOW_CORE_FRAMEWORK_FRAMEWORK_GLOBAL_POOL_H_

#include "stdio.h"
#include <vector>
#include <utility>

namespace tensorflow {

//class AllocationRegion;

typedef size_t ChunkHandle;

struct FreeAR {
	int job_id = -1; // AR's last dedicated job id
	size_t region_id = -1; // last region id for this AR
	size_t size_in_b = -1;
	void* memory_ptr = nullptr; // addr of this AR memory space
	// const AllocationRegion* ar_ptr = nullptr;
	// std::unique_ptr<ChunkHandle[]> handles_;
};


class GlobalPool {
	public:

		bool print_globalpool = true;

		size_t num_bytes = 0;

		std::vector<FreeAR> free_ARs_;

	public:
		GlobalPool();

		~GlobalPool();
		
		void FreeARFromBFC(void* memory_ptr, size_t size_in_b, size_t region_id, int job_id);
		//void FreeARFromBFC(const AllocationRegion* ar_ptr, void* memory_ptr, size_t size_in_b, size_t region_id, int job_id);
		//void FreeARFromBFC(std::unique_ptr<ChunkHandle[]> handles_, void* memory_ptr, size_t size_in_b, size_t region_id, int job_id);

		void AddFreeARToGlobalPool(void* memory_ptr, size_t size_in_b, size_t region_id, int job_id);
		//void AddFreeARToGlobalPool(const AllocationRegion* ar_ptr, void* memory_ptr, size_t size_in_b, size_t region_id, int job_id);
		//void AddFreeARToGlobalPool(std::unique_ptr<ChunkHandle[]> handles_, void* memory_ptr, size_t size_in_b, size_t region_id, int job_id);

		FreeAR AllocARToBFC(size_t rounded_bytes);

		FreeAR EraseFreeARFromGlobalPool(size_t rounded_bytes);

		int num_AR();
};

}  // namespace tensorflow

#endif
