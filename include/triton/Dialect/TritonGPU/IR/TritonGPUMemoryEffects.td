#ifndef TRITON_GPU_MEMORY_EFFECTS
#define TRITON_GPU_MEMORY_EFFECTS

include "mlir/Interfaces/SideEffectInterfaces.td"

def SharedMemory : Resource<"::mlir::triton::gpu::SharedMemory">;
// Upstream attaches the shared-memory kind as a resource parameter, which
// needs LLVM b010a18d ("[mlir][ODS] Allow parameters on side-effect
// resources"). The LLVM 23.1 TableGen this fork builds with predates it, so
// each kind is a sub-resource of SharedMemory instead; see SharedMemoryEffect
// in Dialect.h. Drop this once the shared LLVM carries b010a18d.
def GenericSharedMemory : Resource<"::mlir::triton::gpu::GenericSharedMemory">;
def AsyncSharedMemory : Resource<"::mlir::triton::gpu::AsyncSharedMemory">;
def BarrierSharedMemory : Resource<"::mlir::triton::gpu::BarrierSharedMemory">;

defvar GenericSharedRead = [MemRead<GenericSharedMemory>];
defvar GenericSharedWrite = [MemWrite<GenericSharedMemory>];
defvar GenericSharedReadWrite =
    !listconcat(GenericSharedRead, GenericSharedWrite);
defvar AsyncSharedRead = [MemRead<AsyncSharedMemory>];
defvar AsyncSharedWrite = [MemWrite<AsyncSharedMemory>];
defvar BarrierSharedRead = [MemRead<BarrierSharedMemory>];
defvar BarrierSharedWrite = [MemWrite<BarrierSharedMemory>];
defvar BarrierSharedReadWrite =
    !listconcat(BarrierSharedRead, BarrierSharedWrite);

#endif
