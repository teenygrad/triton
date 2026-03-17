# SPIRV Tools is required to disassemble the spirv module
find_package(SPIRV-Tools CONFIG REQUIRED)
find_package(SPIRV-Tools-opt CONFIG REQUIRED)
find_package(SPIRV-Tools-link CONFIG REQUIRED)

add_triton_library(TritonSPIRV
        SPIRVTranslation.cpp

        # LINK_COMPONENTS
        # Core

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRLLVMIRTransforms
        MLIRSupport
        MLIRTargetLLVMIRExport
        )

# Link SPIRV-Tools as external imported targets after library creation
# (cannot use LINK_LIBS for imported targets with add_mlir_translation_library)
target_link_libraries(TritonSPIRV PRIVATE SPIRV-Tools SPIRV-Tools-link)
