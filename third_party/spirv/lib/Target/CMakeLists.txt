add_subdirectory(SPIRV)

if(TARGET SPIRV-Tools-shared)
  set_target_properties(SPIRV-Tools-shared PROPERTIES EXCLUDE_FROM_ALL ON)
endif()
