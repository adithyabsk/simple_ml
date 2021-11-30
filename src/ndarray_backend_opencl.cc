// #include <CL/cl2.hpp>
// #include <CL/cl.h>
#include <CL/cl.hpp>
// Note: macos has a different opencl location
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <numeric>
#include <string>


std::string GetOpenCLErrorName(cl_int errorCode)
{
	switch (errorCode)
	{
	case CL_SUCCESS:                            return "CL_SUCCESS";
	case CL_DEVICE_NOT_FOUND:                   return "CL_DEVICE_NOT_FOUND";
	case CL_DEVICE_NOT_AVAILABLE:               return "CL_DEVICE_NOT_AVAILABLE";
	case CL_COMPILER_NOT_AVAILABLE:             return "CL_COMPILER_NOT_AVAILABLE";
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case CL_OUT_OF_RESOURCES:                   return "CL_OUT_OF_RESOURCES";
	case CL_OUT_OF_HOST_MEMORY:                 return "CL_OUT_OF_HOST_MEMORY";
	case CL_PROFILING_INFO_NOT_AVAILABLE:       return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case CL_MEM_COPY_OVERLAP:                   return "CL_MEM_COPY_OVERLAP";
	case CL_IMAGE_FORMAT_MISMATCH:              return "CL_IMAGE_FORMAT_MISMATCH";
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case CL_BUILD_PROGRAM_FAILURE:              return "CL_BUILD_PROGRAM_FAILURE";
	case CL_MAP_FAILURE:                        return "CL_MAP_FAILURE";
	case CL_MISALIGNED_SUB_BUFFER_OFFSET:       return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
	case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:    return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
	case CL_COMPILE_PROGRAM_FAILURE:            return "CL_COMPILE_PROGRAM_FAILURE";
	case CL_LINKER_NOT_AVAILABLE:               return "CL_LINKER_NOT_AVAILABLE";
	case CL_LINK_PROGRAM_FAILURE:               return "CL_LINK_PROGRAM_FAILURE";
	case CL_DEVICE_PARTITION_FAILED:            return "CL_DEVICE_PARTITION_FAILED";
	case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:      return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
	case CL_INVALID_VALUE:                      return "CL_INVALID_VALUE";
	case CL_INVALID_DEVICE_TYPE:                return "CL_INVALID_DEVICE_TYPE";
	case CL_INVALID_PLATFORM:                   return "CL_INVALID_PLATFORM";
	case CL_INVALID_DEVICE:                     return "CL_INVALID_DEVICE";
	case CL_INVALID_CONTEXT:                    return "CL_INVALID_CONTEXT";
	case CL_INVALID_QUEUE_PROPERTIES:           return "CL_INVALID_QUEUE_PROPERTIES";
	case CL_INVALID_COMMAND_QUEUE:              return "CL_INVALID_COMMAND_QUEUE";
	case CL_INVALID_HOST_PTR:                   return "CL_INVALID_HOST_PTR";
	case CL_INVALID_MEM_OBJECT:                 return "CL_INVALID_MEM_OBJECT";
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case CL_INVALID_IMAGE_SIZE:                 return "CL_INVALID_IMAGE_SIZE";
	case CL_INVALID_SAMPLER:                    return "CL_INVALID_SAMPLER";
	case CL_INVALID_BINARY:                     return "CL_INVALID_BINARY";
	case CL_INVALID_BUILD_OPTIONS:              return "CL_INVALID_BUILD_OPTIONS";
	case CL_INVALID_PROGRAM:                    return "CL_INVALID_PROGRAM";
	case CL_INVALID_PROGRAM_EXECUTABLE:         return "CL_INVALID_PROGRAM_EXECUTABLE";
	case CL_INVALID_KERNEL_NAME:                return "CL_INVALID_KERNEL_NAME";
	case CL_INVALID_KERNEL_DEFINITION:          return "CL_INVALID_KERNEL_DEFINITION";
	case CL_INVALID_KERNEL:                     return "CL_INVALID_KERNEL";
	case CL_INVALID_ARG_INDEX:                  return "CL_INVALID_ARG_INDEX";
	case CL_INVALID_ARG_VALUE:                  return "CL_INVALID_ARG_VALUE";
	case CL_INVALID_ARG_SIZE:                   return "CL_INVALID_ARG_SIZE";
	case CL_INVALID_KERNEL_ARGS:                return "CL_INVALID_KERNEL_ARGS";
	case CL_INVALID_WORK_DIMENSION:             return "CL_INVALID_WORK_DIMENSION";
	case CL_INVALID_WORK_GROUP_SIZE:            return "CL_INVALID_WORK_GROUP_SIZE";
	case CL_INVALID_WORK_ITEM_SIZE:             return "CL_INVALID_WORK_ITEM_SIZE";
	case CL_INVALID_GLOBAL_OFFSET:              return "CL_INVALID_GLOBAL_OFFSET";
	case CL_INVALID_EVENT_WAIT_LIST:            return "CL_INVALID_EVENT_WAIT_LIST";
	case CL_INVALID_EVENT:                      return "CL_INVALID_EVENT";
	case CL_INVALID_OPERATION:                  return "CL_INVALID_OPERATION";
	case CL_INVALID_GL_OBJECT:                  return "CL_INVALID_GL_OBJECT";
	case CL_INVALID_BUFFER_SIZE:                return "CL_INVALID_BUFFER_SIZE";
	case CL_INVALID_MIP_LEVEL:                  return "CL_INVALID_MIP_LEVEL";
	case CL_INVALID_GLOBAL_WORK_SIZE:           return "CL_INVALID_GLOBAL_WORK_SIZE";
	case CL_INVALID_PROPERTY:                   return "CL_INVALID_PROPERTY";
	case CL_INVALID_IMAGE_DESCRIPTOR:           return "CL_INVALID_IMAGE_DESCRIPTOR";
	case CL_INVALID_COMPILER_OPTIONS:           return "CL_INVALID_COMPILER_OPTIONS";
	case CL_INVALID_LINKER_OPTIONS:             return "CL_INVALID_LINKER_OPTIONS";
	case CL_INVALID_DEVICE_PARTITION_COUNT:     return "CL_INVALID_DEVICE_PARTITION_COUNT";
	case CL_INVALID_PIPE_SIZE:                  return "CL_INVALID_PIPE_SIZE";
	case CL_INVALID_DEVICE_QUEUE:               return "CL_INVALID_DEVICE_QUEUE";

	default:
		return "UNKNOWN ERROR CODE";
	}
}

// from http://www.techdarting.com/2014/01/opencl-errors.html
std::string GetOpenCLErrorDescription(cl_int err) {
	std::string result = "";

	switch (err) {
		case CL_SUCCESS: result += "Everything is good!"; break;
		case CL_DEVICE_NOT_FOUND: result += "No OpenCL devices that matched given device type were found"; break;
		case CL_DEVICE_NOT_AVAILABLE: result += "No OpenCL compatible device was found"; break;
		case CL_COMPILER_NOT_AVAILABLE: result += "OpenCL Compiler perhaps failed to configure itself, or check your OpenCL installation"; break;
		case CL_MEM_OBJECT_ALLOCATION_FAILURE: result += "Failed to allocate memory for buffer object"; break;
		case CL_OUT_OF_RESOURCES: result += "failure to allocate resources required by the OpenCL implementation on the device"; break;
		case CL_OUT_OF_HOST_MEMORY: result += "failure to allocate resources required by the OpenCL implementation on the host"; break;
		case CL_PROFILING_INFO_NOT_AVAILABLE: result += "returned by clGetEventProfilingInfo, if the CL_QUEUE_PROFILING_ENABLE flag is not set for the command-queue and if the profiling information is currently not available"; break;
		case CL_MEM_COPY_OVERLAP: result += "if source and destination buffers are the same buffer object and the source and destination regions overlap"; break;
		case CL_IMAGE_FORMAT_MISMATCH: result += "src and dst image do not use the same image format"; break;
		case CL_IMAGE_FORMAT_NOT_SUPPORTED: result += "the image format is not supported."; break;
		case CL_BUILD_PROGRAM_FAILURE: result += "program build error for given device, Use clGetProgramBuildInfo API call to get the build log of the kernel compilation."; break;
		case CL_MAP_FAILURE: result += "failed to map the requested region into the host address space. This error does not occur for buffer objects created with CL_MEM_USE_HOST_PTR or CL_MEM_ALLOC_HOST_PTR"; break;
		case CL_MISALIGNED_SUB_BUFFER_OFFSET: result += "no devices in given context associated with buffer for which the origin value is aligned to the CL_DEVICE_MEM_BASE_ADDR_ALIGN value"; break;
		case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: result += "returned by clWaitForEvents(), execution status of any of the events in event list is a negative integer value i.e., error"; break;
		case CL_COMPILE_PROGRAM_FAILURE: result += "failed to compile the program source. Error occurs if clCompileProgram does not return until the compile has completed"; break;
		case CL_LINKER_NOT_AVAILABLE: result += "Linker unavailable"; break;
		case CL_LINK_PROGRAM_FAILURE: result += "failed to link the compiled binaries and perhaps libraries"; break;
		case CL_DEVICE_PARTITION_FAILED: result += "given partition name is supported by the implementation but input device couldn't be partitioned further"; break;
		case CL_KERNEL_ARG_INFO_NOT_AVAILABLE: result += "argument information is not available for the given kernel"; break;
		case CL_INVALID_VALUE: result += "values passed in the flags parameter is not valid"; break;
		case CL_INVALID_DEVICE_TYPE: result += "device type specified is not valid, its returned by clCreateContextFromType / clGetDeviceIDs"; break;
		case CL_INVALID_PLATFORM: result += "the specified platform is not a valid platform, its returned by clGetPlatformInfo /clGetDeviceIDs / clCreateContext / clCreateContextFromType"; break;
		case CL_INVALID_DEVICE: result += "device/s specified are not valid"; break;
		case CL_INVALID_CONTEXT: result += "the given context is invalid OpenCL context, or the context associated with certain parameters are not the same"; break;
		case CL_INVALID_QUEUE_PROPERTIES: result += "specified properties are valid but are not supported by the device, its returned by clCreateCommandQueue / clSetCommandQueueProperty"; break;
		case CL_INVALID_COMMAND_QUEUE: result += "the specified command-queue is not a valid command-queue"; break;
		case CL_INVALID_HOST_PTR: result += "host pointer is NULL and CL_MEM_COPY_HOST_PTR or CL_MEM_USE_HOST_PTR are set in flags or if host_ptr is not NULL but CL_MEM_COPY_HOST_PTR or CL_MEM_USE_HOST_PTR are not set in flags. returned by clCreateBuffer / clCreateImage2D / clCreateImage3D"; break;
		case CL_INVALID_MEM_OBJECT: result += "the passed parameter is not a valid memory, image, or buffer object"; break;
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: result += "image format specified is not valid or is NULL, clCreateImage2D /clCreateImage3D returns this."; break;
		case CL_INVALID_IMAGE_SIZE: result += "Its returned by create Image functions 2D/3D, if specified image width or height are outbound or 0"; break;
		case CL_INVALID_SAMPLER: result += "specified sampler is an invalid sampler object"; break;
		case CL_INVALID_BINARY: result += "program binary is not a valid binary for the specified device, returned by clBuildProgram / clCreateProgramWithBinary"; break;
		case CL_INVALID_BUILD_OPTIONS: result += "the given build options are not valid"; break;
		case CL_INVALID_PROGRAM: result += "the given program is an invalid program object, returned by clRetainProgram / clReleaseProgram / clBuildProgram / clGetProgramInfo / clGetProgramBuildInfo / clCreateKernel / clCreateKernelsInProgram"; break;
		case CL_INVALID_PROGRAM_EXECUTABLE: result += "if there is no successfully built executable for program returned by clCreateKernel, there is no device in program then returned by clCreateKernelsInProgram, if no successfully built program executable present for device associated with command queue then returned by clEnqueueNDRangeKernel / clEnqueueTask"; break;
		case CL_INVALID_KERNEL_NAME: result += "mentioned kernel name is not found in program"; break;
		case CL_INVALID_KERNEL_DEFINITION: result += "arguments mismatch for the __kernel function definition and the passed ones, returned by clCreateKernel"; break;
		case CL_INVALID_KERNEL: result += "specified kernel is an invalid kernel object"; break;
		case CL_INVALID_ARG_INDEX: result += "clSetKernelArg if an invalid argument index is specified"; break;
		case CL_INVALID_ARG_VALUE: result += "the argument value specified is NULL, returned by clSetKernelArg"; break;
		case CL_INVALID_ARG_SIZE: result += "the given argument size (arg_size) do not match size of the data type for an argument, returned by clSetKernelArg"; break;
		case CL_INVALID_KERNEL_ARGS: result += "the kernel argument values have not been specified, returned by clEnqueueNDRangeKernel / clEnqueueTask"; break;
		case CL_INVALID_WORK_DIMENSION: result += "given work dimension is an invalid value, returned by clEnqueueNDRangeKernel"; break; 
		case CL_INVALID_WORK_GROUP_SIZE: result += "the specified local workgroup size and number of workitems specified by global workgroup size is not evenly divisible by local workgroup size"; break;
		case CL_INVALID_WORK_ITEM_SIZE: result += "no. of workitems specified in any of local work group sizes is greater than the corresponding values specified by CL_DEVICE_MAX_WORK_ITEM_SIZES in that particular dimension"; break;
		case CL_INVALID_GLOBAL_OFFSET: result += "global_work_offset is not NULL. Must currently be a NULL value. In a future revision of OpenCL, global_work_offset can be used but not until OCL 1.2"; break;
		case CL_INVALID_EVENT_WAIT_LIST: result += "event wait list is NULL and (no. of events in wait list > 0), or event wait list is not NULL and no. of events in wait list is 0, or specified event objects are not valid events"; break;
		case CL_INVALID_EVENT: result += "invalid event objects specified"; break;
		case CL_INVALID_GL_OBJECT: result += "not a valid GL buffer object"; break;
		case CL_INVALID_BUFFER_SIZE: result += "the value of the parameter size is 0 or exceeds CL_DEVICE_MAX_MEM_ALLOC_SIZE for all devices specified in the parameter context, returned by clCreateBuffer"; break;
		case CL_INVALID_GLOBAL_WORK_SIZE: result += "specified global work size is NULL, or any of the values specified in global work dimensions are 0 or exceeds the range given by the sizeof(size_t) for the device on which the kernel will be enqueued, returned by clEnqueueNDRangeKernel"; break;
		case CL_INVALID_PROPERTY: result += "context property name in properties is not a supported property name, returned by clCreateContext"; break;
		case CL_INVALID_IMAGE_DESCRIPTOR: result += "values specified in image description are invalid"; break;
		case CL_INVALID_COMPILER_OPTIONS: result += "compiler options specified by options are invalid, returned by clCompileProgram"; break;
		case CL_INVALID_LINKER_OPTIONS: result += "linker options specified by options are invalid, returned by clLinkProgram"; break;
		default: result += "No description available"; break;
	}

	return result;
}

std::string GetOpenCLErrorInfo(cl_int err) {
	return "Error " + GetOpenCLErrorName(err) + " (" + std::to_string((int)err) + ")\nDescription: " + GetOpenCLErrorDescription(err);
}

// https://stackoverflow.com/a/57336786/3262054
std::vector<cl::Device> devices;
int find_devices() {
    std::vector<cl::Platform> platforms; // get all platforms
    std::vector<cl::Device> devices_available;
    int n = 0; // number of available devices
    cl::Platform::get(&platforms);
    for(int i=0; i<(int)platforms.size(); i++) {
        devices_available.clear();
        platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices_available);
        if(devices_available.size()==0) continue; // no device found in plattform i
        for(int j=0; j<(int)devices_available.size(); j++) {
            n++;
            devices.push_back(devices_available[j]);
        }
    }
    if(platforms.size()==0||devices.size()==0) {
        std::cout << "Error: There are no OpenCL devices available!" << std::endl;
        return -1;
    }
    for(int i=0; i<n; i++) std::cout << "ID: " << i << ", Device: " << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
    return n; // return number of available devices
}

namespace needle {
namespace opencl {

#define ALIGNMENT 256
#define TILE 8
// #define TILE 2  // TODO: REMOVE ME
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);


/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
 * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
 * here by default.
 */
struct OpenCLArray {
  OpenCLArray(const size_t size) {
    cl_int status_code;
    this->mem = new cl::Buffer(
      cl::Context::getDefault(),
      CL_MEM_READ_WRITE,
      size * ELEM_SIZE,
      NULL,
      &status_code
    );
    if (status_code != CL_SUCCESS) {
      throw std::runtime_error(GetOpenCLErrorInfo(status_code));
    }
    this->size = size;
  }
  ~OpenCLArray() { delete(mem); }
  cl::Buffer* mem;
  size_t size;
};

// void Fill(OpenCLArray* out, scalar_t val) {
//   /**
//    * Fill the values of an aligned array with val
//    */
//   for (int i = 0; i < out->size; i++) {
//     out->ptr[i] = val;
//   }
// }

/*
template <typename T>
void PrintVector(std::vector<T> &vec) {
    std::cout << "[ ";
    for (int i = 0; i < vec.size(); i++) {
      std::cout << vec.at(i) << ' ';
    }
    std::cout << " ]" << std::endl;
}

void PrintAlignedArray(const AlignedArray& a) {
    std::cout << "[ ";
    for (size_t i = 0; i < a.size; i++) {
      std::cout << a.ptr[i] << ' ';
    }
    std::cout << " ]" << std::endl;
}

void PrintFloatArray(const float* arr, size_t rows, size_t cols) {
    std::cout << "[ ";
    for(size_t i=0; i < rows; i++) {
      if(i != 0) {
        std::cout << "  ";
      }
      std::cout<<"[ ";
      for(size_t j=0; j<cols; j++) {
        std::cout << arr[i*cols+j] << " ";
      }
      std::cout << " ]";
      if (i != (rows-1)) {
        std::cout << std::endl;
      }
    }
    std::cout << " ]" << std::endl;
}
*/

// void Compact(const OpenCLArray& a, OpenCLArray* out, std::vector<uint32_t> shape,
//              std::vector<uint32_t> strides, size_t offset) {
//   /**
//    * Compact an array in memory
//    *
//    * Args:
//    *   a: non-compact represntation of the array, given as input
//    *   out: compact version of the array to be written
//    *   shape: shapes of each dimension for a and out
//    *   strides: strides of the *a* array (not out, which has compact strides)
//    *   offset: offset of the *a* array (not out, which has zero offset, being compact)
//    *
//    * Returns:
//    *  void (you need to modify out directly, rather than returning anything; this is true for all the
//    *  function will implement here, so we won't repeat this note.)
//    */
//   /// BEGIN YOUR SOLUTION
//   std::vector<uint32_t> counters(shape.size(), 0);
//   auto cnt = 0;
//   auto max_count = std::accumulate(
//     shape.begin(), shape.end(), 1, std::multiplies<uint32_t>()
//   );
//   while (cnt < max_count) {
//     auto prod = std::inner_product(
//       counters.begin(), counters.end(), strides.begin(), 0
//     );
//     out->ptr[cnt++] = a.ptr[offset + prod];

//     // Update the counters
//     auto increment = true;
//     for(int i = counters.size()-1; i >=0; i--){
//       if (increment) {
//         if (++counters[i] == shape[i]) {
//           counters.at(i) = 0;
//           increment = true;
//         } else {
//           increment = false;
//         }
//       }
//     }
//   }
//   /// END YOUR SOLUTION
// }

// void EwiseSetitem(const OpenCLArray& a, OpenCLArray* out, std::vector<uint32_t> shape,
//                   std::vector<uint32_t> strides, size_t offset) {
//   /**
//    * Set items in a (non-compact) array
//    *
//    * Args:
//    *   a: _compact_ array whose items will be written to out
//    *   out: non-compact array whose items are to be written
//    *   shape: shapes of each dimension for a and out
//    *   strides: strides of the *out* array (not a, which has compact strides)
//    *   offset: offset of the *out* array (not a, which has zero offset, being compact)
//    */
//   /// BEGIN YOUR SOLUTION
//   std::vector<uint32_t> counters(shape.size(), 0);
//   auto cnt = 0;
//   auto max_count = std::accumulate(
//     shape.begin(), shape.end(), 1, std::multiplies<uint32_t>()
//   );
//   while (cnt < max_count) {
//     auto prod = std::inner_product(
//       counters.begin(), counters.end(), strides.begin(), 0
//     );
//     out->ptr[offset + prod] = a.ptr[cnt++];

//     // Update the counters
//     auto increment = true;
//     for(int i = counters.size()-1; i >=0; i--){
//       if (increment) {
//         if (++counters[i] == shape[i]) {
//           counters.at(i) = 0;
//           increment = true;
//         } else {
//           increment = false;
//         }
//       }
//     }
//   }
//   /// END YOUR SOLUTION
// }

// void ScalarSetitem(const size_t size, scalar_t val, OpenCLArray* out, std::vector<uint32_t> shape,
//                    std::vector<uint32_t> strides, size_t offset) {
//   /**
//    * Set items is a (non-compact) array
//    *
//    * Args:
//    *   size: number of elements to write in out array (note that this will note be the same as
//    *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
//    *         product of items in shape, but covenient to just pass it here.
//    *   val: scalar value to write to
//    *   out: non-compact array whose items are to be written
//    *   shape: shapes of each dimension of out
//    *   strides: strides of the out array
//    *   offset: offset of the out array
//    */

//   /// BEGIN YOUR SOLUTION
//   std::vector<uint32_t> counters(shape.size(), 0);
//   auto cnt = 0;
//   while (cnt++ < size) {
//     auto prod = std::inner_product(
//       counters.begin(), counters.end(), strides.begin(), 0
//     );
//     out->ptr[offset + prod] = val;

//     // Update the counters
//     auto increment = true;
//     for(int i = counters.size()-1; i >=0; i--){
//       if (increment) {
//         if (++counters[i] == shape[i]) {
//           counters.at(i) = 0;
//           increment = true;
//         } else {
//           increment = false;
//         }
//       }
//     }
//   }
//   /// END YOUR SOLUTION
// }

// void EwiseAdd(const OpenCLArray& a, const OpenCLArray& b, OpenCLArray* out) {
//   /**
//    * Set entries in out to be the sum of correspondings entires in a and b.
//    */
//   for (size_t i = 0; i < a.size; i++) {
//     out->ptr[i] = a.ptr[i] + b.ptr[i];
//   }
// }

// void ScalarAdd(const OpenCLArray& a, scalar_t val, OpenCLArray* out) {
//   /**
//    * Set entries in out to be the sum of correspondings entry in a plus the scalar val.
//    */
//   for (size_t i = 0; i < a.size; i++) {
//     out->ptr[i] = a.ptr[i] + val;
//   }
// }


/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

/// BEGIN YOUR SOLUTION
/*
void EwiseMul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] * b.ptr[i];
  }
}

void ScalarMul(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] * val;
  }
}

void EwiseDiv(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] / b.ptr[i];
  }
}

void ScalarDiv(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] / val;
  }
}

void ScalarPower(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = pow(a.ptr[i], val);
  }
}

void EwiseMaximum(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::max(a.ptr[i], b.ptr[i]);
  }
}

void ScalarMaximum(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::max(a.ptr[i], val);
  }
}

void EwiseEq(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] == b.ptr[i];
  }
}

void ScalarEq(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] == val;
  }
}

void EwiseGe(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] >= b.ptr[i];
  }
}

void ScalarGe(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] >= val;
  }
}

void EwiseLog(const AlignedArray& a, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = log(a.ptr[i]);
  }
}

void EwiseExp(const AlignedArray& a, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = exp(a.ptr[i]);
  }
}

void EwiseTanh(const AlignedArray& a, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = tanh(a.ptr[i]);
  }
}
*/
/// END YOUR SOLUTION

// void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m, uint32_t n,
//             uint32_t p) {
//   /**
//    * Multiply two (compact) matrices into an output (also comapct) matrix.  For this implementation
//    * you can use the "naive" three-loop algorithm.
//    *
//    * Args:
//    *   a: compact 2D array of size m x n
//    *   b: comapct 2D array of size n x p
//    *   out: compact 2D array of size m x p to write the output to
//    *   m: rows of a / out
//    *   n: columns of a / rows of b
//    *   p: coolumns of b / out
//    */

//   /// BEGIN YOUR SOLUTION
//   size_t cnt = 0;
//   for(size_t i = 0; i < m; i++) {
//     for(size_t j = 0; j < p; j++){
//       float sum = 0.0;
//       for(size_t offset = 0; offset < n; offset++) {
//         sum += a.ptr[i*n + offset] * b.ptr[j + p*offset];
//       }
//       out->ptr[cnt++] = sum;
//     }
//   }
//   /// END YOUR SOLUTION
// }

// inline void AlignedDot(const float* __restrict__ a,
//                        const float* __restrict__ b,
//                        float* __restrict__ out) {

//   /**
//    * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
//    * the result to the existing out, which you should not set to zero beforehand).  We are including
//    * the compiler flags here that enable the compile to properly use vector operators to implement
//    * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and
//    * out don't have any overlapping memory (which is necessary in order for vector operations to be
//    * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b,
//    * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the
//    * compiler that the input array siwll be aligned to the appropriate blocks in memory, which also
//    * helps the compiler vectorize the code.
//    *
//    * Args:
//    *   a: compact 2D array of size TILE x TILE
//    *   b: compact 2D array of size TILE x TILE
//    *   out: compact 2D array of size TILE x TILE to write to
//    */

//   a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
//   b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
//   out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

//   // PrintFloatArray(a, TILE, TILE);
//   // PrintFloatArray(b, TILE, TILE);

//   /// BEGIN YOUR SOLUTION
//   size_t cnt = 0;
//   for(size_t i = 0; i < TILE; i++) {
//     for(size_t j = 0; j < TILE; j++){
//       float sum = 0.0;
//       for(size_t offset = 0; offset < TILE; offset++) {
//         // a[i*n + offset] * b[j + p*offset]
//         sum += a[i*TILE + offset] * b[j + TILE*offset];
//       }
//       out[cnt++] += sum;
//     }
//   }
//   // PrintFloatArray(out, TILE, TILE);
//   // std::cout << std::endl;
//   /// END YOUR SOLUTION
// }

// void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m,
//                  uint32_t n, uint32_t p) {
//   /**
//    * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
//    * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
//    *   a[m/TILE][n/TILE][TILE][TILE]
//    * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
//    * function should call `AlignedDot()` implemented above).
//    *
//    * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
//    * assume that this division happens without any remainder.
//    *
//    * Args:
//    *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
//    *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
//    *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
//    *   m: rows of a / out
//    *   n: columns of a / rows of b
//    *   p: columns of b / out
//    *
//    */
//   /// BEGIN YOUR SOLUTION
//   size_t cnt = 0;
//   Fill(out, 0);
//   for(size_t i = 0; i < m/TILE; i++) {
//     for(size_t j = 0; j < p/TILE; j++){
//       for(size_t offset = 0; offset < n/TILE; offset++) {
//         // a[i*n + offset] * b[j + p*offset]
//         // std::cout<<i<<"*"<<n<<"*"<<TILE<<"+"<<offset<<"*"<<TILE<<"*"<<TILE<<std::endl;
//         // std::cout<<j<<"*"<<TILE<<"*"<<TILE<<"+"<<p<<"*"<<offset<<"*"<<TILE<<std::endl;
//         AlignedDot(
//           // i*(n/TILE)*TILE*TILE+offset*TILE*TILE
//           &(a.ptr[i*n*TILE+offset*TILE*TILE]),
//           // j + (p/TILE)*(offset*TILE^2)
//           &(b.ptr[j*TILE*TILE+p*offset*TILE]),
//           &(out->ptr[cnt])
//         );
//       }
//       cnt+=TILE*TILE;
//     }
//   }
//   /// END YOUR SOLUTION
// }

// void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
//   /**
//    * Reduce by taking maximum over `reduce_size` contiguous blocks.
//    *
//    * Args:
//    *   a: compact array of size a.size = out.size * reduce_size to reduce over
//    *   out: compact array to write into
//    *   redice_size: size of the dimension to reduce over
//    */

//   /// BEGIN YOUR SOLUTION
//   auto cnt = 0;
//   for (size_t i = 0; i < a.size; i+=reduce_size) {
//     auto max = a.ptr[i];
//     for (size_t j=1; j < reduce_size; j++) {
//       if (a.ptr[i+j] > max) {
//         max = a.ptr[i+j];
//       }
//     }
//     out->ptr[cnt++] = max;
//   }
//   /// END YOUR SOLUTION
// }

// void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
//   /**
//    * Reduce by taking sum over `reduce_size` contiguous blocks.
//    *
//    * Args:
//    *   a: compact array of size a.size = out.size * reduce_size to reduce over
//    *   out: compact array to write into
//    *   redice_size: size of the dimension to reduce over
//    */

//   /// BEGIN YOUR SOLUTION
//   auto cnt = 0;
//   for (size_t i = 0; i < a.size; i+=reduce_size) {
//     float sum = 0;
//     for (size_t j=0; j < reduce_size; j++) {
//         sum += a.ptr[i+j];
//     }
//     out->ptr[cnt++] = sum;
//   }
//   /// END YOUR SOLUTION
// }

}  // namespace opencl
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_opencl, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace opencl;

  m.attr("__device_name__") = "opencl";
  m.attr("__tile_size__") = TILE;

  py::class_<OpenCLArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      // .def("ptr", &OpenCLArray::ptr_as_int)
      .def_readonly("size", &OpenCLArray::size);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const OpenCLArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    
    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    cl_int status_code = cl::CommandQueue::getDefault().enqueueReadBuffer(
      *(a.mem),
      CL_TRUE,  // blocking
      offset,
      a.size * ELEM_SIZE,
      host_ptr
    );
    if (status_code != CL_SUCCESS)
      throw std::runtime_error(GetOpenCLErrorInfo(status_code));

    // return numpy array
    // https://stackoverflow.com/a/19283829/3262054
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, OpenCLArray* out) {

    cl_int s2;
    cl::CommandQueue queue = cl::CommandQueue::getDefault(&s2);
    if (s2 != CL_SUCCESS) {
      throw std::runtime_error(GetOpenCLErrorInfo(s2));
    }

    cl_int status_code = queue.enqueueWriteBuffer(
      *(out->mem),
      CL_TRUE,  // blocking
      0,
      out->size * ELEM_SIZE,
      a.request().ptr
    );
    if (status_code != CL_SUCCESS) {
      throw std::runtime_error(GetOpenCLErrorInfo(status_code));
    }
  });

  // m.def("fill", Fill);
  // m.def("compact", Compact);
  // m.def("ewise_setitem", EwiseSetitem);
  // m.def("scalar_setitem", ScalarSetitem);
  // m.def("ewise_add", EwiseAdd);
  // m.def("scalar_add", ScalarAdd);

  // m.def("ewise_mul", EwiseMul);
  // m.def("scalar_mul", ScalarMul);
  // m.def("ewise_div", EwiseDiv);
  // m.def("scalar_div", ScalarDiv);
  // m.def("scalar_power", ScalarPower);
  // //
  // m.def("ewise_maximum", EwiseMaximum);
  // m.def("scalar_maximum", ScalarMaximum);
  // m.def("ewise_eq", EwiseEq);
  // m.def("scalar_eq", ScalarEq);
  // m.def("ewise_ge", EwiseGe);
  // m.def("scalar_ge", ScalarGe);
  // //
  // m.def("ewise_log", EwiseLog);
  // m.def("ewise_exp", EwiseExp);
  // m.def("ewise_tanh", EwiseTanh);

  // m.def("matmul", Matmul);
  // m.def("matmul_tiled", MatmulTiled);

  // m.def("reduce_max", ReduceMax);
  // m.def("reduce_sum", ReduceSum);
}