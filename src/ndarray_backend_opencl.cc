#define __CL_ENABLE_EXCEPTIONS
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
	// case CL_INVALID_PIPE_SIZE:                  return "CL_INVALID_PIPE_SIZE";
	// case CL_INVALID_DEVICE_QUEUE:               return "CL_INVALID_DEVICE_QUEUE";

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

void debug_kernel_build(std::string source) {
  // call this function in the module space at the bottom and pass the kernel
  // string code into it to debug.
  // Note: the moment nd.opencl() is called anywhere in the test code, the
  //       runs which means we need to comment out nd.opencl() in the _DEVICES
  //       definition to be able to test the kernel building.
  const cl::Program program(source);
  try {
    program.build();
  } catch (cl::Error err) {
    std::string logs;
    program.getBuildInfo(cl::Device::getDefault(), CL_PROGRAM_BUILD_LOG, &logs);
    std::cerr
        << "ERROR: "
        << err.what()
        << "("
        << GetOpenCLErrorInfo(err.err())
        << "), "
        << "("
        << logs
        << ")"
        << std::endl;
    throw std::runtime_error(GetOpenCLErrorInfo(err.err()));
  }
}

namespace needle {
namespace opencl {

// CL_DEVICE_MAX_WORK_GROUP_SIZE = 4100
#define TILE 8
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
    this->mem = cl::Buffer(
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
  OpenCLArray(const std::vector<uint32_t>& x) {
    std::vector<float> floatVec(x.begin(), x.end());
    cl_int status_code;
    this->mem = cl::Buffer(
      floatVec.begin(),
      floatVec.end(),
      true, // read only
      NULL,
      &status_code
    );
    if (status_code != CL_SUCCESS) {
      throw std::runtime_error(GetOpenCLErrorInfo(status_code));
    }
    this->size = x.size();
  }
  cl::Buffer mem;
  size_t size;
};

#define MAX_THREAD_SIZE 256

struct OpenCLDims {
  OpenCLDims(const size_t size) {
    // this->global = cl::NDRange(CL_DEVICE_MAX_WORK_GROUP_SIZE, 1, 1);
    // this->local = cl::NDRange((size + CL_DEVICE_MAX_WORK_GROUP_SIZE - 1) / CL_DEVICE_MAX_WORK_GROUP_SIZE, 1, 1);
    this->global = cl::NDRange(((size + MAX_THREAD_SIZE - 1) / MAX_THREAD_SIZE) * MAX_THREAD_SIZE, 1, 1);
    this->local = cl::NDRange(MAX_THREAD_SIZE, 1, 1);
  }
  cl::NDRange global;
  cl::NDRange local;
};

#define MAX_VEC_SIZE 8
struct OpenCLVec {
  unsigned int size;
  unsigned int data[MAX_VEC_SIZE];
};

OpenCLVec VecToOpenCL(const std::vector<uint32_t>& x) {
  OpenCLVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded OpenCL supported max dimesions");
  shape.size = (unsigned int)x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = (unsigned int)x[i];
  }
  return shape;
}


// // Alternative fill implementation that is likely faster to the functor
// // approach I took the below approach to learn how to set up and debug OpenCL
// // KernelFunctors since it would be much easier to get fill working, it would
// // be interesting to benchmark these two approaches against each other.
// void Fill(OpenCLArray* out, scalar_t val) {
//   /**
//    * Fill the values of an aligned array with val
//    */
//   cl_int queue_status;
//   cl::CommandQueue queue = cl::CommandQueue::getDefault(&queue_status);
//   if (queue_status != CL_SUCCESS) {
//     throw std::runtime_error(GetOpenCLErrorInfo(queue_status));
//   }
//   cl_int fill_status = queue.enqueueFillBuffer(
//     *(out->mem),
//     val,
//     0,
//     out->size * ELEM_SIZE,
//     NULL
//   );
//   if (fill_status != CL_SUCCESS) {
//     throw std::runtime_error(GetOpenCLErrorInfo(fill_status));
//   }
// }

std::string fill_source =
"__kernel void fill(__global float* out, float val, unsigned int size) {"
"  size_t gid = get_global_id(0);"
"  if (gid < size) out[gid] = val;"
"}";
const cl::Program fill_program(fill_source, true);
auto fill = cl::make_kernel<cl::Buffer, float, unsigned int>(
  fill_program, "fill"
);
void Fill(OpenCLArray* out, float val) {
  OpenCLDims dims(out->size);
  cl::EnqueueArgs eargs(dims.global, dims.local);
  fill(eargs, out->mem, val, (unsigned int)out->size).wait();
}

std::string compact_source =
"__kernel void compact(__global float* a, __global float* out, unsigned int size,"
"                      __global const float* shape, unsigned int shape_size,"
"                      __global const float* strides, unsigned int strides_size,"
"                      unsigned int offset) {"
"  size_t gid = get_global_id(0);"
// shape_prod is not set to size in the case of
// offset, size is just the full size of the output array
// ALSO: you cannot have comments in kernel string code
"  size_t shape_prod = 1;"
"  for (size_t i=0; i<shape_size; i++) {"
"    shape_prod*=shape[i];"
"  }"
"  size_t prod = shape_prod;"
"  if (gid < shape_prod) {"
"    size_t a_idx = offset;"
"    size_t remainder = gid;"
"    for(size_t j=0; j<shape_size; j++){"
"      prod /= shape[j];"
"      a_idx+=(remainder/prod)*strides[j];"
"      remainder %= prod;"
"    }"
"    out[gid] = a[a_idx];"
"  }"
"}";
const cl::Program compact_program(compact_source, true);
auto compact = cl::make_kernel<
  cl::Buffer, cl::Buffer, unsigned int,
  const cl::Buffer, unsigned int,
  const cl::Buffer, unsigned int,
  unsigned int
>(compact_program, "compact");
void Compact(OpenCLArray* a, OpenCLArray* out, std::vector<uint32_t> shape,
              std::vector<uint32_t> strides, size_t offset) {
  OpenCLDims dims(out->size);
  const OpenCLArray shape_cl(shape);
  const OpenCLArray stride_cl(strides);
  cl::EnqueueArgs eargs(dims.global, dims.local);
  try {
    compact(
      eargs, a->mem, out->mem, (unsigned int)(out->size),
      shape_cl.mem, (unsigned int)(shape_cl.size),
      stride_cl.mem, (unsigned int)(stride_cl.size),
      (unsigned int)offset
    ).wait();
  } catch (cl::Error err) {
    std::cerr
        << "ERROR: "
        << err.what()
        << "("
        << GetOpenCLErrorInfo(err.err())
        << ")"
        << std::endl;
    throw std::runtime_error(GetOpenCLErrorInfo(err.err()));
  }
}

std::string ewisesetitem_source =
"__kernel void ewisesetitem(__global float* a, __global float* out, unsigned int size,"
"                      __global const float* shape, unsigned int shape_size,"
"                      __global const float* strides, unsigned int strides_size,"
"                      unsigned int offset) {"
"  size_t gid = get_global_id(0);"
// shape_prod is not set to size in the case of
// offset, size is just the full size of the output array
// ALSO: you cannot have comments in kernel string code
"  size_t shape_prod = 1;"
"  for (size_t i=0; i<shape_size; i++) {"
"    shape_prod*=shape[i];"
"  }"
"  size_t prod = shape_prod;"
"  if (gid < shape_prod) {"
"    size_t a_idx = offset;"
"    size_t remainder = gid;"
"    for(size_t j=0; j<shape_size; j++){"
"      prod /= shape[j];"
"      a_idx+=(remainder/prod)*strides[j];"
"      remainder %= prod;"
"    }"
"    out[a_idx] = a[gid];"
"  }"
"}";
const cl::Program ewisesetitem_program(ewisesetitem_source, true);
auto ewisesetitem = cl::make_kernel<
  cl::Buffer, cl::Buffer, unsigned int,
  const cl::Buffer, unsigned int,
  const cl::Buffer, unsigned int,
  unsigned int
>(ewisesetitem_program, "ewisesetitem");
void EwiseSetitem(OpenCLArray* a, OpenCLArray* out, std::vector<uint32_t> shape,
              std::vector<uint32_t> strides, size_t offset) {
  OpenCLDims dims(out->size);
  const OpenCLArray shape_cl(shape);
  const OpenCLArray stride_cl(strides);
  cl::EnqueueArgs eargs(dims.global, dims.local);
  try {
    ewisesetitem(
      eargs, a->mem, out->mem, (unsigned int)(out->size),
      shape_cl.mem, (unsigned int)(shape_cl.size),
      stride_cl.mem, (unsigned int)(stride_cl.size),
      (unsigned int)offset
    ).wait();
  } catch (cl::Error err) {
    std::cerr
        << "ERROR: "
        << err.what()
        << "("
        << GetOpenCLErrorInfo(err.err())
        << ")"
        << std::endl;
    throw std::runtime_error(GetOpenCLErrorInfo(err.err()));
  }
}

std::string scalarsetitem_source =
"__kernel void scalarsetitem(__global float* a, float val, unsigned int size,"
"                      __global const float* shape, unsigned int shape_size,"
"                      __global const float* strides, unsigned int strides_size,"
"                      unsigned int offset) {"
"  size_t gid = get_global_id(0);"
// shape_prod is not set to size in the case of
// offset, size is just the full size of the output array
// ALSO: you cannot have comments in kernel string code
"  size_t shape_prod = 1;"
"  for (size_t i=0; i<shape_size; i++) {"
"    shape_prod*=shape[i];"
"  }"
"  size_t prod = shape_prod;"
"  if (gid < shape_prod) {"
"    size_t a_idx = offset;"
"    size_t remainder = gid;"
"    for(size_t j=0; j<shape_size; j++){"
"      prod /= shape[j];"
"      a_idx+=(remainder/prod)*strides[j];"
"      remainder %= prod;"
"    }"
"    a[a_idx] = val;"
"  }"
"}";
const cl::Program scalarsetitem_program(scalarsetitem_source, true);
auto scalarsetitem = cl::make_kernel<
  cl::Buffer, float, unsigned int,
  const cl::Buffer, unsigned int,
  const cl::Buffer, unsigned int,
  unsigned int
>(scalarsetitem_program, "scalarsetitem");
void ScalarSetitem(size_t size, scalar_t val, OpenCLArray* a, std::vector<uint32_t> shape,
              std::vector<uint32_t> strides, size_t offset) {
  OpenCLDims dims(size);
  const OpenCLArray shape_cl(shape);
  const OpenCLArray stride_cl(strides);
  cl::EnqueueArgs eargs(dims.global, dims.local);
  try {
    scalarsetitem(
      eargs, a->mem, (float)val, (unsigned int)(size),
      shape_cl.mem, (unsigned int)(shape_cl.size),
      stride_cl.mem, (unsigned int)(stride_cl.size),
      (unsigned int)offset
    ).wait();
  } catch (cl::Error err) {
    std::cerr
        << "ERROR: "
        << err.what()
        << "("
        << GetOpenCLErrorInfo(err.err())
        << ")"
        << std::endl;
    throw std::runtime_error(GetOpenCLErrorInfo(err.err()));
  }
}


std::string ewiseadd_source =
"__kernel void ewiseadd(__global float* a, __global float* b, __global float* out, unsigned int size) {"
"  size_t gid = get_global_id(0);"
"  if (gid < size) out[gid] = a[gid] + b[gid];"
"}";
const cl::Program ewiseadd_program(ewiseadd_source, true);
auto ewiseadd = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, unsigned int>(
  ewiseadd_program, "ewiseadd"
);
void EwiseAdd(OpenCLArray* a, OpenCLArray* b, OpenCLArray* out) {
  OpenCLDims dims(out->size);
  cl::EnqueueArgs eargs(dims.global, dims.local);

  std::cout << "Max: " << CL_DEVICE_MAX_WORK_GROUP_SIZE << std::endl;

  try {
    ewiseadd(eargs, a->mem, b->mem, out->mem, (unsigned int)out->size).wait();
  } catch (cl::Error err) {
    std::cerr
        << "ERROR: "
        << err.what()
        << "("
        << GetOpenCLErrorInfo(err.err())
        << ")"
        << std::endl;
    throw std::runtime_error(GetOpenCLErrorInfo(err.err()));
  }
}


std::string scalaradd_source =
"__kernel void scalaradd(__global float* a, float val, __global float* out, unsigned int size) {"
"  size_t gid = get_global_id(0);"
"  if (gid < size) out[gid] = a[gid] + val;"
"}";
const cl::Program scalaradd_program(scalaradd_source, true);
auto scalaradd = cl::make_kernel<cl::Buffer, float, cl::Buffer, unsigned int>(
  scalaradd_program, "scalaradd"
);
void ScalarAdd(OpenCLArray* a, float val, OpenCLArray* out) {
  OpenCLDims dims(out->size);
  cl::EnqueueArgs eargs(dims.global, dims.local);
  scalaradd(eargs, a->mem, val, out->mem, (unsigned int)out->size).wait();
}


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
std::string ewisemul_source =
"__kernel void ewisemul(__global float* a, __global float* b, __global float* out, unsigned int size) {"
"  size_t gid = get_global_id(0);"
"  if (gid < size) out[gid] = a[gid] * b[gid];"
"}";
const cl::Program ewisemul_program(ewisemul_source, true);
auto ewisemul = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, unsigned int>(
  ewisemul_program, "ewisemul"
);
void EwiseMul(OpenCLArray* a, OpenCLArray* b, OpenCLArray* out) {
  OpenCLDims dims(out->size);
  cl::EnqueueArgs eargs(dims.global, dims.local);
  ewisemul(eargs, a->mem, b->mem, out->mem, (unsigned int)out->size).wait();
}


std::string scalarmul_source =
"__kernel void scalarmul(__global float* a, float val, __global float* out, unsigned int size) {"
"  size_t gid = get_global_id(0);"
"  if (gid < size) out[gid] = a[gid] * val;"
"}";
const cl::Program scalarmul_program(scalarmul_source, true);
auto scalarmul = cl::make_kernel<cl::Buffer, float, cl::Buffer, unsigned int>(
  scalarmul_program, "scalarmul"
);
void ScalarMul(OpenCLArray* a, float val, OpenCLArray* out) {
  OpenCLDims dims(out->size);
  cl::EnqueueArgs eargs(dims.global, dims.local);
  scalarmul(eargs, a->mem, val, out->mem, (unsigned int)out->size).wait();
}


std::string ewisediv_source =
"__kernel void ewisediv(__global float* a, __global float* b, __global float* out, unsigned int size) {"
"  size_t gid = get_global_id(0);"
"  if (gid < size) out[gid] = a[gid] / b[gid];"
"}";
const cl::Program ewisediv_program(ewisediv_source, true);
auto ewisediv = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, unsigned int>(
  ewisediv_program, "ewisediv"
);
void EwiseDiv(OpenCLArray* a, OpenCLArray* b, OpenCLArray* out) {
  OpenCLDims dims(out->size);
  cl::EnqueueArgs eargs(dims.global, dims.local);
  ewisediv(eargs, a->mem, b->mem, out->mem, (unsigned int)out->size).wait();
}


std::string scalardiv_source =
"__kernel void scalardiv(__global float* a, float val, __global float* out, unsigned int size) {"
"  size_t gid = get_global_id(0);"
"  if (gid < size) out[gid] = a[gid] / val;"
"}";
const cl::Program scalardiv_program(scalardiv_source, true);
auto scalardiv = cl::make_kernel<cl::Buffer, float, cl::Buffer, unsigned int>(
  scalardiv_program, "scalardiv"
);
void ScalarDiv(OpenCLArray* a, float val, OpenCLArray* out) {
  OpenCLDims dims(out->size);
  cl::EnqueueArgs eargs(dims.global, dims.local);
  scalardiv(eargs, a->mem, val, out->mem, (unsigned int)out->size).wait();
}


std::string scalarpower_source =
"__kernel void scalarpower(__global float* a, float val, __global float* out, unsigned int size) {"
"  size_t gid = get_global_id(0);"
"  if (gid < size) out[gid] = pow(a[gid], val);"
"}";
const cl::Program scalarpower_program(scalarpower_source, true);
auto scalarpower = cl::make_kernel<cl::Buffer, float, cl::Buffer, unsigned int>(
  scalarpower_program, "scalarpower"
);
void ScalarPower(OpenCLArray* a, float val, OpenCLArray* out) {
  OpenCLDims dims(out->size);
  cl::EnqueueArgs eargs(dims.global, dims.local);
  scalarpower(eargs, a->mem, val, out->mem, (unsigned int)out->size).wait();
}


std::string ewisemaximum_source =
"__kernel void ewisemaximum(__global float* a, __global float* b, __global float* out, unsigned int size) {"
"  size_t gid = get_global_id(0);"
"  if (gid < size) out[gid] = max(a[gid], b[gid]);"
"}";
const cl::Program ewisemaximum_program(ewisemaximum_source, true);
auto ewisemaximum = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, unsigned int>(
  ewisemaximum_program, "ewisemaximum"
);
void EwiseMaximum(OpenCLArray* a, OpenCLArray* b, OpenCLArray* out) {
  OpenCLDims dims(out->size);
  cl::EnqueueArgs eargs(dims.global, dims.local);
  ewisemaximum(eargs, a->mem, b->mem, out->mem, (unsigned int)out->size).wait();
}


std::string scalarmaximum_source =
"__kernel void scalarmaximum(__global float* a, float val, __global float* out, unsigned int size) {"
"  size_t gid = get_global_id(0);"
"  if (gid < size) out[gid] = max(a[gid], val);"
"}";
const cl::Program scalarmaximum_program(scalarmaximum_source, true);
auto scalarmaximum = cl::make_kernel<cl::Buffer, float, cl::Buffer, unsigned int>(
  scalarmaximum_program, "scalarmaximum"
);
void ScalarMaximum(OpenCLArray* a, float val, OpenCLArray* out) {
  OpenCLDims dims(out->size);
  cl::EnqueueArgs eargs(dims.global, dims.local);
  scalarmaximum(eargs, a->mem, val, out->mem, (unsigned int)out->size).wait();
}


std::string ewiseeq_source =
"__kernel void ewiseeq(__global float* a, __global float* b, __global float* out, unsigned int size) {"
"  size_t gid = get_global_id(0);"
"  if (gid < size) out[gid] = a[gid] == b[gid];"
"}";
const cl::Program ewiseeq_program(ewiseeq_source, true);
auto ewiseeq = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, unsigned int>(
  ewiseeq_program, "ewiseeq"
);
void EwiseEq(OpenCLArray* a, OpenCLArray* b, OpenCLArray* out) {
  OpenCLDims dims(out->size);
  cl::EnqueueArgs eargs(dims.global, dims.local);
  ewiseeq(eargs, a->mem, b->mem, out->mem, (unsigned int)out->size).wait();
}


std::string scalareq_source =
"__kernel void scalareq(__global float* a, float val, __global float* out, unsigned int size) {"
"  size_t gid = get_global_id(0);"
"  if (gid < size) out[gid] = a[gid] == val;"
"}";
const cl::Program scalareq_program(scalareq_source, true);
auto scalareq = cl::make_kernel<cl::Buffer, float, cl::Buffer, unsigned int>(
  scalareq_program, "scalareq"
);
void ScalarEq(OpenCLArray* a, float val, OpenCLArray* out) {
  OpenCLDims dims(out->size);
  cl::EnqueueArgs eargs(dims.global, dims.local);
  scalareq(eargs, a->mem, val, out->mem, (unsigned int)out->size).wait();
}


std::string ewisege_source =
"__kernel void ewisege(__global float* a, __global float* b, __global float* out, unsigned int size) {"
"  size_t gid = get_global_id(0);"
"  if (gid < size) out[gid] = a[gid] >= b[gid];"
"}";
const cl::Program ewisege_program(ewisege_source, true);
auto ewisege = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, unsigned int>(
  ewisege_program, "ewisege"
);
void EwiseGe(OpenCLArray* a, OpenCLArray* b, OpenCLArray* out) {
  OpenCLDims dims(out->size);
  cl::EnqueueArgs eargs(dims.global, dims.local);
  ewisege(eargs, a->mem, b->mem, out->mem, (unsigned int)out->size).wait();
}


std::string scalarge_source =
"__kernel void scalarge(__global float* a, float val, __global float* out, unsigned int size) {"
"  size_t gid = get_global_id(0);"
"  if (gid < size) out[gid] = a[gid] >= val;"
"}";
const cl::Program scalarge_program(scalarge_source, true);
auto scalarge = cl::make_kernel<cl::Buffer, float, cl::Buffer, unsigned int>(
  scalarge_program, "scalarge"
);
void ScalarGe(OpenCLArray* a, float val, OpenCLArray* out) {
  OpenCLDims dims(out->size);
  cl::EnqueueArgs eargs(dims.global, dims.local);
  scalarge(eargs, a->mem, val, out->mem, (unsigned int)out->size).wait();
}


std::string ewiselog_source =
"__kernel void ewiselog(__global float* a, __global float* out, unsigned int size) {"
"  size_t gid = get_global_id(0);"
"  if (gid < size) out[gid] = log(a[gid]);"
"}";
const cl::Program ewiselog_program(ewiselog_source, true);
auto ewiselog = cl::make_kernel<cl::Buffer, cl::Buffer, unsigned int>(
  ewiselog_program, "ewiselog"
);
void EwiseLog(OpenCLArray* a, OpenCLArray* out) {
  OpenCLDims dims(out->size);
  cl::EnqueueArgs eargs(dims.global, dims.local);
  ewiselog(eargs, a->mem, out->mem, (unsigned int)out->size).wait();
}


std::string ewiseexp_source =
"__kernel void ewiseexp(__global float* a, __global float* out, unsigned int size) {"
"  size_t gid = get_global_id(0);"
"  if (gid < size) out[gid] = exp(a[gid]);"
"}";
const cl::Program ewiseexp_program(ewiseexp_source, true);
auto ewiseexp = cl::make_kernel<cl::Buffer, cl::Buffer, unsigned int>(
  ewiseexp_program, "ewiseexp"
);
void EwiseExp(OpenCLArray* a, OpenCLArray* out) {
 OpenCLDims dims(out->size);
  cl::EnqueueArgs eargs(dims.global, dims.local);
  ewiseexp(eargs, a->mem, out->mem, (unsigned int)out->size).wait();
}


std::string ewisetanh_source =
"__kernel void ewisetanh(__global float* a, __global float* out, unsigned int size) {"
"  size_t gid = get_global_id(0);"
"  if (gid < size) out[gid] = tanh(a[gid]);"
"}";
const cl::Program ewisetanh_program(ewisetanh_source, true);
auto ewisetanh = cl::make_kernel<cl::Buffer, cl::Buffer, unsigned int>(
  ewisetanh_program, "ewisetanh"
);
void EwiseTanh(OpenCLArray* a, OpenCLArray* out) {
  OpenCLDims dims(out->size);
  cl::EnqueueArgs eargs(dims.global, dims.local);
  ewisetanh(eargs, a->mem, out->mem, (unsigned int)out->size).wait();
}
/// END YOUR SOLUTION

std::string matmul_source =
"__kernel void matmul(__global float* a, __global float* b,"
"                     __global float* out, unsigned int size, unsigned int M,"
"                     unsigned int N, unsigned int P) {"
"  size_t gid = get_global_id(0);"
// Naieve implementation
"  size_t prod = size;"
"  if (gid < size) {"
"    size_t remainder = gid;"
"    prod /= M;"
"    size_t row = remainder/prod;"
"    remainder %= prod;"
"    prod /= P;"
"    size_t column = remainder/prod;"
"    float sum = 0;"
"    for(size_t i=0; i < N; i++) {"
"      sum += a[row*N + i] * b[column + P*i];"
"    }"
"    out[gid] = sum;"
"  }"
"}";
const cl::Program matmul_program(matmul_source, true);
auto matmul = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, unsigned int,
                              unsigned int, unsigned int, unsigned int>(
  matmul_program, "matmul"
);
void Matmul(OpenCLArray* a, OpenCLArray* b, OpenCLArray* out, unsigned int M,
            unsigned int N, unsigned int P) {
  OpenCLDims dims(out->size);
  cl::EnqueueArgs eargs(dims.global, dims.local);
  matmul(
    eargs, a->mem, b->mem, out->mem, (unsigned int)out->size, M, N, P
  ).wait();
}


std::string reducemax_source =
"__kernel void reducemax(__global float* a, __global float* out, unsigned int size, unsigned int reduce_size) {"
"  size_t gid = get_global_id(0);"
"  float result = 0.0f;"
"  if (gid < size)  {"
"    result = a[gid*reduce_size];"
"    for (size_t j = 1; j < reduce_size; j++)  {"
"      result = max(result, a[gid*reduce_size + j]);"
"    }"
"    out[gid] = result;"
"  }"
"}";
const cl::Program reducemax_program(reducemax_source, true);
auto reducemax = cl::make_kernel<cl::Buffer, cl::Buffer, unsigned int, unsigned int>(
  reducemax_program, "reducemax"
);
void ReduceMax(OpenCLArray* a, OpenCLArray* out, unsigned int reduce_size) {
  OpenCLDims dims(out->size);
  cl::EnqueueArgs eargs(dims.global, dims.local);
  reducemax(eargs, a->mem, out->mem, (unsigned int)out->size, reduce_size).wait();
}


std::string reducesum_source =
"__kernel void reducesum(__global float* a, __global float* out, unsigned int size, unsigned int reduce_size) {"
"  size_t gid = get_global_id(0);"
"  float result = 0.0f;"
"  if (gid < size)  {"
"    result = a[gid*reduce_size];"
"    for (size_t j = 1; j < reduce_size; j++)  {"
"      result = result + a[gid*reduce_size + j];"
"    }"
"    out[gid] = result;"
"  }"
"}";
const cl::Program reducesum_program(reducesum_source, true);
auto reducesum = cl::make_kernel<cl::Buffer, cl::Buffer, unsigned int, unsigned int>(
  reducesum_program, "reducesum"
);
void ReduceSum(OpenCLArray* a, OpenCLArray* out, unsigned int reduce_size) {
  OpenCLDims dims(out->size);
  cl::EnqueueArgs eargs(dims.global, dims.local);
  reducesum(eargs, a->mem, out->mem, (unsigned int)out->size, reduce_size).wait();
}


// // Simple setup to debug a Kernel argument definition
// cl::Kernel kernel_fill=cl::Kernel(fill_program, "fill");
// kernel_fill.setArg(0, *(out->mem));
// kernel_fill.setArg(1, val);
// kernel_fill.setArg(2, (unsigned int)out->size);
// try {
//   queue.enqueueNDRangeKernel(kernel_fill, cl::NullRange, dims.global, dims.local);
//   queue.finish();
// } catch (cl::Error err) {
//   std::cerr
//       << "ERROR: "
//       << err.what()
//       << "("
//       << GetOpenCLErrorInfo(err.err())
//       << ")"
//       << std::endl;
//   throw std::runtime_error(GetOpenCLErrorInfo(err.err()));
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
      .def_readonly("size", &OpenCLArray::size);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const OpenCLArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * sizeof(float); });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    try {
      cl_int status_code = cl::CommandQueue::getDefault().enqueueReadBuffer(
        a.mem,
        CL_TRUE,  // blocking
        0,
        a.size * ELEM_SIZE,
        host_ptr
      );
    } catch (cl::Error err) {
      std::cerr
          << "ERROR: "
          << err.what()
          << "("
          << GetOpenCLErrorInfo(err.err())
          << ")"
          << std::endl;
      throw std::runtime_error(GetOpenCLErrorInfo(err.err()));
    }


    // if (status_code != CL_SUCCESS) {
    //   std::cout << "Entered" << std::endl;
    //   std::cerr
    //       << GetOpenCLErrorInfo(status_code)
    //       << std::endl;
    //   throw std::runtime_error(GetOpenCLErrorInfo(status_code));
    // }

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
      out->mem,
      CL_TRUE,  // blocking
      0,
      out->size * ELEM_SIZE,
      a.request().ptr
    );
    if (status_code != CL_SUCCESS) {
      throw std::runtime_error(GetOpenCLErrorInfo(status_code));
    }
  });

  debug_kernel_build(scalarmaximum_source);

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
