#include <CL/cl.h>

#include<iostream>

#include <time.h>

#include<string>

#include<fstream>

#pragma comment (lib,"OpenCL.lib")

#define NDEVS  1

using namespace std;

cl_int ConvertToString(const char *pFileName, std::string &Str)
{
	size_t		uiSize = 0;

	size_t		uiFileSize = 0;

	char		*pStr = NULL;

	std::fstream fFile(pFileName, (std::fstream::in | std::fstream::binary));

	if (fFile.is_open())
	{
		fFile.seekg(0, std::fstream::end);

		uiSize = uiFileSize = (size_t)fFile.tellg();  // 获得文件大小

		fFile.seekg(0, std::fstream::beg);

		pStr = new char[uiSize + 1];

		if (NULL == pStr)

		{	fFile.close();

			return 0;
		}

	fFile.read(pStr, uiFileSize);				// 读取uiFileSize字节

	fFile.close();

	pStr[uiSize] = '\0';

	Str = pStr;

	delete[] pStr;

	return 0;
	}

	cout << "Error: Failed to open cl file\n:" << pFileName << endl;
	
	return -1;
}
// A parallel min() kernel that works well on CPU and GPU
int main()
{
		cl_platform_id   platform;

		int              dev;

		cl_device_type   devs[NDEVS] = { CL_DEVICE_TYPE_GPU };

		cl_uint          *src_ptr;
	
		unsigned int     num_src_items = 4096 * 4096;

		// 1. quick & dirty MWC random init of source buffer.
	
		// Random seed (portable).
	
		time_t ltime;

		time(&ltime);

		src_ptr = (cl_uint *)malloc(num_src_items * sizeof(cl_uint));
	
		cl_uint a = (cl_uint)ltime;
	
		cl_uint b = (cl_uint)ltime;

		cl_uint min = b;

		// Do serial computation of min() for result verification.
	
	for (int i = 0; i < num_src_items; i++)
	
	{		src_ptr[i] = (cl_uint)(b = (a * (b & 65535)) + (b >> 16));
		
			min = src_ptr[i] < min ? src_ptr[i] : min;
	}
	
		// 2. Tell compiler to dump intermediate .il and .isa GPU files.
		
		_putenv("GPU_DUMP_DEVICE_KERNEL=3");
	
		// Get a platform.
	
		clGetPlatformIDs(1, &platform, NULL);
		// 3. Iterate over devices.

	for (dev = 0; dev < NDEVS; dev++)
	
	{
	
			cl_device_id     device;
		
			cl_context       context;
	
			cl_command_queue queue;

			cl_program       program;

			cl_kernel        minp;
	
			cl_kernel        reduce;
	
			cl_mem           src_buf;
	
			cl_mem           dst_buf;
	
			cl_mem           dbg_buf;
	
			cl_uint          *dst_ptr, * dbg_ptr;
			// Find the device.
			clGetDeviceIDs(platform,
	
			devs[dev],
		
			1,
	
			& device,
		
			NULL);
	
			// 4. Compute work sizes.

			cl_uint compute_units;
	
			size_t  global_work_size;
	
			size_t  local_work_size;
		
			size_t  num_groups;
	
			clGetDeviceInfo(device,
			
			CL_DEVICE_MAX_COMPUTE_UNITS,
		
			sizeof(cl_uint),
			
			& compute_units,
			
			NULL);
		
		if (devs[dev] == CL_DEVICE_TYPE_CPU)
			
		{
		
				global_work_size = compute_units * 1;      // 1 thread per core
		
				local_work_size = 1;
			
		}
		
		else
		
		{
		
				cl_uint ws = 64;
			
				global_work_size = compute_units * 7 * ws; // 7 wavefronts per SIMD
			
			while ((num_src_items / 4) % global_work_size != 0)
			
			{
			
					printf("global_work_size = %d\n", global_work_size);
			
					global_work_size += ws;
				
			}
		
				local_work_size = ws;
		
		}
	
			num_groups = global_work_size / local_work_size;
	
			printf("\nglobal_work_size =%d, local_work_size=%d, num_groups = %d\n", global_work_size, local_work_size, num_groups);
		
			// Create a context and command queue on that device.
		
			context = clCreateContext(NULL,
			
			1,
			
			& device,
			
			NULL, NULL, NULL);
	
			queue = clCreateCommandQueue(context,
			
			device,
			
			0, NULL);
	
			// Minimal error check.
			
		if (queue == NULL)
			
		{
			
				printf("Compute device setup failed\n");
		
				return(-1);
			
		};
		
		// Perform runtime source compilation, and obtain kernel entry point.
		string strSource;

		const char *pSource;

		ConvertToString("kernel.cl", strSource);

		pSource = strSource.c_str();			// 获得strSource指针

		size_t uiArrSourceSize = strlen(pSource);
		
		program = clCreateProgramWithSource(context,
			
			1,
			
			& pSource,
		
			NULL, NULL);
	
			cl_int ret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	
			// 5. Print compiler error messages
		
		if (ret != CL_SUCCESS)
			
		{
			
				printf("clBuildProgram failed: %d\n", ret);
		
				char buf[0x10000];
			
				clGetProgramBuildInfo(program,
				
				device,
			
				CL_PROGRAM_BUILD_LOG,
			
				0x10000,
				
				buf,
				
				NULL);
			
				printf("\n%s\n", buf);
			
				return(-1);
			
		}
	
			minp = clCreateKernel(program, "minp", NULL);
	
			reduce = clCreateKernel(program, "reduce", NULL);
	
			// Create input, output and debug buffers.
		
			src_buf = clCreateBuffer(context,
			
			CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			
			num_src_items * sizeof(cl_uint),
			
			src_ptr,
			
			NULL);
		
			dst_buf = clCreateBuffer(context,
		
			CL_MEM_READ_WRITE,
		
			num_groups * sizeof(cl_uint),
			
			NULL, NULL);
		
			dbg_buf = clCreateBuffer(context,
		
			CL_MEM_WRITE_ONLY,
		
			global_work_size * sizeof(cl_uint),
			
			NULL, NULL);
	
			clSetKernelArg(minp, 0, sizeof(void *), (void*)&src_buf);
	
			clSetKernelArg(minp, 1, sizeof(void *), (void*)&dst_buf);
		
			clSetKernelArg(minp, 2, 1 * sizeof(cl_uint), (void*)NULL);
	
			clSetKernelArg(minp, 3, sizeof(void *), (void*)&dbg_buf);
		
			clSetKernelArg(minp, 4, sizeof(num_src_items), (void*)&num_src_items);
	
			clSetKernelArg(minp, 5, sizeof(dev), (void*)&dev);
	
			clSetKernelArg(reduce, 0, sizeof(void *), (void*)&src_buf);
		
			clSetKernelArg(reduce, 1, sizeof(void *), (void*)&dst_buf);
		
	
		
			// 6. Main timing loop.
			
#define NLOOPS 500
		
			cl_event ev;
	
			int nloops = NLOOPS;
		
		while (nloops--)
			
		{
		
				clEnqueueNDRangeKernel(queue,
			
				minp,
			
				1,
				
				NULL,
			
				& global_work_size,
				
				& local_work_size,
			
				0, NULL, &ev);
		
				clEnqueueNDRangeKernel(queue,
			
				reduce,
			
				1,
				
				NULL,
				
				& num_groups,
				
				NULL, 1, &ev, NULL);
		
		}
	
			clFinish(queue);
	
		
	
			printf("B/W %.2f GB/sec, ", ((float)num_src_items *

				sizeof(cl_uint)* NLOOPS));
		
		

			// 7. Look at the results via synchronous buffer map.
		
			dst_ptr = (cl_uint *)clEnqueueMapBuffer(queue,
	
			dst_buf,
	
			CL_TRUE,
	
			CL_MAP_READ,
		
			0,
		
			num_groups * sizeof(cl_uint),
		
			0, NULL, NULL, NULL);
	
			dbg_ptr = (cl_uint *)clEnqueueMapBuffer(queue,
		
			dbg_buf,
		
			CL_TRUE,
	
			CL_MAP_READ,
		
			0,
	
			global_work_size *
	
			sizeof(cl_uint),
		
			0, NULL, NULL, NULL);
	
			// 8. Print some debug info.
		
			printf("%d groups, %d threads, count %d, stride %d\n", dbg_ptr[0],
		
			dbg_ptr[1],
		
			dbg_ptr[2],
		
			dbg_ptr[3]);
		
		if (dst_ptr[0] == min)
			
			printf("result correct\n");
	
		else
	
		printf("result INcorrect\n");
	
	}
	
		printf("\n");

		return 0;
	
}