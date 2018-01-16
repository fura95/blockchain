#include "bc3.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <memory.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>


#define ROTLEFT(a,b) (((a) << (b)) | ((a) >> (32-(b))))
#define ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (32-(b))))

#define PRINT_ERROR() 								\
do { printf("FAILED with errno = %d!\n\n", errno); exit(-1); } while(0)

#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x,2) ^ ROTRIGHT(x,13) ^ ROTRIGHT(x,22))
#define EP1(x) (ROTRIGHT(x,6) ^ ROTRIGHT(x,11) ^ ROTRIGHT(x,25))
#define SIG0(x) (ROTRIGHT(x,7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x,17) ^ ROTRIGHT(x,19) ^ ((x) >> 10))

#define MAX_SOURCE_SIZE (0x100000)

static const WORD k[64] = {
	0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
	0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
	0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
	0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
	0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
	0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
	0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
	0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

/*********************** OPENCL PART ********************************/
#define CHECK(cond,i) if(cond){ printf("Error on CHECK(%d)\n",i); /*cl_free(ctx, plane); */; return NULL; }
cl_ctx* cl_init() {
  cl_ctx* ctx = (cl_ctx*)malloc(sizeof(cl_ctx));
  if(!ctx) return ctx;
  memset(ctx,0,sizeof(cl_ctx));

  cl_int errcode;
  // Get OpenCL platform
  errcode = clGetPlatformIDs(1, &(ctx->cpPlatform), &(ctx->nPlatform));
  CHECK(errcode!=CL_SUCCESS,1);
  // Get GPU or CPU device
  errcode = clGetDeviceIDs(ctx->cpPlatform, CL_DEVICE_TYPE_CPU, 1, &(ctx->cdDevice), NULL);
  CHECK(errcode!=CL_SUCCESS,2);
  // Create the context
  ctx->cxGPUContext = clCreateContext(0, 1, &(ctx->cdDevice), NULL, NULL, &errcode);
  CHECK(errcode!=CL_SUCCESS,3);
  // Create a command-queue
  ctx->cqCommandQueue = clCreateCommandQueue(ctx->cxGPUContext, ctx->cdDevice, 0 , &errcode);
  CHECK(errcode!=CL_SUCCESS,4);

  FILE *fp;
	const char* source_name = "./bc3.cl";
	char* source_code;
  size_t source_size;
  if((fp = fopen(source_name, "r")) == NULL) {
		fprintf(stderr, "Failed to open the file containing the kernel source !\n");
	        exit(EXIT_FAILURE);
	}
	source_code = (char*) malloc (MAX_SOURCE_SIZE * sizeof(char));
  source_size = fread(source_code, sizeof(char), MAX_SOURCE_SIZE, fp);
	fclose(fp);

  //Create bin
  ctx->cProgram = clCreateProgramWithSource(ctx->cxGPUContext, 1, (const char **) &source_code, (const size_t*) &source_size, &errcode);
  CHECK(errcode!=CL_SUCCESS,5);
  //Compile
  char options[10];
  strcpy(options, "-g -O0");
  errcode = clBuildProgram(ctx->cProgram, 1, &(ctx->cdDevice), options, NULL, NULL); 
  if(errcode == CL_BUILD_PROGRAM_FAILURE) {
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(ctx->cProgram, ctx->cdDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        // Allocate memory for the log
        char *log = (char *) malloc(log_size);
        // Get the log
        clGetProgramBuildInfo(ctx->cProgram, ctx->cdDevice, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        // Print the log
        printf("%s\n", log);
  }
 // STATUSCHKMSG("build");
  CHECK(errcode!=CL_SUCCESS,6);
	//Create Kernels
  ctx->cKernel = clCreateKernel(ctx->cProgram, "cl_find_nonce", &errcode);
  CHECK(errcode!=CL_SUCCESS,7);
  
  return ctx;
}

void cl_free(cl_ctx *ctx) {
	if(!ctx) return;
	clReleaseKernel(ctx->cKernel);
	clReleaseProgram(ctx->cProgram);
	clReleaseCommandQueue(ctx->cqCommandQueue);
	clReleaseContext(ctx->cxGPUContext);
	free(ctx);
  }
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++/


/*********************** FUNCTION DEFINITIONS ***********************/
void sha256_transform(SHA256_CTX *ctx, const BYTE data[])
{
	WORD a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];

	for (i = 0, j = 0; i < 16; ++i, j += 4)
		m[i] = (data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) | (data[j + 3]);
	for ( ; i < 64; ++i)
		m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];

	a = ctx->state[0];
	b = ctx->state[1];
	c = ctx->state[2];
	d = ctx->state[3];
	e = ctx->state[4];
	f = ctx->state[5];
	g = ctx->state[6];
	h = ctx->state[7];

	for (i = 0; i < 64; ++i) {
		t1 = h + EP1(e) + CH(e,f,g) + k[i] + m[i];
		t2 = EP0(a) + MAJ(a,b,c);
		h = g;
		g = f;
		f = e;
		e = d + t1;
		d = c;
		c = b;
		b = a;
		a = t1 + t2;
	}

	ctx->state[0] += a;
	ctx->state[1] += b;
	ctx->state[2] += c;
	ctx->state[3] += d;
	ctx->state[4] += e;
	ctx->state[5] += f;
	ctx->state[6] += g;
	ctx->state[7] += h;
} 


void sha256_init(SHA256_CTX *ctx)
{
	ctx->datalen = 0;
	ctx->bitlen = 0;
	ctx->state[0] = 0x6a09e667;
	ctx->state[1] = 0xbb67ae85;
	ctx->state[2] = 0x3c6ef372;
	ctx->state[3] = 0xa54ff53a;
	ctx->state[4] = 0x510e527f;
	ctx->state[5] = 0x9b05688c;
	ctx->state[6] = 0x1f83d9ab;
	ctx->state[7] = 0x5be0cd19;
}

void sha256_update(SHA256_CTX *ctx, const BYTE data[], size_t len)
{
	WORD i;

	for (i = 0; i < len; ++i) {
		ctx->data[ctx->datalen] = data[i];
		ctx->datalen++;
		if (ctx->datalen == 64) {
			sha256_transform(ctx, ctx->data);
			ctx->bitlen += 512;
			ctx->datalen = 0;
		}
	}
}

void sha256_final(SHA256_CTX *ctx, BYTE hash[])
{
	WORD i;

	i = ctx->datalen;

	if (ctx->datalen < 56) {
		ctx->data[i++] = 0x80;
		while (i < 56)
			ctx->data[i++] = 0x00;
	}
	else {
		ctx->data[i++] = 0x80;
		while (i < 64)
			ctx->data[i++] = 0x00;
		sha256_transform(ctx, ctx->data);
		memset(ctx->data, 0, 56);
	}

	ctx->bitlen += ctx->datalen * 8;
	ctx->data[63] = ctx->bitlen;
	ctx->data[62] = ctx->bitlen >> 8;
	ctx->data[61] = ctx->bitlen >> 16;
	ctx->data[60] = ctx->bitlen >> 24;
	ctx->data[59] = ctx->bitlen >> 32;
	ctx->data[58] = ctx->bitlen >> 40;
	ctx->data[57] = ctx->bitlen >> 48;
	ctx->data[56] = ctx->bitlen >> 56;
	sha256_transform(ctx, ctx->data);
	

	for (i = 0; i < 4; ++i) {
		hash[i]      = (ctx->state[0] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 4]  = (ctx->state[1] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 8]  = (ctx->state[2] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 12] = (ctx->state[3] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 16] = (ctx->state[4] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 20] = (ctx->state[5] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 24] = (ctx->state[6] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 28] = (ctx->state[7] >> (24 - i * 8)) & 0x000000ff;
	}
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void clear_chain() {
	fclose(fopen("block_info/block_file.txt", "w"));
	fclose(fopen("block_info/msg.txt", "w"));
	fclose(fopen("block_info/time.txt", "w"));
	fclose(fopen("time/time_inf.txt", "w"));
	fclose(fopen("time/text_info", "w"));
	zero_count_file();
}

void fill_cond_down(WORD *old_condition, WORD *condition, int j) {
	int i = 0;
	for(i = 0; i < j; i++) {
		condition[i] = old_condition[i];
	}
	condition[i] = old_condition[i] >> 1;
	for(i = j + 1; i < 8; i++) {
		condition[i] = old_condition[i];
	}
}

void fill_cond_up(WORD *old_condition, WORD *condition, int j) {
	int i = 0;
	for(i = 0; i < j; i++) {
		condition[i] = old_condition[i];
	}
	condition[i] = (old_condition[i] << 1) | 0x00000001;
	for(i = j + 1; i < 8; i++) {
		condition[i] = old_condition[i];
	}
}

void fill_cond_up_common(WORD *old_condition, WORD *condition, int j) {
	if(old_condition[j] != 0xffffffff) {
		fill_cond_up(old_condition, condition, j);
	} else {
		fill_cond_up(old_condition, condition, j - 1);
	}
}

WORD *get_condition(int size) {

	double time;
	int i = 0;
	WORD *condition = (WORD *)calloc(8, sizeof(WORD));
	WORD old_condition[8];
	if(size == 0) {
		for(i = 0; i < 8; i++) {
			condition[i] = condition[i] | 0xffffffff;
		}
		return condition;
	} 

	FILE *time_file = fopen("block_info/time.txt", "r");
	if(time_file == NULL) {
		printf("\nFunction (get_condition __fopen__): ");
		PRINT_ERROR();
	}
	fscanf(time_file, "%lf", &time);

	for(i = 0; i < 8; i++) {
		fscanf(time_file, "%u", &old_condition[i]);
	}

	if(fclose(time_file) == EOF) {
		printf("\nFunction (get_condition __fclose__): ");
		PRINT_ERROR();
	}

	if(time < 180.0) { // 540
		if(old_condition[0] > 0x00000000) {
			fill_cond_down(old_condition, condition, 0);
		} else if(old_condition[1] > 0x00000000) {
			fill_cond_down(old_condition, condition, 1);
		} else if(old_condition[2] > 0x00000000) {
			fill_cond_down(old_condition, condition, 2);
		} else if(old_condition[3] > 0x00000000) {
			fill_cond_down(old_condition, condition, 3);
		} else if(old_condition[4] > 0x00000000) {
			fill_cond_down(old_condition, condition, 4);
		} else if(old_condition[5] > 0x00000000) {
			fill_cond_down(old_condition, condition, 5);
		} else if(old_condition[6] > 0x00000000) {
			fill_cond_down(old_condition, condition, 6);
		} else if(old_condition[7] > 0x00000000) {
			fill_cond_down(old_condition, condition, 7);
		} //TODO: all elements of condition should be 0, but calloc by default makes the all equal to 0.
	} else if(time > 300.0){ //900
		if(old_condition[0] > 0x00000000) {
			if(old_condition[0] != 0xffffffff) {
				fill_cond_up(old_condition, condition, 0);
			}
		} else if(old_condition[1] > 0x00000000) {
			fill_cond_up_common(old_condition, condition, 1);
		} else if(old_condition[2] > 0x00000000) {
			fill_cond_up_common(old_condition, condition, 2);
		} else if(old_condition[3] > 0x00000000) {
			fill_cond_up_common(old_condition, condition, 3);
		} else if(old_condition[4] > 0x00000000) {
			fill_cond_up_common(old_condition, condition, 4);
		} else if(old_condition[5] > 0x00000000) {
			fill_cond_up_common(old_condition, condition, 5);
		} else if(old_condition[6] > 0x00000000) {
			fill_cond_up_common(old_condition, condition, 6);
		} else if(old_condition[7] > 0x00000000) {
			fill_cond_up_common(old_condition, condition, 7);
		} else {
			for(i = 0; i < 7; i++) {
				condition[i] = old_condition[i];
			}
			condition[i] = condition[i] | 0x00000001;
		}
	}
	return condition;
}

int comparison_result(WORD* condition, block_t *block) {
	int i = 0;
	int j = 0;
	while(condition[i] == 0) {
		i++;
	}
	j = i;
	for(; i > 0; i--) {
		if((block -> ctx_current).state[i - 1] != 0) {
			block -> nonce += 1;
			return 1;
		}
	}
	if(condition[j] > (block -> ctx_current).state[j]) {
		return 0;
	} else {
		block -> nonce += 1;
		return 1;
	}
}

void insert_block(char *msg) {

	block_t curr_block;
	block_t prev_block;
	clock_t start, stop;
	int i = 0;
	int size = get_count_from_file();
	curr_block.nonce = 0;
	WORD *condition = get_condition(size);
	if(size == 0) {
		curr_block.nonce += 1;
		curr_block.msg = msg;
		curr_block.number = 1;
		curr_block.length_of_msg = strlen(msg) + 1;
		start = clock();
		set_ctx(NULL, &curr_block, 0);
		stop = clock();
		for(i = 0; i < 8; i++) {
			curr_block.ctx_previous.state[i] = 0;
		}
		print_block_to_file(&curr_block);
		up_count_in_count_file();
	} else {
		curr_block.msg = msg;
		curr_block.number = size + 1;
		curr_block.length_of_msg = strlen(msg) + 1;
		read_block_from_file(&prev_block, size);
		start = clock();

		do {
		curr_block.nonce += 1;
		set_ctx(&prev_block, &curr_block, 1);
		} while(comparison_result(condition, &curr_block)); //need to parall

		stop = clock();
		curr_block.ctx_previous = prev_block.ctx_current;
		print_block_to_file(&curr_block);
		up_count_in_count_file();
	}

	FILE *time_file = fopen("block_info/time.txt", "w");
	if(time_file == NULL) {
		printf("\nFunction (insert_file __fopen__): ");
		PRINT_ERROR();
	}
	fprintf(time_file, "%lf ", (double)(stop - start)/CLOCKS_PER_SEC);
	for(i = 0; i < 8; i++) {
		fprintf(time_file, "%u ", condition[i]);
	}
	if(fclose(time_file) == EOF) {
		printf("\nFunction (insert_file __fclose__): ");
		PRINT_ERROR();
	}
	
	FILE *time_file_1 = fopen("time/time_inf.txt", "a");
	double time = (double)(stop - start)/CLOCKS_PER_SEC;
	if(time_file_1 == NULL) {
		printf("\nFunction (insert_file __fopen__): ");
		PRINT_ERROR();
	}
	fprintf(time_file_1, "Time = %f s\nHashrate = %f h/s\n", time, curr_block.nonce/time);
	if(fclose(time_file_1) == EOF) {
		printf("\nFunction (insert_file __fclose__): ");
		PRINT_ERROR();
	}
	
	FILE *time_file_2 = fopen("time/text_info", "a");
	if(time_file_2 == NULL) {
		printf("\nFunction (insert_file __fopen__): ");
		PRINT_ERROR();
	}
	fprintf(time_file_2, "%d %f\n", curr_block.number, time);
	if(fclose(time_file_2) == EOF) {
		printf("\nFunction (insert_file __fclose__): ");
		PRINT_ERROR();
	}
	free(condition);
}

void print_hex(WORD number) {
	int i = 0;
	BYTE num = 0;
	for(i = 4; i > 0; i--) {
		num = (BYTE)(number >> (i - 1) * 4);
		printf("%x ", num & 0xff);
	}
}

void print_chain() {
	printf("\n");
	int size = get_count_from_file();
	int i = 0;
	int j = 0;
	int offset = 0;
	block_t block;

	for(i = 0; i < size; i++) {
		read_block_from_file(&block, i + 1);
		printf("%d | nonce = %lld | ", block.number, block.nonce);
		char *msg = read_string_from_file_by_offset(offset, block.length_of_msg);
		printf("%s\n", msg);
		free(msg);
		printf("Current hash:\t");
		for(j = 0; j < 8; j++) {
			print_hex(block.ctx_current.state[j]);
		}
		printf("\nPrevious hash:\t");
		for(j = 0; j < 8; j++) {
			print_hex(block.ctx_previous.state[j]);
		}
		printf("\n\n");
		offset += block.length_of_msg - 1;
	}
	printf("\n");
}

void set_ctx(block_t *prev_block, block_t *curr_block, int condition) {

	BYTE buf[SHA256_BLOCK_SIZE];
	SHA256_CTX ctx;
	int str_len = strlen(curr_block -> msg);
	int size_ull = (int)sizeof(unsigned long long);

	if(condition == 0) {
		BYTE instance[str_len + size_ull];
		memcpy(instance, (BYTE *)(curr_block -> msg), str_len);
		memcpy(instance + str_len, (BYTE *)(&(curr_block -> nonce)), size_ull);
		sha256_init(&ctx);
		sha256_update(&ctx, instance, str_len + size_ull);
	}
	if(condition == 1) {
		BYTE instance[str_len + 32 + size_ull];
		memcpy(instance, (BYTE *)(curr_block -> msg), str_len);
		memcpy(instance + str_len, (BYTE *)((prev_block -> ctx_current).state), 32);
		memcpy(instance + str_len + 32, (BYTE *)(&(curr_block -> nonce)), size_ull);
		sha256_init(&ctx);
		sha256_update(&ctx, instance, str_len + 32 + size_ull); 
	}
	sha256_final(&ctx, buf);
	curr_block -> ctx_current = ctx;
}

void up_count_in_count_file() {
	FILE *count_file = fopen("block_info/count.txt", "r");
	if(count_file == NULL) {
		printf("\nFunction (up_count_in_count_file __fopen__): ");
		PRINT_ERROR();
	}
	int count = 0;
	fscanf(count_file, "%d", &count);
	if(fclose(count_file) == EOF) {
		printf("\nFunction (up_count_in_count_file __fclose__): ");
		PRINT_ERROR();
	}
	count_file = fopen("block_info/count.txt", "w");
	if(count_file == NULL) {
		printf("\nFunction (up_count_in_count_file __fopen__): ");
		PRINT_ERROR();
	}
	count++;
	fprintf(count_file, "%d", count);
	if(fclose(count_file) == EOF) {
		printf("\nFunction (up_count_in_count_file __fclose__): ");
		PRINT_ERROR();
	}
}

void print_block_to_file(block_t * block) { 
	int fd = open("block_info/block_file.txt", O_APPEND|O_WRONLY);
	if(fd == -1) {
		printf("\nFunction (print_block_to_file): ");
		PRINT_ERROR();
	}
	write(fd, block, sizeof(struct block));
	if(close(fd) == -1) {
		printf("\nFunction (print_block_to_file __ close(fd)__): ");
		PRINT_ERROR();
	}
	print_string_to_file(block -> msg);
}

void print_string_to_file(char* msg) {
	int fd = open("block_info/msg.txt", O_APPEND|O_WRONLY);
	if(fd == -1) {
		printf("\nFunction (print_string_to_file): ");
		PRINT_ERROR();
	}
	write(fd, msg, strlen(msg));
	if(close(fd) == -1) {
		printf("\nFunction (print_string_to_file __ close(fd)__): ");
		PRINT_ERROR();
	}
}

void zero_count_file() {
	FILE *count_file = fopen("block_info/count.txt", "w");
	if(count_file == NULL) {
		printf("\nFunction (zero_count_file __fopen__): ");
		PRINT_ERROR();
	}
	fprintf(count_file, "%d", 0);
	if(fclose(count_file) == EOF) {
		printf("\nFunction (zero_count_file __fclose__): ");
		PRINT_ERROR();
	}
}

int get_count_from_file() {
	int count = 0;
	FILE *count_file = fopen("block_info/count.txt", "r");
	if(count_file == NULL) {
		printf("\nFunction (get_count_from_file __fopen__): ");
		PRINT_ERROR();
	}
	fscanf(count_file, "%d", &count);
	if(fclose(count_file) == EOF) {
		printf("\nFunction (get_count_from_file __fclose__): ");
		PRINT_ERROR();
	}
	return count;
}

void read_block_from_file(block_t *block, int number) {
	int fd = open("block_info/block_file.txt", O_RDONLY);
	if(fd == -1) {
		printf("\nFunction (read_block_from_file __ open__): ");
		PRINT_ERROR();
	}
	pread(fd, block, sizeof(struct block), (number - 1)*sizeof(struct block));
	if(close(fd) == -1) {
		printf("\nFunction (read_block_from_file __ close(fd)__): ");
		PRINT_ERROR();
	}
}

char *read_string_from_file_by_offset(int offset, int length) {
	char *msg = (char *)calloc(length, sizeof(char));
	if((msg == NULL) && (length != 0)) {
		printf("\nFunction (read_string_from_file_by_offset __ calloc__): ");
		PRINT_ERROR();
	}
	int fd = open("block_info/msg.txt", O_RDONLY);
	if(fd == -1) {
		printf("\nFunction (read_string_from_file_by_offset __ open__): ");
		PRINT_ERROR();
	}
	pread(fd, msg, length - 1, offset);
	if(close(fd) == -1) {
		printf("\nFunction (read_string_from_file_by_offset __ close(fd)__): ");
		PRINT_ERROR();
	}
	msg[length - 1] = '\0';
	return msg;
}

int check_ctx(block_t *prev_block, block_t *curr_block) {

	block_t test_block;
	int i = 0;

	test_block.msg = curr_block -> msg;
	test_block.nonce = curr_block -> nonce;
	if(prev_block == NULL) {
		set_ctx(NULL, &test_block, 0);
		for(i = 0; i < 8; i++) {
			if((curr_block -> ctx_current).state[i] != test_block.ctx_current.state[i]) {
				return 1;
			}
		}
	} else {
		set_ctx(prev_block, &test_block, 1);
		for(i = 0; i < 8; i++) {
			if((curr_block -> ctx_current).state[i] != test_block.ctx_current.state[i]) {
				return 1;
			}
		}
	}
	return 0;
}

void verify_chain() {
	int size = get_count_from_file();
	int i = 0;
	block_t curr_block;
	block_t prev_block;
	int offset = 0;

	if(size == 0) {
		printf("\nBlockchain is empty, verification complete!\n\n");
		return;
	} else {
		for(i = 0; i < size; i++) {
			if(i == 0) {
				read_block_from_file(&curr_block, i + 1);
				curr_block.msg = read_string_from_file_by_offset(offset, curr_block.length_of_msg);
				if(check_ctx(NULL, &curr_block) != 0) {
					printf("\nVerification failed: blockchain was changed!\n\n");
					return;
				}
				free(curr_block.msg);
			} else {
				read_block_from_file(&prev_block, i);
				read_block_from_file(&curr_block, i + 1);
				prev_block.msg = read_string_from_file_by_offset(offset, prev_block.length_of_msg);
				offset += prev_block.length_of_msg - 1;
				curr_block.msg = read_string_from_file_by_offset(offset, curr_block.length_of_msg);
				if(check_ctx(&prev_block, &curr_block) != 0) {
					printf("\nVerification failed: blockchain was changed!\n\n");
					return;
				}
				free(prev_block.msg);
				free(curr_block.msg);
			}
		}
	}
	printf("\nVerification successfully complete!\n\n");
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void do_it(char *msg) {
	//verify_chain();
	insert_block(msg);
	print_chain();
}

int main(int argc, char *argv[]) {
	if(!strcmp(argv[1], "clean")) {
	clear_chain();
	printf("Chain cleaned\n");
	} else 
	do_it(argv[1]); 
	return 0;
}
