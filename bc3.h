#ifndef __BLOCKCHAIN__
#define __BLOCKCHAIN__

#include <stddef.h>
#include "CL/cl.h"

typedef struct {
	cl_uint nPlatform; //OpenCL number of platforms
	cl_platform_id cpPlatform;       //OpenCL platform
	cl_device_id   cdDevice;         //OpenCL device
	cl_context       cxGPUContext;   //OpenCL context
	cl_command_queue cqCommandQueue; //OpenCL command que
	cl_program cProgram; //OpenCL program
	cl_kernel  cKernel;  //OpenCL kernel
  } cl_ctx; //OpenCL struct

//						SHA256 PART STARTS

#define SHA256_BLOCK_SIZE 32

typedef unsigned char BYTE;
typedef unsigned int  WORD;


typedef struct {
	BYTE data[64];
	WORD datalen;
	unsigned long long bitlen;
	WORD state[8];
} SHA256_CTX;

void sha256_init(SHA256_CTX *ctx);
void sha256_update(SHA256_CTX *ctx, const BYTE data[], size_t len);
void sha256_final(SHA256_CTX *ctx, BYTE hash[]);

//						SHA256 PART ENDS

//						BLOCKCHAIN PART STARTS

typedef struct block {
	unsigned long long nonce;
	int number; // Number will be from 1 to ...
	int length_of_msg; 
	SHA256_CTX ctx_current;
	SHA256_CTX ctx_previous;
	char* msg;
} block_t;

//API STARTS

void clear_chain();
void print_chain();
void insert_block(char *msg);
void verify_chain();

//API ENDS

void read_block_from_file(block_t *block, int number);
void set_ctx(block_t *prev_block, block_t *curr_block, int condition);
void up_count_in_count_file(); 
void print_block_to_file(block_t * block);
void print_string_to_file(char * msg); 
void zero_count_file();
int get_count_from_file();
char *read_string_from_file_by_offset(int offset, int length);

WORD *get_condition();
void print_hex(WORD number);
WORD *get_condition(int size);
void fill_cond_down(WORD *old_condition, WORD *condition, int j);
int comparison_result(WORD* condition, block_t *block);
void fill_cond_up_common(WORD *old_condition, WORD *condition, int j);
void fill_cond_up(WORD *old_condition, WORD *condition, int j);

//						BLOCKCHAIN PART ENDS

#endif
