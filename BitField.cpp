#include "BitField.hpp"

#include <cuda_runtime.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cassert>

//TODO:
//	solves everything 	-> 	implement registration of defragmentation callbacks (to make it possible to move pages, to reduce total fieldsize)
//	reduces unused space 	-> 	maybe circumvent by implementing counting of free space before reaching occupied by "to relocated data" <-- should be very quick to implement maybe
//
//	implement update_all_pages flag/parameter/option
//
//	remember some empty line configurations for faster new allocations when bit field is big
//	if >= certain size skip to index, etc.

void bit_field_init(struct bit_field *bf, unsigned int pages, unsigned int pagesize) {
	bf->data = (unsigned int *) malloc((pages*(pagesize+1))*sizeof(unsigned int));
	memset(bf->data, 0, (pages*(pagesize+1))*sizeof(unsigned int));
	bf->pages = pages;
	bf->pagesize = pagesize;
	bf->invalidators_c = 0;
	bf->devices_c = 0;

	bf->biggest_tracked_allocated_page = 0;
}

/* INTERNAL */

void bit_field_set_invalidators(struct bit_field *bf, unsigned int page, unsigned int page_count) {
	for (int i = 0; i < bf->invalidators_c; i++) {
		for (int j = 0; j < page_count; j++) {
			(&(*bf->invalidators[i]))[(page+j)/8] |= 1 << ((page+j) % 8);
		}
	}
}

void bit_field_add_pages(struct bit_field *bf, unsigned int pages) {
	bf->data = (unsigned int *) realloc(bf->data, (bf->pages+pages)*(bf->pagesize+1)*sizeof(unsigned int));
	memset(&bf->data[bf->pages*(bf->pagesize+1)], 0, pages*(bf->pagesize+1)*sizeof(unsigned int));
	for (int i = 0; i < bf->invalidators_c; i++) {
		bf->invalidators[i] = (unsigned char *) realloc(&(*bf->invalidators[i]), (bf->pages+7+pages+7)/8 * sizeof(unsigned char));
		memset(&(bf->invalidators[i])[(bf->pages+7)/8], 1, (pages+7)/8 * sizeof(unsigned char));
	}
	for (int i = 0; i < bf->devices_c; i++) {
		unsigned int *device_ptr = NULL;
		cudaError_t err = cudaSuccess;
	        err = cudaSetDevice(bf->device_ids[i]);
		err = cudaMalloc((void **)&device_ptr, (bf->pages+pages)*(bf->pagesize+1)* sizeof(int));
		err = cudaMemcpy(device_ptr, bf->device_data[i], bf->pages*(bf->pagesize+1)*sizeof(int), cudaMemcpyDeviceToDevice);
		cudaFree(bf->device_data[i]);												//maybe do some delayed free
		bf->device_data[i] = device_ptr;
	}
	bf->pages += pages;
}

unsigned int bit_field_get_pagetype(const struct bit_field *bf, const unsigned int page) {
	return (bf->data[page*(bf->pagesize+1)]);
}

unsigned int bit_field_get_pagetype_from_index(const struct bit_field *bf, const unsigned int index) {
	return (bf->data[index-(index % (bf->pagesize+1))]);
}

void bit_field_set_pagetype(struct bit_field *bf, const unsigned int page, const unsigned int type) {
	bf->data[page*(bf->pagesize+1)] = type;
}

unsigned int bit_field_get_value(const struct bit_field *bf, const unsigned int page, const unsigned int position) {
	return bf->data[page*(bf->pagesize+1)+1+position];
}

unsigned int bit_field_get_index(const struct bit_field *bf, const unsigned int page, const unsigned int position) {
	return page*(bf->pagesize+1)+1+position;
}

unsigned int bit_field_get_page_from_index(const struct bit_field *bf, const unsigned int index) {
	return (index / (bf->pagesize+1));
}

unsigned int bit_field_get_free_location(struct bit_field *bf, const unsigned int size, const unsigned int skip) {
	int skip_ac = skip;
	if (bf->pages > 10000 && skip_ac == 0) {
		skip_ac = bf->biggest_tracked_allocated_page;
	}
	for (int i = skip_ac; i < bf->pages; i++) {
		if (bit_field_get_pagetype(bf, i) == 0) {
			int type = 1;
			while (type < size+1) type *= 2;
			if (type > bf->pagesize) {
				int occupied = 0;
				for (int j = 1; j < ceil(type / (float)bf->pagesize); j++) {
					while (i+j >= bf->pages) bit_field_add_pages(bf, bf->pages/2);
					if (bit_field_get_pagetype(bf, i+j) != 0) {
						occupied = j;
						break;
					}
				}
				if (occupied != 0) {
					i += (occupied-1);
					continue;
				}
			}
			bit_field_set_pagetype(bf, i, type);
			if (i > bf->biggest_tracked_allocated_page) {
				bf->biggest_tracked_allocated_page = i;
//				printf("bf->pages: %i, fpol: %i\r\n", bf->pages, bf->biggest_tracked_allocated_page);
			}
		}

		int type = bit_field_get_pagetype(bf, i);
		if (type >= size+1 && type < bf->pagesize) {
			for (int j = 0; j < bf->pagesize-type+1; j += type) {
                                if (bit_field_get_value(bf, i, j) == 0) {
					bit_field_set_invalidators(bf, i, 1);
                                        return bit_field_get_index(bf, i, j);
                                }
                        }
		} else if (type == bf->pagesize) {
			if (bit_field_get_value(bf, i, 0) == 0) {
				bit_field_set_invalidators(bf, i, 1);
				return bit_field_get_index(bf, i, 0);
			}
		} else if (type > bf->pagesize) {
			if (bit_field_get_value(bf, i, 0) == 0) {
				bit_field_set_invalidators(bf, i, ceil(type/(float)(bf->pagesize+1)));
				return bit_field_get_index(bf, i, 0);
			}
			i += floor((type/(float)(bf->pagesize+1)));
		}
	}
	int old_pages = bf->pages;
	bit_field_add_pages(bf, bf->pages/2);
	return bit_field_get_free_location(bf, size, old_pages);
}
/* END INTERNAL */

unsigned int bit_field_add_data(struct bit_field *bf, const unsigned int datum) {
	int index = bit_field_get_free_location(bf, 1, 0);
	bf->data[index] = 1;
	bf->data[index+1] = datum;
	return index;
}

unsigned int bit_field_add_bulk(struct bit_field *bf, const unsigned int *data, const unsigned int data_len) {
	int index = bit_field_get_free_location(bf, data_len, 0);
	bf->data[index] = data_len;
	memcpy(&bf->data[index+1], data, data_len*sizeof(unsigned int));
	return index;
}

//TODO: a lot of space for improvement
unsigned int bit_field_add_data_to_segment(struct bit_field *bf, const unsigned int index, const unsigned int datum) {
	int page = bit_field_get_page_from_index(bf, index);

	int pagetype = bit_field_get_pagetype_from_index(bf, index);
	int size = bf->data[index];
	if (size+1+1 < pagetype) {
		bf->data[index+1+size] = datum;
		bf->data[index]++;
		bit_field_set_invalidators(bf, page, 1);
		if (bit_field_get_page_from_index(bf, index+1+size) != page) {
			bit_field_set_invalidators(bf, bit_field_get_page_from_index(bf, index+1+size), 1);
		}
		return index;
	}

	int new_index = bit_field_get_free_location(bf, size+1, 0);
	memcpy(&bf->data[new_index], &bf->data[index], (size+1)*sizeof(unsigned int));

	//delete old line/s
	memset(&bf->data[index], 0, (size+1)*sizeof(unsigned int));
	bit_field_set_invalidators(bf, page, ceil(pagetype/(float)(bf->pagesize+1)));

	//clear pagetype if old page empty
	if (pagetype <= bf->pagesize) {
		int is_empty = 1;
		for (int i = page*(bf->pagesize+1)+1; i < page*(bf->pagesize+1)+1 + bf->pagesize; i += pagetype) {
                        if (bf->data[i] != 0) {
				is_empty = 0;
				break;
                        }
                }
		if (is_empty == 1) {
			bit_field_set_pagetype(bf, page, 0);
		}
	} else {
		bit_field_set_pagetype(bf, page, 0);
	}
	//invalidate old line/s
//	bit_field_set_invalidators(bf, page, ceil(pagetype/(float)(bf->pagesize+1)));

	//update data (shouled be invalidated due to get_free_location call)
	bf->data[new_index] = size+1;
	bf->data[new_index+1+size] = datum;

	return new_index;
}

unsigned int bit_field_add_bulk_to_segment(struct bit_field *bf, const unsigned int index, const unsigned int *data, const unsigned int data_len) {
	int page = bit_field_get_page_from_index(bf, index);

	int pagetype = bit_field_get_pagetype_from_index(bf, index);
        int size = bf->data[index];
        if (size+1+data_len < pagetype) {
		memcpy(&bf->data[index+1+size], data, data_len*sizeof(unsigned int));
		bf->data[index] += data_len;

		//greedy invalidators must be improved, to only invalidate the actual invalid lines
		bit_field_set_invalidators(bf, page, ceil(pagetype/(float)(bf->pagesize+1)));

                return index;
        }

        int new_index = bit_field_get_free_location(bf, size+data_len, 0);
        memcpy(&bf->data[new_index], &bf->data[index], (size+1)*sizeof(unsigned int));

	//delete old line/s
        memset(&bf->data[index], 0, (size+1)*sizeof(unsigned int));
	bit_field_set_invalidators(bf, page, ceil(pagetype/(float)(bf->pagesize+1)));

        //clear pagetype if old page/s empty
        if (pagetype <= bf->pagesize) {
                int is_empty = 1;
                for (int i = page*(bf->pagesize+1)+1; i < page*(bf->pagesize+1)+1 + bf->pagesize; i += pagetype) {
                        if (bf->data[i] != 0) {
                                is_empty = 0;
                                break;
                        }
                }
                if (is_empty == 1) {
                        bit_field_set_pagetype(bf, page, 0);
                }
        } else {
                bit_field_set_pagetype(bf, page, 0);
        }
	//invalidate old line/s
//	bit_field_set_invalidators(bf, page, ceil(pagetype/(float)(bf->pagesize+1)));

        //update data (should be invalidated due to get_free_location call)
	memcpy(&bf->data[new_index+1+size], data, data_len*sizeof(unsigned int));
	bf->data[new_index] = size+data_len;

        return new_index;
}

void bit_field_update_data(struct bit_field *bf, const unsigned int index, const unsigned int datum) {
	int page = bit_field_get_page_from_index(bf, index);
	bf->data[index] = datum;
	bit_field_set_invalidators(bf, page, 1);
}

unsigned int bit_field_remove_data_from_segment(struct bit_field *bf, const unsigned int index, const unsigned int datum) {
	int page = bit_field_get_page_from_index(bf, index);
	int pagetype = bit_field_get_pagetype_from_index(bf, index);
	int size = bf->data[index];
	if (size == 1) {
		bf->data[index] = 0;
		assert(bf->data[index+1] == datum);
		bf->data[index+1] = 0;
		bit_field_set_invalidators(bf, page, 1);

		//clear pagetype if old page/s empty
	        if (pagetype <= bf->pagesize) {
        	        int is_empty = 1;
                	for (int i = page*(bf->pagesize+1)+1; i < page*(bf->pagesize+1)+1 + bf->pagesize; i += pagetype) {
                        	if (bf->data[i] != 0) {
                                	is_empty = 0;
	                                break;
        	                }
	                }
        	        if (is_empty == 1) {
                	        bit_field_set_pagetype(bf, page, 0);
	                }
        	} else {
	                bit_field_set_pagetype(bf, page, 0);
        	}

		return 0;
	}
	for (int i = 0; i < size; i++) {
		if (bf->data[index+1+i] == datum) {
			if (i == size-1) {
				bf->data[index+1+i] = 0;
			} else {
				bf->data[index+1+i] = bf->data[index+1+size-1];
				bf->data[index+1+size-1] = 0;
			}
			bf->data[index]--;

			int page_removed_item = bit_field_get_page_from_index(bf, index+1+i);
			int page_moved_item = bit_field_get_page_from_index(bf, index+1+size-1);
			bit_field_set_invalidators(bf, page, 1);
			if (page_removed_item != page) {
				bit_field_set_invalidators(bf, page_removed_item, 1);
			}
			if (page_moved_item != page && page_moved_item != page_removed_item) {
				bit_field_set_invalidators(bf, page_moved_item, 1);
			}
			return index;
		}
	}
	assert(index == 0); //should not be reached
	return index;
}

/* INTERNAL */
unsigned int bit_field_register_invalidator(struct bit_field *bf) {
	if (bf->invalidators_c == 0) {
		bf->invalidators = (unsigned char **) malloc(sizeof(unsigned char *));
	} else {
		bf->invalidators = (unsigned char **) realloc(bf->invalidators, (bf->invalidators_c+1)*sizeof(unsigned char *));
	}
	bf->invalidators[bf->invalidators_c++] = (unsigned char *) malloc((bf->pages+7)/8 * sizeof(unsigned char));
	memset(bf->invalidators[bf->invalidators_c-1], 1, (bf->pages+7)/8 * sizeof(unsigned char));
	return bf->invalidators_c-1;
}
/* END INTERNAL */

unsigned int bit_field_register_device(struct bit_field *bf, unsigned int device_id) {
	if (bf->devices_c == 0) {
		bf->device_data = (unsigned int **) malloc(sizeof(unsigned int *));
		bf->device_ids = (unsigned int *) malloc(sizeof(unsigned int));
	} else {
		bf->device_data = (unsigned int **) realloc(bf->device_data, (bf->devices_c+1)*sizeof(unsigned int *));
		bf->device_ids = (unsigned int *) realloc(bf->device_ids, (bf->devices_c+1)*sizeof(unsigned int));
	}
	bit_field_register_invalidator(bf);

	cudaError_t err = cudaSuccess;
	err = cudaSetDevice(device_id);
	bf->device_data[bf->devices_c] = NULL;
	err = cudaMalloc((void **)&bf->device_data[bf->devices_c], bf->pages*(bf->pagesize+1)*sizeof(unsigned int));
	if (err != cudaSuccess) {
                fprintf(stderr, "Error allocating device_data (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
        }
	bf->device_ids[bf->devices_c++] = device_id;
	return bf->devices_c-1;
}

void bit_field_update_device(const struct bit_field *bf, unsigned int device_id) {
	cudaError_t err = cudaSuccess;
	err = cudaSetDevice(device_id);
	for (int i = 0; i < bf->devices_c; i++) {
		if (bf->device_ids[i] == device_id) {
			int cp_size = 0;
			int cp_startpage = 0;
			int cp_started = 0;
			for (int j = 0; j < (bf->pages+7)/8; j++) {
//				printf("%c\r\n", (&(*bf->invalidators[i])[j]));
				if ((&(*bf->invalidators[i]))[j] > 0) {
					if (!cp_started) {
						cp_started = 1;
						cp_startpage = j;
					}
					if ((j*8)+7 >= bf->pages-1) {
						cp_size += (bf->pages - (j*8));
					} else {
						cp_size += 8;
					}
				} else {
					if (cp_started) {
						err = cudaMemcpy(&bf->device_data[i][(cp_startpage*8)*(bf->pagesize+1)], &bf->data[(cp_startpage*8)*(bf->pagesize+1)], cp_size*(bf->pagesize+1)*sizeof(unsigned int), cudaMemcpyHostToDevice);
	                                        if (err != cudaSuccess) {
        	                                        fprintf(stderr, "Error copying device_data (error code %s)!\n", cudaGetErrorString(err));
                	                                exit(EXIT_FAILURE);
                        	                }
						cp_started = 0;
						cp_size = 0;
					}
				}
			}
			if (cp_started) {
                               	err = cudaMemcpy(&bf->device_data[i][(cp_startpage*8)*(bf->pagesize+1)], &bf->data[(cp_startpage*8)*(bf->pagesize+1)], cp_size*(bf->pagesize+1)*sizeof(unsigned int), cudaMemcpyHostToDevice);
                                if (err != cudaSuccess) {
                                       	fprintf(stderr, "Error copying device_data (error code %s)!\n", cudaGetErrorString(err));
                                        exit(EXIT_FAILURE);
                                }
                        }
			memset(bf->invalidators[i], 0, (bf->pages+7)/8 * sizeof(unsigned char));
			break;
		}
	}
}

void bit_field_dump(const struct bit_field *bf) {
	printf("dumping bitfield, pages: %i\r\n", bf->pages);
	for (int i = 0; i < bf->pages; i++) {
		printf("%i\t", bit_field_get_pagetype(bf, i));
		for (int j = 0; j < bf->pagesize; j++) {
			printf("%i\t", bit_field_get_value(bf, i, j));
		}
		printf("\r\n");
	}
	for (int i = 0; i < bf->invalidators_c; i++) {
		for (int j = 0; j < (bf->pages+7)/8; j++) {
			for (int k = 0; k < 8; k++) {
				if (((&(*bf->invalidators[i]))[j] >> k) & 0x1) {
					printf("1");
				} else {
					printf("0");
				}
			}
		}
		printf("\r\n");
	}
	printf("end dumping bitfield\r\n");
}
