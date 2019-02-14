#include "Catalog.hpp"

#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"

long total_vectors_loaded = 0;

char *get_config(const char *name, char *fname) {
    char *value = (char *) malloc(1024);
    FILE *configfile = fopen(fname, "r");
    value[0] = '\0';

    if (configfile != NULL) {
        while (1) {
            char configname[1024];
            char tempvalue[1024];

            int status = fscanf(configfile, " %1023[^= ] = %s ", configname, tempvalue); //Parse key=value

            if (status == EOF){
                break;
            }

            if (strcmp(configname, name) == 0){
                strncpy(value, tempvalue, strlen(tempvalue)+1);
                break;
            }
        }
        fclose(configfile);
    }
    return value;
}

unsigned int catalog_load_vectors_into_bit_field(char *name, struct bit_field *bf, float *out_radius) {
	char *vec_c_c = get_config("vec_c", name);
	int vec_c = atoi(vec_c_c);

	vec_c = (vec_c - (vec_c % 3));

	if (vec_c <= 2) return 0;

	float *vecs = (float *) malloc(vec_c*sizeof(float));
	for (int i = 0; i < vec_c; i++) {
		char buf[16];
		sprintf(buf, "vec[%i]", i);

		char *val = get_config(buf, name);
		if (val[0] != '\0') {
			vecs[i] = (float) atof(val);
		} else {
			free(val);
			free(vecs);
			return 0;
		}
		free(val);
	}
	*out_radius = 0;
	for (int i = 0; i < vec_c/3; i++) {
		float dist = sqrt((vecs[(i*3)]*vecs[(i*3)]) + (vecs[(i*3)+1]*vecs[(i*3)+1]) + (vecs[(i*3)+2]*vecs[(i*3)+2]));
		if (dist > *out_radius) {
			*out_radius = dist;
		}
	}
	if ((total_vectors_loaded + vec_c/3)/(1000*3) > total_vectors_loaded/(1000*3)) {
		printf("total vectors loaded: %lu\r\n", total_vectors_loaded / 3);
	}
	total_vectors_loaded += (vec_c/3);
	unsigned int pos = bit_field_add_bulk(bf, (unsigned int *) vecs, ceil(vec_c*sizeof(float)/sizeof(unsigned int)));
	free(vecs);
	return pos;
}

float *catalog_load_position(char *name) {
	float *vecs = (float *) malloc(3*sizeof(float));
	for (int i = 0; i < 3; i++) {
		char buf[16];
		sprintf(buf, "position[%i]", i);
		char *val = get_config(buf, name);
		vecs[i] = (float) atof(val);
		free(val);
	}
	return vecs;
}
