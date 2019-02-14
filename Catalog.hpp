#ifndef CATALOG_HPP
#define CATALOG_HPP

#include "BitField.hpp"

unsigned int catalog_load_vectors_into_bit_field(char *name, struct bit_field *bf, float *out_radius);

float *catalog_load_position(char *name);

#endif
