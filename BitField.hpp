#ifndef BITFIELD_HPP
#define BITFIELD_HPP

struct bit_field {
	unsigned int pages;
	unsigned int pagesize;
	unsigned int *data;

	unsigned char **invalidators;
	unsigned int invalidators_c;

	unsigned int **device_data;
	unsigned int *device_ids;
	unsigned int devices_c;

	unsigned int biggest_tracked_allocated_page;
};

void bit_field_init(struct bit_field *bf, unsigned int pages, unsigned int pagesize);

unsigned int bit_field_add_data(struct bit_field *bf, const unsigned int datum);
unsigned int bit_field_add_bulk(struct bit_field *bf, const unsigned int *data, const unsigned int data_len);

unsigned int bit_field_add_data_to_segment(struct bit_field *bf, const unsigned int index, const unsigned int datum);
unsigned int bit_field_add_bulk_to_segment(struct bit_field *bf, const unsigned int index, const unsigned int *data, const unsigned int data_len);

void bit_field_update_data(struct bit_field *bf, const unsigned int index, const unsigned int datum);

unsigned int bit_field_remove_data_from_segment(struct bit_field *bf, const unsigned int index, const unsigned int datum);

unsigned int bit_field_register_device(struct bit_field *bf, unsigned int device_id);
void bit_field_update_device(const struct bit_field *bf, unsigned int device_id);

void bit_field_dump(const struct bit_field *bf);

#endif /* BITFIELD_HPP */
