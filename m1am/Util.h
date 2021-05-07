#pragma once

#include <string>
#include <vector>
#include "Vector3.h"
#include <type_traits>

using namespace std;

const float M_PI = 3.14159265358979323846;

string& util_ltrim(string& str, const string& chars);
string& util_rtrim(string& str, const string& chars);
string& util_trim(string& str, const string& chars);

vector<string> util_file_read_lines(const string filepath);

bool util_starts_with(const std::string& str, const string& with);

template <typename T> struct vector3<T> util_get_vector3(const std::string& str) {
	struct vector3<T> result;
	int start = 0;
	int end = str.find_first_of(",", start);
	int idx = 0;
	while (end != std::string::npos) {
		string tmp_s = str.substr(start, end - start);
		const char *tmp = tmp_s.c_str();
		if (is_same<T, float>::value) {
			result[idx] = stof(tmp);
		} else if (is_same<T, int>::value) {
			result[idx] = stoi(tmp);
		}
		start = end + 1;
		end = str.find_first_of(",", start);
		idx++;
	}
	string tmp_s = str.substr(start);
	const char* tmp = tmp_s.c_str();
	if (is_same<T, float>::value) {
		result[idx] = stof(tmp);
	} else if (is_same<T, int>::value) {
		result[idx] = stoi(tmp);
	}
	return result;
}

vector<string>	util_split(const std::string& str, const std::string& separator = ",");

void util_read_binary(string filename, unsigned char* bin, size_t* out_length);
void util_write_binary(string filename, unsigned char* bin, size_t length);