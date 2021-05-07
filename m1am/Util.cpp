#include "Util.h"

#include <fstream>
#include "windows.h"
#include <sstream>

string& util_ltrim(string& str, const string& chars) {
	str.erase(0, str.find_first_not_of(chars));
	return str;
}

string& util_rtrim(string& str, const string& chars) {
	str.erase(str.find_last_not_of(chars) + 1);
	return str;
}

string& util_trim(string& str, const string& chars) {
	return util_ltrim(util_rtrim(str, chars), chars);
}

vector<string> util_file_read_lines(const string filepath) {
	vector<string> result;

	ifstream file(filepath);
	string filecontent;
	string chars(" \n\r\t");
	if (file.is_open()) {
		while (getline(file, filecontent)) {
			if (filecontent.size() > 0) {
				string name = filecontent;
				name = util_trim(name, chars);
				result.push_back(name);
			}
		}
	}
	return result;
}

bool util_starts_with(const string& str, const string& with) {
	if (str.length() < with.length()) return false;
	string tmp = str.substr(0, with.length());
	return strcmp(tmp.c_str(), with.c_str()) == 0;
}

vector<string>	util_split(const std::string& str, const std::string& separator) {
	vector<string> result;
	int start = 0;
	int end = str.find_first_of(separator, start);
	while (end != std::string::npos) {
		result.push_back(str.substr(start, end - start));
		start = end + 1;
		end = str.find_first_of(separator, start);
	}
	result.push_back(str.substr(start));
	return result;
}


void util_read_binary(string filename, unsigned char* bin, size_t* out_length) {
	HANDLE file_handle = CreateFileA(filename.c_str(),
		FILE_GENERIC_READ,
		0,
		NULL,
		OPEN_EXISTING,
		FILE_ATTRIBUTE_NORMAL,
		NULL);

	(*out_length) = 0;

	if (file_handle != INVALID_HANDLE_VALUE) {
		char buffer[1024];
		memset(buffer, 0, 1024);

		DWORD dwBytesRead;

		unsigned char* bin_tmp = bin;

		while (ReadFile(file_handle, buffer, 1024, &dwBytesRead, NULL)) {
			if (dwBytesRead != 0) {
				memcpy(bin_tmp, buffer, dwBytesRead);
				bin_tmp += dwBytesRead;
				(*out_length) += dwBytesRead;
			}
			else {
				break;
			}
		}
	}

	CloseHandle(file_handle);
}

void util_write_binary(string filename, unsigned char* bin, size_t length) {
	HANDLE file_handle = CreateFileA(filename.c_str(),
		FILE_GENERIC_WRITE,
		0,
		NULL,
		OPEN_ALWAYS,
		FILE_ATTRIBUTE_NORMAL,
		NULL);

	char buffer[1024];
	memset(buffer, 0, 1024);

	DWORD dwBytesWritten;

	int ct = 0;
	while (ct < length) {
		int bytes_to_write = 1024;
		if (length - ct < 1024) {
			bytes_to_write = length - ct;
		}
		memcpy(buffer, &bin[ct], bytes_to_write);
		int part_write = 0;
		while (part_write < bytes_to_write) {
			WriteFile(file_handle, &buffer[part_write], bytes_to_write - part_write, &dwBytesWritten, NULL);
			part_write += dwBytesWritten;
			ct += dwBytesWritten;
		}
	}
	CloseHandle(file_handle);
}