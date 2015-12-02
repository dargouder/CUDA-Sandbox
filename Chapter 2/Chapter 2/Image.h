#pragma once
#include <cmath>
#include <string>
#include <fstream>
#include "RGBColour.h"


	class Image {
	public:
		RGBColour** image;
		int width;
		int height;
		int size;
	public:
		Image();
		~Image();
		Image(int p_width, int p_height);
		Image(int p_width, int p_height, RGBColour p_background);

		bool set(int p_x, int p_y, const RGBColour& p_colour);
		RGBColour get(int p_x, int p_y);
		void gammaCorrect(float p_gamma);
		
		void writePPM(const std::string &p_strImageFile);
		void readPPM(std::string p_file_name);



	};
