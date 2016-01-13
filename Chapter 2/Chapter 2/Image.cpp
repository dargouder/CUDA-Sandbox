#include "Image.h"
#include <algorithm>
#include <iostream>


Image::Image() {

}

Image::~Image(){

	delete image;
}

Image::Image(int p_width, int p_height){
	width = p_width;
	height = p_height;
	size = p_width * p_height * (sizeof(float) * 3);
	image = new RGBColour[width*height];
}




Image::Image(int p_width, int p_height, RGBColour p_background){
	width = p_width;
	height = p_height;

	image = new RGBColour[width*height];
	for (int i = 0; i < width; i++){
		for (int j = 0; j < height; j++){
			int index = GetIndex(i, j);
			image[index] = p_background;
		}
	}
}

bool Image::set(int p_x, int p_y, const RGBColour& p_colour){
	if (0 > p_x || p_x > width) return false;
	if (0 > p_y || p_y > height) return false;

	image[GetIndex(p_x,p_y)] = p_colour;
	return true;
}
RGBColour Image::get(int p_x, int p_y){
	if (0 > p_x || p_x > width) return false;
	if (0 > p_y || p_y > height) return false;

	return image[GetIndex(p_x, p_y)];
}

void Image::gammaCorrect(float gamma){
	RGBColour temp;
	float power = 1.0 / gamma;
	for (int i = 0; i < width; i++){
		for (int j = 0; j < height; j++){
			int index = GetIndex(i, j);
			temp = image[index];
			image[index] = RGBColour(pow(temp.r, power), pow(temp.g, power), pow(temp.b, power));
		}
	}
}

void Image::writePPM(const std::string &p_strImageFile){

	std::ofstream imageFile;
	imageFile.open(p_strImageFile.c_str(), std::ios::binary);

	imageFile << "P6" << std::endl;
	imageFile << width << " " << height << std::endl;
	imageFile << "255" << std::endl;
	int i, j;

	unsigned int ired, igreen, iblue;
	unsigned char red, green, blue;



	for (i = 0; i < height; i++) {
		for (j = width - 1; j >= 0; j--){
			int index = GetIndex(j, i);
			red = static_cast<unsigned char>(std::min(1.f, image[index].r) * 255);
			green = static_cast<unsigned char>(std::min(1.f, image[index].g) * 255);
			blue = static_cast<unsigned char>(std::min(1.f, image[index].b) * 255);


			imageFile << red << green << blue;
		}
	}

	imageFile.close();

}

void Image::readPPM(std::string p_filename){
	std::ifstream in;
	in.open(p_filename.c_str());
	if (!in.is_open()){
		std::cerr << "ERROR -- Couldn't open file \ '" << p_filename << "\'.\n";
		exit(-1);
	}


	char ch, type;
	char red, green, blue;
	int i, j, cols, rows;
	int num;

	in.get(ch);
	in.get(type);

	in >> cols >> rows >> num;

	width = cols;
	height = rows;

	image = new RGBColour[width * height];

	in.get(ch);

	for (i = height - 1; i > 0; i--){
		for (j = 0; j < width; j++){
			in.get(red);
			in.get(green);
			in.get(blue);
			int index = GetIndex(j, i);
			image[index] = RGBColour((float)((unsigned char)red) / 255.0,
				(float)((unsigned char)green) / 255.0,
				(float)((unsigned char)blue) / 255.0);

		}
	}

}