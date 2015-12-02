#pragma once

class RGBColour {
public:
	float r, g, b;
public:
	RGBColour() { r = g = b = 0; }
	inline RGBColour(float red, float green, float blue) : r(red), g(green), b(blue) {}
	inline RGBColour(float p_col) : r(p_col), g(p_col), b(p_col) {}
	RGBColour(const RGBColour& original) { *this = original; }

	void setRed(float red) { r = red; }
	void setGreen(float green) { g = green; }
	void setBlue(float blue) { b = blue; }

	inline RGBColour& operator=(const RGBColour& right){
		r = right.r;
		g = right.g;
		b = right.b;

		return *this;
	}


	inline RGBColour& operator+=(const RGBColour& right) {
		*this = *this + right;
		return *this;
	}
	inline RGBColour& operator*=(const RGBColour& right){
		*this = *this * right;
		return *this;
	}
	inline RGBColour& operator/=(const RGBColour& right)
	{
		*this = *this / right;
		return *this;
	}
	inline RGBColour& operator*=(float right){
		*this = *this * right;
		return *this;
	}
	inline RGBColour& operator/=(float right){
		*this = *this / right;
		return *this;
	}

	RGBColour operator+() const { return *this; }
	RGBColour operator-() const { return RGBColour(-r, -g, -b); }



	inline void clamp(){
		if (r > 1.0f) r = 1.0f;
		if (g > 1.0f) g = 1.0f;
		if (b > 1.0f) b = 1.0f;

		if (r < 0.0f) r = 0.0f;
		if (g < 0.0f) g = 0.0f;
		if (b < 0.0f) b = 0.0f;
	}

	inline friend std::ostream& operator<<(std::ostream &out, const RGBColour& rgb){
		out << rgb.r << ' ' << rgb.g << ' ' << rgb.b << ' ';
		return out;
	}

	inline friend RGBColour operator*(const RGBColour& c, float f)
	{
		return RGBColour(c.r*f, c.g*f, c.b*f);
	}

	inline friend RGBColour operator*(float f, const RGBColour& c){
		return RGBColour(c.r*f, c.g*f, c.b*f);
	}
	inline friend RGBColour operator/(const RGBColour& c, float f)
	{
		return RGBColour(c.r / f, c.g / f, c.b / f);
	}
	inline friend RGBColour operator*(const RGBColour& c1, const RGBColour& c2){
		return RGBColour(c1.r * c2.r, c1.g * c2.g, c1.b * c2.b);
	}
	inline friend RGBColour operator/(const RGBColour& c1, const RGBColour& c2){
		return RGBColour(c1.r / c2.r, c1.g / c2.g, c1.b / c2.b);
	}
	inline friend RGBColour operator+(const RGBColour& c1, const RGBColour& c2){
		return RGBColour(c1.r + c2.r, c1.g + c2.g, c1.b + c2.b);
	}


	inline friend bool operator==(const RGBColour& r1, const RGBColour& r2){
		if (r1.r != r2.r) return false;
		if (r1.g != r2.g) return false;
		if (r1.b != r2.b) return false;
		return true;
	}

	inline bool IsBlack() {
		return r == 0 && g == 0 && b == 0;
	}
};
