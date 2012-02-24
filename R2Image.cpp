 // Source file for image class



// Include files 

#include "R2/R2.h"
#include "R2Pixel.h"
#include "R2Image.h"
#include <vector>
#include <iostream>




////////////////////////////////////////////////////////////////////////
// Constructors/Destructors
////////////////////////////////////////////////////////////////////////


R2Image::
R2Image(void)
  : pixels(NULL),
    npixels(0),
    width(0), 
    height(0)
{
}



R2Image::
R2Image(const char *filename)
  : pixels(NULL),
    npixels(0),
    width(0), 
    height(0)
{
  // Read image
  Read(filename);
}



R2Image::
R2Image(int width, int height)
  : pixels(NULL),
    npixels(width * height),
    width(width), 
    height(height)
{
  // Allocate pixels
  pixels = new R2Pixel [ npixels ];
  assert(pixels);
}



R2Image::
R2Image(int width, int height, const R2Pixel *p)
  : pixels(NULL),
    npixels(width * height),
    width(width), 
    height(height)
{
  // Allocate pixels
  pixels = new R2Pixel [ npixels ];
  assert(pixels);

  // Copy pixels 
  for (int i = 0; i < npixels; i++) 
    pixels[i] = p[i];
}



R2Image::
R2Image(const R2Image& image)
  : pixels(NULL),
    npixels(image.npixels),
    width(image.width), 
    height(image.height)
    
{
  // Allocate pixels
  pixels = new R2Pixel [ npixels ];
  assert(pixels);

  // Copy pixels 
  for (int i = 0; i < npixels; i++) 
    pixels[i] = image.pixels[i];
}



R2Image::
~R2Image(void)
{
  // Free image pixels
  if (pixels) delete [] pixels;
}



R2Image& R2Image::
operator=(const R2Image& image)
{
  // Delete previous pixels
  if (pixels) { delete [] pixels; pixels = NULL; }

  // Reset width and height
  npixels = image.npixels;
  width = image.width;
  height = image.height;

  // Allocate new pixels
  pixels = new R2Pixel [ npixels ];
  assert(pixels);

  // Copy pixels 
  for (int i = 0; i < npixels; i++) 
    pixels[i] = image.pixels[i];

  // Return image
  return *this;
}



////////////////////////////////////////////////////////////////////////
// Image processing functions
// YOU IMPLEMENT THE FUNCTIONS IN THIS SECTION
////////////////////////////////////////////////////////////////////////

// Per-pixel Operations ////////////////////////////////////////////////

void R2Image::
Brighten(double factor)
{
  // Brighten the image by multiplying each pixel component by the factor,
  // then clamping the result to a valid range.

   int i;
	for (i = 0; i < npixels; i++)
	{
		R2Pixel cur = pixels[i];
		double red = cur.Red();
		double green = cur.Green();
		double blue = cur.Blue();
		double alpha = cur.Alpha();

		red *= factor;
		green *= factor;
		blue *= factor;
		
		pixels[i].Reset(red, green, blue, alpha);
	 	pixels[i].Clamp(1.0);
	}

}

void R2Image::
AddNoise(double factor)
{
  // Add noise to an image.  The amount of noise is given by the factor
  // in the range [0.0..1.0].  0.0 adds no noise.  1.0 adds a lot of noise.

  fprintf(stderr, "AddNoise(%g) not implemented\n", factor);
}

void R2Image::
ChangeContrast(double factor)
{
  // Change the contrast of an image by interpolating between the image
  // and a constant gray image with the average luminance.
  // Interpolation reduces constrast, extrapolation boosts constrast,
  // and negative factors generate inverted images.

   int i;
	double total_lumin = 0;
	for (i = 0; i < npixels; i++)
	{
		total_lumin += pixels[i].Luminance();
	}
	total_lumin /= npixels;

	for (i = 0; i < npixels; i++)
	{
		double red = pixels[i].Red();
		double green = pixels[i].Green();
		double blue = pixels[i].Blue();
		double alpha = pixels[i].Alpha();

		red *= factor;
		green *= factor;
		blue *= factor;

		red += (1 - factor) * total_lumin;
		green += (1 - factor) * total_lumin;
		blue += (1 - factor) * total_lumin;

		pixels[i].Reset(red, green, blue, alpha);
		pixels[i].Clamp();
	}


}

void R2Image::
ChangeSaturation(double factor)
{
  // Changes the saturation of an image by interpolating between the
  // image and a gray level version of the image.  Interpolation
  // decreases saturation, extrapolation increases it, negative factors
  // preserve luminance  but invert the hue of the input image.

  //generate black and white image
  int i;

	for (i = 0; i < npixels; i++)
	{
		double red = pixels[i].Red();
		double green = pixels[i].Green();
		double blue = pixels[i].Blue();
		double alpha = pixels[i].Alpha();

		red *= factor;
		green *= factor;
		blue *= factor;

		red += (1 - factor) * pixels[i].Luminance();
		green += (1 - factor) * pixels[i].Luminance();
		blue += (1 - factor) * pixels[i].Luminance();

		pixels[i].Reset(red, green, blue, alpha);
		pixels[i].Clamp();
	}  
}

void R2Image::
ApplyGamma(double exponent)
{
  // Apply a gamma correction with exponent to each pixel
  int i;
  for (i = 0; i < npixels; i++)
  {
	  double red = pixels[i].Red();
		double green = pixels[i].Green();
		double blue = pixels[i].Blue();

		red = pow (red, exponent);
		green = pow (green, exponent);
		blue = pow (blue, exponent);

		pixels[i].Reset(red, green, blue, pixels[i].Alpha());
		pixels[i].Clamp();
  }

}

void R2Image::
BlackAndWhite(void)
{
  // Replace each pixel with its luminance value
  // Put this in each channel,  so the result is grayscale
   int i;
	for (i = 0; i < npixels; i++)
	{
		double alpha = pixels[i].Alpha();
		double lumin = pixels[i].Luminance();
		
		pixels[i].Reset(lumin, lumin, lumin, alpha);
	}


}

// Linear filtering ////////////////////////////////////////////////


void R2Image::
Blur(double sigma)
{
	// Blur an image with a Gaussian filter with a given sigma.
	int n = ceil(3* sigma) * 2 + 1;
	vector <vector <double> > filter = gaussian_filter(n, sigma); 

	R2Image copy(*this);
	copy.ApplyGamma(2.2);
	ApplyGamma(2.2);

	for (int i = 0; i < npixels; i++)
	{
		R2Pixel pixel(0,0,0,1);
		int x = i / height; // width;
		int y = i % height;/// width;

		
		double total = 0;
		for (int srcx = x - n/2; srcx < x + n/2; srcx++)
		{
			for(int srcy = y - n/2; srcy < y + n/2; srcy++)
			{
				//edge cases
				if(srcx < 0 || srcx >= width) continue;
				if(srcy < 0 || srcy >= height) continue;

				total += filter[abs(srcx - x)][abs(srcy - y)];
				pixel += copy.Pixel(srcx, srcy)*filter[abs(srcx-x)][abs(srcy-y)];

			}

		}
		pixels[i] = pixel/total;

	}
	ApplyGamma(1/2.2);
}


double R2Image::
gaussian (int row, int col, double sigma, int n)
{
	double distance = dist(col, row);

	return exp(-1* pow(distance, 2)/ ( 2 * pow(sigma , 2)));
}

double R2Image::
dist(int x, int y)
{
	return sqrt(pow(x, 2) + pow(y, 2));
}


vector < vector<double> > R2Image::
gaussian_filter(int n, double sigma)
{
	vector<vector <double> > filter (n , vector<double> (n));

	double sum = 0;
	for (int row = 0; row < filter.size(); row++)
	{
		for (int col = 0; col < filter[row].size(); col++)
		{
			filter[row][col] = gaussian(row, col, sigma, 0);
			sum += filter[row][col];
		}
	}

	//normalize
	for (int row = 0; row < filter.size(); row++)
	{
		for (int col = 0; col < filter[row].size(); col++) {
			filter[row][col] /= sum;
		}
	}

	return filter;
}


void R2Image::
Sharpen()
{
  // Sharpen an image using a linear filter
  R2Image copy(*this);
	copy.Blur(4.0);

  int i;
	int factor = 3;

	for (i = 0; i < npixels; i++)
	{
		double red = pixels[i].Red();
		double green = pixels[i].Green();
		double blue = pixels[i].Blue();
		double alpha = pixels[i].Alpha();

		red *= factor;
		green *= factor;
		blue *= factor;

		red += (1 - factor) * copy.pixels[i].Red();
		green += (1 - factor) * copy.pixels[i].Green();
		blue += (1 - factor) * copy.pixels[i].Blue();

		pixels[i].Reset(red, green, blue, alpha);
		pixels[i].Clamp();
	}  
	
}


void R2Image::
EdgeDetect(void)
{
  // Detect edges in an image.
	int n = 3;
	int filter [3][3] = {{-1,-1,-1}, {-1, 8, -1}, {-1, -1, -1}};
  R2Image copy(*this);

	ApplyGamma(2.2);
	copy.ApplyGamma(2.2);

	for (int i = 0; i < npixels; i++)
	{
		R2Pixel pixel(0,0,0,1);
		int x = i / height; 
		int y = i % height;

		double total = 0;
		for (int srcy = 0; srcy < n; srcy++)
		{
			for (int srcx = 0; srcx < n; srcx++)
			{
				if (x + srcx - 1 < 0 || x + srcx - 1 > width) continue;
				if (y + srcy - 1 < 0 || y + srcy - 1 > height) continue;
				int current_filter = filter[srcx][srcy];
				pixel += copy.Pixel(x + srcx - 1, y + srcy - 1) * current_filter; 
			}
		}

		pixels[i] = pixel/9.0;
		pixels[i].Clamp();
	}
  ApplyGamma(1/2.2);
}



void R2Image::
MotionBlur(int amount)
{
  // Perform horizontal motion blur

  // convolve in X direction with a linear ramp of amount non-zero pixels
  // the image should be strongest on the right hand side (see example)
	ApplyGamma(2.2);
	ApplyGamma(1/2.2);
  
}


// Non-Linear filtering ////////////////////////////////////////////////

void R2Image::
MedianFilter(double sigma)
{
  // Perform median filtering with a given width
	R2Image copy (*this);
	ApplyGamma(2.2);
	copy.ApplyGamma(2.2);

	int window = sigma * 2 + 1;

	for (int y = 0; y < height; y++)
	{	
		for (int x = 0; x < width; x++)
		{
			R2Pixel cur_pix = Pixel(x,y);
			for (int channel = 0; channel < 4; channel++)
			{
				vector <double> color_array;

				for (int fy = y - window/2; fy < y + window; fy++)
				{
					for (int fx = x - window/2; fx < x + window; fx++)
					{
						if (fy < 0 || fy > height) continue;
						if (fx < 0 || fx > width) continue;
						color_array.push_back(Pixel(fx,fy)[channel]);
					}

				}
				sort(color_array.begin(), color_array.end());
				if (window % 2 != 0)
				{
					Pixel(x, y)[channel] = 
						color_array[color_array.size()/2];
				}
				else
				{
					double median_fst = color_array[color_array.size()/2];
					double median_snd = color_array[color_array.size()/2 + 1]; 
					Pixel(x,y)[channel] = (median_fst + median_snd)/2;
					pixels[y * width + x].Clamp();
				}
			}

		}
	}

	ApplyGamma(1/2.2);
}


void R2Image::
BilateralFilter(double rangesigma, double domainsigma)
{
  // Perform bilateral filtering with a given range and domain widths.
	ApplyGamma(2.2);
	R2Image copy(*this);

	int window = ceil(3* domainsigma) * 2 + 1;
	vector <vector <double> > filter = gaussian_filter(window, domainsigma);

  for (int y = 0; y < height; y++)
	{	
		for (int x = 0; x < width; x++)
		{
			R2Pixel pixel(0,0,0,1);

//			for (int channel = 0; channel < 4; channel++)
//			{
				double total = 0;
				for (int fy = y - window/2; fy < y + window/2; fy++)
				{
					for (int fx = x - window/2; fx < x + window/2; fx++)
					{
						if (fy < 0 || fy > height) continue;
						if (fx < 0 || fx > width) continue;

						double red_sq = pow((copy.Pixel(x,y) 
									- copy.Pixel(fx, fy))[0], 2.0);
						double green_sq = pow((copy.Pixel(x,y) 
									- copy.Pixel(fx, fy))[1], 2.0);
						double blue_sq = pow((copy.Pixel(x,y) 
									- copy.Pixel(fx, fy))[2], 2.0);


						double similarity = sqrt(red_sq + green_sq + blue_sq);					
						double filt_val = filter[abs(fx - x)][abs(fy - y)] 
							       *exp(-similarity*similarity/2.0/rangesigma/rangesigma);

					//	printf("%f\n", filt_val);

						total += filt_val;
						pixel += copy.Pixel(fx, fy) * filt_val;
					}
				}
				Pixel(x,y) = pixel/total;
				Pixel(x,y).Clamp();
			}
		}
		ApplyGamma(1/2.2);
}


// Resampling operations  ////////////////////////////////////////////////


void R2Image::
Scale(double sx, double sy, int sampling_method)
{
  // Scale an image in x by sx, and y by sy.

}


void R2Image::
Rotate(double angle, int sampling_method)
{
  // Rotate an image by the given angle.

}


void R2Image::
Fun(int sampling_method)
{
  // Warp an image using a creative filter of your choice.

}


// Dither operations ////////////////////////////////////////////////

void R2Image::
Quantize (int nbits)
{
  // Quantizes an image with "nbits" bits per channel.
   int i;
	double levels = pow(2, nbits) -1;
	for (i = 0; i < npixels; i++)
	{
		for(int channel = 0; channel < 4; channel++)
		{
			pixels[i][channel] = round(pixels[i][channel]*levels)/levels;
		}
	}

}



void R2Image::
RandomDither(int nbits)
{
  // Converts and image to nbits per channel using random dither.
   int i;
	double levels = pow(2, nbits) -1;
	for (i = 0; i < npixels; i++)
	{
		for(int channel=0; channel<4; channel++)
		{
			double noise = 1;
			while (noise > 0.5)
				noise = 2 * (double) rand()/ (double) RAND_MAX - 0.5;
			pixels[i][channel] = 
				round(pixels[i][channel]*levels + noise)/levels;
		}
	}

  
}



void R2Image::
OrderedDither(int nbits)
{
  // Converts an image to nbits per channel using ordered dither, 
  // with a 4x4 Bayer's pattern matrix.
	
  int bayer[4][4] = {{15, 7, 13, 5}, {3, 11, 1, 9},  
	  {12, 4, 14, 6}, {0, 8 , 2, 10}};
  

   int i;
	double levels = pow(2, nbits) -1;
	for (i = 0; i < npixels; i++)
	{
		int y = i % height;
		int x = i / height;
		int x_temp = x % 4;
		int y_temp = y % 4;
		for(int channel=0; channel<4; channel++)
		{
			pixels[i][channel] = 
				round(pixels[i][channel]*levels + 0.5 
						+ bayer[x_temp][y_temp]/16.0)/levels;
		}
	}


}



void R2Image::
FloydSteinbergDither(int nbits)
{
  // Converts an image to nbits per channel using Floyd-Steinberg dither.
  // with error diffusion.
	double matrix[4] = {7.0/16, 3.0/16, 5.0/16, 1.0/16};
	double levels = pow(2, nbits) -1;

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			R2Pixel cur_pix = Pixel(x,y);
			for (int channel = 0; channel < 4; channel++)
			{
				double cur_val = cur_pix[channel];
				double quant_val = round(cur_val * levels)/levels;
				Pixel(x,y)[channel] = quant_val;
				//Pixel(x,y).Clamp();
				double error = cur_val - quant_val;
				
				if (x + 1 <	width)
				{
					Pixel(x+1, y)[channel] += matrix[0] * error;
					Pixel(x+1, y).Clamp();
				}
				if (y + 1 < height && x - 1 >= 0)
				{
					Pixel(x - 1, y + 1)[channel] += matrix[1] * error;
					Pixel(x-1, y+1).Clamp();
				}
				if (y + 1 < height)
				{
					Pixel(x, y + 1)[channel] += matrix[2] * error;
					Pixel(x, y + 1).Clamp();
				}
				if (y + 1 < height && x + 1 < width)
				{
					Pixel(x + 1, y + 1)[channel] += matrix[3] * error;
				  Pixel(x + 1, y + 1).Clamp();
				}

			}
		}
	}
}



// Miscellaneous operations ////////////////////////////////////////////////

void R2Image::
Crop(int x, int y, int w, int h)
{
  // Extracts a sub image from the image, 
  // at position (x, y), width w, and height h.
  
  if (x + w > width)
	  return;
	if (y + h > height)
		return;

	//get the starting pixel
	int new_npixels = w * h;
	R2Pixel new_pic [new_npixels];
	
	int i;
	int counter = 0;
	for (i = y; i < y + h; i++)
	{
		int arr_pos = width * i + x;
		int j;
		for (j = arr_pos; j < w + arr_pos; j++)
		{
			new_pic[counter] = pixels[j];			
			counter++;
		}
	}
	pixels = new_pic;
	npixels = new_npixels;
	width = w;
	height = h;
}

void R2Image::
Composite(const R2Image& top, int operation)
{
  // Composite passed image on top of this one using operation (e.g., OVER)

  // FILL IN IMPLEMENTATION HERE (REMOVE PRINT STATEMENT WHEN DONE)
  fprintf(stderr, "Composite not implemented\n");
}

void R2Image::
ExtractChannel(int channel)
{
  // Extracts a channel of an image (e.g., R2_IMAGE_RED_CHANNEL).  
  // Leaves the specified channel intact, 
  // and sets all the other ones to zero.

  int i;
  switch (channel)
  {
  		case R2_IMAGE_RED_CHANNEL:
			  for (i = 0; i < npixels; i++)
	  		  		pixels[i].Reset(pixels[i].Red(), 0, 0, 0);
		 	  break;
	   case R2_IMAGE_GREEN_CHANNEL:
			  for (i = 0; i < npixels; i++)
			  	   pixels[i].Reset(0, pixels[i].Green(), 0, 0);
			  break;
		case R2_IMAGE_BLUE_CHANNEL:
			  for (i = 0; i < npixels; i++)
			  		pixels[i].Reset(0, 0, pixels[i].Blue(), 0);
		  	  break;
	   case R2_IMAGE_ALPHA_CHANNEL:
			  for (i = 0; i < npixels; i++)
					pixels[i].Reset(0, 0, 0, pixels[i].Alpha());
			  break;
			
  }
}

void R2Image::
CopyChannel(const R2Image& from_image, int from_channel, int to_channel)
{
  // Copies one channel of an image (e.g., R2_IMAGE_RED_CHANNEL).  
  // to another channel

  // IMPLEMENT AS PREREQUISITE OF THE Composite() ASSIGNMENT  (REMOVE PRINT STATEMENT WHEN DONE)
  fprintf(stderr, "CopyChannel not implemented\n");
}

void R2Image::
Add(const R2Image& image)
{
  // Add passed image pixel-by-pixel.

  // MAY FILL IN IMPLEMENTATION HERE (REMOVE PRINT STATEMENT WHEN DONE) (NO CREDIT FOR ASSIGNMENT)
  fprintf(stderr, "Add not implemented\n");
}

void R2Image::
Subtract(const R2Image& image)
{
  // Subtract passed image from this image.

  // MAY FILL IN IMPLEMENTATION HERE (REMOVE PRINT STATEMENT WHEN DONE) (NO CREDIT FOR ASSIGNMENT)
  fprintf(stderr, "Subtract not implemented\n");
}

void R2Image::
Morph(const R2Image& target, 
  R2Segment *source_segments, R2Segment *target_segments, int nsegments, 
  double t, int sampling_method)
{
  // Morph this source image towards a passed target image by t using pairwise line segment correspondences

  // FILL IN IMPLEMENTATION HERE (REMOVE PRINT STATEMENT WHEN DONE)
  fprintf(stderr, "Morph not implemented\n");
}


////////////////////////////////////////////////////////////////////////
// I/O Functions
////////////////////////////////////////////////////////////////////////

int R2Image::
Read(const char *filename)
{
  // Initialize everything
  if (pixels) { delete [] pixels; pixels = NULL; }
  npixels = width = height = 0;

  // Parse input filename extension
  char *input_extension;
  if (!(input_extension = (char*)strrchr(filename, '.'))) {
    fprintf(stderr, "Input file has no extension (e.g., .jpg).\n");
    return 0;
  }
  
  // Read file of appropriate type
  if (!strncmp(input_extension, ".bmp", 4)) return ReadBMP(filename);
  else if (!strncmp(input_extension, ".ppm", 4)) return ReadPPM(filename);
  else if (!strncmp(input_extension, ".jpg", 4)) return ReadJPEG(filename);
  else if (!strncmp(input_extension, ".jpeg", 5)) return ReadJPEG(filename);
  
  // Should never get here
  fprintf(stderr, "Unrecognized image file extension");
  return 0;
}



int R2Image::
Write(const char *filename) const
{
  // Parse input filename extension
  char *input_extension;
  if (!(input_extension = (char*)strrchr(filename, '.'))) {
    fprintf(stderr, "Input file has no extension (e.g., .jpg).\n");
    return 0;
  }
  
  // Write file of appropriate type
  if (!strncmp(input_extension, ".bmp", 4)) return WriteBMP(filename);
  else if (!strncmp(input_extension, ".ppm", 4)) return WritePPM(filename, 1);
  else if (!strncmp(input_extension, ".jpg", 5)) return WriteJPEG(filename);
  else if (!strncmp(input_extension, ".jpeg", 5)) return WriteJPEG(filename);

  // Should never get here
  fprintf(stderr, "Unrecognized image file extension");
  return 0;
}



////////////////////////////////////////////////////////////////////////
// BMP I/O
////////////////////////////////////////////////////////////////////////

#if !defined(_WIN32)

typedef struct tagBITMAPFILEHEADER {
  unsigned short int bfType;
  unsigned int bfSize;
  unsigned short int bfReserved1;
  unsigned short int bfReserved2;
  unsigned int bfOffBits;
} BITMAPFILEHEADER;

typedef struct tagBITMAPINFOHEADER {
  unsigned int biSize;
  int biWidth;
  int biHeight;
  unsigned short int biPlanes;
  unsigned short int biBitCount;
  unsigned int biCompression;
  unsigned int biSizeImage;
  int biXPelsPerMeter;
  int biYPelsPerMeter;
  unsigned int biClrUsed;
  unsigned int biClrImportant;
} BITMAPINFOHEADER;

typedef struct tagRGBTRIPLE {
  unsigned char rgbtBlue;
  unsigned char rgbtGreen;
  unsigned char rgbtRed;
} RGBTRIPLE;

typedef struct tagRGBQUAD {
  unsigned char rgbBlue;
  unsigned char rgbGreen;
  unsigned char rgbRed;
  unsigned char rgbReserved;
} RGBQUAD;

#endif

#define BI_RGB        0L
#define BI_RLE8       1L
#define BI_RLE4       2L
#define BI_BITFIELDS  3L

#define BMP_BF_TYPE 0x4D42 /* word BM */
#define BMP_BF_OFF_BITS 54 /* 14 for file header + 40 for info header (not sizeof(), but packed size) */
#define BMP_BI_SIZE 40 /* packed size of info header */


static unsigned short int WordReadLE(FILE *fp)
{
  // Read a unsigned short int from a file in little endian format 
  unsigned short int lsb, msb;
  lsb = getc(fp);
  msb = getc(fp);
  return (msb << 8) | lsb;
}



static void WordWriteLE(unsigned short int x, FILE *fp)
{
  // Write a unsigned short int to a file in little endian format
  unsigned char lsb = (unsigned char) (x & 0x00FF); putc(lsb, fp); 
  unsigned char msb = (unsigned char) (x >> 8); putc(msb, fp);
}



static unsigned int DWordReadLE(FILE *fp)
{
  // Read a unsigned int word from a file in little endian format 
  unsigned int b1 = getc(fp);
  unsigned int b2 = getc(fp);
  unsigned int b3 = getc(fp);
  unsigned int b4 = getc(fp);
  return (b4 << 24) | (b3 << 16) | (b2 << 8) | b1;
}



static void DWordWriteLE(unsigned int x, FILE *fp)
{
  // Write a unsigned int to a file in little endian format 
  unsigned char b1 = (x & 0x000000FF); putc(b1, fp);
  unsigned char b2 = ((x >> 8) & 0x000000FF); putc(b2, fp);
  unsigned char b3 = ((x >> 16) & 0x000000FF); putc(b3, fp);
  unsigned char b4 = ((x >> 24) & 0x000000FF); putc(b4, fp);
}



static int LongReadLE(FILE *fp)
{
  // Read a int word from a file in little endian format 
  int b1 = getc(fp);
  int b2 = getc(fp);
  int b3 = getc(fp);
  int b4 = getc(fp);
  return (b4 << 24) | (b3 << 16) | (b2 << 8) | b1;
}



static void LongWriteLE(int x, FILE *fp)
{
  // Write a int to a file in little endian format 
  char b1 = (x & 0x000000FF); putc(b1, fp);
  char b2 = ((x >> 8) & 0x000000FF); putc(b2, fp);
  char b3 = ((x >> 16) & 0x000000FF); putc(b3, fp);
  char b4 = ((x >> 24) & 0x000000FF); putc(b4, fp);
}



int R2Image::
ReadBMP(const char *filename)
{
  // Open file
  FILE *fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Unable to open image file: %s\n", filename);
    return 0;
  }

  /* Read file header */
  BITMAPFILEHEADER bmfh;
  bmfh.bfType = WordReadLE(fp);
  bmfh.bfSize = DWordReadLE(fp);
  bmfh.bfReserved1 = WordReadLE(fp);
  bmfh.bfReserved2 = WordReadLE(fp);
  bmfh.bfOffBits = DWordReadLE(fp);
  
  /* Check file header */
  assert(bmfh.bfType == BMP_BF_TYPE);
  /* ignore bmfh.bfSize */
  /* ignore bmfh.bfReserved1 */
  /* ignore bmfh.bfReserved2 */
  assert(bmfh.bfOffBits == BMP_BF_OFF_BITS);
  
  /* Read info header */
  BITMAPINFOHEADER bmih;
  bmih.biSize = DWordReadLE(fp);
  bmih.biWidth = LongReadLE(fp);
  bmih.biHeight = LongReadLE(fp);
  bmih.biPlanes = WordReadLE(fp);
  bmih.biBitCount = WordReadLE(fp);
  bmih.biCompression = DWordReadLE(fp);
  bmih.biSizeImage = DWordReadLE(fp);
  bmih.biXPelsPerMeter = LongReadLE(fp);
  bmih.biYPelsPerMeter = LongReadLE(fp);
  bmih.biClrUsed = DWordReadLE(fp);
  bmih.biClrImportant = DWordReadLE(fp);
  
  // Check info header 
  assert(bmih.biSize == BMP_BI_SIZE);
  assert(bmih.biWidth > 0);
  assert(bmih.biHeight > 0);
  assert(bmih.biPlanes == 1);
  assert(bmih.biBitCount == 24);  /* RGB */
  assert(bmih.biCompression == BI_RGB);   /* RGB */
  int lineLength = bmih.biWidth * 3;  /* RGB */
  if ((lineLength % 4) != 0) lineLength = (lineLength / 4 + 1) * 4;
  assert(bmih.biSizeImage == (unsigned int) lineLength * (unsigned int) bmih.biHeight);

  // Assign width, height, and number of pixels
  width = bmih.biWidth;
  height = bmih.biHeight;
  npixels = width * height;

  // Allocate unsigned char buffer for reading pixels
  int rowsize = 3 * width;
  if ((rowsize % 4) != 0) rowsize = (rowsize / 4 + 1) * 4;
  int nbytes = bmih.biSizeImage;
  unsigned char *buffer = new unsigned char [nbytes];
  if (!buffer) {
    fprintf(stderr, "Unable to allocate temporary memory for BMP file");
    fclose(fp);
    return 0;
  }

  // Read buffer 
  fseek(fp, (long) bmfh.bfOffBits, SEEK_SET);
  if (fread(buffer, 1, bmih.biSizeImage, fp) != bmih.biSizeImage) {
    fprintf(stderr, "Error while reading BMP file %s", filename);
    return 0;
  }

  // Close file
  fclose(fp);

  // Allocate pixels for image
  pixels = new R2Pixel [ width * height ];
  if (!pixels) {
    fprintf(stderr, "Unable to allocate memory for BMP file");
    fclose(fp);
    return 0;
  }

  // Assign pixels
  for (int j = 0; j < height; j++) {
    unsigned char *p = &buffer[j * rowsize];
    for (int i = 0; i < width; i++) {
      double b = (double) *(p++) / 255;
      double g = (double) *(p++) / 255;
      double r = (double) *(p++) / 255;
      R2Pixel pixel(r, g, b, 1);
      SetPixel(i, j, pixel);
    }
  }

  // Free unsigned char buffer for reading pixels
  delete [] buffer;

  // Return success
  return 1;
}



int R2Image::
WriteBMP(const char *filename) const
{
  // Open file
  FILE *fp = fopen(filename, "wb");
  if (!fp) {
    fprintf(stderr, "Unable to open image file: %s\n", filename);
    return 0;
  }

  // Compute number of bytes in row
  int rowsize = 3 * width;
  if ((rowsize % 4) != 0) rowsize = (rowsize / 4 + 1) * 4;

  // Write file header 
  BITMAPFILEHEADER bmfh;
  bmfh.bfType = BMP_BF_TYPE;
  bmfh.bfSize = BMP_BF_OFF_BITS + rowsize * height;
  bmfh.bfReserved1 = 0;
  bmfh.bfReserved2 = 0;
  bmfh.bfOffBits = BMP_BF_OFF_BITS;
  WordWriteLE(bmfh.bfType, fp);
  DWordWriteLE(bmfh.bfSize, fp);
  WordWriteLE(bmfh.bfReserved1, fp);
  WordWriteLE(bmfh.bfReserved2, fp);
  DWordWriteLE(bmfh.bfOffBits, fp);

  // Write info header 
  BITMAPINFOHEADER bmih;
  bmih.biSize = BMP_BI_SIZE;
  bmih.biWidth = width;
  bmih.biHeight = height;
  bmih.biPlanes = 1;
  bmih.biBitCount = 24;       /* RGB */
  bmih.biCompression = BI_RGB;    /* RGB */
  bmih.biSizeImage = rowsize * (unsigned int) bmih.biHeight;  /* RGB */
  bmih.biXPelsPerMeter = 2925;
  bmih.biYPelsPerMeter = 2925;
  bmih.biClrUsed = 0;
  bmih.biClrImportant = 0;
  DWordWriteLE(bmih.biSize, fp);
  LongWriteLE(bmih.biWidth, fp);
  LongWriteLE(bmih.biHeight, fp);
  WordWriteLE(bmih.biPlanes, fp);
  WordWriteLE(bmih.biBitCount, fp);
  DWordWriteLE(bmih.biCompression, fp);
  DWordWriteLE(bmih.biSizeImage, fp);
  LongWriteLE(bmih.biXPelsPerMeter, fp);
  LongWriteLE(bmih.biYPelsPerMeter, fp);
  DWordWriteLE(bmih.biClrUsed, fp);
  DWordWriteLE(bmih.biClrImportant, fp);

  // Write image, swapping blue and red in each pixel
  int pad = rowsize - width * 3;
  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      const R2Pixel& pixel = (*this)[i][j];
      double r = 255.0 * pixel.Red();
      double g = 255.0 * pixel.Green();
      double b = 255.0 * pixel.Blue();
      if (r >= 255) r = 255;
      if (g >= 255) g = 255;
      if (b >= 255) b = 255;
      fputc((unsigned char) b, fp);
      fputc((unsigned char) g, fp);
      fputc((unsigned char) r, fp);
    }

    // Pad row
    for (int i = 0; i < pad; i++) fputc(0, fp);
  }
  
  // Close file
  fclose(fp);

  // Return success
  return 1;  
}



////////////////////////////////////////////////////////////////////////
// PPM I/O
////////////////////////////////////////////////////////////////////////

int R2Image::
ReadPPM(const char *filename)
{
  // Open file
  FILE *fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Unable to open image file: %s\n", filename);
    return 0;
  }

  // Read PPM file magic identifier
  char buffer[128];
  if (!fgets(buffer, 128, fp)) {
    fprintf(stderr, "Unable to read magic id in PPM file");
    fclose(fp);
    return 0;
  }

  // skip comments
  int c = getc(fp);
  while (c == '#') {
    while (c != '\n') c = getc(fp);
    c = getc(fp);
  }
  ungetc(c, fp);

  // Read width and height
  if (fscanf(fp, "%d%d", &width, &height) != 2) {
    fprintf(stderr, "Unable to read width and height in PPM file");
    fclose(fp);
    return 0;
  }

  npixels = width * height;
	
  // Read max value
  double max_value;
  if (fscanf(fp, "%lf", &max_value) != 1) {
    fprintf(stderr, "Unable to read max_value in PPM file");
    fclose(fp);
    return 0;
  }
	
  // Allocate image pixels
  pixels = new R2Pixel [ width * height ];
  if (!pixels) {
    fprintf(stderr, "Unable to allocate memory for PPM file");
    fclose(fp);
    return 0;
  }

  // Check if raw or ascii file
  if (!strcmp(buffer, "P6\n")) {
    // Read up to one character of whitespace (\n) after max_value
    int c = getc(fp);
    if (!isspace(c)) putc(c, fp);

    // Read raw image data 
    // First ppm pixel is top-left, so read in opposite scan-line order
    for (int j = height-1; j >= 0; j--) {
      for (int i = 0; i < width; i++) {
        double r = (double) getc(fp) / max_value;
        double g = (double) getc(fp) / max_value;
        double b = (double) getc(fp) / max_value;
        R2Pixel pixel(r, g, b, 1);
        SetPixel(i, j, pixel);
      }
    }
  }
  else {
    // Read asci image data 
    // First ppm pixel is top-left, so read in opposite scan-line order
    for (int j = height-1; j >= 0; j--) {
      for (int i = 0; i < width; i++) {
	// Read pixel values
	int red, green, blue;
	if (fscanf(fp, "%d%d%d", &red, &green, &blue) != 3) {
	  fprintf(stderr, "Unable to read data at (%d,%d) in PPM file", i, j);
	  fclose(fp);
	  return 0;
	}

	// Assign pixel values
	double r = (double) red / max_value;
	double g = (double) green / max_value;
	double b = (double) blue / max_value;
        R2Pixel pixel(r, g, b, 1);
        SetPixel(i, j, pixel);
      }
    }
  }

  // Close file
  fclose(fp);

  // Return success
  return 1;
}



int R2Image::
WritePPM(const char *filename, int ascii) const
{
  // Check type
  if (ascii) {
    // Open file
    FILE *fp = fopen(filename, "w");
    if (!fp) {
      fprintf(stderr, "Unable to open image file: %s\n", filename);
      return 0;
    }

    // Print PPM image file 
    // First ppm pixel is top-left, so write in opposite scan-line order
    fprintf(fp, "P3\n");
    fprintf(fp, "%d %d\n", width, height);
    fprintf(fp, "255\n");
    for (int j = height-1; j >= 0 ; j--) {
      for (int i = 0; i < width; i++) {
        const R2Pixel& p = (*this)[i][j];
        int r = (int) (255 * p.Red());
        int g = (int) (255 * p.Green());
        int b = (int) (255 * p.Blue());
        fprintf(fp, "%-3d %-3d %-3d  ", r, g, b);
        if (((i+1) % 4) == 0) fprintf(fp, "\n");
      }
      if ((width % 4) != 0) fprintf(fp, "\n");
    }
    fprintf(fp, "\n");

    // Close file
    fclose(fp);
  }
  else {
    // Open file
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
      fprintf(stderr, "Unable to open image file: %s\n", filename);
      return 0;
    }
    
    // Print PPM image file 
    // First ppm pixel is top-left, so write in opposite scan-line order
    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", width, height);
    fprintf(fp, "255\n");
    for (int j = height-1; j >= 0 ; j--) {
      for (int i = 0; i < width; i++) {
        const R2Pixel& p = (*this)[i][j];
        int r = (int) (255 * p.Red());
        int g = (int) (255 * p.Green());
        int b = (int) (255 * p.Blue());
        fprintf(fp, "%c%c%c", r, g, b);
      }
    }
    
    // Close file
    fclose(fp);
  }

  // Return success
  return 1;  
}



////////////////////////////////////////////////////////////////////////
// JPEG I/O
////////////////////////////////////////////////////////////////////////


extern "C" { 
#   define XMD_H // Otherwise, a conflict with INT32
#   undef FAR // Otherwise, a conflict with windows.h
#   include "jpeg/jpeglib.h"
};



int R2Image::
ReadJPEG(const char *filename)
{
  // Open file
  FILE *fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Unable to open image file: %s\n", filename);
    return 0;
  }

  // Initialize decompression info
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, fp);
  jpeg_read_header(&cinfo, TRUE);
  jpeg_start_decompress(&cinfo);

  // Remember image attributes
  width = cinfo.output_width;
  height = cinfo.output_height;
  npixels = width * height;
  int ncomponents = cinfo.output_components;

  // Allocate pixels for image
  pixels = new R2Pixel [ npixels ];
  if (!pixels) {
    fprintf(stderr, "Unable to allocate memory for BMP file");
    fclose(fp);
    return 0;
  }

  // Allocate unsigned char buffer for reading image
  int rowsize = ncomponents * width;
  if ((rowsize % 4) != 0) rowsize = (rowsize / 4 + 1) * 4;
  int nbytes = rowsize * height;
  unsigned char *buffer = new unsigned char [nbytes];
  if (!buffer) {
    fprintf(stderr, "Unable to allocate temporary memory for JPEG file");
    fclose(fp);
    return 0;
  }

  // Read scan lines 
  // First jpeg pixel is top-left, so read pixels in opposite scan-line order
  while (cinfo.output_scanline < cinfo.output_height) {
    int scanline = cinfo.output_height - cinfo.output_scanline - 1;
    unsigned char *row_pointer = &buffer[scanline * rowsize];
    jpeg_read_scanlines(&cinfo, &row_pointer, 1);
  }

  // Free everything
  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);

  // Close file
  fclose(fp);

  // Assign pixels
  for (int j = 0; j < height; j++) {
    unsigned char *p = &buffer[j * rowsize];
    for (int i = 0; i < width; i++) {
      double r, g, b, a;
      if (ncomponents == 1) {
        r = g = b = (double) *(p++) / 255;
        a = 1;
      }
      else if (ncomponents == 1) {
        r = g = b = (double) *(p++) / 255;
        a = 1;
        p++;
      }
      else if (ncomponents == 3) {
        r = (double) *(p++) / 255;
        g = (double) *(p++) / 255;
        b = (double) *(p++) / 255;
        a = 1;
      }
      else if (ncomponents == 4) {
        r = (double) *(p++) / 255;
        g = (double) *(p++) / 255;
        b = (double) *(p++) / 255;
        a = (double) *(p++) / 255;
      }
      else {
        fprintf(stderr, "Unrecognized number of components in jpeg image: %d\n", ncomponents);
        return 0;
      }
      R2Pixel pixel(r, g, b, a);
      SetPixel(i, j, pixel);
    }
  }

  // Free unsigned char buffer for reading pixels
  delete [] buffer;

  // Return success
  return 1;
}


	

int R2Image::
WriteJPEG(const char *filename) const
{
  // Open file
  FILE *fp = fopen(filename, "wb");
  if (!fp) {
    fprintf(stderr, "Unable to open image file: %s\n", filename);
    return 0;
  }

  // Initialize compression info
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);
  jpeg_stdio_dest(&cinfo, fp);
  cinfo.image_width = width; 	/* image width and height, in pixels */
  cinfo.image_height = height;
  cinfo.input_components = 3;		/* # of color components per pixel */
  cinfo.in_color_space = JCS_RGB; 	/* colorspace of input image */
  cinfo.dct_method = JDCT_ISLOW;
  jpeg_set_defaults(&cinfo);
  cinfo.optimize_coding = TRUE;
  jpeg_set_quality(&cinfo, 75, TRUE);
  jpeg_start_compress(&cinfo, TRUE);
	
  // Allocate unsigned char buffer for reading image
  int rowsize = 3 * width;
  if ((rowsize % 4) != 0) rowsize = (rowsize / 4 + 1) * 4;
  int nbytes = rowsize * height;
  unsigned char *buffer = new unsigned char [nbytes];
  if (!buffer) {
    fprintf(stderr, "Unable to allocate temporary memory for JPEG file");
    fclose(fp);
    return 0;
  }

  // Fill buffer with pixels
  for (int j = 0; j < height; j++) {
    unsigned char *p = &buffer[j * rowsize];
    for (int i = 0; i < width; i++) {
      const R2Pixel& pixel = (*this)[i][j];
      int r = (int) (255 * pixel.Red());
      int g = (int) (255 * pixel.Green());
      int b = (int) (255 * pixel.Blue());
      if (r > 255) r = 255;
      if (g > 255) g = 255;
      if (b > 255) b = 255;
      *(p++) = r;
      *(p++) = g;
      *(p++) = b;
    }
  }



  // Output scan lines
  // First jpeg pixel is top-left, so write in opposite scan-line order
  while (cinfo.next_scanline < cinfo.image_height) {
    int scanline = cinfo.image_height - cinfo.next_scanline - 1;
    unsigned char *row_pointer = &buffer[scanline * rowsize];
    jpeg_write_scanlines(&cinfo, &row_pointer, 1);
  }

  // Free everything
  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);

  // Close file
  fclose(fp);

  // Free unsigned char buffer for reading pixels
  delete [] buffer;

  // Return number of bytes written
  return 1;
}






