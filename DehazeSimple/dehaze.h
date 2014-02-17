/************************************************************************/
/* FileName:	dehaze.h												*/
/* Author:		Pang Liang												*/
/* Dependence:	OpenCV 2.3.1	&	SparseLib++					        */
/* Date:        2012-12-19                                              */
/************************************************************************/

/************************************************************************/
/* Paper: "Single Image Haze Removal Using Dark Channel Prior"         */
/*        I(x)=J(x)t(x)+A(1-t(x))                                       */
/************************************************************************/
#include "iostream"
#include <algorithm>
#include "time.h"
#include "string.h"
#include "io.h"

/****** OpenCV 2.3.1 *******/
#include "opencv2/opencv.hpp"

#define MAX_INT 20000000

using namespace std;
using namespace cv;

//Type of Min and Max value
typedef struct _MinMax
{
	double min;
	double max;
}MinMax;

Mat ReadImage();
void rerange();
void fill_x_y();
int find_table( int y );
void locate(int l1,int l2,double l3);
void getL(Mat img);
Mat hazefree(Mat img,Mat t,Vec<float,3> a);
Vec<float,3> Airlight(Mat img, Mat dark);
Mat TransmissionMat(Mat dark);
Mat DarkChannelPrior(Mat img);
MinMax MaxAndMinOfMatirx( Mat x );
void RefineTrans(Mat trans);

void writeLFile();
void printMat(char * name,Mat m);