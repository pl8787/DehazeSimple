/************************************************************************/
/* FileName:	dehaze.cpp												*/
/* Author:		Pang Liang												*/
/* Dependence:	OpenCV 2.3.1	&	SparseLib++					        */
/* Date:        2012-12-19                                              */
/************************************************************************/

/************************************************************************/
/* Paper: "Single Image Haze Removal Using Dark Channel Prior"         */
/*        I(x)=J(x)t(x)+A(1-t(x))                                       */
/************************************************************************/

#include "dehaze.h"

using namespace std;
using namespace cv;

// Define Const
float lambda=0.0001;	//lambda
int _PriorSize=15;		//the window size of dark channel
double _topbright=0.01;//the top rate of bright pixel in dark channel
double _w=0.95;			//w
float t0=0.01;			//lowest transmission
int SizeH=0;			//image Height
int SizeW=0;			//image Width
int SizeH_W=0;			//total number of pixels

Mat trans_refine;		//refine of trans			

// Define Rows and Cols index of L
int idx_x[MAX_INT];				//Rows
int idx_y[MAX_INT];				//Cols
double idx_v[MAX_INT]={0.0};	//Values
int idx_l=0;					//total number of non-zero value of L

// Define fast convert table
int convert_table[25];

char img_name[100]="example.bmp";
char trans_name[100]="printMatInfosdfasdfasdf;";
char out_name[100]="";

//Read Image
Mat ReadImage()
{
	Mat img=imread(img_name);

	SizeH = img.rows;
	SizeW = img.cols;
	SizeH_W = img.rows*img.cols;

	Mat real_img(img.rows,img.cols,CV_32FC3);
	img.convertTo(real_img,CV_32FC3);
	real_img=real_img/255;
	return real_img;
}

//Read TransImage
Mat ReadTransImage()
{
	Mat img=imread(trans_name,0);

	Mat real_img(img.rows,img.cols,CV_32FC1);
	img.convertTo(real_img,CV_32FC1);
	real_img=real_img/255;
	return real_img;
}

//Calculate Dark Channel
//J^{dark}(x)=min( min( J^c(y) ) )
Mat DarkChannelPrior(Mat img) 
{
	Mat dark=Mat::zeros(img.rows,img.cols,CV_32FC1);
	Mat dark_out=Mat::zeros(img.rows,img.cols,CV_32FC1);
	for(int i=0;i<img.rows;i++)
	{
		for(int j=0;j<img.cols;j++)
		{
			dark.at<float>(i,j)=min(min(img.at<Vec<float,3>>(i,j)[0],img.at<Vec<float,3>>(i,j)[1]),min(img.at<Vec<float,3>>(i,j)[0],img.at<Vec<float,3>>(i,j)[2]));
		}
	}
	erode(dark,dark_out,Mat::ones(_PriorSize,_PriorSize,CV_32FC1));
	return dark_out;
}

//Calculate Airlight
Vec<float,3> Airlight(Mat img, Mat dark)
{
	int n_bright=_topbright*SizeH_W;
	Mat dark_1=dark.reshape(1,SizeH_W);
	Vector<int> max_idx;
	float max_num=0;
	int max_pos=0;
	Vec<float,3> a;
	Vec<float,3> A(0,0,0);
	Mat RGBPixcels=Mat::ones(n_bright,1,CV_32FC3);
	Mat HLSPixcels=Mat::ones(n_bright,1,CV_32FC3);
	Mat IdxPixcels=Mat::ones(n_bright,1,CV_32SC1);

	for(int i=0;i<n_bright;i++)
	{
		max_num=0;
		max_idx.push_back(max_num);
		for(float * p = (float *)dark_1.datastart;p!=(float *)dark_1.dataend;p++)
		{
			if(*p>max_num)
			{
				max_num = *p;
				max_idx[i] = (p-(float *)dark_1.datastart);
				RGBPixcels.at<Vec<float,3>>(i,0) = ((Vec<float,3> *)img.data)[max_idx[i]];
				IdxPixcels.at<int>(i,0) = (p-(float *)dark_1.datastart);
				//((Vec<float,3> *)img.data)[max_idx[i]] = Vec<float,3>(0,0,1);
			}
		}
		((float *)dark_1.data)[max_idx[i]]=0;
	}

	float maxL=0.0;
	//int maxIdx=0;
	for(int j=0; j<n_bright; j++)
	{
		A[0]+=RGBPixcels.at<Vec<float,3>>(j,0)[0];
		A[1]+=RGBPixcels.at<Vec<float,3>>(j,0)[1];
		A[2]+=RGBPixcels.at<Vec<float,3>>(j,0)[2];
	}

	A[0]/=n_bright;
	A[1]/=n_bright;
	A[2]/=n_bright;

	return A;
}


//Calculate Transmission Matrix
Mat TransmissionMat(Mat dark)
{

	return 1-_w*dark;
}

//Calculate Haze Free Image
Mat hazefree(Mat img,Mat t,Vec<float,3> a,float exposure = 0)
{
	Mat freeimg=Mat::zeros(SizeH,SizeW,CV_32FC3);
	img.copyTo(freeimg);
	Vec<float,3> * p=(Vec<float,3> *)freeimg.datastart;
	float * q=(float *)t.datastart;
	for(;p<(Vec<float,3> *)freeimg.dataend && q<(float *)t.dataend;p++,q++)
	{
		(*p)[0]=((*p)[0]-a[0])/std::max(*q,t0)+a[0] + exposure;
		(*p)[1]=((*p)[1]-a[1])/std::max(*q,t0)+a[1] + exposure;
		(*p)[2]=((*p)[2]-a[2])/std::max(*q,t0)+a[2] + exposure;
	}
	return freeimg;
}

//************* Utility Functions **********
//Print Matrix
void printMat(char * name,Mat m)
{
	cout<<name<<"\n"<<m<<endl;
}

//Print Matrix Information
void printMatInfo(char * name,Mat m)
{
	cout<<name<<":"<<endl;
	cout<<"\t"<<"cols="<<m.cols<<endl;
	cout<<"\t"<<"rows="<<m.rows<<endl;
	cout<<"\t"<<"channels="<<m.channels()<<endl;
}

//Write Matrix to File, so that Matlab can read
void writeMatToFile(char * filename,Mat m)
{
	FILE * fout = fopen(filename,"w");
	int count=0;
	for(float * p=(float *)m.datastart;p<(float *)m.dataend;p++)
	{
		fprintf(fout,"%f ",*p);
		count++;
		if(count%m.cols==0) fprintf(fout,"\n");
	}
	fclose(fout);
}

//Write L Sparse Matrix to File
void writeLFile()
{
	FILE * foutx = fopen("idx_x.txt","w");
	FILE * fouty = fopen("idx_y.txt","w");
	FILE * foutv = fopen("idx_v.txt","w");
	for(int i=0;i<idx_l;i++)
	{
		fprintf(foutx,"%d ",idx_x[i]+1);
		fprintf(fouty,"%d ",idx_y[i]+1);
		fprintf(foutv,"%f ",idx_v[i]);
	}
	fclose(foutx);
	fclose(fouty);
	fclose(foutv);
}

//Calculate Min and Max value of Matrix
MinMax MaxAndMinOfMatirx( Mat x ) 
{
	MinMax rtn;
	rtn.max=0;
	rtn.min=1000;
	for(float * p=(float *)x.datastart; p<(float *)x.dataend; p++)
	{
		if(*p>rtn.max)	rtn.max=*p;
		if(*p<rtn.min)  rtn.min=*p;
	}
	return rtn;
}

//Process Args from CMD
void processArgs(int argc, char * argv[])
{
	cout<<"/************************************************************************/"<<endl;
	cout<<"/* FileName:   Dehaze.exe                                               */"<<endl;
	cout<<"/* Author:     Pang Liang                                               */"<<endl;
	cout<<"/* Dependence: OpenCV 2.3.1  &  SparseLib++                             */"<<endl;
	cout<<"/* Date:       2012-12-20                                               */"<<endl;
	cout<<"/* Usage:      [ImageName] [-t TransName] [-o OutputName]               */"<<endl;
	cout<<"/*             -o:        Path of Output Image.                         */"<<endl;
	cout<<"/*             ImageName: Path of Image.                                */"<<endl;	
	cout<<"/*             -t:        Path of Outside TransImage.                   */"<<endl;
	cout<<"/************************************************************************/"<<endl;
	for(int i=1;i<argc;i++)
	{
		if(strcmp(argv[i],"-o")==0)
		{
			i++;
			strcpy(out_name,argv[i]);
		}
		else if(strcmp(argv[i],"-t")==0)
		{
			i++;
			strcpy(trans_name,argv[i]);
		}
		else
		{
			strcpy(img_name,argv[i]);
		}
	}
}

//Main Function
int main(int argc, char * argv[])
{
	Mat dark_channel;
	Mat trans;
	Mat img;
	Mat free_img;
	char filename[100];

	processArgs(argc,argv);

	while(access(img_name,0)!=0)
	{
		cout<<"The image "<<img_name<<" don't exist."<<endl<<"Please enter another one:"<<endl;
		cin>>filename;
		//img_name=filename;
	}

	clock_t start , finish ;
	double duration1,duration2,duration3,duration4,duration5,duration6,duration7;

	//Read image
	cout<<"Reading Image ..."<<endl;
			start=clock();
	img=ReadImage();
	//imshow("Original Image",img);
	printMatInfo("img",img);
		finish=clock();
		duration1=( double )( finish - start )/ CLOCKS_PER_SEC ;
		cout<<"Time Cost: "<<duration1<<"s"<<endl;
	waitKey(1000);
	cout<<endl;

	//Calculate DarkChannelPrior
	cout<<"Calculating Dark Channel Prior ..."<<endl;
		start=clock();
	dark_channel=DarkChannelPrior(img);
	//imshow("Dark Channel Prior",dark_channel);
	printMatInfo("dark_channel",dark_channel);
		finish=clock();
		duration3=( double )( finish - start )/ CLOCKS_PER_SEC ;
		cout<<"Time Cost: "<<duration3<<"s"<<endl;
	waitKey(1000);
	cout<<endl;
	
	//Calculate Airlight	
	cout<<"Calculating Airlight ..."<<endl;
		start=clock();
	Vec<float,3> a=Airlight(img,dark_channel);
	cout<<"Airlight:\t"<<" B:"<<a[0]<<" G:"<<a[1]<<" R:"<<a[2]<<endl;
		finish=clock();
		duration4=( double )( finish - start )/ CLOCKS_PER_SEC ;
		cout<<"Time Cost: "<<duration4<<"s"<<endl;
	cout<<endl;

	//Reading Refine Trans
	cout<<"Reading Refine Transmission..."<<endl;
	trans_refine=ReadTransImage();
	printMatInfo("trans_refine",trans_refine);
	//imshow("Refined Transmission Mat",trans_refine);
	cout<<endl;

	//Haze Free
	cout<<"Calculating Haze Free Image ..."<<endl;
		start=clock();

	free_img=hazefree(img,trans_refine,a,0.2);
	//imshow("Haze Free",free_img);
	
	printMatInfo("free_img",free_img);
		finish=clock();
		duration7=( double )( finish - start )/ CLOCKS_PER_SEC ;
		cout<<"Time Cost: "<<duration7<<"s"<<endl;
	
		//cout<<"Total Time Cost: "<<duration1+duration2+duration3+duration4+duration5+duration6+duration7<<"s"<<endl;

	//Save Image
	//char img_name_dark[100]="Dark_";
	//char img_name_step[100]="Step_";
	//char img_name_free[100]="Hazefree_";
	//strcat(img_name_free,img_name);
	//strcat(img_name_step,img_name);
	//strcat(img_name_dark,img_name);
	imwrite(out_name,free_img*255);
	//imwrite(img_name_step,trans_refine*255);
	//imwrite(img_name_dark,trans*255);
	cout<<"Image saved as "<<out_name<<endl;
	//waitKey();
	cout<<endl;

	return 0;
}