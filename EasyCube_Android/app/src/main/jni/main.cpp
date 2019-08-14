#include <jni.h>
#include "com_delilah_easycube_MainActivity.h"

#include <opencv2/opencv.hpp>

using namespace cv;


extern "C" {
    JNIEXPORT void JNICALL
    Java_com_delilah_easycube_MainActivity_ConvertRGBtoGray(
    JNIEnv *env,
    jobject instance,
    jlong matAddInput,
    jlong matAddResult){

    Mat &matInput = *(Mat *)matAddInput;
    Mat &matResult = *(Mat *)matAddResult;

    cvtColor(matInput, matResult, COLOR_RGBA2GRAY);
    }
}