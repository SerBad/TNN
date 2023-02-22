//
// Created by zhou on 2023/2/16.
// LatentEncode
//

#ifndef ANDROID_LATENT_ENCODE_H
#define ANDROID_LATENT_ENCODE_H

#include <jni.h>
#define TNN_LATENT_ENCODE(sig) Java_com_tencent_tnn_demo_LatentEncode_##sig
#ifdef __cplusplus
extern "C" {
#endif
JNIEXPORT JNICALL jint TNN_LATENT_ENCODE(init)(JNIEnv *env, jobject thiz,  jstring modelPath, jint computUnitType);
JNIEXPORT JNICALL jint TNN_LATENT_ENCODE(deinit)(JNIEnv *env, jobject thiz);
JNIEXPORT JNICALL jboolean TNN_LATENT_ENCODE(checkNpu)(JNIEnv *env, jobject thiz, jstring modelPath);
JNIEXPORT JNICALL jdoubleArray TNN_LATENT_ENCODE(detectFromImage)(JNIEnv *env, jobject thiz, jobject imageSource );

#ifdef __cplusplus
}
#endif

#endif //ANDROID_LATENT_ENCODE_H
