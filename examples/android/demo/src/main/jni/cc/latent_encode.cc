//
// Created by zhou on 2023/2/16.
//
// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.


#include "latent_encode.h"

#include <jni.h>
#include <typeinfo>

#include "helper_jni.h"

#include <android/bitmap.h>
#include <iomanip>

#include "tnn/core/macro.h"
#include "tnn/core/tnn.h"
#include "tnn/core/common.h"
#include "tnn/core/blob.h"

std::shared_ptr<tnn::TNN> net_ = nullptr;
std::shared_ptr<tnn::Instance> instance_ = nullptr;

std::string model_path_str_ = "";

static int gComputeUnitType = 0; // 0 is cpu, 1 is gpu, 2 is huawei_npu

JNIEXPORT JNICALL jint TNN_LATENT_ENCODE(init)(JNIEnv *env, jobject thiz, jstring modelPath, jint computUnitType) {
    LOGE("开始初始化1");
    std::string protoContent, modelContent;
    std::string modelPathStr(jstring2string(env, modelPath));
    LOGE("开始初始化2");
    // NDK无法读取存在Asset下的文件
    protoContent = fdLoadFile(modelPathStr + "/faces_w_encoder.opt.tnnproto");
    modelContent = fdLoadFile(modelPathStr + "/faces_w_encoder.opt.tnnmodel");
    LOGI("proto content size %lu model content size %lu", protoContent.length(), modelContent.length());
    gComputeUnitType = computUnitType;
    LOGE("开始初始化3");
    TNN_NS::Status status;
    if (!net_) {
        auto net = std::make_shared<TNN_NS::TNN>();
        TNN_NS::ModelConfig model_config;
        model_config.model_type = TNN_NS::MODEL_TYPE_TNN;
        model_config.params = {protoContent, modelContent, modelPathStr};
        LOGE("开始初始化4");
        status = net->Init(model_config);

        LOGE("instance.net init aaa %d  %d ", (int) status, TNN_NS::TNN_OK);
        if (status != TNN_NS::TNN_OK) {
            LOGE("instance.net init failed %d", (int) status);
            return status;
        }
        LOGE("初始化成功 %d", (int) status);
        net_ = net;

        if (!net_) return 0;
        TNN_NS::NetworkConfig network_config;

        if (gComputeUnitType == 2) {
            network_config.network_type = TNN_NS::NETWORK_TYPE_HUAWEI_NPU;
            network_config.device_type = TNN_NS::DEVICE_HUAWEI_NPU;
        } else if (gComputeUnitType == 1) {
            network_config.network_type = TNN_NS::NETWORK_TYPE_DEFAULT;
            network_config.device_type = TNN_NS::DEVICE_OPENCL;
        } else {
            network_config.network_type = TNN_NS::NETWORK_TYPE_DEFAULT;
            network_config.device_type = TNN_NS::DEVICE_ARM;
        }
        LOGE("初始化成功1  network_config.device_type %d ", network_config.device_type);
        std::vector<int> nchw = {1, 3, 256, 256};
        TNN_NS::InputShapesMap input_shapes = {};
        LOGE("初始化成功2");
        input_shapes.insert(std::pair<std::string, TNN_NS::DimsVector>("modelInput", nchw));
        LOGE("初始化成功3 net_ %d ", net_ != NULL);
        instance_ = net_->CreateInst(network_config, status, input_shapes);
        LOGE("instance.net CreateInst success %d instance_ %d ", (int) status, instance_ != NULL);
    }
    return TNN_NS::TNN_OK;
}

JNIEXPORT JNICALL jboolean TNN_LATENT_ENCODE(checkNpu)(JNIEnv *env, jobject thiz, jstring modelPath) {
//    TNN_NS::UltraFaceDetector tmpDetector;
//    std::string protoContent, modelContent;
//    std::string modelPathStr(jstring2string(env, modelPath));
//    protoContent = fdLoadFile(modelPathStr + "/version-slim-320_simplified.tnnproto");
//    modelContent = fdLoadFile(modelPathStr + "/version-slim-320_simplified.tnnmodel");
//    auto option = std::make_shared<TNN_NS::UltraFaceDetectorOption>();
//    option->compute_units = TNN_NS::TNNComputeUnitsHuaweiNPU;
//    option->library_path = "";
//    option->proto_content = protoContent;
//    option->model_content = modelContent;
//    option->input_height = 240;
//    option->input_width = 320;
//    tmpDetector.setNpuModelPath(modelPathStr + "/");
//    tmpDetector.setCheckNpuSwitch(true);
//    TNN_NS::Status ret = tmpDetector.Init(option);
//    return ret == TNN_NS::TNN_OK;


    return 0;
}

JNIEXPORT JNICALL jint TNN_LATENT_ENCODE(deinit)(JNIEnv *env, jobject thiz) {
    net_ = nullptr;
    instance_ = nullptr;
    model_path_str_ = nullptr;
    return 0;
}


JNIEXPORT JNICALL jfloatArray TNN_LATENT_ENCODE(detectFromImage)(JNIEnv *env, jobject thiz, jobject imageSource) {
    AndroidBitmapInfo sourceInfocolor;
    void *sourcePixelscolor;
    if (!instance_) {
        return env->NewFloatArray(0);
    }
    if (AndroidBitmap_getInfo(env, imageSource, &sourceInfocolor) < 0) {
        LOGE("detectFromImage fail1 ");
        return env->NewFloatArray(0);
    }
    LOGE("detectFromImage sourceInfocolor.format %d", sourceInfocolor.format);
    if (sourceInfocolor.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        return env->NewFloatArray(0);
    }

    if (AndroidBitmap_lockPixels(env, imageSource, &sourcePixelscolor) < 0) {
        LOGE("detectFromImage fail2 ");
        return env->NewFloatArray(0);
    }
    LOGE("detectFromImage image ");
    TNN_NS::DeviceType dt = TNN_NS::DEVICE_ARM;
    if (gComputeUnitType == 2) {
        dt = TNN_NS::DEVICE_HUAWEI_NPU;
    } else if (gComputeUnitType == 1) {
        dt = TNN_NS::DEVICE_OPENCL;
    } else {
        dt = TNN_NS::DEVICE_ARM;
    }
    TNN_NS::DimsVector target_dims = {1, 3, 256, 256};

    auto input_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::N8UC4, target_dims, sourcePixelscolor);

    std::shared_ptr<TNN_NS::Mat> output_mat = nullptr;
    LOGE(" output_mat.get()1 %d %d ", sourceInfocolor.height, sourceInfocolor.width);
    LOGE(" output_mat.get()1 input_mat1  %d %d batch: %d", input_mat->GetWidth(), input_mat->GetHeight(), input_mat->GetBatch());
    LOGE(" output_mat.get()1 input_mat2  {%d , %d , %d , %d}", input_mat->GetDim(0), input_mat->GetDim(1), input_mat->GetDim(2), input_mat->GetDim(3));

    TNN_NS::MatConvertParam input_convert_param;
    float scale = 2.0 / 255.0;
    float bias = -1.0;
    input_convert_param.scale = {scale, scale, scale, scale};
    input_convert_param.bias = {bias, bias, bias, bias};
    const uint8_t *input_uint8_t_data = static_cast<uint8_t *>(input_mat->GetData());
    LOGE(" output_mat.get()1 input_mat3 input_uint8_t_data1   {%f , %f , %f  }", (input_uint8_t_data[0]) * scale + bias, input_uint8_t_data[1] * scale + bias, input_uint8_t_data[2] * scale + bias);
    LOGE(" output_mat.get()1 input_mat3 input_uint8_t_data2   {%d , %d , %d  }", input_uint8_t_data[0], input_uint8_t_data[1], input_uint8_t_data[2]);


    LOGE("detectFromImage image2 input_mat %d instance_ %d ", input_mat != NULL, instance_ != NULL);
    TNN_NS::Status status = instance_->SetInputMat(input_mat, input_convert_param, "modelInput");

    LOGE("detectFromImage image3  %d", (int) status);
    instance_->Forward();
    LOGE("detectFromImage image4 ");
    instance_->GetOutputMat(output_mat);
    LOGE("detectFromImage image5 ");
    AndroidBitmap_unlockPixels(env, imageSource);
    const float *data = static_cast<float *>(output_mat->GetData());
    LOGE(" output_mat.get()5 %f =1= %16f ", ((double) data[1]), ((double) data[45]));
    int len = output_mat->GetDim(0) * output_mat->GetDim(1) * output_mat->GetDim(2);
    jfloatArray result = env->NewFloatArray(len);
    env->SetFloatArrayRegion(result, 0, len, data);
    return result;
}

JNIEXPORT JNICALL jfloatArray TNN_LATENT_ENCODE(detectFromImageForFloatArray)(JNIEnv *env, jobject thiz, jfloatArray imageSource) {
    TNN_NS::DeviceType dt = TNN_NS::DEVICE_ARM;
    if (gComputeUnitType == 2) {
        dt = TNN_NS::DEVICE_HUAWEI_NPU;
    } else if (gComputeUnitType == 1) {
        dt = TNN_NS::DEVICE_OPENCL;
    } else {
        dt = TNN_NS::DEVICE_ARM;
    }
    TNN_NS::DimsVector target_dims = {1, 3, 256, 256};

    auto input_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::NCHW_FLOAT, target_dims, env->GetFloatArrayElements(imageSource, JNI_FALSE));

    std::shared_ptr<TNN_NS::Mat> output_mat = nullptr;

    LOGE(" output_mat.get()1 input_mat1  %d %d batch: %d", input_mat->GetWidth(), input_mat->GetHeight(), input_mat->GetBatch());
    LOGE(" output_mat.get()1 input_mat2  {%d , %d , %d , %d}", input_mat->GetDim(0), input_mat->GetDim(1), input_mat->GetDim(2), input_mat->GetDim(3));

    TNN_NS::MatConvertParam input_convert_param;
    float scale = 1.0;
    float bias = 0;
    input_convert_param.scale = {scale, scale, scale};
    input_convert_param.bias = {bias, bias, bias};
    const float *input_data = static_cast<float *>(input_mat->GetData());

    LOGE(" output_mat.get()1 input_mat3 input_data1   {%f , %f , %f  }", (input_data[0]) * scale + bias, input_data[1] * scale + bias, input_data[2] * scale + bias);
    LOGE(" output_mat.get()1 input_mat3 input_data2   {%f , %f , %f  }", input_data[0], input_data[1], input_data[2]);

    LOGE("detectFromImage image2 input_mat %d instance_ %d ", input_mat != NULL, instance_ != NULL);
    TNN_NS::Status status = instance_->SetInputMat(input_mat, input_convert_param, "modelInput");

    LOGE("detectFromImage image3  %d", (int) status);
    instance_->Forward();
    LOGE("detectFromImage image4 ");
    instance_->GetOutputMat(output_mat);
    LOGE("detectFromImage image5 ");

    const float *data = static_cast<float *>(output_mat->GetData());
    LOGE(" output_mat.get()5 %f =1= %16f ", ((double) data[1]), ((double) data[45]));
    int len = output_mat->GetDim(0) * output_mat->GetDim(1) * output_mat->GetDim(2);
    jfloatArray result = env->NewFloatArray(len);
    env->SetFloatArrayRegion(result, 0, len, data);
    return result;
}