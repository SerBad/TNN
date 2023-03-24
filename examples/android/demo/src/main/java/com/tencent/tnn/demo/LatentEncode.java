package com.tencent.tnn.demo;


import android.graphics.Bitmap;

public class LatentEncode {

    public native int init(String modelPath, int computeType);

    public native boolean checkNpu(String modelPath);

    /*
     * 释放内存
     * */
    public native int deinit();

    public native double[] detectFromImage(Bitmap bitmap);
}
