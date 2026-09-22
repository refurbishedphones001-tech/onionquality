package com.example.onionqualityai;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Base64;
import android.util.Log;
import android.webkit.JavascriptInterface;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.Iterator;
import java.util.Map;

public class AndroidInference {
    private static final String TAG = "OnionInference";
    private static final int SIZE = 224;
    private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] STD = {0.229f, 0.224f, 0.225f};

    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;
    private final String outputName;

    public AndroidInference(android.content.Context context) throws Exception {
        env = OrtEnvironment.getEnvironment();
        byte[] model = readAll(context.getAssets().open("onion_quality_model.onnx"));
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setIntraOpNumThreads(2);
        options.setInterOpNumThreads(1);
        session = env.createSession(model, options);
        inputName = session.getInputNames().iterator().next();
        outputName = session.getOutputNames().iterator().next();
    }

    @JavascriptInterface
    public synchronized String analyze(String dataUrl) {
        long started = System.currentTimeMillis();
        Bitmap resized = null;
        try {
            String payload = dataUrl;
            int comma = payload.indexOf(',');
            if (comma >= 0) payload = payload.substring(comma + 1);
            byte[] bytes = Base64.decode(payload, Base64.DEFAULT);

            resized = decodeDownsampled(bytes, SIZE);
            if (resized == null) throw new IOException("Could not decode image");
            if (resized.getWidth() != SIZE || resized.getHeight() != SIZE) {
                Bitmap scaled = Bitmap.createScaledBitmap(resized, SIZE, SIZE, true);
                if (scaled != resized) {
                    resized.recycle();
                    resized = scaled;
                }
            }

            float[] input = new float[3 * SIZE * SIZE];
            int[] pixels = new int[SIZE * SIZE];
            resized.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE);
            int plane = SIZE * SIZE;
            for (int i = 0; i < plane; i++) {
                int pixel = pixels[i];
                float r = ((pixel >> 16) & 0xFF) / 255f;
                float g = ((pixel >> 8) & 0xFF) / 255f;
                float b = (pixel & 0xFF) / 255f;
                input[i] = (r - MEAN[0]) / STD[0];
                input[plane + i] = (g - MEAN[1]) / STD[1];
                input[2 * plane + i] = (b - MEAN[2]) / STD[2];
            }

            try (OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), new long[]{1, 3, SIZE, SIZE});
                 OrtSession.Result result = session.run(Collections.singletonMap(inputName, tensor))) {
                OnnxValue outputValue = result.get(outputName).orElseThrow(() -> new IllegalStateException("Missing model output"));
                Object raw = outputValue.getValue();
                float[] logits;
                if (raw instanceof float[][]) {
                    logits = ((float[][]) raw)[0];
                } else if (raw instanceof float[]) {
                    logits = (float[]) raw;
                } else {
                    throw new IllegalStateException("Unexpected model output type: " + raw.getClass().getName());
                }

                float[] probs = softmax(logits);
                JSONObject json = new JSONObject();
                json.put("ok", true);
                JSONArray arr = new JSONArray();
                for (float p : probs) arr.put((double) p);
                json.put("probs", arr);
                json.put("ms", System.currentTimeMillis() - started);
                return json.toString();
            }
        } catch (Throwable e) {
            // Catches OutOfMemoryError too (it's an Error, not an Exception) so a single
            // bad/huge photo can never crash the app or leave the JS call hanging.
            Log.e(TAG, "Inference failed", e);
            try {
                JSONObject json = new JSONObject();
                json.put("ok", false);
                json.put("error", e.getMessage() == null ? e.toString() : e.getMessage());
                json.put("ms", System.currentTimeMillis() - started);
                return json.toString();
            } catch (Exception ignored) {
                return "{\"ok\":false,\"error\":\"Native inference failed\"}";
            }
        } finally {
            if (resized != null) resized.recycle();
        }
    }

    /** Decode straight to roughly targetSize, so a large photo never needs a full-resolution
     *  bitmap in memory first (the main cause of intermittent "could not analyze" failures). */
    private static Bitmap decodeDownsampled(byte[] bytes, int targetSize) {
        BitmapFactory.Options bounds = new BitmapFactory.Options();
        bounds.inJustDecodeBounds = true;
        BitmapFactory.decodeByteArray(bytes, 0, bytes.length, bounds);

        int sample = 1;
        int w = bounds.outWidth, h = bounds.outHeight;
        while (w / (sample * 2) >= targetSize && h / (sample * 2) >= targetSize) {
            sample *= 2;
        }

        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inSampleSize = sample;
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;
        try {
            return BitmapFactory.decodeByteArray(bytes, 0, bytes.length, options);
        } catch (OutOfMemoryError e) {
            // Retry once, more aggressively downsampled.
            options.inSampleSize = sample * 2;
            return BitmapFactory.decodeByteArray(bytes, 0, bytes.length, options);
        }
    }

    public synchronized void close() {
        try { session.close(); } catch (Exception ignored) { }
    }

    private static float[] softmax(float[] logits) {
        float max = logits[0];
        for (float v : logits) if (v > max) max = v;
        float[] exp = new float[logits.length];
        float sum = 0f;
        for (int i = 0; i < logits.length; i++) {
            exp[i] = (float) Math.exp(logits[i] - max);
            sum += exp[i];
        }
        for (int i = 0; i < exp.length; i++) exp[i] /= sum;
        return exp;
    }

    private static byte[] readAll(InputStream in) throws IOException {
        try (InputStream input = in; ByteArrayOutputStream out = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[8192];
            int n;
            while ((n = input.read(buffer)) != -1) out.write(buffer, 0, n);
            return out.toByteArray();
        }
    }
}
