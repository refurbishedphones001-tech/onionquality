# OnionQualityAI Android APK

This project packages the supplied OnionQualityWeb app into a native Android WebView shell and moves ONNX inference into ONNX Runtime Android. The trained model and model information are bundled in `app/src/main/assets`.

## GitHub build
1. Create a GitHub repository and upload this folder.
2. Open **Actions** → **Build OnionQualityAI APK**.
3. Run the workflow.
4. Download the `OnionQualityAI-debug-apk` artifact and install `app-debug.apk` on an Android phone.

The debug APK is suitable for demo/testing. It is not a Play Store release-signed APK.
