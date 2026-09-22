package com.example.onionqualityai;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.ActivityNotFoundException;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Bundle;
import android.print.PrintAttributes;
import android.print.PrintManager;
import android.webkit.JavascriptInterface;
import android.webkit.PermissionRequest;
import android.webkit.ValueCallback;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceRequest;
import android.webkit.WebResourceResponse;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Toast;

import androidx.activity.OnBackPressedCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.PickVisualMediaRequest;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.webkit.WebViewAssetLoader;

import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    private static final int CAMERA_REQ = 10;
    private static final String APP_HOST = "appassets.androidplatform.net";

    private WebView webView;
    private PermissionRequest pendingPermissionRequest;
    private ValueCallback<Uri[]> fileCallback;
    private AndroidInference inference;

    // Gallery pickers (system Photo Picker, falls back to the document picker automatically)
    private ActivityResultLauncher<PickVisualMediaRequest> singlePicker;
    private ActivityResultLauncher<PickVisualMediaRequest> multiPicker;

    // "Save CSV" support (WebView cannot download blob: links by itself)
    private ActivityResultLauncher<String> createCsvLauncher;
    private String pendingCsv;

    @SuppressLint("SetJavaScriptEnabled")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        registerLaunchers();

        webView = new WebView(this);
        setContentView(webView);

        try {
            inference = new AndroidInference(this);
        } catch (Exception e) {
            Toast.makeText(this, "Could not load onion model: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }

        if ((getApplicationInfo().flags & android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE) != 0) {
            WebView.setWebContentsDebuggingEnabled(true);
        }

        WebViewAssetLoader assetLoader = new WebViewAssetLoader.Builder()
                .addPathHandler("/assets/", new WebViewAssetLoader.AssetsPathHandler(this))
                .build();

        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setAllowFileAccess(false);
        settings.setAllowContentAccess(true);
        settings.setMediaPlaybackRequiresUserGesture(false);

        if (inference != null) webView.addJavascriptInterface(inference, "AndroidInference");
        webView.addJavascriptInterface(new AppBridge(), "AndroidApp");

        webView.setWebViewClient(new WebViewClient() {
            @Override
            public WebResourceResponse shouldInterceptRequest(WebView view, WebResourceRequest request) {
                return assetLoader.shouldInterceptRequest(request.getUrl());
            }

            @Override
            public WebResourceResponse shouldInterceptRequest(WebView view, String url) {
                return assetLoader.shouldInterceptRequest(Uri.parse(url));
            }
        });

        webView.setWebChromeClient(new WebChromeClient() {
            @Override
            public void onPermissionRequest(final PermissionRequest request) {
                boolean trusted = request.getOrigin() != null && APP_HOST.equals(request.getOrigin().getHost());
                if (trusted && containsVideoCapture(request.getResources())) {
                    runOnUiThread(() -> {
                        if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.CAMERA)
                                == PackageManager.PERMISSION_GRANTED) {
                            request.grant(new String[]{PermissionRequest.RESOURCE_VIDEO_CAPTURE});
                        } else {
                            pendingPermissionRequest = request;
                            requestPermissions(new String[]{Manifest.permission.CAMERA}, CAMERA_REQ);
                        }
                    });
                } else {
                    request.deny();
                }
            }

            @Override
            public boolean onShowFileChooser(WebView view, ValueCallback<Uri[]> callback, FileChooserParams params) {
                if (fileCallback != null) fileCallback.onReceiveValue(null);
                fileCallback = callback;

                PickVisualMediaRequest request = new PickVisualMediaRequest.Builder()
                        .setMediaType(ActivityResultContracts.PickVisualMedia.ImageOnly.INSTANCE)
                        .build();
                try {
                    if (params != null && params.getMode() == FileChooserParams.MODE_OPEN_MULTIPLE) {
                        multiPicker.launch(request);
                    } else {
                        singlePicker.launch(request);
                    }
                } catch (ActivityNotFoundException | IllegalStateException e) {
                    finishFileChooser(null);
                    Toast.makeText(MainActivity.this, "No gallery app available", Toast.LENGTH_SHORT).show();
                    return false;
                }
                return true;
            }
        });

        getOnBackPressedDispatcher().addCallback(this, new OnBackPressedCallback(true) {
            @Override
            public void handleOnBackPressed() {
                if (webView.canGoBack()) {
                    webView.goBack();
                } else {
                    setEnabled(false);
                    getOnBackPressedDispatcher().onBackPressed();
                }
            }
        });

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, CAMERA_REQ);
        }

        webView.loadUrl("https://" + APP_HOST + "/assets/index.html");
    }

    private void registerLaunchers() {
        singlePicker = registerForActivityResult(new ActivityResultContracts.PickVisualMedia(), uri -> {
            finishFileChooser(uri == null ? null : new Uri[]{uri});
        });

        multiPicker = registerForActivityResult(new ActivityResultContracts.PickMultipleVisualMedia(), (List<Uri> uris) -> {
            finishFileChooser(uris == null || uris.isEmpty() ? null : uris.toArray(new Uri[0]));
        });

        createCsvLauncher = registerForActivityResult(new ActivityResultContracts.CreateDocument("text/csv"), uri -> {
            String csv = pendingCsv;
            pendingCsv = null;
            if (uri == null || csv == null) return;
            try (OutputStream out = getContentResolver().openOutputStream(uri)) {
                if (out == null) throw new java.io.IOException("Cannot open file");
                out.write(new byte[]{(byte) 0xEF, (byte) 0xBB, (byte) 0xBF}); // UTF-8 BOM for Excel
                out.write(csv.getBytes(StandardCharsets.UTF_8));
                Toast.makeText(this, "CSV saved", Toast.LENGTH_SHORT).show();
            } catch (Exception e) {
                Toast.makeText(this, "Could not save CSV: " + e.getMessage(), Toast.LENGTH_LONG).show();
            }
        });
    }

    private void finishFileChooser(Uri[] results) {
        if (fileCallback != null) {
            fileCallback.onReceiveValue(results);
            fileCallback = null;
        }
    }

    /** Functions the web page can call for things a plain WebView cannot do. */
    private class AppBridge {
        @JavascriptInterface
        public void saveCsv(String fileName, String content) {
            runOnUiThread(() -> {
                pendingCsv = content;
                try {
                    createCsvLauncher.launch(fileName == null || fileName.isEmpty() ? "onion-quality.csv" : fileName);
                } catch (ActivityNotFoundException e) {
                    pendingCsv = null;
                    Toast.makeText(MainActivity.this, "No file manager available to save the CSV", Toast.LENGTH_LONG).show();
                }
            });
        }

        @JavascriptInterface
        public void printPage() {
            runOnUiThread(() -> {
                PrintManager pm = (PrintManager) getSystemService(PRINT_SERVICE);
                if (pm == null) return;
                pm.print("OnionQualityAI Report",
                        webView.createPrintDocumentAdapter("OnionQualityAI_Report"),
                        new PrintAttributes.Builder().build());
            });
        }
    }

    private static boolean containsVideoCapture(String[] resources) {
        for (String resource : resources) {
            if (PermissionRequest.RESOURCE_VIDEO_CAPTURE.equals(resource)) return true;
        }
        return false;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_REQ && pendingPermissionRequest != null) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                pendingPermissionRequest.grant(new String[]{PermissionRequest.RESOURCE_VIDEO_CAPTURE});
            } else {
                pendingPermissionRequest.deny();
            }
            pendingPermissionRequest = null;
        }
    }

    @Override
    protected void onDestroy() {
        finishFileChooser(null);
        if (webView != null) {
            webView.removeJavascriptInterface("AndroidInference");
            webView.removeJavascriptInterface("AndroidApp");
            webView.destroy();
        }
        if (inference != null) inference.close();
        super.onDestroy();
    }
}
