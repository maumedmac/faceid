package bo.edu.cba.faceid;

import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.objdetect.FaceDetectorYN;
import org.opencv.objdetect.FaceRecognizerSF;
import org.opencv.imgproc.Imgproc;

import android.os.Bundle;
import android.util.Log;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.Toast;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Collections;
import java.util.List;

public class MainActivity extends CameraActivity implements CvCameraViewListener2 {

    private static final String    TAG  = "OCVSample::Activity";

    private static final Scalar    BOX_COLOR         = new Scalar(0, 255, 0);
    private static final Scalar    RIGHT_EYE_COLOR   = new Scalar(255, 0, 0);
    private static final Scalar    LEFT_EYE_COLOR    = new Scalar(0, 0, 255);
    private static final Scalar    NOSE_TIP_COLOR    = new Scalar(0, 255, 0);
    private static final Scalar    MOUTH_RIGHT_COLOR = new Scalar(255, 0, 255);
    private static final Scalar    MOUTH_LEFT_COLOR  = new Scalar(0, 255, 255);
    private static final Scalar    MATCH_COLOR       = new Scalar(0, 255, 0);

    private Mat                    mRgba;
    private Mat                    mBgr;
    private Mat                    mBgrScaled;
    private Size                   mInputSize = null;
    private final float            mScale = 2.f;
    private FaceDetectorYN         mFaceDetector;
    private FaceRecognizerSF       mFaceRecognizer;
    private Mat                    mFaces;
    private Mat                    mRegisteredFaceFeature;

    private CameraBridgeViewBase   mOpenCvCameraView;
    private int mCameraId = CameraBridgeViewBase.CAMERA_ID_BACK;

    public MainActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);

        if (OpenCVLoader.initLocal()) {
            Log.i(TAG, "OpenCV loaded successfully");
        } else {
            Log.e(TAG, "OpenCV initialization failed!");
            (Toast.makeText(this, "OpenCV initialization failed!", Toast.LENGTH_LONG)).show();
            return;
        }

        loadFaceModels();

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);

        mOpenCvCameraView = findViewById(R.id.camera_view);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCameraIndex(mCameraId);
        mOpenCvCameraView.setCvCameraViewListener(this);

        Button switchCameraButton = findViewById(R.id.switch_camera_button);
        switchCameraButton.setOnClickListener(v -> swapCamera());

        Button registerButton = findViewById(R.id.button_register);
        registerButton.setOnClickListener(v -> registerFace());

        Button loadDataButton = findViewById(R.id.button_load_data);
        loadDataButton.setOnClickListener(v -> loadData());
    }

    private String getPathFromRawResource(int resourceId, String filename) {
        File file = new File(getCacheDir(), filename);
        if (!file.exists()) {
            try (InputStream is = getResources().openRawResource(resourceId);
                 FileOutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4096];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
            } catch (IOException e) {
                Log.e(TAG, "Failed to load model from raw resource: " + e.getMessage());
                return null;
            }
        }
        return file.getAbsolutePath();
    }

    private void loadFaceModels() {
        String fdModelPath = getPathFromRawResource(R.raw.face_detection_yunet_2023mar, "face_detection_yunet_2023mar.onnx");
        if (fdModelPath != null) {
            mFaceDetector = FaceDetectorYN.create(fdModelPath, "", new Size(320, 320));
            if (mFaceDetector != null) {
                Log.i(TAG, "FaceDetectorYN initialized successfully!");
            } else {
                Log.e(TAG, "Failed to create FaceDetectorYN!");
            }
        } else {
            Log.e(TAG, "Failed to get path for face detection model.");
        }

        String frModelPath = getPathFromRawResource(R.raw.face_recognition_sface_2021dec, "face_recognition_sface_2021dec.onnx");
        if(frModelPath != null) {
            mFaceRecognizer = FaceRecognizerSF.create(frModelPath, "");
            if (mFaceRecognizer != null) {
                Log.i(TAG, "FaceRecognizerSF initialized successfully!");
            } else {
                Log.e(TAG, "Failed to create FaceRecognizerSF!");
            }
        } else {
            Log.e(TAG, "Failed to get path for face recognition model.");
        }
    }

    private void registerFace() {
        if (mFaceRecognizer == null) {
            Toast.makeText(this, "El reconocedor facial no está listo.", Toast.LENGTH_SHORT).show();
            return;
        }
        if (mFaces == null || mFaces.empty()) {
            Toast.makeText(this, "No se detectó ninguna cara para registrar.", Toast.LENGTH_SHORT).show();
            return;
        }

        if (mFaces.rows() > 1) {
            Toast.makeText(this, "Se detectó más de una cara. Por favor, asegúrese de que solo haya una.", Toast.LENGTH_SHORT).show();
            return;
        }

        Mat bgrCopy = mBgr.clone();
        Mat facesCopy = mFaces.clone();

        Mat alignedFace = new Mat();
        mFaceRecognizer.alignCrop(bgrCopy, facesCopy.row(0), alignedFace);

        Mat feature = new Mat();
        mFaceRecognizer.feature(alignedFace, feature);

        bgrCopy.release();
        facesCopy.release();

        saveFaceFeature(feature);
    }

    private void saveFaceFeature(Mat feature) {
        File storageDir = getExternalFilesDir(null);
        if (storageDir == null) {
            Log.e(TAG, "External storage is not available.");
            Toast.makeText(this, "Almacenamiento externo no disponible.", Toast.LENGTH_SHORT).show();
            return;
        }

        File file = new File(storageDir, "face_feature.bin");
        try (FileOutputStream fos = new FileOutputStream(file); ObjectOutputStream oos = new ObjectOutputStream(fos)) {
            float[] featureArray = new float[feature.cols() * feature.rows()];
            feature.get(0, 0, featureArray);
            oos.writeObject(featureArray);
            Toast.makeText(this, "Huella facial guardada en " + file.getAbsolutePath(), Toast.LENGTH_LONG).show();
        } catch (IOException e) {
            Log.e(TAG, "Error al guardar la huella facial", e);
            Toast.makeText(this, "Error al guardar la huella facial.", Toast.LENGTH_SHORT).show();
        }
    }

    private void loadData() {
        File storageDir = getExternalFilesDir(null);
        if (storageDir == null) {
            Log.e(TAG, "External storage is not available.");
            Toast.makeText(this, "Almacenamiento externo no disponible.", Toast.LENGTH_SHORT).show();
            return;
        }

        File file = new File(storageDir, "face_feature.bin");
        if (!file.exists()) {
            Toast.makeText(this, "No hay ninguna huella facial registrada.", Toast.LENGTH_SHORT).show();
            return;
        }

        try (FileInputStream fis = new FileInputStream(file); ObjectInputStream ois = new ObjectInputStream(fis)) {
            float[] featureArray = (float[]) ois.readObject();
            mRegisteredFaceFeature = new Mat(1, featureArray.length, 5); // CV_32F
            mRegisteredFaceFeature.put(0, 0, featureArray);
            Toast.makeText(this, "Huella facial cargada correctamente.", Toast.LENGTH_SHORT).show();
        } catch (IOException | ClassNotFoundException e) {
            Log.e(TAG, "Error al cargar la huella facial", e);
            Toast.makeText(this, "Error al cargar la huella facial.", Toast.LENGTH_SHORT).show();
        }
    }

    private void swapCamera() {
        mCameraId = mCameraId == CameraBridgeViewBase.CAMERA_ID_BACK ? CameraBridgeViewBase.CAMERA_ID_FRONT : CameraBridgeViewBase.CAMERA_ID_BACK;
        mOpenCvCameraView.disableView();
        mOpenCvCameraView.setCameraIndex(mCameraId);
        mOpenCvCameraView.enableView();
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.enableView();
    }

    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(mOpenCvCameraView);
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat();
        mBgr = new Mat();
        mBgrScaled = new Mat();
        mFaces = new Mat();
    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
        mBgr.release();
        mBgrScaled.release();
        mFaces.release();
    }

    private void visualizeAndVerify(Mat rgba, Mat faces) {
        if (faces.empty() || mFaceRecognizer == null) {
            return;
        }

        int thickness = 2;
        float[] faceData = new float[faces.cols() * faces.channels()];

        for (int i = 0; i < faces.rows(); i++) {
            faces.get(i, 0, faceData);

            Rect box = new Rect(Math.round(mScale*faceData[0]), Math.round(mScale*faceData[1]),
                    Math.round(mScale*faceData[2]), Math.round(mScale*faceData[3]));
            Imgproc.rectangle(rgba, box, BOX_COLOR, thickness);

            if (mRegisteredFaceFeature != null) {
                Mat alignedFace = new Mat();
                mFaceRecognizer.alignCrop(mBgr, faces.row(i), alignedFace);
                Mat currentFeature = new Mat();
                mFaceRecognizer.feature(alignedFace, currentFeature);

                double cosScore = mFaceRecognizer.match(mRegisteredFaceFeature, currentFeature, FaceRecognizerSF.FR_COSINE);
                double l2Score = mFaceRecognizer.match(mRegisteredFaceFeature, currentFeature, FaceRecognizerSF.FR_NORM_L2);

                double cosThreshold = 0.363;
                double l2Threshold = 1.128;

                if (cosScore > cosThreshold && l2Score < l2Threshold) {
                    Imgproc.putText(rgba, "Identificado", new Point(box.x, box.y - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.9, MATCH_COLOR, 2);
                }
            }
        }
    }

    @Override
    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();

        if (mCameraId == CameraBridgeViewBase.CAMERA_ID_FRONT) {
            Core.flip(mRgba, mRgba, 1);
        }

        if (mFaceDetector == null) {
            return mRgba;
        }

        Size inputSize = new Size(Math.round(mRgba.cols()/mScale), Math.round(mRgba.rows()/mScale));
        if (mInputSize == null || !mInputSize.equals(inputSize)) {
            mInputSize = inputSize;
            mFaceDetector.setInputSize(mInputSize);
        }

        Imgproc.cvtColor(mRgba, mBgr, Imgproc.COLOR_RGBA2BGR);
        Imgproc.resize(mBgr, mBgrScaled, mInputSize);

        mFaceDetector.detect(mBgrScaled, mFaces);
        visualizeAndVerify(mRgba, mFaces);

        return mRgba;
    }
}
