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

import android.app.AlertDialog;
import android.os.Bundle;
import android.text.InputType;
import android.util.Log;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MainActivity extends CameraActivity implements CvCameraViewListener2 {

    private static final String    TAG  = "OCVSample::Activity";
    private static final String    DATABASE_FILE_NAME = "face_database.bin";

    private static final Scalar    BOX_COLOR         = new Scalar(0, 255, 0);
    private static final Scalar    MATCH_COLOR       = new Scalar(0, 255, 0);

    private Mat                    mRgba;
    private Mat                    mBgr;
    private Mat                    mBgrScaled;
    private Size                   mInputSize = null;
    private final float            mScale = 2.f;
    private FaceDetectorYN         mFaceDetector;
    private FaceRecognizerSF       mFaceRecognizer;
    private Mat                    mFaces;
    private final HashMap<String, Mat> mRegisteredFaces = new HashMap<>();

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
        registerButton.setOnClickListener(v -> startRegistrationProcess());

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
            Log.i(TAG, "FaceDetectorYN initialized successfully!");
        } else {
            Log.e(TAG, "Failed to create FaceDetectorYN!");
        }

        String frModelPath = getPathFromRawResource(R.raw.face_recognition_sface_2021dec, "face_recognition_sface_2021dec.onnx");
        if(frModelPath != null) {
            mFaceRecognizer = FaceRecognizerSF.create(frModelPath, "");
            Log.i(TAG, "FaceRecognizerSF initialized successfully!");
        } else {
            Log.e(TAG, "Failed to create FaceRecognizerSF!");
        }
    }

    private void startRegistrationProcess() {
        if (mFaceRecognizer == null || mFaces == null || mFaces.empty()) {
            Toast.makeText(this, "No face detected for registration.", Toast.LENGTH_SHORT).show();
            return;
        }
        if (mFaces.rows() > 1) {
            Toast.makeText(this, "Multiple faces detected. Please ensure only one.", Toast.LENGTH_SHORT).show();
            return;
        }

        // Extract feature from the detected face
        final Mat currentFeature = new Mat();
        Mat alignedFace = new Mat();
        mFaceRecognizer.alignCrop(mBgr, mFaces.row(0), alignedFace);
        mFaceRecognizer.feature(alignedFace, currentFeature);
        alignedFace.release();

        // Check if the face is already registered
        String matchedName = findMatchingFace(currentFeature);

        if (matchedName != null) {
            Toast.makeText(this, "This face is already registered as '" + matchedName + "'.", Toast.LENGTH_LONG).show();
            currentFeature.release(); // Release the feature as it's not needed anymore
            return;
        }

        // Face is not registered, proceed with asking for a name.
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Enter Name for New Face");

        final EditText input = new EditText(this);
        input.setInputType(InputType.TYPE_CLASS_TEXT);
        builder.setView(input);

        builder.setPositiveButton("Register", (dialog, which) -> {
            String name = input.getText().toString().trim();
            if (name.isEmpty()) {
                Toast.makeText(this, "Name cannot be empty.", Toast.LENGTH_SHORT).show();
            } else if (mRegisteredFaces.containsKey(name)) {
                Toast.makeText(this, "Name '" + name + "' already exists. Please choose a different name.", Toast.LENGTH_LONG).show();
            } else {
                addFaceToDatabase(name, currentFeature);
            }
        });

        builder.setNegativeButton("Cancel", (dialog, which) -> dialog.cancel());

        AlertDialog dialog = builder.create();
        dialog.setOnDismissListener(d -> currentFeature.release());
        dialog.show();
    }

    private String findMatchingFace(Mat currentFeature) {
        if (mRegisteredFaces.isEmpty()) {
            return null;
        }

        for (Map.Entry<String, Mat> entry : mRegisteredFaces.entrySet()) {
            double cosScore = mFaceRecognizer.match(entry.getValue(), currentFeature, FaceRecognizerSF.FR_COSINE);
            double cosThreshold = 0.363; // Recommended threshold for SFace

            if (cosScore > cosThreshold) {
                return entry.getKey(); // Match found
            }
        }
        return null; // No match found
    }


    private void addFaceToDatabase(String name, Mat feature) {
        File storageDir = getExternalFilesDir(null);
        if (storageDir == null) {
            Toast.makeText(this, "External storage not available.", Toast.LENGTH_SHORT).show();
            return;
        }

        File dbFile = new File(storageDir, DATABASE_FILE_NAME);
        HashMap<String, float[]> faceDatabase = loadFaceDatabase(dbFile);

        float[] featureArray = new float[feature.cols() * feature.rows()];
        feature.get(0, 0, featureArray);
        faceDatabase.put(name, featureArray);

        try (FileOutputStream fos = new FileOutputStream(dbFile); ObjectOutputStream oos = new ObjectOutputStream(fos)) {
            oos.writeObject(faceDatabase);
            mRegisteredFaces.put(name, feature.clone()); // Update in-memory map
            Toast.makeText(this, "Face of '" + name + "' registered successfully.", Toast.LENGTH_LONG).show();
        } catch (IOException e) {
            Log.e(TAG, "Error saving face database", e);
            Toast.makeText(this, "Error saving face feature.", Toast.LENGTH_SHORT).show();
        }
    }

    @SuppressWarnings("unchecked")
    private HashMap<String, float[]> loadFaceDatabase(File dbFile) {
        if (!dbFile.exists()) {
            return new HashMap<>(); // Return a new map if file doesn't exist
        }
        try (FileInputStream fis = new FileInputStream(dbFile); ObjectInputStream ois = new ObjectInputStream(fis)) {
            return (HashMap<String, float[]>) ois.readObject();
        } catch (IOException | ClassNotFoundException e) {
            Log.e(TAG, "Error loading face database", e);
            return new HashMap<>(); // Return a new map on error
        }
    }

    private void loadData() {
        File storageDir = getExternalFilesDir(null);
        if (storageDir == null) {
            Toast.makeText(this, "External storage is not available.", Toast.LENGTH_SHORT).show();
            return;
        }

        File dbFile = new File(storageDir, DATABASE_FILE_NAME);
        HashMap<String, float[]> faceDatabase = loadFaceDatabase(dbFile);

        mRegisteredFaces.clear();
        for (Map.Entry<String, float[]> entry : faceDatabase.entrySet()) {
            Mat faceFeature = new Mat(1, entry.getValue().length, 5); // CV_32F
            faceFeature.put(0, 0, entry.getValue());
            mRegisteredFaces.put(entry.getKey(), faceFeature);
        }
        Toast.makeText(this, mRegisteredFaces.size() + " registered face(s) loaded.", Toast.LENGTH_SHORT).show();
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
        for(Mat mat : mRegisteredFaces.values()) {
            mat.release();
        }
    }

    private void visualizeAndVerify(Mat rgba, Mat faces) {
        if (mFaceRecognizer == null || faces.empty()) {
            return;
        }

        for (int i = 0; i < faces.rows(); i++) {
            float[] faceData = new float[faces.cols() * faces.channels()];
            faces.get(i, 0, faceData);

            Rect box = new Rect(Math.round(mScale*faceData[0]), Math.round(mScale*faceData[1]),
                    Math.round(mScale*faceData[2]), Math.round(mScale*faceData[3]));
            Imgproc.rectangle(rgba, box, BOX_COLOR, 2);

            String identifiedName = null;

            if (!mRegisteredFaces.isEmpty()) {
                Mat alignedFace = new Mat();
                mFaceRecognizer.alignCrop(mBgr, faces.row(i), alignedFace);
                Mat currentFeature = new Mat();
                mFaceRecognizer.feature(alignedFace, currentFeature);

                identifiedName = findMatchingFace(currentFeature);

                currentFeature.release();
                alignedFace.release();
            }

            if (identifiedName != null) {
                Imgproc.putText(rgba, identifiedName, new Point(box.x, box.y - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.9, MATCH_COLOR, 2);
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
