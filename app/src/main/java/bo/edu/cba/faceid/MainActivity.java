package bo.edu.cba.faceid;

import com.parse.ParseObject;
import com.parse.ParseQuery;

import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
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
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MainActivity extends CameraActivity implements CvCameraViewListener2 {

    private static final String    TAG  = "OCVSample::Activity";

    private static final Scalar    BOX_COLOR         = new Scalar(0, 255, 0);
    private static final Scalar    MATCH_COLOR       = new Scalar(0, 255, 0);
    private static final int       NUM_REGISTRATION_SAMPLES = 5;

    private Mat                    mRgba;
    private Mat                    mBgr;
    private Mat                    mBgrScaled;
    private Size                   mInputSize = null;
    private final float            mScale = 2.f;
    private FaceDetectorYN         mFaceDetector;
    private FaceRecognizerSF       mFaceRecognizer;
    private Mat                    mFaces;
    private final HashMap<String, Mat> mRegisteredFaces = new HashMap<>();
    private final List<Mat>        registrationFeatures = new ArrayList<>();

    private CameraBridgeViewBase   mOpenCvCameraView;
    private int mCameraId = CameraBridgeViewBase.CAMERA_ID_BACK;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if (OpenCVLoader.initLocal()) {
            Log.i(TAG, "OpenCV loaded successfully");
        } else {
            Log.e(TAG, "¡La inicialización de OpenCV falló!");
            Toast.makeText(this, "¡La inicialización de OpenCV falló!", Toast.LENGTH_LONG).show();
            return;
        }

        loadFaceModels();
        loadData();

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
            Toast.makeText(this, "No se detectó ningún rostro para iniciar el registro.", Toast.LENGTH_SHORT).show();
            return;
        }
        if (mFaces.rows() > 1) {
            Toast.makeText(this, "Se detectaron múltiples rostros. Por favor, asegúrese de que solo haya uno.", Toast.LENGTH_SHORT).show();
            return;
        }

        final Mat currentFeature = new Mat();
        Mat alignedFace = new Mat();
        mFaceRecognizer.alignCrop(mBgr, mFaces.row(0), alignedFace);
        mFaceRecognizer.feature(alignedFace, currentFeature);
        alignedFace.release();

        String matchedName = findMatchingFace(currentFeature);
        currentFeature.release();

        if (matchedName != null) {
            Toast.makeText(this, "Este rostro parece ya estar registrado como '" + matchedName + "'.", Toast.LENGTH_LONG).show();
            return;
        }

        startGuidedRegistration();
    }

    private void startGuidedRegistration() {
        for (Mat feature : registrationFeatures) {
            feature.release();
        }
        registrationFeatures.clear();
        captureStep(0);
    }

    private void captureStep(int step) {
        if (step >= NUM_REGISTRATION_SAMPLES) {
            processAndSaveTemplate();
            return;
        }

        String[] instructions = {
                "Paso 1/5: Mire al Frente",
                "Paso 2/5: Gire Ligeramente a la Izquierda",
                "Paso 3/5: Gire Ligeramente a la Derecha",
                "Paso 4/5: Mire Ligeramente Hacia Arriba",
                "Paso 5/5: Mire Ligeramente Hacia Abajo"
        };

        new AlertDialog.Builder(this)
                .setTitle("Registrar Nuevo Rostro")
                .setMessage(instructions[step])
                .setPositiveButton("Capturar", (dialog, which) -> {
                    if (mFaces == null || mFaces.empty() || mFaces.rows() > 1) {
                        Toast.makeText(this, "Por favor, asegúrese de que solo un rostro esté claramente visible.", Toast.LENGTH_SHORT).show();
                        captureStep(step);
                        return;
                    }

                    Mat alignedFace = new Mat();
                    mFaceRecognizer.alignCrop(mBgr, mFaces.row(0), alignedFace);
                    Mat feature = new Mat();
                    mFaceRecognizer.feature(alignedFace, feature);
                    registrationFeatures.add(feature);
                    alignedFace.release();

                    Toast.makeText(this, "¡Captura " + (step + 1) + " exitosa!", Toast.LENGTH_SHORT).show();

                    captureStep(step + 1);
                })
                .setNegativeButton("Cancelar", (dialog, which) -> {
                    for (Mat f : registrationFeatures) {
                        f.release();
                    }
                    registrationFeatures.clear();
                    dialog.cancel();
                })
                .setCancelable(false)
                .show();
    }

    private void processAndSaveTemplate() {
        Mat averagedFeature = new Mat();
        registrationFeatures.get(0).copyTo(averagedFeature);

        for (int i = 1; i < registrationFeatures.size(); i++) {
            Core.add(averagedFeature, registrationFeatures.get(i), averagedFeature);
        }
        Core.divide(averagedFeature, new Scalar(registrationFeatures.size()), averagedFeature);

        Core.normalize(averagedFeature, averagedFeature);

        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Ingrese el Nombre para el Rostro Registrado");
        final EditText input = new EditText(this);
        input.setInputType(InputType.TYPE_CLASS_TEXT);
        builder.setView(input);

        builder.setPositiveButton("Guardar", (dialog, which) -> {
            String name = input.getText().toString().trim();
            if (name.isEmpty()) {
                Toast.makeText(this, "El nombre no puede estar vacío.", Toast.LENGTH_SHORT).show();
            } else if (mRegisteredFaces.containsKey(name)) {
                Toast.makeText(this, "El nombre '" + name + "' ya existe.", Toast.LENGTH_LONG).show();
            } else {
                float[] featureArray = new float[averagedFeature.cols() * averagedFeature.rows()];
                averagedFeature.get(0, 0, featureArray);

                addUserToDatabase(name, featureArray);

                mRegisteredFaces.put(name, averagedFeature.clone());
                Toast.makeText(this, "Rostro de '" + name + "' registrado exitosamente.", Toast.LENGTH_LONG).show();
            }
        });

        builder.setNegativeButton("Cancelar", (dialog, which) -> dialog.cancel());

        AlertDialog dialog = builder.create();
        dialog.setOnDismissListener(d -> {
            averagedFeature.release();
            for (Mat feature : registrationFeatures) {
                feature.release();
            }
            registrationFeatures.clear();
        });
        dialog.show();
    }


    private String findMatchingFace(Mat currentFeature) {
        if (mRegisteredFaces.isEmpty()) {
            return null;
        }

        String bestMatchName = null;
        double maxCosineScore = 0.0;
        double cosThreshold = 0.363;

        for (Map.Entry<String, Mat> entry : mRegisteredFaces.entrySet()) {
            double cosScore = mFaceRecognizer.match(entry.getValue(), currentFeature, FaceRecognizerSF.FR_COSINE);

            if (cosScore > cosThreshold && cosScore > maxCosineScore) {
                maxCosineScore = cosScore;
                bestMatchName = entry.getKey();
            }
        }
        return bestMatchName;
    }

    private void addUserToDatabase(String name, float[] featureArray) {
        List<Double> featureList = new ArrayList<>();
        for (float v : featureArray) {
            featureList.add((double) v);
        }

        ParseObject user = new ParseObject("UserFace");
        user.put("name", name);
        user.put("faceEmbedding", featureList);
        user.saveInBackground(e -> {
            if (e == null) {
                Log.d(TAG, "Usuario '" + name + "' guardado en Back4App.");
            } else {
                Log.e(TAG, "Error al guardar el usuario en Back4App", e);
            }
        });
    }

    private void loadData() {
        ParseQuery<ParseObject> query = ParseQuery.getQuery("UserFace");
        query.findInBackground((users, e) -> {
            if (e == null) {
                for (Mat mat : mRegisteredFaces.values()) {
                    mat.release();
                }
                mRegisteredFaces.clear();

                for (ParseObject user : users) {
                    String name = user.getString("name");
                    List<Double> featureList = user.getList("faceEmbedding");
                    if (name != null && featureList != null) {
                        float[] featureArray = new float[featureList.size()];
                        for (int i = 0; i < featureList.size(); i++) {
                            featureArray[i] = featureList.get(i).floatValue();
                        }

                        Mat faceFeature = new Mat(1, featureArray.length, CvType.CV_32F);
                        faceFeature.put(0, 0, featureArray);
                        mRegisteredFaces.put(name, faceFeature);
                    }
                }
                Toast.makeText(this, mRegisteredFaces.size() + " rostro(s) registrado(s) cargado(s) desde la nube.", Toast.LENGTH_SHORT).show();
            } else {
                Log.e(TAG, "Error al cargar los usuarios desde Back4App", e);
            }
        });
    }

    private void swapCamera() {
        mCameraId = mCameraId == CameraBridgeViewBase.CAMERA_ID_BACK ? CameraBridgeViewBase.CAMERA_ID_FRONT : CameraBridgeViewBase.CAMERA_ID_BACK;
        mOpenCvCameraView.disableView();
        mOpenCvCameraView.setCameraIndex(mCameraId);
        mOpenCvCameraView.enableView();
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
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
        for(Mat mat : registrationFeatures) {
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
