package br.com.macsfilho.facedetection;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;

import org.apache.http.HttpEntity;
import org.apache.http.HttpResponse;
import org.apache.http.NameValuePair;
import org.apache.http.client.HttpClient;
import org.apache.http.client.entity.UrlEncodedFormEntity;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.impl.client.DefaultHttpClient;
import org.apache.http.message.BasicNameValuePair;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.objdetect.CascadeClassifier;

import android.app.Activity;
import android.content.Context;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Vibrator;
import android.telephony.TelephonyManager;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.WindowManager;


public class FdActivity extends Activity implements CvCameraViewListener2 {

    //TAG for Log
    private static final String    TAG                 = "FaceDetector";

    private static final Scalar    FACE_RECT_COLOR     = new Scalar(255,0,0,150);
    public static final int        JAVA_DETECTOR       = 0;
    public static final int        NATIVE_DETECTOR     = 1;


    long dot=300;
    long dash=650;
    long sGap=250;
    Vibrator mVibrator;

    long timeout = 600000000;//nanosec
    long timeoutV=20000 ;//ms
    Calendar c;
    SimpleDateFormat df=new SimpleDateFormat("yyyy-MM-dd");
    SimpleDateFormat tf=new SimpleDateFormat("HH:mm:ss");

    private MenuItem               mItemFace50;
    private MenuItem               mItemFace40;
    private MenuItem               mItemFace30;
    private MenuItem               mItemFace20;
    private MenuItem               mItemType;

    long start;
    long startV;
    long[] pattern;

    private Mat                    mRgba;
    private Mat                    mGray;
    private File                   mCascadeFile;
    private CascadeClassifier      mJavaDetector;
    private DetectionBasedTracker  mNativeDetector;

    private int                    mDetectorType       =NATIVE_DETECTOR;
    private String[]               mDetectorName;

    String imei;
    ArrayList<NameValuePair> data = new ArrayList<NameValuePair>(2);


    private float                  mRelativeFaceSize   = 0.2f;
    private int                    mAbsoluteFaceSize   = 0;

    private CameraBridgeViewBase   mOpenCvCameraView;

    private static final ScheduledExecutorService worker =
            Executors.newSingleThreadScheduledExecutor();



    class SendData extends AsyncTask<String, Void , Void>{

        @Override
        protected Void doInBackground(String... params) {

            data.add(new BasicNameValuePair("nFaces",params[0]));
            data.add(new BasicNameValuePair("imei",params[1]));
            data.add(new BasicNameValuePair("date",params[2]));
            data.add(new BasicNameValuePair("time",params[3]));

            //http post
            try{
                HttpClient httpclient = new DefaultHttpClient();
                HttpPost httppost = new
                        HttpPost("http://www.facescount.tk/webservices/receive.php");
                        //HttpPost("http://10.0.2.2:4040/projects/androidProject/webservices/receive.php");
                        //HttpPost("http://192.168.43.187:4040/projects/androidProject/webservices/receive.php");
                httppost.setEntity(new UrlEncodedFormEntity(data));
                HttpResponse response = httpclient.execute(httppost);
                HttpEntity entity = response.getEntity();
                InputStream is = entity.getContent();
                Log.i("postData", response.getStatusLine().toString());
            }
            catch(Exception e)
            {
                Log.e("log_tag", "Error in http connection "+e.toString());
            }

            return null;
        }

        @Override
        protected void onPreExecute() {
            super.onPreExecute();

        }


        @Override
        protected void onPostExecute(Void aVoid) {
            super.onPostExecute(aVoid);
        }
    }


    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {



        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");

                    // Load native library after(!) OpenCV initialization
                    System.loadLibrary("detection_based_tracker");

                    try {
                        // load cascade file from application resources
                        InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
                        FileOutputStream os = new FileOutputStream(mCascadeFile);

                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        is.close();
                        os.close();

                        mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                        if (mJavaDetector.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier");
                            mJavaDetector = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

                        mNativeDetector = new DetectionBasedTracker(mCascadeFile.getAbsolutePath(), 0);

                        cascadeDir.delete();

                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }

                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public FdActivity() {
        mDetectorName = new String[2];
        mDetectorName[JAVA_DETECTOR] = "Java";
        mDetectorName[NATIVE_DETECTOR] = "Native (tracking)";

    }


    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        start = System.nanoTime();
        startV = System.currentTimeMillis();
        setContentView(R.layout.face_detect_surface_view);
        if (!OpenCVLoader.initDebug()) {
            // Handle initialization error
        }
        TelephonyManager tm = (TelephonyManager)getSystemService(Context.TELEPHONY_SERVICE);
        imei=tm.getDeviceId();
        mVibrator=(Vibrator)getSystemService(Context.VIBRATOR_SERVICE);
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause(){
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume(){
        super.onResume();
        if(!OpenCVLoader.initDebug()){
            OpenCVLoader.initAsync("3.0.0", this, mLoaderCallback);
        }else{
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy(){
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame){
            mRgba = inputFrame.rgba();
            mGray = inputFrame.gray();

            if (mAbsoluteFaceSize == 0) {
                int height = mGray.rows();
                if (Math.round(height * mRelativeFaceSize) > 0) {
                    mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
                }
                mNativeDetector.setMinFaceSize(mAbsoluteFaceSize);
            }

            MatOfRect faces = new MatOfRect();

            if (mDetectorType == JAVA_DETECTOR) {
                if (mJavaDetector != null)
                    mJavaDetector.detectMultiScale(mGray, faces, 1.1, 2, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                            new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
            } else if (mDetectorType == NATIVE_DETECTOR) {
                if (mNativeDetector != null)
                    mNativeDetector.detect(mGray, faces);
            } else {
                Log.e(TAG, "Not selected!");
            }
            Rect[] facesArray = faces.toArray();
/*

        if(startV+timeoutV<System.currentTimeMillis()){
            startV=System.currentTimeMillis();
            pattern =new long[2+(facesArray.length*2)];
            pattern[0]=0;
            pattern[1]=dash;
            for(int i=2;i<pattern.length;i+=2){
                pattern[i]=sGap;
                pattern[i+1]=dot;
            }
            mVibrator.vibrate(pattern,-1);
        }

        if(start+timeout<System.nanoTime()  ){
            if(facesArray.length!=0) {
                start = System.nanoTime();
                c = Calendar.getInstance();
                Log.e("Number of Faces : ", Integer.toString(facesArray.length) + " ; " + imei);
                new SendData().execute(Integer.toString(facesArray.length), imei, df.format(c.getTime()), tf.format(c.getTime()));
            }
        }
*/
            for (int i = 0; i < facesArray.length; i++) {
                Core.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 3);
            }

            return mRgba;

    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        mItemFace50 = menu.add("50%");
        mItemFace40 = menu.add("40%");
        mItemFace30 = menu.add("30%");
        mItemFace20 = menu.add("20%");
        mItemType   = menu.add(mDetectorName[mDetectorType]);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        if (item == mItemFace50)
            setMinFaceSize(0.5f);
        else if (item == mItemFace40)
            setMinFaceSize(0.4f);
        else if (item == mItemFace30)
            setMinFaceSize(0.3f);
        else if (item == mItemFace20)
            setMinFaceSize(0.2f);
        else if (item == mItemType) {
            int tmpDetectorType = (mDetectorType + 1) % mDetectorName.length;
            item.setTitle(mDetectorName[tmpDetectorType]);
            setDetectorType(tmpDetectorType);
        }
        return true;
    }

    private void setMinFaceSize(float faceSize) {
        mRelativeFaceSize = faceSize;
        mAbsoluteFaceSize = 0;
    }

    private void setDetectorType(int type) {
        if (mDetectorType != type) {
            mDetectorType = type;

            if (type == NATIVE_DETECTOR) {
                mNativeDetector.start();
            } else {
                mNativeDetector.stop();
            }
        }
    }
}




