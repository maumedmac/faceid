package bo.edu.cba.faceid;

import android.app.Application;
import com.parse.Parse;

public class App extends Application {
    @Override
    public void onCreate() {
        super.onCreate();
        Parse.initialize(new Parse.Configuration.Builder(this)
                .applicationId("29or4z0wCBRgX6nH5t9I24FDbkcuVAXYQS8jaciy") // Reemplaza con tu App ID
                .clientKey("LgBf5uqnRHmPvw5BIviERdB0HIy36E6IyYfxZIgG") // Reemplaza con tu Client Key
                .server("https://parseapi.back4app.com/")
                .build());
    }
}
