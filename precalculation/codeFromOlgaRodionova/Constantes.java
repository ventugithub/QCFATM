package TRAFFIC;

public class Constantes {
 
    public static double NM64_TO_METER = 1852.0/64;

    // constante de conversion
    public static double NM_TO_METER = 1852.0;
    public static double KT_TO_MS = 1852.0/3600.0;
    public static double FEET_TO_METER = 0.3048006096012;
    public static double FL_TO_METER = 30.48006096012;
    public static double FTM_TO_MS = 0.3048006096012/60.0;   
    public static double DEG_TO_RAD = Math.PI/180;
    public static double RAD_TO_DEG = 180 / Math.PI;
    public static double MATH_PI_2  = Math.PI * 2;
    public static double NM_TO_DEGREE = 1.0 / 60.0;
    
    public static double EARTH_RADIUS = 6371210; // in meters
    public static double SECONDS_PER_DAY = 24 * 60 * 60;
    public static int DELAY_MAX  = 30 * 60; // en secondes
    public static int DELAY_SLOT = 60; //secondes (toujours plus grand ou egal a DeltaT)
    public static int NB_SLOTS   = DELAY_MAX / DELAY_SLOT;
    
     // Global Grid Limits
    public static double LON_MIN = -180.0;   // in degree
    public static double LON_MAX = 180.0;  
    public static double LAT_MIN = -90.0;
    public static double LAT_MAX = 90.0;
    public static double ALT_MIN = 24000;  // in feet
    public static double ALT_MAX = 44000;
    public static double T_MIN   = 0.0;   // in seconds
    public static double T_MAX   = 172800.0; // 
    
    // separation norms
    public static double SEP_NORM_HOR_NM      = 30; // in nautical miles
    public static double SEP_NORM_HOR_METERS  = SEP_NORM_HOR_NM * NM_TO_METER; // in meters
    public static double SEP_NORM_HOR_DEGREES = SEP_NORM_HOR_NM * NM_TO_DEGREE; // in degrees
    public static double SEP_NORM_VERTICAL    = 1000; // in feet
    public static int    SEP_NORM_TIME        = 180; // in seconds
    
    // grid steps
    public static double STEP_LON = SEP_NORM_HOR_DEGREES * 2; // in degrees
    public static double STEP_LAT = SEP_NORM_HOR_DEGREES;
    public static double STEP_ALT = SEP_NORM_VERTICAL; // in feet
    public static double STEP_T   = 60; // in seconds
   
    // number of grid cells
    public static long NB_LON    = (long)((LON_MAX - LON_MIN) / STEP_LON);
    public static long NB_LAT    = (long)((LAT_MAX - LAT_MIN) / STEP_LAT);
    public static long NB_ALT    = (long)((ALT_MAX - ALT_MIN) / STEP_ALT);
    public static long NB_T      = (long)((T_MAX - T_MIN) / STEP_T);
    public static long NB_MULT_I = NB_T;
    public static long NB_MULT_J = NB_MULT_I * NB_LON;
    public static long NB_MULT_K = NB_MULT_J * NB_LAT;

    public static boolean EXTENSION_SPATIALE_PROGRESSIVE = true;
    public static boolean ACTION_SPATIALE_AUTHORIZED = false;
    public static boolean ACTION_TEMPORAL_AUTHORIZED = false;
    public static boolean CUT_TRAJECTORY_TO_SPACE    = false;
        
    public static int VERIFY_ON_LEVELS                = 0;
    public static int VERIFY_CONTINIOUSLY             = 1;
    public static int MODIFY_PARTICULAR_TRAJECTORY    = 0;
    public static int MODIFY_INTERACTING_TRAJECTORIES = 1;
    public static int MODIFY_PT_AND_IT                = 2;
    
    public static int    NB_TRANSITIONS_COOLING   = 100;
    public static int    NB_TRANSITIONS_HEATING   = 100;
    public static double ALPHA_COOLING            = 0.99;
    public static double INITIAL_ACCEPTANCE_PROBA = 0.3;
    
    public static int NAT_MIN_LONGITUDE = -70; // in degrees
    public static int NAT_MAX_LONGITUDE = 0;
    public static int NAT_MIN_LATITUDE  = 0;
    public static int NAT_MAX_LATITUDE  = 90;    
    
    public static double MAX_CURVE_DEVIATION = 0.1; // in ratio to 1, ex. 10%
    
    public static int SEUIL_TYPE = 0; // 0 - seuil = 0.5 * mean * (1 - tempRatio)
                                      // 1 - seuil = mean * (1 - tempRatio)
    								  // 2 - seuil = 0.5 * mean
                                      // 3 - seuil = 0.25 * mean
    
    public static int fileNumber;
    public static String fileType = "ADSB";
    
    public static int TIME_STEP_FOR_CONFLICT_RECORD = 30 * 60; // in seconds
    
}