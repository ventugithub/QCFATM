package TRAFFIC;

public class GeoPoint4D {

	//private int    id;
	private double time;   // in seconds
	private double lat;    // in degree
	//private double latRad; // in radian
	private double sinLat;
	private double cosLat;
	private double lon;    // in degree
	//private double lonRad; // in radian
	private double sinLon;
	private double cosLon;
	private double alt;    // in feet 
	private double speed;  // in m/s
    //public int tendency;
    //public int mode;
	private GridCoordinate gridCoordinate;
	private GeoPoint4D[]   middlePointsOnGreatCircle = null;
	private double         distToNext                = 0;
	
	private boolean        isInSpace = false;
    
	public void updateGridCoordinates()
    {
    	int i = (int)((lon - Constantes.LON_MIN) / Constantes.STEP_LON);
        int j = (int)((lat - Constantes.LAT_MIN) / Constantes.STEP_LAT);
        int k = (int)((alt - Constantes.ALT_MIN) / Constantes.STEP_ALT);
        int t = (int)((time - Constantes.T_MIN) / Constantes.STEP_T);
        gridCoordinate = new  GridCoordinate(i, j, k, t);
    }
    
    public long getKey()
    {
    	return gridCoordinate.getKey();
    }
    
    public int getFlightLevelId()
    {
    	return gridCoordinate.getK();
    }
}
