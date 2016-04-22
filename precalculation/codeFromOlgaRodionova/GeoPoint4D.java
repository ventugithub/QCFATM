package TRAFFIC;

public class GeoPoint4D {
	
	private double time;   // in seconds
	private double lat;    // in degree	
	private double sinLat;
	private double cosLat;
	private double lon;    // in degree	
	private double sinLon;
	private double cosLon;
	private double alt;    // in feet 
	private double speed;  // in m/s
    private GridCoordinate gridCoordinate;
	private GeoPoint4D[]   middlePointsOnGreatCircle = null;
	private double         distToNext                = 0;	
	private boolean        isInSpace = false;
    
	public long getKey()
    {
    	return gridCoordinate.getKey();
    }
	 
	public boolean isInSpace()
    {
    	return isInSpace;
    }
	 
    /////////////////////////////////////////////////////////////////////////
    // check if two points are in conflict
    ///////////////////////////////////////////////////////////////////////////    
    
    public void setMiddlePointsOnGreatCircle(GeoPoint4D pNext, int numPoints)
    {
    	if (numPoints < 1)
    		return;
    	
    	GeoPoint4D[] middlePoints = new GeoPoint4D[numPoints];
    	double       altitudeStep = (pNext.alt - alt) / (numPoints + 1);
    	double       altCur       = alt;
    	double       timeStep     = (pNext.time - time) / (numPoints + 1);
    	double       timeCur      = time;
    	
    	if (lat == pNext.lat)
    	{
    		double lonDistDeg = pNext.lon - lon;
    		double lonStepDeg = lonDistDeg / (numPoints + 1);
    		double lonCur     = lon;
    		    		
    		for (int i = 0; i < numPoints; i++)
    		{
    			lonCur  += lonStepDeg;
    			altCur  += altitudeStep;
    			timeCur += timeStep;
    			middlePoints[i] = new GeoPoint4D(lonCur, lat, altCur, timeCur);     			
    		}
    	}
    	else if (lon == pNext.lon)
    	{
    		double latDistDeg = pNext.lat - lat;
    		double latStepDeg = latDistDeg / (numPoints + 1);
    		double latCur     = lat;
    		    		
    		for (int i = 0; i < numPoints; i++)
    		{
    			latCur  += latStepDeg;
    			altCur  += altitudeStep;
    			timeCur += timeStep;
    			middlePoints[i] = new GeoPoint4D(lon, latCur, altCur, timeCur);    			
    		}
    	}
    	else
    	{
    		double cosPhi1     = cosLat;
        	double sinPhi1     = sinLat;
        	double tanPhi1     = sinPhi1 / cosPhi1;
        	double cosPhi2     = pNext.cosLat;
        	double sinPhi2     = pNext.sinLat;
        	double tanPhi2     = sinPhi2 / cosPhi2;
        	double sinLambda12 = pNext.sinLon * cosLon - pNext.cosLon * sinLon; 
        	double cosLambda12 = pNext.cosLon * cosLon + pNext.sinLon * sinLon; 
        	double alpha1      = Math.atan2(sinLambda12, cosPhi1 * tanPhi2 - sinPhi1 * cosLambda12);
        	double alpha2      = Math.atan2(sinLambda12, -cosPhi2 * tanPhi1 + sinPhi2 * cosLambda12);
        	double sigma01     = Math.atan2(tanPhi1, Math.cos(alpha1));
        	double sigma02     = Math.atan2(tanPhi2, Math.cos(alpha2));
        	double sigma12     = sigma02 - sigma01;
        	double stepSigma   = sigma12 / (numPoints + 1);
        	double sinAlpha0   = Math.sin(alpha1) * cosPhi1;
        	double cosAlpha0   = Math.cos(Math.asin(sinAlpha0));
        	double lambda01    = Math.atan2(sinAlpha0 * Math.sin(sigma01), Math.cos(sigma01));
        	double lambda0     = lon * Constantes.DEG_TO_RAD - lambda01;
        	double sigma       = sigma01;
        	
        	for (int i = 0; i < numPoints; i++)
        	{
        		sigma   += stepSigma;
        		altCur  += altitudeStep;
        		timeCur += timeStep;
        		
        		double sinSigma    = Math.sin(sigma);
        		double sinPhi      = cosAlpha0 * sinSigma;
        		double phi         = Math.asin(sinPhi);
        		double deltaLambda = Math.atan2(sinAlpha0 * sinSigma, Math.cos(sigma));
        		double lambda      = lambda0 + deltaLambda;  
        		
        		middlePoints[i] = new GeoPoint4D(0, 0, altCur, timeCur);         		
        		middlePoints[i].setLongitudeRad(lambda);
        		middlePoints[i].setLatitudeRad(phi);        		
        	}
    	}
    	
    	middlePointsOnGreatCircle = middlePoints;
    	distToNext                = distance2D(pNext);
    }
    
    public static double inConflict2D(GeoPoint4D p1, GeoPoint4D p1Next, GeoPoint4D p2, GeoPoint4D p2Next, boolean checkBetween)
    {
    	double conflict = p1.checkConflict2Dtime(p2);
    	
    	if (conflict > 0)
    		return conflict;
    	
		if (!checkBetween)
			return 0;		
		
		GeoPoint4D[] p1quaterPoints  = p1.middlePointsOnGreatCircle;
		GeoPoint4D[] p2quaterPoints  = p2.middlePointsOnGreatCircle;
		int          numMiddlePoints = p1quaterPoints.length;		
		
		for (int i = 0; i < numMiddlePoints; i++)
		{
			conflict = p1quaterPoints[i].checkConflict2Dtime(p2quaterPoints[i]);
			if (conflict > 0)
				return conflict;		    
	    }
		return 0;
	}  
    
    private double checkConflict2Dtime(GeoPoint4D p)
    {
    	return checkConflictTime(p) * checkConflict2D(p);    	
    }
    
    private double checkConflictTime(GeoPoint4D p)
    {
    	double t = Constantes.SEP_NORM_TIME - Math.abs(time - p.time);
    	if (t > 0)
    		return 1;
    	else
    		return 0;    	
    }
    
    private double checkConflict2D(GeoPoint4D p)
    {
    	double dist = Constantes.SEP_NORM_HOR_METERS - distance2D(p);    	
    	if (dist > 0)
    		return 1;
    	else
    		return 0;
    } 
    
    public double distance2D(GeoPoint4D p)
    {
    	double CosD = sinLat * p.sinLat + cosLat * p.cosLat * (cosLon * p.cosLon + sinLon * p.sinLon);
		CosD = Math.min(CosD, 1);
		CosD = Math.max(CosD, -1);
		return Constantes.EARTH_RADIUS * Math.acos(CosD);
    }
    
    public double distanceAlt(GeoPoint4D p)
    {
    	return Math.abs(alt - p.alt);
    }
 }
