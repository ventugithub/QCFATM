package DETECTION;
//import java.io.*;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.*;

import TRAFFIC.*;
import WIND.DayWind;
import RECUIT.*;

public abstract class GridAL {
	
	private int                               nbTrajs;
	private int                               nbTrajsModif;
    private HashMap<Long,ArrayList<InfoPlot>> tableHash = new HashMap<Long,ArrayList<InfoPlot>>();
    private TrajectoryInteractions[]          tableFlag;    
    private Trajectory4DSet                   trajSet;    
    private DayWind                           winds =  null;
    private double                            congestionMax;
    private double                            congestionMin;
    private double                            congestionMean;
    
    private NATFIRS                           oceanicFIRs;
    private FIR                               oceanicArea = null;                        
    
    private int                               worstTraj = -1;
    
    public void resetCongestion()
    {
    	congestionMax  = 0.0; 
		congestionMin  = 1000000.0;
		congestionMean = 0.0;
		worstTraj      = -1;
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////
    // trajectory management functions
    ////////////////////////////////////////////////////////////////////////////////////////
    
    public void putAllTrajectories()
    {
    	tableHash.clear();
    	for (int i = 0; i < nbTrajs; i++)
    	{
    		Trajectory4D traj = trajSet.getTrajectory(i);
    		putAllTrajectoryPoints(i, traj);
    	}     	
    }
    
    private void putAllTrajectoryPoints(int trajId, Trajectory4D traj)
    {
    	int nbPts = traj.getNumPoints();
    	for (int j = 0; j < nbPts; j++)
		{
			InfoPlot info = new InfoPlot(trajId, j);			
			if (traj.getPoint4D(j).isInSpace())
			{
				long key = traj.getPoint4D(j).getKey();
				addPoint3D(info, key);				
			}			
		}		
    }    

    public void addPoint3D(InfoPlot info, long key) 
    {	
		if (tableHash.get(key) == null)
			tableHash.put(key, new ArrayList<InfoPlot>());	    		
	    tableHash.get(key).add(info);	
    }     
    
    ////////////////////////////////////////////////////////////////////////////////////////
    // evaluate functions
    ////////////////////////////////////////////////////////////////////////////////////////

    public double evaluateFullTrajSet(Decision[] tableDecision)
    {
		double congestionGlobale = 0.0;		
		resetCongestion();
				
		for (int i = 0; i < nbTrajs; i++)
		{
		    tableDecision[i].resetCongestion();
		    tableFlag[i].clear();
		}
				
		//calcul du nombre de voisins en chaque point et sommation
		for (int i = 0; i < nbTrajs; i++)
		{
			TrajectoryInteractions listTrajInteractions = tableFlag[i];
			Decision               uneDecision          = tableDecision[i];	
		    Trajectory4D           traj                 = trajSet.getTrajectory(i);	
		    int                    nbPts                = traj.getNumPoints();		   	
		    
		    for (int j = 0; j < nbPts; j++)
			{
		    	GeoPoint4D curPt = traj.getPoint4D(j);
		    	if (curPt.isInSpace())
		    	{
			    	long                key          = curPt.getKey();
				    InfoPlot            info         = new InfoPlot(i, j);
				    ArrayList<InfoPlot> listeVoisins = new ArrayList<InfoPlot>();
				    ArrayList<Double>   conflictRate = new ArrayList<Double>();
				    
				    searchNeighbors(info, key, listeVoisins, conflictRate);
				    uneDecision.addCongestion(conflictRate, listeVoisins.size());
				    for (int k = 0; k < listeVoisins.size(); k++)
					{		  
				    	listTrajInteractions.addInteractions(listeVoisins.get(k).getIdTraj(), conflictRate.get(k), 1);								
				    }
		    	}
			}	 
		    double congestionCur = uneDecision.getCongestionLevel();
		    congestionGlobale += congestionCur;
		    if (congestionCur < congestionMin)
		    {
		    	congestionMin = congestionCur;
		    }
		    if (congestionCur > 0)
		    {
		    	if (congestionCur > congestionMax)
			    {
			    	congestionMax = congestionCur;
			    	worstTraj     = i;
			    }
		    }
		}		
		congestionMean = congestionGlobale / nbTrajs;
		return congestionGlobale;
    }    
    
    protected void searchNeighbors(InfoPlot info, long key, ArrayList<InfoPlot> bufferList, 
			ArrayList<Double> conflictRate) 
    {   
		GridCoordinate      g          = new GridCoordinate(key);				
		searchNeighborsOnLevel(info, g, bufferList, conflictRate); 			
    }
    
    protected void searchNeighborsOnLevel(InfoPlot info, GridCoordinate g, ArrayList<InfoPlot> bufferList, 
    		ArrayList<Double> conflictRate) 
    {   
		int  j        = g.getJ();
		int  i        = g.getI();
		int  t        = g.getT();	    
		long indKmult = g.getK() * Constantes.NB_MULT_K;
		
		for (long indJ = j - 1; indJ <= j + 1; indJ++) // check neighbor cells
		{
			long indJmult = indJ * Constantes.NB_MULT_J;
			for (long indI = i - 1; indI <= i + 1; indI++)
			{
				long indImult = indI * Constantes.NB_MULT_I;
			    for (long indT = t - 3; indT <= t + 3; indT++)
			    { 
			    	long newKey = indKmult + indJmult + indImult + indT;
				   	searchNeighborsInCell(info, newKey, bufferList, conflictRate);									    
				}
			}			
		}  		
    }
    
    protected void searchNeighborsInCell(InfoPlot info, long key, ArrayList<InfoPlot> bufferList, 
    		ArrayList<Double> conflictRate)
    {
    	ArrayList<InfoPlot> liste = tableHash.get(key);
    	if (liste == null) // there are no plots in this cell
    		return;
    	
		int idPointVol   = info.getIdPt();
		int indexTrajVol = info.getIdTraj();
	    for (InfoPlot plot : liste)
	    {
	    	boolean checkConflictBetween = false;
	    	int     idVoisin             = plot.getIdPt();
	    	int     indexTrajVoisin      = plot.getIdTraj();	
	    	
	    	if (indexTrajVol != indexTrajVoisin) // if it is not the same flight
	    	{		      		       
	    		GeoPoint4D p1     = trajSet.getTrajectoryPoint(indexTrajVoisin, idVoisin);
	    		GeoPoint4D p2     = trajSet.getTrajectoryPoint(indexTrajVol, idPointVol);		    
	    		GeoPoint4D p1Next = trajSet.getTrajectoryPoint(indexTrajVoisin, idVoisin + 1);
	    		GeoPoint4D p2Next = trajSet.getTrajectoryPoint(indexTrajVol, idPointVol + 1);
	    		
			    if (p1Next != null && p2Next != null && p1Next.isInSpace() && p2Next.isInSpace())
			    	checkConflictBetween = true;
			    double curConflict = inCoflict(p1, p1Next, p2, p2Next, checkConflictBetween);
			    if (curConflict > 0)
			    {
				    bufferList.add(plot);	
				    conflictRate.add(curConflict);
			    }
	    	}		
	    }	
    } 
    
    protected double inCoflict(GeoPoint4D p1, GeoPoint4D p1Next, GeoPoint4D p2, GeoPoint4D p2Next, boolean checkBetween)
	{
		if (p1.distanceAlt(p2) == 0) 
	   	    return GeoPoint4D.inConflict2D(p1, p1Next, p2, p2Next, checkBetween);	
		
	   	return 0;
	}
    
}