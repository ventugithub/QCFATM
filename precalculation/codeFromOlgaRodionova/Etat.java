package RECUIT;

import java.util.*;
import java.io.*;

import TRAFFIC.*;
import DETECTION.*;

public abstract class Etat {

	protected int                                      dimEtatTotal;
	protected int                                      dimEtatModif;
	protected Decision[]                               tableDecision;
	protected ComeBackInfo[]                           listModif = new ComeBackInfo[2];   
	    
	protected GridAL              grid;
	protected static Random       generateur = new Random(345);
    
    public double calculCriterSimple()
    {
    	grid.putAllTrajectories();
    	double res = grid.evaluateFullTrajSet(tableDecision);
		return res;
    }    
}