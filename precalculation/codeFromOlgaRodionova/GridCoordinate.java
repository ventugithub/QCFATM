package TRAFFIC;

public class GridCoordinate {
    private int  i;
    private int  j;
    private int  k;
    private int  t;
    private long key;
    private long keyTempo;
    private long keySpace;

    public GridCoordinate(int i, int j, int k, int t)
    {
    	this.i = i;
    	this.j = j;
    	this.k = k;
    	this.t = t;
    	computeKey();
    }
    
    public GridCoordinate(long newKey)
    {
    	key = newKey;
    	computeIndices();
    	keyTempo = (long)t;
    	keySpace = key - keyTempo;
    }

    public static long computeKey(int ii, int jj, int kk, int tt)
    {
    	long I = (long)ii;
    	long J = (long)jj;
    	long K = (long)kk;
    	long T = (long)tt;
    	
    	return K * Constantes.NB_MULT_K + J * Constantes.NB_MULT_J + I * Constantes.NB_MULT_I + T;	    
    }

    public void computeKey()
    {
		long I = (long)i;
		long J = (long)j;
		long K = (long)k;
		long T = (long)t;
		
	    keySpace = K * Constantes.NB_MULT_K + J * Constantes.NB_MULT_J + I * Constantes.NB_MULT_I;
	    keyTempo = T;
	    key      = keySpace + keyTempo;	
    }

    public void computeIndices()
    {
    	long buffer = key;
    	k = (int)(key / Constantes.NB_MULT_K);
		buffer = buffer - k * Constantes.NB_MULT_K;
		j = (int)(buffer / Constantes.NB_MULT_J);
		buffer = buffer - j * Constantes.NB_MULT_J;
		i = (int)(buffer / Constantes.NB_MULT_I);
		t = (int)(buffer - i * Constantes.NB_MULT_I);	
    }

    public void spaceUpdateKey()
    {
    	long I = (long)i;
    	long J = (long)j;
    	long K = (long)k;
    	
    	keySpace = K * Constantes.NB_MULT_K + J * Constantes.NB_MULT_J + I * Constantes.NB_MULT_I;
	    key      = keySpace + keyTempo;	    
    }
    
    public void tempoUpdateKey()
    {
    	keyTempo = (long)t;
    	key      = keySpace + keyTempo;    	
    }

    public void copierGridCoorinate(GridCoordinate in)
    {
    	i        = in.i;
    	j        = in.j;
    	k        = in.k;
    	t        = in.t;
    	key      = in.key;
    	keySpace = in.keySpace;
    	keyTempo = in.keyTempo;
    }
    
    public static void copierGridCoorinate(GridCoordinate in, GridCoordinate out)
    {
    	out.copierGridCoorinate(in);    	
    }   
    
    public long getKey()
    {
    	return key;
    }
    
    public long getKeyTempo()
    {
    	return keyTempo;
    }
    
    public long getKeySpace()
    {
    	return keySpace;
    }
    
    public int getI()
    {
    	return i;
    }
    
    public int getJ()
    {
    	return j;
    }
    
    public int getK()
    {
    	return k;
    }
    
    public int getT()
    {
    	return t;
    }
}