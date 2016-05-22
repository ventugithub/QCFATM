Dear Tobias,

Here are the files with code for conflict detection. I removed everything
that was not related to conflict detection directly.
 
You should start to look at file Etat.jave ("etat" means "state” in
french, some of my functions have strange french names…). It has just one
function calculCriterSimple that uses the GridAL class from GridAL.java
file to calculate the value of criteria of optimization (in this case -
simply the number of conflicts).

First we put all points to the grid (grid.putAllTrajectories)
and next we calculate the conflicts between them (grid.evaluateFullTrajSet)

Function evaluateFullTrajSet adresses in its turn to the class
GeoPoint4D.java, to verify if a pair of particular points is in conflict.

I hope that could help you. If you have any question, don’t hesitate.

Best regards,
Olga




