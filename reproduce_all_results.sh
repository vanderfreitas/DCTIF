
# timeseries generation
echo 'TIMESERIES GENERATION...'
cd timeseries_generation
python timeseries_generation_henon_map.py
python timeseries_generation_logistic_map.py
python lyapunov_henon.py
python lyapunov_logistic.py



################### DCSD Algorithm ##############

# DCSD computation
echo 'DCSD COMPUTATION...'
cd ../DCSD

echo '    HENON MAP...'
python DCSD_Henon.py
echo '    LOGISTIC MAP...'
python DCSD_Logistic.py

# DCSD Figures
echo '    PLOTTING LOGISTIC MAP...'
cd Logistic
python plot_bifurcation_diagram.py
python plot_network_measures.py

echo '    PLOTTING HENON MAP...'
cd ../Henon
python plot_bifurcation_diagram.py
python plot_network_measures.py




################### DCTIF Algorithm ##############

# DCTIF computation
echo 'DCTIF COMPUTATION...'
cd ../../DCTIF

echo '    HENON MAP...'
python DCTIF_Henon.py
echo '    LOGISTIC MAP...'
python DCTIF_Logistic.py

# DCTIF Figures
echo '    PLOTTING LOGISTIC MAP...'
cd Logistic
python DCTIF_logistic_plot.py

echo '    PLOTTING HENON MAP...'
cd ../Henon
python DCTIF_henon_plot.py



echo 'ALL DONE!'
