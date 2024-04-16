# file: fit_csv.gnuplot
#set terminal pdfcairo
#set output 'plot.pdf'

set datafile separator ','

d=0

f(x) = A*exp(-((x-u)/r)**2/2)

A=1000
u=140
r=35

fit f(x) 'confirmados_fecha_2.csv' using 1:3 every :::0::d via A,u,r

set xrange [0:400]

plot 'confirmados_fecha_2.csv' using 1:3, 'confirmados_fecha_2.csv' every :::0::d using 1:3, f(x)


#unset terminal