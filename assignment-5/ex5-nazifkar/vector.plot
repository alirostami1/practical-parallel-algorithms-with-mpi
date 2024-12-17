set terminal png size 1800,768 enhanced font ,12
set output 'velocity.png'
set datafile separator whitespace

plot 'velocity.dat' using 1:2:3:4:5 with vectors filled head size 0.01,20,60 lc palette
