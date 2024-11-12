#!/usr/bin/env perl
use strict;
use warnings;
use utf8;

my ($N, $ITER) = @ARGV;

if (not defined $N) {
  die "Usage: $0 <N> <ITER>\n";
}

my $DEBUG = 0;
my $CMD = "./exe-ICC $N $ITER 2> /dev/null";
my $NM = 4;
my $NPM = 18;
print "# N:$N IT:$ITER\n";

foreach my $np ( 1 ... $NM ) {
	$np *= $NPM;
	if ( $DEBUG ) { print "likwid-mpirun -mpi slurm  -n $np  -nperdomain M:$NPM  $CMD\n"; }
	my $result = `likwid-mpirun --mpiopts  \"--cpu-freq=2000000-2000000:performance\" -mpi slurm  -n $np  -nperdomain M:$NPM  $CMD`;
	if ( $DEBUG ) { print $result; }
	my @col = split(' ', $result);
	my $p = $col[2];

	print "$np $p\n";
}
